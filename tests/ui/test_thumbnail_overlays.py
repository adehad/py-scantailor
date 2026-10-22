"""Each stage's result is drawn over its thumbnails, and cached with them."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui

from scantailor.app.ui.page_renderer import render_thumbnail
from scantailor.app.ui.thumbnails.overlays import paint_stage_overlay, view_stage
from scantailor.app.ui.thumbnails.render_memo import ThumbnailMemo
from scantailor.core import pipeline
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, Margins, PageId, SubPage
from scantailor.core.page_geometry import build_transformation
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.filters.select_content.params import PageDetectionMode

WIDTH, HEIGHT = 400, 300
DPI = 300.0
SOURCE = ((WIDTH, HEIGHT), DPI)
PAGE = PageId(image_id=ImageId(file_path=Path("spread.png")))


def _blank(width: int = WIDTH // 2, height: int = HEIGHT // 2) -> QtGui.QImage:
    image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_Grayscale8)
    image.fill(QtGui.QColor("white"))
    return image


def _pixels(image: QtGui.QImage) -> NDArray[np.int_]:
    rgb = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    array = np.frombuffer(rgb.constBits(), np.uint8, rgb.sizeInBytes())
    rows = array.reshape(rgb.height(), rgb.bytesPerLine())
    return rows[:, : rgb.width() * 3].reshape(rgb.height(), rgb.width(), 3).astype(int)


def _split(stages: StageSequence, page: PageId = PAGE) -> None:
    stages.page_split_filter.set_manual_split(page, WIDTH / 2, WIDTH, HEIGHT)


def _is_white(pixel: NDArray[np.int_]) -> bool:
    return bool((pixel > 245).all())


class TestStagesWithoutAnOverlay:
    @pytest.mark.parametrize("stage", [FilterStage.FIX_ORIENTATION, FilterStage.OUTPUT])
    def test_the_thumbnail_comes_back_untouched(self, stage):
        thumbnail = _blank()

        overlaid = paint_stage_overlay(thumbnail, stage, PAGE, StageSequence(), SOURCE)

        assert overlaid is thumbnail

    def test_a_page_without_a_split_layout_gets_no_split_overlay(self):
        thumbnail = _blank()

        overlaid = paint_stage_overlay(
            thumbnail, FilterStage.PAGE_SPLIT, PAGE, StageSequence(), SOURCE
        )

        assert overlaid is thumbnail

    def test_the_input_thumbnail_is_never_drawn_on(self):
        stages = StageSequence()
        _split(stages)
        thumbnail = _blank()

        paint_stage_overlay(thumbnail, FilterStage.PAGE_SPLIT, PAGE, stages, SOURCE)

        assert (_pixels(thumbnail) == 255).all()


class TestPageSplit:
    def test_the_two_pages_get_their_own_tints(self):
        stages = StageSequence()
        _split(stages)

        pixels = _pixels(
            paint_stage_overlay(_blank(), FilterStage.PAGE_SPLIT, PAGE, stages, SOURCE)
        )

        left, right = pixels[75, 40], pixels[75, 160]
        assert left[1] > left[0] and left[1] > left[2]
        assert right[2] > right[0] and right[2] > right[1]

    def test_the_cutter_is_drawn_down_the_middle(self):
        stages = StageSequence()
        _split(stages)

        pixels = _pixels(
            paint_stage_overlay(_blank(), FilterStage.PAGE_SPLIT, PAGE, stages, SOURCE)
        )

        column = pixels[20:130, 99:101]
        assert (column[..., 2] - column[..., 0] > 80).all()

    def test_a_removed_half_gets_the_offcut_tint(self):
        stages = StageSequence()
        _split(stages)

        pixels = _pixels(
            paint_stage_overlay(
                _blank(),
                FilterStage.PAGE_SPLIT,
                PAGE,
                stages,
                SOURCE,
                removed_halves=(True, False),
            )
        )

        left = pixels[75, 40]
        assert left[0] > left[1] and left[0] > left[2]


class TestDeskew:
    def test_a_grid_is_laid_over_the_page(self):
        pixels = _pixels(
            paint_stage_overlay(
                _blank(), FilterStage.DESKEW, PAGE, StageSequence(), SOURCE
            )
        )

        lines_across = [
            y for y in range(pixels.shape[0]) if not _is_white(pixels[y, 3])
        ]
        assert len(lines_across) >= 5
        assert _is_white(pixels[lines_across[0] + 3, 3])


class TestSelectContent:
    def test_the_content_box_is_tinted_and_the_rest_left_alone(self):
        stages = StageSequence()
        stages.select_content_filter.set_manual_content(
            PAGE, ContentBox(x=100, y=100, width=100, height=50)
        )

        pixels = _pixels(
            paint_stage_overlay(
                _blank(), FilterStage.SELECT_CONTENT, PAGE, stages, SOURCE
            )
        )

        inside = pixels[62, 75]
        assert inside[2] > inside[0] + 20
        assert _is_white(pixels[10, 10])
        assert _is_white(pixels[120, 150])

    def test_the_page_box_is_outlined_only_when_page_detection_is_on(self):
        stages = StageSequence()
        filter_ = stages.select_content_filter
        filter_.set_manual_content(PAGE, ContentBox(x=100, y=100, width=100, height=50))
        filter_.set_params(
            PAGE,
            filter_.get_params(PAGE).model_copy(
                update={
                    "page_box": PageBox(x=20, y=20, width=360, height=260),
                    "page_detection_mode": PageDetectionMode.MANUAL,
                }
            ),
        )
        with_page = _pixels(
            paint_stage_overlay(
                _blank(), FilterStage.SELECT_CONTENT, PAGE, stages, SOURCE
            )
        )
        filter_.set_params(
            PAGE,
            filter_.get_params(PAGE).model_copy(
                update={"page_detection_mode": PageDetectionMode.DISABLED}
            ),
        )
        without_page = _pixels(
            paint_stage_overlay(
                _blank(), FilterStage.SELECT_CONTENT, PAGE, stages, SOURCE
            )
        )

        edge = with_page[75, 9:12]
        assert (edge[:, 0] > edge[:, 2] + 40).any()
        assert all(_is_white(pixel) for pixel in without_page[75, 9:12])


def _laid_out(aligned: bool) -> tuple[StageSequence, QtGui.QImage]:
    """Lay the page out with 10 mm margins; return the stages and a blank view."""
    stages = StageSequence()
    stages.select_content_filter.set_manual_content(
        PAGE, ContentBox(x=100, y=80, width=200, height=140)
    )
    layout = stages.page_layout_filter
    layout.set_margins(PAGE, Margins.uniform(10.0))
    alignment = Alignment.centered().model_copy(update={"is_null": not aligned})
    layout.set_params(PAGE, layout.get_params(PAGE).with_alignment(alignment))
    pipeline.process_page(
        np.zeros((HEIGHT, WIDTH), np.uint8),
        PAGE,
        stages,
        pipeline.PipelineOptions(
            end_stage=FilterStage.PAGE_LAYOUT, dpi=DPI, render_image=False
        ),
    )
    rect = build_transformation(
        stages, PAGE, (WIDTH, HEIGHT), DPI, FilterStage.PAGE_LAYOUT
    ).resulting_rect
    return stages, _blank(round(rect.width), round(rect.height))


class TestMargins:
    def test_the_margins_are_tinted_and_the_content_left_clear(self):
        stages, view = _laid_out(aligned=True)

        pixels = _pixels(
            paint_stage_overlay(view, FilterStage.PAGE_LAYOUT, PAGE, stages, SOURCE)
        )

        height, width = pixels.shape[:2]
        assert _is_white(pixels[height // 2, width // 2])
        assert not _is_white(pixels[4, width // 2])
        assert not _is_white(pixels[height // 2, 4])

    def test_a_page_not_matched_to_the_others_has_another_colour(self):
        aligned_stages, aligned_view = _laid_out(aligned=True)
        own_stages, own_view = _laid_out(aligned=False)

        aligned = _pixels(
            paint_stage_overlay(
                aligned_view, FilterStage.PAGE_LAYOUT, PAGE, aligned_stages, SOURCE
            )
        )[4, 30]
        own = _pixels(
            paint_stage_overlay(
                own_view, FilterStage.PAGE_LAYOUT, PAGE, own_stages, SOURCE
            )
        )[4, 30]

        assert aligned[0] > aligned[1]
        assert own[2] > own[0]


def test_page_split_thumbnails_show_the_whole_image():
    assert view_stage(FilterStage.PAGE_SPLIT) == FilterStage.FIX_ORIENTATION
    assert view_stage(FilterStage.DESKEW) == FilterStage.DESKEW


class TestRenderedThumbnails:
    @pytest.fixture
    def spread(self, tmp_path: Path) -> ImageId:
        path = tmp_path / "spread.tif"
        pixels = np.full((HEIGHT, WIDTH, 3), 235, dtype=np.uint8)
        for x in (60, 260):
            cv2.rectangle(pixels, (x, 60), (x + 80, 240), (40, 40, 40), -1)
        write_image(path, pixels, Dpi.uniform(round(DPI)))
        return ImageId(file_path=path)

    @pytest.mark.parametrize("sub_page", [SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE])
    def test_each_page_of_a_spread_shows_the_whole_spread_at_page_split(
        self, spread, sub_page
    ):
        stages = StageSequence()
        page = PageId(image_id=spread, sub_page=sub_page)
        _split(stages, page)

        thumbnail = render_thumbnail(
            page,
            stages,
            FilterStage.PAGE_SPLIT,
            QtCore.QSize(100, 100),
            memo=ThumbnailMemo(),
        )

        assert (thumbnail.width(), thumbnail.height()) == (100, 75)

    def test_a_removed_half_makes_another_thumbnail(self, spread):
        stages, memo = StageSequence(), ThumbnailMemo()
        page = PageId(image_id=spread, sub_page=SubPage.RIGHT_PAGE)
        _split(stages, page)
        box = QtCore.QSize(100, 100)

        kept = render_thumbnail(page, stages, FilterStage.PAGE_SPLIT, box, memo=memo)
        removed = render_thumbnail(
            page,
            stages,
            FilterStage.PAGE_SPLIT,
            box,
            memo=memo,
            removed_halves=(True, False),
        )

        assert kept != removed
        assert (
            render_thumbnail(page, stages, FilterStage.PAGE_SPLIT, box, memo=memo)
            is kept
        )

    def test_the_overlay_is_kept_with_the_cached_thumbnail(self, spread):
        stages, memo = StageSequence(), ThumbnailMemo()
        page = PageId(image_id=spread, sub_page=SubPage.LEFT_PAGE)
        _split(stages, page)
        box = QtCore.QSize(100, 100)

        first = render_thumbnail(page, stages, FilterStage.PAGE_SPLIT, box, memo=memo)
        again = render_thumbnail(page, stages, FilterStage.PAGE_SPLIT, box, memo=memo)

        assert again is first
        left = _pixels(first)[37, 5]
        assert left[1] > left[0]
