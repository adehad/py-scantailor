"""Tests that pipeline stages crop, rotate and pad the image."""

import math
from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, Margins, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.settings import Settings as OutputSettings
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.params import ContentSize
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout
from scantailor.filters.select_content.content_box import ContentBox

# 254 dpi makes one millimetre exactly ten pixels.
DPI = 254.0
PX_PER_MM = 10

SPREAD_WIDTH = 400
SPREAD_HEIGHT = 300
GUTTER_X = 200

LEFT_BACKGROUND = 200
RIGHT_BACKGROUND = 120
LEFT_INK = 50
RIGHT_INK = 10

# (x, y, width, height) of the ink block on the left page, in source pixels.
LEFT_BLOCK = (40, 60, 110, 140)


@pytest.fixture(autouse=True)
def _output_at_source_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep output pixels one to one with the source so sizes stay in millimetres."""
    monkeypatch.setattr(OutputSettings, "DEFAULT_DPI", int(DPI))


def _page_id(sub_page: SubPage, name: str = "spread.tiff") -> PageId:
    return PageId(image_id=ImageId(file_path=Path(name)), sub_page=sub_page)


def _spread() -> np.ndarray:
    image = np.full((SPREAD_HEIGHT, SPREAD_WIDTH), LEFT_BACKGROUND, dtype=np.uint8)
    image[:, GUTTER_X:] = RIGHT_BACKGROUND
    x, y, w, h = LEFT_BLOCK
    image[y : y + h, x : x + w] = LEFT_INK
    image[y : y + h, GUTTER_X + x : GUTTER_X + x + w] = RIGHT_INK
    return image


def _configure_spread(sequence: StageSequence, sub_page: SubPage) -> PageId:
    """Split at the gutter, no skew, content fixed to the left block."""
    for side in (SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE):
        sequence.page_split_filter.set_manual_split(
            _page_id(side), GUTTER_X, SPREAD_WIDTH, SPREAD_HEIGHT
        )
    page_id = _page_id(sub_page)
    sequence.deskew_filter.set_manual_angle(page_id, 0.0)
    x, y, w, h = LEFT_BLOCK
    sequence.select_content_filter.set_manual_content(
        page_id, ContentBox(x=x, y=y, width=w, height=h)
    )
    return page_id


def _run(image: np.ndarray, page_id: PageId, sequence: StageSequence, end: FilterStage):
    options = PipelineOptions(end_stage=end, dpi=DPI)
    return process_page(image, page_id, sequence, options)


class TestPageSplitCrop:
    """Page split selects one half of a spread."""

    def test_left_page_is_cropped_to_gutter(self) -> None:
        """The left sub-page has the gutter's width and the full height."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.LEFT_PAGE)

        result = _run(_spread(), page_id, sequence, FilterStage.PAGE_SPLIT)

        assert result.image.shape == (SPREAD_HEIGHT, GUTTER_X)
        np.testing.assert_array_equal(result.image, _spread()[:, :GUTTER_X])

    def test_right_page_is_cropped_to_the_other_side(self) -> None:
        """The right sub-page holds exactly the pixels right of the gutter."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.RIGHT_PAGE)

        result = _run(_spread(), page_id, sequence, FilterStage.PAGE_SPLIT)

        assert result.image.shape == (SPREAD_HEIGHT, SPREAD_WIDTH - GUTTER_X)
        np.testing.assert_array_equal(result.image, _spread()[:, GUTTER_X:])

    def test_offcut_layout_keeps_only_area_between_cutters(self) -> None:
        """Single page with offcut drops the strips outside the cutters."""
        sequence = StageSequence()
        page_id = _page_id(SubPage.SINGLE_PAGE)
        sequence.page_split_filter.set_manual_cutters(
            page_id, 30, 350, SPREAD_WIDTH, SPREAD_HEIGHT
        )

        result = _run(_spread(), page_id, sequence, FilterStage.PAGE_SPLIT)

        np.testing.assert_array_equal(result.image, _spread()[:, 30:350])

    def test_input_image_is_not_modified(self) -> None:
        """Cropping works on a copy of the caller's array."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.LEFT_PAGE)
        image = _spread()

        _run(image, page_id, sequence, FilterStage.PAGE_LAYOUT)

        np.testing.assert_array_equal(image, _spread())


class TestPageLayoutMargins:
    """Page layout places content inside margins and sets the page size."""

    def test_output_size_is_content_plus_margins(self) -> None:
        """Hard margins in mm grow the page around the content box."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.LEFT_PAGE)
        sequence.page_layout_filter.set_margins(
            page_id, Margins(left=2, right=3, top=1, bottom=4)
        )

        result = _run(_spread(), page_id, sequence, FilterStage.PAGE_LAYOUT)

        _x, _y, w, h = LEFT_BLOCK
        assert result.image.shape == (h + 10 + 40, w + 20 + 30)
        assert result.metadata["PAGE_LAYOUT"]["page_size_px"] == (w + 50, h + 50)

    def test_content_lands_at_the_margin_offset(self) -> None:
        """The ink block sits at (left margin, top margin) with page pixels around."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.LEFT_PAGE)
        sequence.page_layout_filter.set_margins(
            page_id, Margins(left=2, right=3, top=1, bottom=4)
        )

        image = _run(_spread(), page_id, sequence, FilterStage.PAGE_LAYOUT).image

        _x, _y, w, h = LEFT_BLOCK
        left, top = 2 * PX_PER_MM, 1 * PX_PER_MM
        assert np.all(image[top : top + h, left : left + w] == LEFT_INK)
        assert image[0, 0] == LEFT_BACKGROUND
        assert image[-1, -1] == LEFT_BACKGROUND
        ink_pixels = np.count_nonzero(image == LEFT_INK)
        assert ink_pixels == w * h

    def test_margin_past_the_gutter_is_white_not_the_facing_page(self) -> None:
        """A margin reaching beyond the split line shows white, never the other page."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.LEFT_PAGE)
        # Content ends at x=150, so 8 mm reaches x=230, past the gutter at 200.
        sequence.page_layout_filter.set_margins(
            page_id, Margins(left=1, right=8, top=1, bottom=1)
        )

        image = _run(_spread(), page_id, sequence, FilterStage.PAGE_LAYOUT).image

        assert not np.isin(image, [RIGHT_BACKGROUND, RIGHT_INK]).any()
        gutter_column = GUTTER_X - (LEFT_BLOCK[0] - 1 * PX_PER_MM)
        assert np.all(image[:, gutter_column - 1] == LEFT_BACKGROUND)
        assert np.all(image[:, gutter_column:] == 255)

    def test_auto_margins_keep_the_page_size(self) -> None:
        """With auto margins the page rectangle, not the content, sets the size."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.LEFT_PAGE)
        layout = sequence.page_layout_filter
        layout.set_params(page_id, layout.get_params(page_id).with_auto_margins())

        image = _run(_spread(), page_id, sequence, FilterStage.PAGE_LAYOUT).image

        np.testing.assert_array_equal(image, _spread()[:, :GUTTER_X])

    def test_alignment_pads_smaller_page_up_to_the_largest(self) -> None:
        """Soft margins make a smaller page as large as the biggest aligned one."""
        sequence = StageSequence()
        left_id = _configure_spread(sequence, SubPage.LEFT_PAGE)
        other_id = _page_id(SubPage.SINGLE_PAGE, "other.tiff")
        layout = sequence.page_layout_filter
        layout.set_margins(left_id, Margins(left=2, right=3, top=1, bottom=4))
        layout.set_alignment(left_id, Alignment.top_left())

        layout.set_params(
            other_id,
            layout.get_params(other_id).model_copy(
                update={
                    "hard_margins_mm": Margins(),
                    "content_size_mm": ContentSize(width_mm=30.0, height_mm=40.0),
                    "alignment": Alignment.top_left(),
                }
            ),
        )

        image = _run(_spread(), left_id, sequence, FilterStage.PAGE_LAYOUT).image

        # This page is 16 x 19 mm with margins against an aggregate of 30 x 40 mm,
        # so top-left alignment adds 14 mm on the right and 21 mm at the bottom.
        _x, _y, w, h = LEFT_BLOCK
        assert image.shape == (400, 300)
        assert np.all(image[10 : 10 + h, 20 : 20 + w] == LEFT_INK)
        assert image[0, 0] == LEFT_BACKGROUND
        # The soft margins show source pixels while they stay inside the left
        # page (it ends at x=200, y=300) and white beyond it.
        page_right = GUTTER_X - (LEFT_BLOCK[0] - 2 * PX_PER_MM)
        page_bottom = SPREAD_HEIGHT - (LEFT_BLOCK[1] - 1 * PX_PER_MM)
        assert np.all(image[:, page_right:] == 255)
        assert np.all(image[page_bottom:, :] == 255)
        assert image[0, page_right - 1] == LEFT_BACKGROUND
        assert image[page_bottom - 1, 0] == LEFT_BACKGROUND


class TestDeskewRotation:
    """Deskew rotates through the transformation."""

    @staticmethod
    def _barred_page() -> np.ndarray:
        image = np.full((300, 400), 255, dtype=np.uint8)
        image[145:155, 20:380] = 0
        return image

    @staticmethod
    def _bar_centre_row(image: np.ndarray, column: int) -> float:
        rows = np.flatnonzero(image[:, column] < 128)
        return float(rows.mean())

    def test_positive_angle_rotates_clockwise_and_grows_the_canvas(self) -> None:
        """A 3 degree correction tilts a flat bar so its right end sits lower."""
        sequence = StageSequence()
        page_id = _page_id(SubPage.SINGLE_PAGE, "single.tiff")
        sequence.page_split_filter.set_params(
            page_id,
            sequence.page_split_filter.get_params(page_id).with_manual_layout(
                PageLayout.single_page_uncut(400, 300)
            ),
        )
        sequence.deskew_filter.set_manual_angle(page_id, 3.0)
        sequence.select_content_filter.disable_content_detection(page_id)

        image = _run(self._barred_page(), page_id, sequence, FilterStage.DESKEW).image

        theta = math.radians(3.0)
        expected_w = 400 * math.cos(theta) + 300 * math.sin(theta)
        expected_h = 400 * math.sin(theta) + 300 * math.cos(theta)
        assert image.shape[1] == pytest.approx(expected_w, abs=1)
        assert image.shape[0] == pytest.approx(expected_h, abs=1)

        left = self._bar_centre_row(image, 60)
        right = self._bar_centre_row(image, image.shape[1] - 60)
        slope = (right - left) / (image.shape[1] - 120)
        assert slope == pytest.approx(math.tan(theta), abs=0.01)

    def test_corners_exposed_by_rotation_are_white(self) -> None:
        """The wedges that rotation uncovers are filled with white."""
        sequence = StageSequence()
        page_id = _page_id(SubPage.SINGLE_PAGE, "single.tiff")
        sequence.deskew_filter.set_manual_angle(page_id, 5.0)
        dark = np.zeros((300, 400), dtype=np.uint8)

        image = _run(dark, page_id, sequence, FilterStage.DESKEW).image

        assert image[0, 0] == 255
        assert image[-1, -1] == 255
        assert image[image.shape[0] // 2, image.shape[1] // 2] == 0


class TestSinglePagePassThrough:
    """An unsplit page with no skew and no cropping is unchanged."""

    def test_image_comes_back_identical(self) -> None:
        """With nothing to do, the stages leave pixels and size alone."""
        sequence = StageSequence()
        page_id = _page_id(SubPage.SINGLE_PAGE, "single.tiff")
        sequence.page_split_filter.set_layout_type(
            page_id, LayoutType.SINGLE_PAGE_UNCUT
        )
        sequence.deskew_filter.set_manual_angle(page_id, 0.0)
        sequence.select_content_filter.disable_content_detection(page_id)
        sequence.page_layout_filter.set_margins(page_id, Margins())
        image = _spread()

        result = _run(image, page_id, sequence, FilterStage.PAGE_LAYOUT)

        np.testing.assert_array_equal(result.image, image)

    def test_only_deskew_changes_a_single_page(self) -> None:
        """A skewed single page grows to the rotated canvas with white wedges."""
        sequence = StageSequence()
        page_id = _page_id(SubPage.SINGLE_PAGE, "single.tiff")
        sequence.page_split_filter.set_layout_type(
            page_id, LayoutType.SINGLE_PAGE_UNCUT
        )
        sequence.deskew_filter.set_manual_angle(page_id, 2.0)
        sequence.select_content_filter.disable_content_detection(page_id)
        sequence.page_layout_filter.set_margins(page_id, Margins())
        image = _spread()

        result = _run(image, page_id, sequence, FilterStage.PAGE_LAYOUT)

        theta = math.radians(2.0)
        assert result.image.shape[1] == pytest.approx(
            SPREAD_WIDTH * math.cos(theta) + SPREAD_HEIGHT * math.sin(theta), abs=1
        )
        assert result.image.shape[0] == pytest.approx(
            SPREAD_WIDTH * math.sin(theta) + SPREAD_HEIGHT * math.cos(theta), abs=1
        )
        assert result.image[0, 0] == 255
        assert result.image[-1, -1] == 255


class TestFullPipelineOnSpread:
    """The output stage receives the transformed image."""

    def test_output_is_binary_and_sized_like_the_page_layout(self) -> None:
        """Output runs on the margin-padded left page, not on the whole spread."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.LEFT_PAGE)
        sequence.page_layout_filter.set_margins(
            page_id, Margins(left=2, right=3, top=1, bottom=4)
        )

        result = process_page(_spread(), page_id, sequence, PipelineOptions(dpi=DPI))

        _x, _y, w, h = LEFT_BLOCK
        assert result.image.shape[:2] == (h + 50, w + 50)
        assert len(np.unique(result.image)) <= 2

    def test_detect_only_returns_the_untransformed_image(self) -> None:
        """Detect-only reports geometry but hands back the source pixels."""
        sequence = StageSequence()
        page_id = _configure_spread(sequence, SubPage.LEFT_PAGE)
        options = PipelineOptions(dpi=DPI, detect_only=True)

        result = process_page(_spread(), page_id, sequence, options)

        np.testing.assert_array_equal(result.image, _spread())
        assert "page_size_px" in result.metadata["PAGE_LAYOUT"]


def _ruled_page(skew_degrees: float) -> np.ndarray:
    """Text-like rules, rotated counter-clockwise by skew_degrees (cv2 convention)."""
    import cv2

    page = np.full((600, 400), 255, dtype=np.uint8)
    for y in range(100, 500, 30):
        page[y : y + 8, 60:340] = 0
    matrix = cv2.getRotationMatrix2D((200, 300), skew_degrees, 1.0)
    return np.asarray(
        cv2.warpAffine(page, matrix, (400, 600), borderValue=255), dtype=np.uint8
    )


def _rule_sharpness(image: np.ndarray) -> int:
    """Ink count of the densest row; level rules concentrate ink in few rows."""
    return int((image < 128).sum(axis=1).max())


class TestAutoDeskewDirection:
    """Detected skew must be corrected, not doubled."""

    @pytest.mark.parametrize("skew", [3.0, -3.0])
    def test_auto_detected_angle_levels_the_rules(self, skew: float) -> None:
        """Rules end up more level after auto deskew than before."""
        sequence = StageSequence()
        page_id = _page_id(SubPage.SINGLE_PAGE, "skewed.tiff")
        page = _ruled_page(skew)
        options = PipelineOptions(
            start_stage=FilterStage.DESKEW, end_stage=FilterStage.DESKEW, dpi=300.0
        )

        result = process_page(page, page_id, sequence, options)

        assert _rule_sharpness(result.image) > _rule_sharpness(page)

    @pytest.mark.parametrize("correction", [3.0, -3.0])
    def test_manual_angle_levels_the_rules(self, correction: float) -> None:
        """A manual clockwise-positive angle matching the skew levels the rules."""
        sequence = StageSequence()
        page_id = _page_id(SubPage.SINGLE_PAGE, "skewed.tiff")
        sequence.deskew_filter.set_manual_angle(page_id, correction)
        page = _ruled_page(correction)
        options = PipelineOptions(
            start_stage=FilterStage.DESKEW, end_stage=FilterStage.DESKEW, dpi=300.0
        )

        result = process_page(page, page_id, sequence, options)

        assert _rule_sharpness(result.image) > _rule_sharpness(page)
