"""Scans whose horizontal and vertical resolutions differ are processed square.

Each test scans the same generated page twice: once at 300 dpi on both axes,
once at 300 dpi across and 150 dpi down, as a scanner with a coarse vertical
step would. Both must give the same physical sizes and the same output.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.models import Dpi, ImageId, OrthogonalRotation, PageId
from scantailor.core.page_geometry import build_transformation, view_dpi
from scantailor.core.pipeline import PipelineOptions, PipelineResult, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.split_finder import detect_split

PAGE_DPI = 300
OUTPUT_DPI = Dpi.uniform(300)
SQUARE = Dpi.uniform(PAGE_DPI)
UNEQUAL = Dpi(horizontal=PAGE_DPI, vertical=PAGE_DPI // 2)


def straight_text_page() -> NDArray[np.uint8]:
    """A 6 x 8 inch page at ``PAGE_DPI`` with a block of text clear of its edges."""
    page = np.full((2400, 1800), 235, dtype=np.uint8)
    for baseline in range(300, 2100, 72):
        cv2.putText(
            page,
            "The quick brown fox jumps over the lazy dog",
            (225, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            20,
            4,
            cv2.LINE_AA,
        )
    return page


def _page_id(name: str) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(f"{name}.png")))


def _squash(page: NDArray[np.uint8], x_factor: float, y_factor: float) -> NDArray:
    height, width = page.shape[:2]
    size = (round(width * x_factor), round(height * y_factor))
    return np.asarray(cv2.resize(page, size, interpolation=cv2.INTER_AREA))


def _scan_at_unequal_dpi(page: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return a page scanned at ``SQUARE`` as if scanned at ``UNEQUAL``."""
    return _squash(page, 1.0, UNEQUAL.vertical / SQUARE.vertical)


def _run(
    image: NDArray[np.uint8],
    dpi: Dpi,
    name: str,
    end_stage: FilterStage = FilterStage.OUTPUT,
    rotation: OrthogonalRotation | None = None,
) -> tuple[PipelineResult, StageSequence, PageId]:
    page = _page_id(name)
    sequence = StageSequence()
    sequence.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    if rotation is not None:
        sequence.fix_orientation_filter.set_rotation(page.image_id, rotation)
    output = sequence.output_filter.get_params(page)
    sequence.output_filter.set_params(page, output.with_output_dpi(OUTPUT_DPI))
    result = process_page(
        image, page, sequence, PipelineOptions(end_stage=end_stage, dpi=dpi)
    )
    return result, sequence, page


def _ink_size(image: NDArray[np.uint8]) -> tuple[int, int]:
    """Width and height of the box around the dark pixels of an output page."""
    columns = np.flatnonzero((image < 128).any(axis=0))
    rows = np.flatnonzero((image < 128).any(axis=1))
    return int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1)


class TestPhysicalSize:
    def test_the_content_size_in_millimetres_matches_a_square_pixel_scan(
        self,
    ) -> None:
        page = straight_text_page()
        _, square_stages, square_page = _run(
            page, SQUARE, "square", FilterStage.SELECT_CONTENT
        )
        _, unequal_stages, unequal_page = _run(
            _scan_at_unequal_dpi(page), UNEQUAL, "unequal", FilterStage.SELECT_CONTENT
        )

        square = square_stages.select_content_filter.get_params(square_page)
        unequal = unequal_stages.select_content_filter.get_params(unequal_page)

        assert square.content_size_mm.width_mm > 100
        assert unequal.content_size_mm.width_mm == pytest.approx(
            square.content_size_mm.width_mm, abs=1.0
        )
        assert unequal.content_size_mm.height_mm == pytest.approx(
            square.content_size_mm.height_mm, abs=1.0
        )

    def test_the_content_box_is_in_square_pixels_at_the_smaller_resolution(
        self,
    ) -> None:
        page = straight_text_page()
        _, square_stages, square_page = _run(
            page, SQUARE, "square", FilterStage.SELECT_CONTENT
        )
        _, unequal_stages, unequal_page = _run(
            _scan_at_unequal_dpi(page), UNEQUAL, "unequal", FilterStage.SELECT_CONTENT
        )

        square = square_stages.select_content_filter.get_params(square_page)
        unequal = unequal_stages.select_content_filter.get_params(unequal_page)

        half = UNEQUAL.vertical / SQUARE.vertical
        assert unequal.content_box.width == pytest.approx(
            square.content_box.width * half, abs=2.0
        )
        assert unequal.content_box.height == pytest.approx(
            square.content_box.height * half, abs=2.0
        )

    def test_the_view_resolution_is_the_smaller_axis(self) -> None:
        assert view_dpi(UNEQUAL) == UNEQUAL.vertical
        assert view_dpi(300.0) == 300.0


class TestOutput:
    def test_the_output_page_has_the_size_of_a_square_pixel_scan(self) -> None:
        page = straight_text_page()

        square, _, _ = _run(page, SQUARE, "square")
        unequal, _, _ = _run(_scan_at_unequal_dpi(page), UNEQUAL, "unequal")

        assert square.dpi == unequal.dpi
        height, width = square.image.shape[:2]
        assert unequal.image.shape[1] == pytest.approx(width, rel=0.01)
        assert unequal.image.shape[0] == pytest.approx(height, rel=0.01)

    def test_the_text_keeps_its_proportions(self) -> None:
        page = straight_text_page()

        square, _, _ = _run(page, SQUARE, "square")
        unequal, _, _ = _run(_scan_at_unequal_dpi(page), UNEQUAL, "unequal")

        square_width, square_height = _ink_size(square.image)
        unequal_width, unequal_height = _ink_size(unequal.image)
        assert unequal_width == pytest.approx(square_width, rel=0.01)
        assert unequal_height == pytest.approx(square_height, rel=0.01)

    def test_a_turned_scan_swaps_the_resolutions_with_its_axes(self) -> None:
        page = straight_text_page()
        # Scanned sideways: the page's width runs down the file at 150 dpi.
        sideways = _squash(np.rot90(page, 1), 1.0, 0.5)

        square, _, _ = _run(page, SQUARE, "square")
        turned, _, _ = _run(
            sideways,
            UNEQUAL,
            "turned",
            rotation=OrthogonalRotation(degrees=270),
        )

        height, width = square.image.shape[:2]
        assert turned.image.shape[1] == pytest.approx(width, rel=0.01)
        assert turned.image.shape[0] == pytest.approx(height, rel=0.01)


class TestPageGeometry:
    @pytest.mark.parametrize(
        "rotation",
        [OrthogonalRotation(degrees=0), OrthogonalRotation(degrees=90)],
        ids=["upright", "turned"],
    )
    def test_the_views_rebuild_the_transformation_the_pipeline_used(
        self, rotation: OrthogonalRotation
    ) -> None:
        image = _scan_at_unequal_dpi(straight_text_page())
        result, sequence, page = _run(
            image, UNEQUAL, "unequal", FilterStage.SELECT_CONTENT, rotation
        )
        height, width = image.shape[:2]
        oriented = rotation.rotate_dimensions(width, height)
        oriented_size = (round(oriented[0]), round(oriented[1]))

        rebuilt = build_transformation(
            sequence, page, oriented_size, UNEQUAL, FilterStage.DESKEW
        )

        assert result.transformation is not None
        np.testing.assert_allclose(
            rebuilt.transform, result.transformation.transform, atol=1e-9
        )


class TestPageSplit:
    def test_the_split_line_of_a_spread_lands_between_the_pages(self) -> None:
        single = straight_text_page()
        spread = np.hstack([single, single])
        square = detect_split(spread, LayoutType.TWO_PAGES, SQUARE)

        unequal = detect_split(
            _scan_at_unequal_dpi(spread), LayoutType.TWO_PAGES, UNEQUAL
        )

        square_x = square.layout.get_split_line_x()
        unequal_x = unequal.layout.get_split_line_x()
        assert square_x is not None and unequal_x is not None
        assert unequal_x == pytest.approx(square_x, abs=0.01 * spread.shape[1])

    def test_the_layout_outline_is_the_whole_unequal_image(self) -> None:
        image = _scan_at_unequal_dpi(np.hstack([straight_text_page()] * 2))

        result = detect_split(image, LayoutType.TWO_PAGES, UNEQUAL)

        height, width = image.shape
        assert result.layout.outline == pytest.approx((0, 0, width, height))


class TestPostScale:
    def test_a_quarter_turn_scales_each_axis_by_its_own_resolution(self) -> None:
        # 2 x 2 inches: 600 px at 300 dpi across, 300 px at 150 dpi down.
        xform = ImageTransformation.from_size(
            600, 300, Dpi(horizontal=300, vertical=150)
        )
        xform.set_pre_rotation(OrthogonalRotation(degrees=90))

        xform.post_scale_to_dpi(Dpi.uniform(300))

        rect = xform.resulting_rect
        assert rect.width == pytest.approx(600)
        assert rect.height == pytest.approx(600)

    def test_square_pixels_double_at_double_the_resolution(self) -> None:
        xform = ImageTransformation.from_size(600, 300, Dpi.uniform(300))
        xform.set_post_rotation(10.0)

        xform.post_scale_to_dpi(Dpi.uniform(600))

        at_source = ImageTransformation.from_size(600, 300, Dpi.uniform(300))
        at_source.set_post_rotation(10.0)
        assert xform.resulting_rect.width == pytest.approx(
            at_source.resulting_rect.width * 2
        )
        assert xform.resulting_rect.height == pytest.approx(
            at_source.resulting_rect.height * 2
        )
