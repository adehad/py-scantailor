"""Split detection on the region a manual trim keeps and with a stored threshold."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.page_split.dependencies import Dependencies
from scantailor.filters.page_split.filter import Filter
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayoutType
from scantailor.filters.page_split.page_layout_estimator import estimate_page_layout
from scantailor.filters.page_split.split_finder import detect_split

from . import scan_pages as pages

SPREAD_WIDTH = 1500
HEIGHT = 1000
BED = 1100


def _page_id() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("/trim.tiff")), sub_page=SubPage.SINGLE_PAGE
    )


def _spread_beside_a_scanner_bed() -> NDArray[np.uint8]:
    """A clean spread with 1100 columns of unrelated text to its left."""
    bed = pages.blank(BED, HEIGHT)
    pages.add_text(bed, 20, BED - 20, 100, HEIGHT - 100, seed=11)
    return np.hstack([bed, pages.two_page_spread(SPREAD_WIDTH, HEIGHT)])


def _two_text_columns() -> NDArray[np.uint8]:
    page = pages.blank(1500, 1000)
    pages.add_text(page, 110, 500, 100, 900, seed=1)
    pages.add_text(page, 800, 1390, 100, 900, seed=2)
    return page


def _trim_polygon(x: int, y: int, width: int, height: int) -> NDArray[np.float64]:
    return np.array(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float64,
    )


class TestRegion:
    def test_the_split_is_found_in_the_region_and_reported_in_the_whole_image(
        self,
    ) -> None:
        image = _spread_beside_a_scanner_bed()

        layout = estimate_page_layout(
            image,
            LayoutType.TWO_PAGES,
            300.0,
            region=(BED, 0, SPREAD_WIDTH, HEIGHT),
        )

        assert layout.get_split_line_x() == pytest.approx(BED + 750.0, abs=1.0)
        assert layout.outline == (0.0, 0.0, float(image.shape[1]), float(HEIGHT))

    def test_the_whole_image_is_analysed_without_a_region(self) -> None:
        image = _spread_beside_a_scanner_bed()

        layout = estimate_page_layout(image, LayoutType.TWO_PAGES, 300.0)

        assert layout.get_split_line_x() != pytest.approx(BED + 750.0, abs=50.0)

    def test_the_cutters_span_the_height_of_the_whole_image(self) -> None:
        image = _spread_beside_a_scanner_bed()

        layout = estimate_page_layout(
            image,
            LayoutType.TWO_PAGES,
            300.0,
            region=(BED, 100, SPREAD_WIDTH, HEIGHT - 200),
        )

        assert layout.cutter1 is not None
        assert layout.cutter1[0][1] == 0.0
        assert layout.cutter1[1][1] == float(HEIGHT)

    def test_a_region_agrees_with_the_cropped_image_shifted_back(self) -> None:
        image = _spread_beside_a_scanner_bed()

        in_region = detect_split(
            image, LayoutType.TWO_PAGES, 300.0, region=(BED, 0, SPREAD_WIDTH, HEIGHT)
        ).layout
        cropped = detect_split(image[:, BED:], LayoutType.TWO_PAGES, 300.0).layout

        assert cropped.get_split_line_x() is not None
        assert in_region.get_split_line_x() == pytest.approx(
            cropped.get_split_line_x() + BED
        )

    def test_a_region_beyond_the_image_is_clipped(self) -> None:
        image = _spread_beside_a_scanner_bed()

        layout = estimate_page_layout(
            image,
            LayoutType.TWO_PAGES,
            300.0,
            region=(BED, -50, 10 * SPREAD_WIDTH, 2 * HEIGHT),
        )

        assert layout.get_split_line_x() == pytest.approx(BED + 750.0, abs=1.0)

    def test_an_uncut_request_ignores_the_region(self) -> None:
        image = _spread_beside_a_scanner_bed()

        layout = estimate_page_layout(
            image, LayoutType.SINGLE_PAGE_UNCUT, 300.0, region=(BED, 0, 300, 300)
        )

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT
        assert layout.outline == (0.0, 0.0, float(image.shape[1]), float(HEIGHT))

    def test_an_offcut_is_cut_off_inside_the_region(self) -> None:
        wide = np.hstack(
            [
                pages.blank(300, 1400),
                pages.single_page_with_neighbours_letters_on_left_edge(),
            ]
        )

        layout = estimate_page_layout(
            wide, LayoutType.AUTO_LAYOUT_TYPE, 300.0, region=(300, 0, 1000, 1400)
        )

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter1 is not None
        assert 300.0 + 70.0 < 0.5 * (layout.cutter1[0][0] + layout.cutter1[1][0])


class TestStoredThreshold:
    def test_a_stored_threshold_replaces_otsus_in_the_filter(self) -> None:
        filter_ = Filter()
        page_id = _page_id()
        filter_.set_layout_type(page_id, LayoutType.TWO_PAGES)
        page = _two_text_columns()

        by_default = filter_.detect_split(page, page_id, store_result=False)
        # Nothing is darker than 10, so there is no content and the split falls
        # back to the middle of the image.
        stored = filter_.detect_split(
            page, page_id, store_result=False, bw_threshold=10
        )

        assert by_default.layout.get_split_line_x() == pytest.approx(650.0, abs=1.0)
        assert stored.layout.get_split_line_x() == pytest.approx(750.0)

    def test_process_passes_the_threshold_on(self) -> None:
        filter_ = Filter()
        page_id = _page_id()
        filter_.set_layout_type(page_id, LayoutType.TWO_PAGES)

        params = filter_.process(_two_text_columns(), page_id, bw_threshold=10)

        assert params.page_layout is not None
        assert params.page_layout.get_split_line_x() == pytest.approx(750.0)


class TestFilterWithTrim:
    def test_the_trimmed_region_is_what_is_analysed(self) -> None:
        image = _spread_beside_a_scanner_bed()
        filter_ = Filter()
        page_id = _page_id()

        params = filter_.process(
            image, page_id, trim_area=_trim_polygon(BED, 0, SPREAD_WIDTH, HEIGHT)
        )

        assert params.page_layout is not None
        assert params.page_layout.get_split_line_x() == pytest.approx(
            BED + 750.0, abs=1.0
        )
        assert params.dependencies.trim == (BED, 0, SPREAD_WIDTH, HEIGHT)

    def test_a_trim_covering_the_whole_image_is_no_trim(self) -> None:
        image = pages.two_page_spread()
        filter_ = Filter()
        page_id = _page_id()

        params = filter_.process(
            image, page_id, trim_area=_trim_polygon(0, 0, 1500, 1000)
        )

        assert params.dependencies.trim is None

    def test_a_changed_trim_detects_again(self) -> None:
        image = _spread_beside_a_scanner_bed()
        filter_ = Filter()
        page_id = _page_id()
        filter_.process(
            image, page_id, trim_area=_trim_polygon(BED, 0, SPREAD_WIDTH, HEIGHT)
        )

        params = filter_.process(
            image, page_id, trim_area=_trim_polygon(BED + 100, 0, 1400, HEIGHT)
        )

        assert params.dependencies.trim == (BED + 100, 0, 1400, HEIGHT)
        assert params.page_layout is not None
        assert params.page_layout.get_split_line_x() == pytest.approx(
            BED + 750.0, abs=5.0
        )

    def test_an_unchanged_trim_keeps_the_stored_layout(self) -> None:
        image = _spread_beside_a_scanner_bed()
        filter_ = Filter()
        page_id = _page_id()
        trim = _trim_polygon(BED, 0, SPREAD_WIDTH, HEIGHT)
        first = filter_.process(image, page_id, trim_area=trim)

        second = filter_.process(image, page_id, trim_area=trim)

        assert second == first


class TestDependenciesWithTrim:
    TRIM = (10, 20, 300, 400)

    def _known(self, trim: tuple[int, int, int, int] | None) -> Dependencies:
        return Dependencies(
            image_size=(500, 600), layout_type=LayoutType.AUTO_LAYOUT_TYPE, trim=trim
        )

    def test_the_same_trim_is_compatible(self) -> None:
        assert self._known(self.TRIM).is_compatible_with(
            self._known(self.TRIM), manual_split_line=False
        )

    def test_a_different_trim_is_incompatible_for_a_detected_layout(self) -> None:
        assert not self._known(self.TRIM).is_compatible_with(
            self._known(None), manual_split_line=False
        )
        assert not self._known(None).is_compatible_with(
            self._known(self.TRIM), manual_split_line=False
        )

    def test_a_layout_placed_by_hand_survives_a_changed_trim(self) -> None:
        assert self._known(None).is_compatible_with(
            self._known(self.TRIM), manual_split_line=True
        )

    def test_projects_stored_without_a_trim_load_as_untrimmed(self) -> None:
        stored = Dependencies.model_validate(
            {"image_size": (500, 600), "rotation_degrees": 0}
        )

        assert stored.trim is None
