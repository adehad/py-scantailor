"""Tests for the page split detection entry point."""

import time

import cv2
import numpy as np
import pytest

from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayoutType
from scantailor.filters.page_split.split_finder import SplitResult, detect_split

from . import scan_pages as pages


class TestDetectSplit:
    def test_returns_the_layout_of_the_estimator(self) -> None:
        result = detect_split(pages.two_page_spread())

        assert isinstance(result, SplitResult)
        assert result.layout.layout_type is PageLayoutType.TWO_PAGES
        assert result.layout.get_split_line_x() == 750.0

    def test_a_colour_image_gives_the_same_layout_as_its_gray_version(self) -> None:
        gray = pages.spread_with_folding_line()
        colour = cv2.merge([gray, gray, gray])
        with_alpha = cv2.merge([gray, gray, gray, np.full_like(gray, 255)])

        expected = detect_split(gray).layout

        assert detect_split(colour).layout == expected
        assert detect_split(with_alpha).layout == expected

    def test_honours_the_requested_layout_type(self) -> None:
        page = pages.two_page_spread()

        uncut = detect_split(page, LayoutType.SINGLE_PAGE_UNCUT).layout
        cut = detect_split(page, LayoutType.PAGE_PLUS_OFFCUT).layout

        assert uncut.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT
        assert cut.layout_type is PageLayoutType.SINGLE_PAGE_CUT

    def test_a_blank_image_still_gets_a_layout_of_the_requested_type(self) -> None:
        blank = np.zeros((100, 100), dtype=np.uint8)

        for layout_type, expected in [
            (LayoutType.TWO_PAGES, PageLayoutType.TWO_PAGES),
            (LayoutType.PAGE_PLUS_OFFCUT, PageLayoutType.SINGLE_PAGE_CUT),
            (LayoutType.SINGLE_PAGE_UNCUT, PageLayoutType.SINGLE_PAGE_UNCUT),
        ]:
            assert detect_split(blank, layout_type).layout.layout_type is expected

    def test_the_resolution_sets_the_scale_of_the_analysis(self) -> None:
        page = pages.two_page_spread(750, 500)

        at_150 = detect_split(page, LayoutType.TWO_PAGES, dpi=150.0).layout

        assert at_150.get_split_line_x() == 375.0

    def test_a_threshold_can_be_given(self) -> None:
        page = pages.two_page_spread()

        result = detect_split(page, LayoutType.TWO_PAGES, bw_threshold=128)

        assert result.layout.get_split_line_x() == 750.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "layout_type", [LayoutType.AUTO_LAYOUT_TYPE, LayoutType.TWO_PAGES]
)
def test_an_a4_spread_at_300_dpi_is_analysed_in_a_few_seconds(
    layout_type: LayoutType,
) -> None:
    page = pages.two_page_spread(3508, 2480)
    detect_split(page[:300, :500], layout_type)  # compile kernels before timing

    started = time.perf_counter()
    detect_split(page, layout_type, dpi=300.0)

    assert time.perf_counter() - started < 5.0
