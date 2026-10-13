"""The crop and page rectangles that the pipeline and the views both use."""

import numpy as np
import pytest

from scantailor.core.models import SubPage
from scantailor.core.page_geometry import page_crop_area, selection_rects
from scantailor.core.transformation import Rect
from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.filters.select_content.params import Params as SelectParams

WIDTH, HEIGHT = 600, 800
SIZE = (WIDTH, HEIGHT)


def _two_pages(split: float = 300.0) -> PageLayout:
    return PageLayout(
        outline=(0, 0, WIDTH, HEIGHT),
        layout_type=PageLayoutType.TWO_PAGES,
        cutter1=((split, 0), (split, HEIGHT)),
    )


def _bounds(polygon: np.ndarray) -> tuple[float, float, float, float]:
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


class TestPageCropArea:
    def test_each_page_of_a_split_keeps_its_side_of_the_cutter(self):
        left = page_crop_area(_two_pages(), SubPage.LEFT_PAGE, SIZE, None)
        right = page_crop_area(_two_pages(), SubPage.RIGHT_PAGE, SIZE, None)

        assert left is not None
        assert right is not None
        assert _bounds(left) == pytest.approx((0, 0, 300, HEIGHT))
        assert _bounds(right) == pytest.approx((300, 0, WIDTH, HEIGHT))

    def test_an_outline_beyond_the_image_is_clipped_to_it(self):
        layout = PageLayout(
            outline=(-50, -50, WIDTH + 100, HEIGHT + 100),
            layout_type=PageLayoutType.SINGLE_PAGE_UNCUT,
        )

        area = page_crop_area(layout, SubPage.SINGLE_PAGE, SIZE, None)

        assert area is not None
        assert _bounds(area) == pytest.approx((0, 0, WIDTH, HEIGHT))

    def test_the_manual_trim_narrows_the_page(self):
        trim = np.array([[100, 50], [WIDTH, 50], [WIDTH, HEIGHT], [100, HEIGHT]], float)

        area = page_crop_area(_two_pages(), SubPage.LEFT_PAGE, SIZE, trim)

        assert area is not None
        assert _bounds(area) == pytest.approx((100, 50, 300, HEIGHT))

    def test_cutters_that_cross_inside_the_image_keep_the_whole_image(self):
        crossing = PageLayout(
            outline=(0, 0, WIDTH, HEIGHT),
            layout_type=PageLayoutType.SINGLE_PAGE_CUT,
            cutter1=((100, 0), (300, HEIGHT)),
            cutter2=((300, 0), (100, HEIGHT)),
        )

        area = page_crop_area(crossing, SubPage.SINGLE_PAGE, SIZE, None)

        assert area is not None
        assert _bounds(area) == pytest.approx((0, 0, WIDTH, HEIGHT))

    def test_an_outline_outside_the_image_is_degenerate_and_gives_none(self):
        outside = PageLayout(
            outline=(WIDTH + 10, HEIGHT + 10, 50, 50),
            layout_type=PageLayoutType.SINGLE_PAGE_UNCUT,
        )

        assert page_crop_area(outside, SubPage.SINGLE_PAGE, SIZE, None) is None

    def test_a_sub_page_the_layout_does_not_have_gives_none(self):
        single = PageLayout(
            outline=(0, 0, WIDTH, HEIGHT),
            layout_type=PageLayoutType.SINGLE_PAGE_UNCUT,
        )

        assert page_crop_area(single, SubPage.RIGHT_PAGE, SIZE, None) is None


class TestSelectionRects:
    VIEW = Rect(x=0, y=0, width=500, height=700)

    def test_stored_boxes_become_rectangles(self):
        params = SelectParams(
            content_box=ContentBox(x=40, y=50, width=300, height=400),
            page_box=PageBox(x=10, y=20, width=450, height=650),
        )

        page, content = selection_rects(params, self.VIEW)

        assert (page.x, page.y, page.width, page.height) == (10, 20, 450, 650)
        assert (content.x, content.y, content.width, content.height) == (
            40,
            50,
            300,
            400,
        )

    def test_without_a_valid_page_box_the_view_stands_in_for_the_page(self):
        page, content = selection_rects(SelectParams(), self.VIEW)

        assert page == self.VIEW
        assert content.width == 0
