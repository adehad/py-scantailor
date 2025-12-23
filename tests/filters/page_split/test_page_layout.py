"""Tests for PageLayout class."""

import numpy as np
import pytest

from scantailor.core import SubPage
from scantailor.filters.page_split import PageLayout, PageLayoutType


class TestPageLayout:
    """Tests for PageLayout class."""

    def test_single_page_uncut_creation(self):
        """Create single page uncut layout."""
        layout = PageLayout.single_page_uncut(100, 200)
        assert layout.outline == (0, 0, 100, 200)
        assert layout.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT
        assert layout.cutter1 is None
        assert layout.cutter2 is None

    def test_single_page_uncut_with_offset(self):
        """Create single page uncut with offset."""
        layout = PageLayout.single_page_uncut(100, 200, x=10, y=20)
        assert layout.outline == (10, 20, 100, 200)

    def test_two_pages_creation(self):
        """Create two pages layout."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        assert layout.outline == (0, 0, 200, 100)
        assert layout.layout_type == PageLayoutType.TWO_PAGES
        assert layout.cutter1 is not None
        assert layout.cutter2 is not None

    def test_two_pages_split_line(self):
        """Two pages split line is at correct position."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        assert layout.get_split_line_x() == 100

    def test_single_page_cut_creation(self):
        """Create single page cut layout."""
        layout = PageLayout.single_page_cut(200, 100, left_x=20, right_x=180)
        assert layout.outline == (0, 0, 200, 100)
        assert layout.layout_type == PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter1 is not None
        assert layout.cutter2 is not None

    def test_num_cutters_single_uncut(self):
        """Single uncut has 0 cutters."""
        layout = PageLayout.single_page_uncut(100, 100)
        assert layout.num_cutters == 0

    def test_num_cutters_two_pages(self):
        """Two pages has 1 cutter."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        assert layout.num_cutters == 1

    def test_num_cutters_single_cut(self):
        """Single cut has 2 cutters."""
        layout = PageLayout.single_page_cut(200, 100, left_x=20, right_x=180)
        assert layout.num_cutters == 2

    def test_num_sub_pages_single(self):
        """Single page layouts have 1 sub-page."""
        layout = PageLayout.single_page_uncut(100, 100)
        assert layout.num_sub_pages == 1

    def test_num_sub_pages_two(self):
        """Two page layout has 2 sub-pages."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        assert layout.num_sub_pages == 2

    def test_outline_polygon(self):
        """Get outline as polygon."""
        layout = PageLayout.single_page_uncut(100, 200)
        polygon = layout.get_outline_polygon()
        assert polygon.shape == (4, 2)
        np.testing.assert_array_equal(polygon[0], [0, 0])
        np.testing.assert_array_equal(polygon[1], [100, 0])
        np.testing.assert_array_equal(polygon[2], [100, 200])
        np.testing.assert_array_equal(polygon[3], [0, 200])

    def test_left_page_outline(self):
        """Get left page outline for two pages."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        left = layout.get_left_page_outline()
        assert left is not None
        assert left.shape == (4, 2)
        np.testing.assert_array_equal(left[0], [0, 0])
        np.testing.assert_array_equal(left[1], [100, 0])
        np.testing.assert_array_equal(left[2], [100, 100])
        np.testing.assert_array_equal(left[3], [0, 100])

    def test_right_page_outline(self):
        """Get right page outline for two pages."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        right = layout.get_right_page_outline()
        assert right is not None
        assert right.shape == (4, 2)
        np.testing.assert_array_equal(right[0], [100, 0])
        np.testing.assert_array_equal(right[1], [200, 0])
        np.testing.assert_array_equal(right[2], [200, 100])
        np.testing.assert_array_equal(right[3], [100, 100])

    def test_left_page_outline_single_returns_none(self):
        """Single page layout returns None for left outline."""
        layout = PageLayout.single_page_uncut(100, 100)
        assert layout.get_left_page_outline() is None

    def test_right_page_outline_single_returns_none(self):
        """Single page layout returns None for right outline."""
        layout = PageLayout.single_page_uncut(100, 100)
        assert layout.get_right_page_outline() is None

    def test_single_page_outline_uncut(self):
        """Single uncut returns full outline."""
        layout = PageLayout.single_page_uncut(100, 200)
        single = layout.get_single_page_outline()
        assert single is not None
        polygon = layout.get_outline_polygon()
        np.testing.assert_array_equal(single, polygon)

    def test_single_page_outline_cut(self):
        """Single cut returns area between cutters."""
        layout = PageLayout.single_page_cut(200, 100, left_x=20, right_x=180)
        single = layout.get_single_page_outline()
        assert single is not None
        np.testing.assert_array_equal(single[0], [20, 0])
        np.testing.assert_array_equal(single[1], [180, 0])
        np.testing.assert_array_equal(single[2], [180, 100])
        np.testing.assert_array_equal(single[3], [20, 100])

    def test_get_page_outline_left(self):
        """Get page outline for LEFT_PAGE."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        outline = layout.get_page_outline(SubPage.LEFT_PAGE)
        expected = layout.get_left_page_outline()
        np.testing.assert_array_equal(outline, expected)

    def test_get_page_outline_right(self):
        """Get page outline for RIGHT_PAGE."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        outline = layout.get_page_outline(SubPage.RIGHT_PAGE)
        expected = layout.get_right_page_outline()
        np.testing.assert_array_equal(outline, expected)

    def test_get_page_outline_single(self):
        """Get page outline for SINGLE_PAGE."""
        layout = PageLayout.single_page_uncut(100, 100)
        outline = layout.get_page_outline(SubPage.SINGLE_PAGE)
        expected = layout.get_single_page_outline()
        np.testing.assert_array_equal(outline, expected)

    def test_with_split_line(self):
        """Create layout with new split line."""
        layout = PageLayout.two_pages(200, 100, split_x=100)
        new_layout = layout.with_split_line(120)
        assert new_layout.get_split_line_x() == 120
        assert new_layout.layout_type == PageLayoutType.TWO_PAGES

    def test_with_cutters(self):
        """Create layout with new cutters."""
        layout = PageLayout.single_page_cut(200, 100, left_x=20, right_x=180)
        new_layout = layout.with_cutters(30, 170)
        single = new_layout.get_single_page_outline()
        assert single is not None
        assert single[0][0] == 30
        assert single[1][0] == 170

    def test_invalid_outline_raises(self):
        """Invalid outline dimensions raise error."""
        with pytest.raises(ValueError, match="positive"):
            PageLayout(
                outline=(0, 0, 0, 100),
                layout_type=PageLayoutType.SINGLE_PAGE_UNCUT,
            )

    def test_frozen_model(self):
        """Layout is immutable."""
        layout = PageLayout.single_page_uncut(100, 100)
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            layout.outline = (0, 0, 200, 200)  # type: ignore[misc]
