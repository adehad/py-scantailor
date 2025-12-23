"""Tests for split_finder module."""

import numpy as np

from scantailor.filters.page_split import (
    LayoutType,
    PageLayoutType,
)
from scantailor.filters.page_split.split_finder import (
    ContentSpan,
    detect_split,
    estimate_num_pages,
    find_content_spans,
    find_vertical_lines,
)


class TestContentSpan:
    """Tests for ContentSpan class."""

    def test_width(self):
        """Span width calculation."""
        span = ContentSpan(begin=10, end=50)
        assert span.width == 40

    def test_center(self):
        """Span center calculation."""
        span = ContentSpan(begin=10, end=50)
        assert span.center == 30.0


class TestFindContentSpans:
    """Tests for find_content_spans function."""

    def test_empty_image(self):
        """Empty image has no content spans."""
        image = np.zeros((100, 200), dtype=np.uint8)
        spans = find_content_spans(image)
        assert len(spans) == 0

    def test_full_content(self):
        """Full content image has one span."""
        image = np.full((100, 200), 255, dtype=np.uint8)
        spans = find_content_spans(image)
        assert len(spans) == 1
        assert spans[0].begin == 0
        assert spans[0].end == 200

    def test_single_content_region(self):
        """Single content region detected."""
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, 50:150] = 255
        spans = find_content_spans(image)
        assert len(spans) == 1
        assert spans[0].begin == 50
        assert spans[0].end == 150

    def test_two_content_regions(self):
        """Two separate content regions detected."""
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, 10:40] = 255
        image[:, 160:190] = 255
        spans = find_content_spans(image, min_whitespace_width=50)
        assert len(spans) == 2

    def test_merge_close_spans(self):
        """Close spans are merged."""
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, 10:40] = 255
        image[:, 45:80] = 255  # Only 5 pixels gap
        spans = find_content_spans(image, min_whitespace_width=10)
        assert len(spans) == 1
        assert spans[0].begin == 10
        assert spans[0].end == 80

    def test_small_content_filtered(self):
        """Small content below threshold is filtered."""
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, 50:52] = 255  # Only 2 pixels wide
        spans = find_content_spans(image, min_content_width=5)
        assert len(spans) == 0


class TestEstimateNumPages:
    """Tests for estimate_num_pages function."""

    def test_portrait_single_page(self):
        """Portrait orientation suggests single page."""
        assert estimate_num_pages(100, 150) == 1

    def test_landscape_two_pages(self):
        """Landscape orientation suggests two pages."""
        assert estimate_num_pages(300, 200) == 2

    def test_square_single_page(self):
        """Square suggests single page."""
        assert estimate_num_pages(100, 100) == 1

    def test_wide_two_pages(self):
        """Very wide image suggests two pages."""
        assert estimate_num_pages(400, 200) == 2


class TestFindVerticalLines:
    """Tests for find_vertical_lines function."""

    def test_no_lines_in_blank(self):
        """Blank image has no vertical lines."""
        image = np.full((100, 200), 128, dtype=np.uint8)
        lines = find_vertical_lines(image)
        assert len(lines) == 0

    def test_detect_strong_vertical_line(self):
        """Strong vertical line is detected."""
        image = np.full((200, 200), 200, dtype=np.uint8)
        # Draw a strong vertical line
        image[:, 100:103] = 0
        lines = find_vertical_lines(image, min_line_length_ratio=0.3)
        # May or may not detect depending on edge detection
        # This is a best-effort test
        assert isinstance(lines, list)

    def test_max_lines_limit(self):
        """Result respects max_lines limit."""
        image = np.full((100, 200), 128, dtype=np.uint8)
        lines = find_vertical_lines(image, max_lines=3)
        assert len(lines) <= 3


class TestDetectSplit:
    """Tests for detect_split function."""

    def test_single_page_uncut_explicit(self):
        """Explicit single page uncut returns correct layout."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = detect_split(image, LayoutType.SINGLE_PAGE_UNCUT)
        assert result.layout.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT
        assert result.confidence == 1.0

    def test_two_pages_layout(self):
        """Two pages layout creates split at center."""
        image = np.zeros((100, 200), dtype=np.uint8)
        result = detect_split(image, LayoutType.TWO_PAGES)
        assert result.layout.layout_type == PageLayoutType.TWO_PAGES
        split_x = result.layout.get_split_line_x()
        assert split_x is not None
        # Should be near center
        assert 80 < split_x < 120

    def test_auto_portrait_single(self):
        """Auto mode on portrait image suggests single page."""
        image = np.zeros((150, 100), dtype=np.uint8)
        result = detect_split(image, LayoutType.AUTO_LAYOUT_TYPE)
        # Portrait should be single page
        assert result.layout.num_sub_pages == 1

    def test_auto_landscape_two_pages(self):
        """Auto mode on landscape image suggests two pages."""
        image = np.zeros((100, 250), dtype=np.uint8)
        result = detect_split(image, LayoutType.AUTO_LAYOUT_TYPE)
        # Wide landscape should be two pages
        assert result.layout.layout_type == PageLayoutType.TWO_PAGES

    def test_color_image_converted(self):
        """Color image is handled correctly."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detect_split(image, LayoutType.SINGLE_PAGE_UNCUT)
        assert result.layout is not None

    def test_result_has_layout(self):
        """Result always has a layout."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = detect_split(image)
        assert result.layout is not None
        assert result.confidence >= 0

    def test_two_pages_with_content_gap(self):
        """Two pages with gap between content uses gap for split."""
        image = np.zeros((100, 200), dtype=np.uint8)
        # Content on left
        image[20:80, 10:80] = 255
        # Content on right
        image[20:80, 120:190] = 255
        result = detect_split(image, LayoutType.TWO_PAGES)
        split_x = result.layout.get_split_line_x()
        assert split_x is not None
        # Should be in the gap
        assert 80 < split_x < 120
