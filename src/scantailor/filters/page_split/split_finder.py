"""Split line detection for page split filter.

This module provides functions for automatically detecting where to split
a scanned image into pages, based on content analysis and whitespace detection.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.imageproc.binarize import binarize_otsu

from .layout_type import LayoutType
from .page_layout import PageLayout


@dataclass
class ContentSpan:
    """A horizontal span of content."""

    begin: int
    end: int

    @property
    def width(self) -> int:
        """Return the width of this span."""
        return self.end - self.begin

    @property
    def center(self) -> float:
        """Return the center x coordinate."""
        return (self.begin + self.end) / 2.0


@dataclass
class SplitResult:
    """Result of split line detection."""

    layout: PageLayout
    confidence: float = 0.0
    vertical_lines: list[float] = field(default_factory=list)


def find_vertical_lines(
    image: NDArray[np.uint8],
    max_lines: int = 8,
    min_line_length_ratio: float = 0.5,
) -> list[float]:
    """Find strong vertical lines in an image.

    These could be fold lines in a book or edges of content.

    Args:
        image: Grayscale image.
        max_lines: Maximum number of lines to return.
        min_line_length_ratio: Minimum line length as ratio of image height.

    Returns:
        List of x coordinates where vertical lines were detected,
        sorted left to right.
    """
    h, w = image.shape[:2]
    min_line_length = int(h * min_line_length_ratio)

    # Use Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Use probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    if lines is None:
        return []

    vertical_lines: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check if line is near-vertical (within 5 degrees)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > 0 and dx / dy < 0.1:  # tan(5°) ≈ 0.087
            # Use average x as the line position
            x_center = (x1 + x2) / 2.0
            vertical_lines.append(x_center)

    if not vertical_lines:
        return []

    # Cluster nearby lines and take their centers
    vertical_lines.sort()
    clustered: list[float] = []
    cluster: list[float] = [vertical_lines[0]]

    for x in vertical_lines[1:]:
        if x - cluster[-1] < w * 0.02:  # 2% of width threshold
            cluster.append(x)
        else:
            clustered.append(sum(cluster) / len(cluster))
            cluster = [x]
    clustered.append(sum(cluster) / len(cluster))

    # Return up to max_lines, sorted by x
    return sorted(clustered[:max_lines])


def find_content_spans(
    image: NDArray[np.uint8],
    min_content_width: int = 5,
    min_whitespace_width: int = 20,
) -> list[ContentSpan]:
    """Find horizontal spans of content in a binary image.

    Args:
        image: Binary image (0 = background, 255 = content).
        min_content_width: Minimum width to consider as content.
        min_whitespace_width: Minimum gap width to split spans.

    Returns:
        List of ContentSpan objects.
    """
    # Calculate column sums (histogram)
    col_sums = np.sum(image > 0, axis=0)

    spans: list[ContentSpan] = []
    in_content = False
    span_begin = 0

    for x, count in enumerate(col_sums):
        if count > 0:  # Has content
            if not in_content:
                span_begin = x
                in_content = True
        elif in_content:
            # End of content span
            if x - span_begin >= min_content_width:
                spans.append(ContentSpan(span_begin, x))
            in_content = False

    # Handle span that extends to the edge
    if in_content and len(col_sums) - span_begin >= min_content_width:
        spans.append(ContentSpan(span_begin, len(col_sums)))

    # Merge spans that are close together
    if len(spans) <= 1:
        return spans

    merged: list[ContentSpan] = [spans[0]]
    for span in spans[1:]:
        if span.begin - merged[-1].end < min_whitespace_width:
            # Merge with previous span
            merged[-1] = ContentSpan(merged[-1].begin, span.end)
        else:
            merged.append(span)

    return merged


def estimate_num_pages(width: int, height: int, dpi: float = 300.0) -> int:
    """Estimate whether an image contains one or two pages.

    Based on aspect ratio heuristics.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        dpi: Image resolution.

    Returns:
        1 or 2 indicating estimated number of pages.
    """
    aspect = width / height if height > 0 else 1.0

    # Typical single page is taller than wide (portrait)
    # Two pages side by side would be wider than tall (landscape)
    # Threshold around 1.2 seems reasonable
    if aspect > 1.2:
        return 2
    return 1


def detect_split(
    image: NDArray[np.uint8],
    layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE,
    dpi: float = 300.0,
) -> SplitResult:
    """Detect how to split an image into pages.

    Args:
        image: Grayscale or color image.
        layout_type: Requested layout type. AUTO_LAYOUT_TYPE will auto-detect.
        dpi: Image resolution.

    Returns:
        SplitResult with detected layout and confidence.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    else:
        gray = image

    h, w = gray.shape[:2]

    # Handle explicit single page uncut
    if layout_type == LayoutType.SINGLE_PAGE_UNCUT:
        return SplitResult(
            layout=PageLayout.single_page_uncut(w, h),
            confidence=1.0,
        )

    # Find vertical lines (potential fold/cut lines)
    lines = find_vertical_lines(gray)

    # Binarize for content analysis
    binary = binarize_otsu(gray)

    # Find content spans
    spans = find_content_spans(binary)

    # Determine number of pages based on layout type or auto-detect
    if layout_type == LayoutType.TWO_PAGES:
        num_pages = 2
    elif layout_type == LayoutType.PAGE_PLUS_OFFCUT:
        num_pages = 1
    else:  # AUTO
        num_pages = estimate_num_pages(w, h, dpi)

    if num_pages == 2:
        return _detect_two_page_split(w, h, lines, spans)
    return _detect_single_page_split(w, h, lines, spans, layout_type)


def _detect_two_page_split(
    width: int,
    height: int,
    lines: list[float],
    spans: list[ContentSpan],
) -> SplitResult:
    """Detect split line for two-page layout.

    Args:
        width: Image width.
        height: Image height.
        lines: Detected vertical lines.
        spans: Content spans.

    Returns:
        SplitResult with two-page layout.
    """
    center = width / 2.0

    # Filter lines that are too close to edges
    center_lines = [x for x in lines if 0.2 * width < x < 0.8 * width]

    split_x: float
    confidence: float

    if center_lines:
        # Find line closest to center
        split_x = min(center_lines, key=lambda x: abs(x - center))
        distance_from_center = abs(split_x - center) / center
        confidence = max(0.0, 1.0 - distance_from_center)
    elif spans:
        # Find the widest gap between spans near the center
        best_gap_x = center
        best_gap_width = 0.0

        for i in range(len(spans) - 1):
            gap_begin = spans[i].end
            gap_end = spans[i + 1].begin
            gap_center = (gap_begin + gap_end) / 2.0
            gap_width = gap_end - gap_begin

            # Prefer gaps near center
            center_weight = 1.0 - abs(gap_center - center) / center
            weighted_width = gap_width * (0.5 + 0.5 * center_weight)

            if weighted_width > best_gap_width:
                best_gap_width = weighted_width
                best_gap_x = gap_center

        split_x = best_gap_x
        confidence = min(1.0, best_gap_width / (0.05 * width))
    else:
        # No content detected, split at center
        split_x = center
        confidence = 0.5

    return SplitResult(
        layout=PageLayout.two_pages(width, height, split_x),
        confidence=confidence,
        vertical_lines=lines,
    )


def _detect_single_page_split(
    width: int,
    height: int,
    lines: list[float],
    spans: list[ContentSpan],
    layout_type: LayoutType,
) -> SplitResult:
    """Detect layout for single page (with or without cut).

    Args:
        width: Image width.
        height: Image height.
        lines: Detected vertical lines.
        spans: Content spans.
        layout_type: Requested layout type.

    Returns:
        SplitResult with single page layout.
    """
    center = width / 2.0

    # Check for offcut (garbage at edges)
    has_left_edge_content = _check_edge_content(spans, 0, width * 0.1)
    has_right_edge_content = _check_edge_content(spans, width * 0.9, width)

    # If layout type is AUTO and no edge content, return uncut
    if layout_type == LayoutType.AUTO_LAYOUT_TYPE:
        if not has_left_edge_content and not has_right_edge_content:
            return SplitResult(
                layout=PageLayout.single_page_uncut(width, height),
                confidence=0.8,
                vertical_lines=lines,
            )

    # Determine cut positions
    left_x = 0.0
    right_x = float(width)

    # Find left cutter
    if has_left_edge_content:
        # Look for a line or gap on the left side
        left_lines = [x for x in lines if x < center * 0.6]
        if left_lines:
            left_x = left_lines[-1]  # Rightmost of left-side lines
        elif spans:
            # Use left edge of first significant span
            for span in spans:
                if span.width > width * 0.1:
                    left_x = max(0, span.begin - 10)
                    break

    # Find right cutter
    if has_right_edge_content:
        # Look for a line or gap on the right side
        right_lines = [x for x in lines if x > width - center * 0.6]
        if right_lines:
            right_x = right_lines[0]  # Leftmost of right-side lines
        elif spans:
            # Use right edge of last significant span
            for span in reversed(spans):
                if span.width > width * 0.1:
                    right_x = min(width, span.end + 10)
                    break

    # If we found cutters, return cut layout
    if left_x > 0 or right_x < width:
        return SplitResult(
            layout=PageLayout.single_page_cut(width, height, left_x, right_x),
            confidence=0.7,
            vertical_lines=lines,
        )

    # Default to uncut
    return SplitResult(
        layout=PageLayout.single_page_uncut(width, height),
        confidence=0.6,
        vertical_lines=lines,
    )


def _check_edge_content(
    spans: list[ContentSpan],
    edge_start: float,
    edge_end: float,
) -> bool:
    """Check if there's significant content at an edge.

    Args:
        spans: Content spans.
        edge_start: Start of edge region.
        edge_end: End of edge region.

    Returns:
        True if there's content at the edge that might be garbage.
    """
    for span in spans:
        # Check if span overlaps with edge region
        if span.begin < edge_end and span.end > edge_start:
            # Small spans at the edge might be garbage
            if span.width < (edge_end - edge_start) * 0.5:
                return True
    return False
