"""Automatic page layout detection.

A folding line (a thin dark vertical line) decides the layout when one can be
found. Otherwise the page is binarised at 300 dpi, cleaned of shadows and
garbage, deskewed and reduced to 150 dpi, and the layout follows from the
vertical bands of content and whitespace.

Coordinates are those of the input image, which is already upright. A region of
it can be analysed alone; the layout is still in the coordinates of the whole
image.
"""

import math
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.imageproc.area_scale import scale_to_gray
from scantailor.imageproc.binary_morphology import open_brick, seed_fill
from scantailor.imageproc.binary_shear import h_shear
from scantailor.imageproc.binary_threshold import otsu_threshold
from scantailor.imageproc.reduce_threshold import reduce_threshold
from scantailor.imageproc.skew_finder import Skew, find_skew_binary

from .content_span_finder import Span, find_content_spans
from .layout_type import LayoutType
from .page_layout import PageLayout, PageLayoutType
from .vert_line_finder import Line, find_lines

_DEG2RAD = math.pi / 180.0
_MAX_FOLDING_LINES = 8
_NEAR_EDGE_FOR_UNCUT = 0.65
_ONE_SIDED_LINES_THRESHOLD = 0.3
_TWO_PAGE_EDGE_LIMIT = 0.6
_TARGET_DPI = 300.0
_DPI_TOLERANCE = 0.1
_MIN_COMPONENT_SIZE = 5
_MAX_COMPONENT_ASPECT = 6
_OFFCUT_MARGIN = 2
_OFFCUT_WIDTH = 3
_MIN_CONTENT_WIDTH = 2
_MIN_WHITESPACE_WIDTH = 8
_MIN_BALANCE_RATIO = 0.25
_ACCEPTABLE_BALANCE_FRACTION = 0.90
_EDGE_SPAN_FRACTION = 15
_SINGLE_SPAN_CLEARANCE = 15
_OFFCUT_CUT_MARGIN = 20
_OFFCUT_MAX_WIDTH_FRACTION = 4
_MIN_OFFCUT_GUTTER = 20
_OFFCUT_GUTTER_RATIO = 2
_SKEW_ACCURACY = 0.5

_Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class _Cutters:
    """Layout kind and its cutter lines, before conversion to a PageLayout."""

    layout_type: PageLayoutType
    lines: tuple[Line, ...] = ()


def estimate_page_layout(
    gray: NDArray[np.uint8],
    layout_type: LayoutType,
    dpi: float,
    bw_threshold: int | None = None,
    region: tuple[int, int, int, int] | None = None,
) -> PageLayout:
    """Detect how an upright page image divides into pages.

    The result is always of the requested type; automatic detection may return
    any type.

    Args:
        gray (NDArray[np.uint8]): (H, W) gray image.
        layout_type (LayoutType): Requested layout.
        dpi (float): Resolution of the image, the same on both axes.
        bw_threshold (int | None): Gray level below which a pixel is black. By
            default Otsu's threshold of the analysed part of ``gray``.
        region (tuple[int, int, int, int] | None): ``(x, y, width, height)`` of
            the only part of ``gray`` to analyse, such as what a manual trim
            keeps. It is clipped to the image. By default the whole image.

    Returns:
        PageLayout: The layout of the whole image, with cutter lines spanning
        its height.
    """
    full_height, full_width = gray.shape
    x0, y0, x1, y1 = _clip_region(region, full_width, full_height)
    analysed = gray[y0:y1, x0:x1]
    height, width = analysed.shape
    if layout_type == LayoutType.SINGLE_PAGE_UNCUT:
        return PageLayout.single_page_uncut(full_width, full_height)

    folding = _try_cut_at_folding_line(layout_type, analysed, dpi)
    if folding is not None:
        layout = _to_page_layout(folding, width, height)
    else:
        if bw_threshold is None:
            bw_threshold = otsu_threshold(analysed)
        layout = _cut_at_whitespace(layout_type, analysed, dpi, bw_threshold)
    if (x0, y0, width, height) == (0, 0, full_width, full_height):
        return layout
    return _in_whole_image(layout, x0, y0, full_width, full_height)


def _clip_region(
    region: tuple[int, int, int, int] | None, width: int, height: int
) -> tuple[int, int, int, int]:
    """Return the region as ``(left, top, right, bottom)`` inside the image."""
    if region is None:
        return 0, 0, width, height
    x, y, region_width, region_height = region
    left, top = min(max(x, 0), width - 1), min(max(y, 0), height - 1)
    right = min(max(x + region_width, left + 1), width)
    bottom = min(max(y + region_height, top + 1), height)
    return left, top, right, bottom


def _in_whole_image(
    layout: PageLayout, x: int, y: int, width: int, height: int
) -> PageLayout:
    """Move a layout found in a region to the coordinates of the whole image."""
    if layout.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT:
        return PageLayout.single_page_uncut(width, height)

    def moved(line: Line | None) -> Line | None:
        if line is None:
            return None
        (x1, y1), (x2, y2) = line
        return _across_height(((x1 + x, y1 + y), (x2 + x, y2 + y)), height)

    return PageLayout(
        outline=(0.0, 0.0, float(width), float(height)),
        layout_type=layout.layout_type,
        cutter1=moved(layout.cutter1),
        cutter2=moved(layout.cutter2),
    )


def _num_pages(layout_type: LayoutType, width: int, height: int) -> int:
    if layout_type == LayoutType.AUTO_LAYOUT_TYPE:
        return 2 if width > height else 1
    if layout_type == LayoutType.TWO_PAGES:
        return 2
    return 1


def _center_x(line: Line) -> float:
    return 0.5 * (line[0][0] + line[1][0])


def _left_edge(height: int) -> Line:
    return ((0.0, 0.0), (0.0, float(height)))


def _right_edge(width: int, height: int) -> Line:
    return ((float(width), 0.0), (float(width), float(height)))


def _to_page_layout(cutters: _Cutters, width: int, height: int) -> PageLayout:
    """Build the layout, stretching each cutter to the top and bottom of the image."""
    lines = tuple(_across_height(line, height) for line in cutters.lines)
    outline = (0.0, 0.0, float(width), float(height))
    if cutters.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT:
        return PageLayout.single_page_uncut(width, height)
    if cutters.layout_type == PageLayoutType.TWO_PAGES:
        return PageLayout(
            outline=outline,
            layout_type=PageLayoutType.TWO_PAGES,
            cutter1=lines[0],
            cutter2=lines[0],
        )
    return PageLayout(
        outline=outline,
        layout_type=PageLayoutType.SINGLE_PAGE_CUT,
        cutter1=lines[0],
        cutter2=lines[1],
    )


def _across_height(line: Line, height: int) -> Line:
    """Extend a line along its own direction to y = 0 and y = height.

    A layout reads the middle of its cutter as the cutter's position, so a
    cutter must span the image for that to be its position at mid height.
    """
    (x1, y1), (x2, y2) = line
    if y1 == y2:
        return line
    slope = (x2 - x1) / (y2 - y1)
    return ((x1 - y1 * slope, 0.0), (x1 + (height - y1) * slope, float(height)))


def _try_cut_at_folding_line(
    layout_type: LayoutType, gray: NDArray[np.uint8], dpi: float
) -> _Cutters | None:
    """Cut at the folding lines found in the image, if there are usable ones."""
    height, width = gray.shape
    num_pages = _num_pages(layout_type, width, height)
    lines = find_lines(gray, dpi, _MAX_FOLDING_LINES)
    lines.sort(key=_center_x)
    center = 0.5 * width

    if num_pages == 1:
        lines = _keep_one_sided_lines_apart(lines, center)
        return _single_page_from_lines(layout_type, lines, width, height)

    limit = _TWO_PAGE_EDGE_LIMIT * center
    lines = [line for line in lines if abs(_center_x(line) - center) <= limit]
    return _two_pages_from_lines(lines, center)


def _keep_one_sided_lines_apart(lines: list[Line], center: float) -> list[Line]:
    """Reduce lines that all lie on one side of the centre to the one nearest it.

    They cannot be the two edges of a page, so at most one of them is real.
    """
    if len(lines) <= 1:
        return lines
    left, right = lines[0], lines[-1]
    left_dist = center - _center_x(left)
    right_dist = center - _center_x(right)
    if (left_dist < 0) != (right_dist < 0):
        return lines
    left_dist, right_dist = abs(left_dist), abs(right_dist)
    threshold = _ONE_SIDED_LINES_THRESHOLD * center
    if left_dist < threshold or right_dist < threshold:
        return lines
    return [left if left_dist < right_dist else right]


def _single_page_from_lines(
    layout_type: LayoutType, lines: list[Line], width: int, height: int
) -> _Cutters | None:
    center = 0.5 * width
    # With no line the whitespace analysis decides, which is what can find an
    # offcut; it returns an uncut layout when there is none.
    if not lines:
        return None
    if layout_type != LayoutType.PAGE_PLUS_OFFCUT and _lines_leave_page_uncut(
        lines, center
    ):
        return _Cutters(PageLayoutType.SINGLE_PAGE_UNCUT)

    if len(lines) > 1:
        return _Cutters(PageLayoutType.SINGLE_PAGE_CUT, (lines[0], lines[-1]))
    line = lines[0]
    if _center_x(line) < center:
        return _Cutters(
            PageLayoutType.SINGLE_PAGE_CUT, (line, _right_edge(width, height))
        )
    return _Cutters(PageLayoutType.SINGLE_PAGE_CUT, (_left_edge(height), line))


def _lines_leave_page_uncut(lines: list[Line], center: float) -> bool:
    """Whether no line is close enough to an edge to suggest a cut page."""
    limit = _NEAR_EDGE_FOR_UNCUT * center
    if lines and abs(center - _center_x(lines[0])) > limit:
        return False
    return not (len(lines) > 1 and abs(center - _center_x(lines[-1])) > limit)


def _two_pages_from_lines(lines: list[Line], center: float) -> _Cutters | None:
    if not lines:
        return None
    if len(lines) == 1:
        return _Cutters(PageLayoutType.TWO_PAGES, (lines[0],))
    best = min(lines, key=lambda line: abs(_center_x(line) - center))
    return _Cutters(PageLayoutType.TWO_PAGES, (best,))


def _to_300_dpi_binary(
    gray: NDArray[np.uint8], dpi: float, threshold: int
) -> tuple[NDArray[np.bool_], _Matrix]:
    """Threshold the image at 300 dpi.

    Returns:
        The black pixels and the matrix taking input coordinates to theirs.
    """
    factor = _TARGET_DPI / dpi
    if abs(factor - 1.0) < _DPI_TOLERANCE:
        return gray < threshold, np.eye(3)

    height, width = gray.shape
    new_size = (
        max(1, math.ceil(factor * width)),
        max(1, math.ceil(factor * height)),
    )
    scaled = scale_to_gray(gray, new_size)
    return scaled < threshold, _scaling(factor, factor)


def _remove_garbage_and_halve(black: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Halve to 150 dpi and drop specks and the shadows of the page borders."""
    reduced = reduce_threshold(black, 2)

    # A real mark contains a bar at least 4 pixels long, either way.
    bars = open_brick(reduced, 4, 1) | open_brick(reduced, 1, 4)
    reduced = seed_fill(bars, reduced)

    wide_shadow = open_brick(reduced, 200, 14, black_surroundings=True)
    tall_shadow = open_brick(reduced, 14, 300, black_surroundings=True)
    grown = cv2.dilate(reduced.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
        np.bool_
    )
    shadows = seed_fill(wide_shadow | tall_shadow, grown)
    return reduced & ~shadows


def _has_left_offcut(img: NDArray[np.bool_]) -> bool:
    height, _width = img.shape
    rows = slice(_OFFCUT_MARGIN, max(0, height - _OFFCUT_MARGIN))
    return bool(img[rows, _OFFCUT_MARGIN : _OFFCUT_MARGIN + _OFFCUT_WIDTH].any())


def _has_right_offcut(img: NDArray[np.bool_]) -> bool:
    height, width = img.shape
    rows = slice(_OFFCUT_MARGIN, max(0, height - _OFFCUT_MARGIN))
    first = max(0, width - _OFFCUT_MARGIN - _OFFCUT_WIDTH)
    return bool(img[rows, first : first + _OFFCUT_WIDTH].any())


def _scaling(x: float, y: float) -> _Matrix:
    return np.array([[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, 1.0]])


def _translation(dx: float, dy: float) -> _Matrix:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [dx, dy, 1.0]])


def _x_shear(factor: float) -> _Matrix:
    """Matrix of ``x' = x + factor * y`` for row vectors."""
    return np.array([[1.0, 0.0, 0.0], [factor, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _deskew(img: NDArray[np.bool_], skew: Skew) -> tuple[NDArray[np.bool_], _Matrix]:
    """Shear the page upright and trim the corners the shear left empty.

    Returns:
        The new image and the matrix taking old pixel coordinates to new ones;
        the identity when the skew is too small, uncertain or too large to trim.
    """
    height, width = img.shape
    tangent = math.tan(skew.angle * _DEG2RAD)
    margin = math.ceil(abs(0.5 * height * tangent))
    new_width = width - margin * 2
    if new_width <= 0:
        return img, np.eye(3)

    sheared = h_shear(img, tangent, 0.5 * height)
    matrix = (
        _translation(-0.5 * width, -0.5 * height)
        @ _x_shear(tangent)
        @ _translation(0.5 * width - margin, 0.5 * height)
    )
    return sheared[:, margin : margin + new_width], matrix


def _cut_at_whitespace(
    layout_type: LayoutType,
    gray: NDArray[np.uint8],
    dpi: float,
    bw_threshold: int,
) -> PageLayout:
    """Find whitespace to cut through; always returns a layout of the requested type."""
    height, width = gray.shape
    black, to_300 = _to_300_dpi_binary(gray, dpi, bw_threshold)
    img = _remove_garbage_and_halve(black)
    to_150 = to_300 @ _scaling(0.5, 0.5)

    left_offcut = _has_left_offcut(img)
    right_offcut = _has_right_offcut(img)

    skew = find_skew_binary(
        img, coarse_reduction=0, fine_reduction=0, accuracy=_SKEW_ACCURACY
    )
    if skew.angle != 0.0 and skew.confidence >= Skew.GOOD_CONFIDENCE:
        img, deskew_matrix = _deskew(img, skew)
        to_150 = to_150 @ deskew_matrix

    num_pages = _num_pages(layout_type, width, height)
    cutters = _cut_at_whitespace_deskewed_150(
        layout_type, num_pages, img, left_offcut, right_offcut
    )

    back = np.asarray(np.linalg.inv(to_150), dtype=np.float64)
    mapped = tuple(_map_line(line, back) for line in cutters.lines)
    return _to_page_layout(_Cutters(cutters.layout_type, mapped), width, height)


def _map_line(line: Line, matrix: _Matrix) -> Line:
    """Transform a line's end points as row vectors by a 3x3 matrix."""
    points = np.array([[*line[0], 1.0], [*line[1], 1.0]]) @ matrix
    return (float(points[0, 0]), float(points[0, 1])), (
        float(points[1, 0]),
        float(points[1, 1]),
    )


def _content_columns(img: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Count, per column, the kept components whose bounding box covers it.

    Components that are tiny or much taller than wide (rules, edges) are not
    content. Only whether a column is covered matters downstream.
    """
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=8
    )
    left = stats[1:, cv2.CC_STAT_LEFT]
    box_width = stats[1:, cv2.CC_STAT_WIDTH]
    box_height = stats[1:, cv2.CC_STAT_HEIGHT]
    keep = (
        (box_width >= _MIN_COMPONENT_SIZE)
        & (box_height >= _MIN_COMPONENT_SIZE)
        & (box_height / np.maximum(box_width, 1) <= _MAX_COMPONENT_ASPECT)
    )
    edges = np.zeros(img.shape[1] + 1, dtype=np.int64)
    np.add.at(edges, left[keep], 1)
    np.add.at(edges, (left + box_width)[keep], -1)
    return np.cumsum(edges)[:-1]


def _cut_at_whitespace_deskewed_150(
    layout_type: LayoutType,
    num_pages: int,
    img: NDArray[np.bool_],
    left_offcut: bool,
    right_offcut: bool,
) -> _Cutters:
    """Choose cutters from the content spans of a deskewed 150 dpi page."""
    height, width = img.shape
    spans = find_content_spans(
        _content_columns(img).tolist(), _MIN_CONTENT_WIDTH, _MIN_WHITESPACE_WIDTH
    )
    if num_pages == 1:
        return _process_spans_single_page(
            layout_type, spans, width, height, left_offcut, right_offcut
        )
    # One page may hold nothing but a little garbage.
    _remove_insignificant_edge_spans(spans)
    return _process_spans_two_pages(spans, width)


def _remove_insignificant_edge_spans(spans: list[Span]) -> None:
    """Drop small outer spans while together they stay under 1/15 of the content."""
    if not spans:
        return
    may_be_removed = sum(span.width for span in spans) // _EDGE_SPAN_FRACTION
    while len(spans) > 1:
        first, last = spans[0], spans[-1]
        if first.width < last.width:
            if first.width > may_be_removed:
                break
            may_be_removed -= first.width
            spans.pop(0)
        else:
            if last.width > may_be_removed:
                break
            may_be_removed -= last.width
            spans.pop()


def _vert_line(x: float) -> Line:
    return ((x, 0.0), (x, 1.0))


def _process_spans_single_page(
    layout_type: LayoutType,
    spans: list[Span],
    width: int,
    height: int,
    left_offcut: bool,
    right_offcut: bool,
) -> _Cutters:
    auto = layout_type == LayoutType.AUTO_LAYOUT_TYPE
    if left_offcut and not right_offcut and auto:
        x = _left_cut_position(spans, width)
        if x is not None:
            return _Cutters(
                PageLayoutType.SINGLE_PAGE_CUT,
                (_vert_line(x), _right_edge(width, height)),
            )
    if right_offcut and not left_offcut and auto:
        x = _right_cut_position(spans, width)
        if x is not None:
            return _Cutters(
                PageLayoutType.SINGLE_PAGE_CUT,
                (_left_edge(height), _vert_line(x)),
            )

    if layout_type == LayoutType.PAGE_PLUS_OFFCUT:
        return _Cutters(
            PageLayoutType.SINGLE_PAGE_CUT,
            (_left_edge(height), _right_edge(width, height)),
        )
    return _Cutters(PageLayoutType.SINGLE_PAGE_UNCUT)


def _left_cut_position(spans: list[Span], width: int) -> float | None:
    """Where to cut off garbage at the left edge; None if the content is there."""
    if not spans:
        return 0.0
    first = spans[0]
    if first.begin > 0:
        return 0.5 * first.begin
    if first.width > width // 2:
        # Most likely the page's content; the garbage is at the other side.
        return None
    if len(spans) > 1:
        gutter = Span.between(first, spans[1])
        if not _is_offcut_strip(first, gutter, spans[1:], width):
            return None
        return gutter.center
    return float(min(first.end + _OFFCUT_CUT_MARGIN, width))


def _right_cut_position(spans: list[Span], width: int) -> float | None:
    """Where to cut off garbage at the right edge; None if the content is there."""
    if not spans:
        return float(width)
    last = spans[-1]
    if last.end < width:
        return Span.between(last, width).center
    if last.width > width // 2:
        return None
    if len(spans) > 1:
        gutter = Span.between(spans[-2], last)
        if not _is_offcut_strip(last, gutter, spans[:-1], width):
            return None
        return gutter.center
    return float(max(last.begin - _OFFCUT_CUT_MARGIN, 0))


def _is_offcut_strip(
    strip: Span, gutter: Span, content: list[Span], width: int
) -> bool:
    """Tell a neighbour's strip at the image edge from the page's own text.

    Upstream cuts at the first gap before any edge span narrower than half the
    page, which takes the last word of a page whose text runs off the scan for
    an offcut. A neighbour's strip is narrow, and the gutter between it and the
    content is a real margin: wider than any word gap inside the content.

    Args:
        strip (Span): The span touching the image edge.
        gutter (Span): The whitespace between the strip and the content.
        content (list[Span]): The remaining spans, in column order.
        width (int): Width of the image.

    Returns:
        bool: True when the strip should be cut off.
    """
    if strip.width > width // _OFFCUT_MAX_WIDTH_FRACTION:
        return False
    widest_inside = max(
        (Span.between(left, right).width for left, right in pairwise(content)),
        default=0,
    )
    return (
        gutter.width >= _MIN_OFFCUT_GUTTER
        and gutter.width >= _OFFCUT_GUTTER_RATIO * widest_inside
    )


def _process_spans_two_pages(spans: list[Span], width: int) -> _Cutters:
    if not spans:
        return _Cutters(PageLayoutType.TWO_PAGES, (_vert_line(0.5 * width),))
    if len(spans) == 1:
        return _two_pages_with_single_span(spans[0], width)

    content_begin = spans[0].begin
    content_end = spans[-1].end
    # Content on each side of a gap, measured from the outer ends of the content.
    before = [spans[i].end - content_begin for i in range(len(spans) - 1)]
    after = [content_end - spans[i + 1].begin for i in range(len(spans) - 1)]

    def balance(i: int) -> float:
        return min(before[i], after[i]) / max(before[i], after[i])

    best_gap = 0
    best_ratio = 0.0
    for i in range(len(before)):
        ratio = balance(i)
        if ratio > best_ratio:
            best_ratio = ratio
            best_gap = i

    if best_ratio < _MIN_BALANCE_RATIO:
        # Probably one of the pages is empty.
        return _two_pages_with_single_span(Span(content_begin, content_end), width)

    acceptable = best_ratio * _ACCEPTABLE_BALANCE_FRACTION
    widest_gap = best_gap
    max_width = Span.between(spans[best_gap], spans[best_gap + 1]).width
    # Look for a wider gap among the neighbours that are nearly as balanced,
    # first towards the left and then towards the right of the best gap.
    for neighbours in (range(best_gap - 1, -1, -1), range(best_gap + 1, len(before))):
        for i in neighbours:
            if balance(i) < acceptable:
                break
            gap_width = Span.between(spans[i], spans[i + 1]).width
            if gap_width > max_width:
                max_width = gap_width
                widest_gap = i
    gap = Span.between(spans[widest_gap], spans[widest_gap + 1])
    return _Cutters(PageLayoutType.TWO_PAGES, (_vert_line(gap.center),))


def _two_pages_with_single_span(span: Span, width: int) -> _Cutters:
    """Split a spread whose content is a single span."""
    page_center = 0.5 * width
    distance_to_center = abs(page_center - span.center) - 0.5 * span.width
    if distance_to_center > _SINGLE_SPAN_CLEARANCE:
        x = page_center
    else:
        left_whitespace = Span.between(0, span)
        right_whitespace = Span.between(span, width)
        if left_whitespace.width > right_whitespace.width:
            x = float(max(0, span.begin - _SINGLE_SPAN_CLEARANCE))
        else:
            x = float(min(width, span.end + _SINGLE_SPAN_CLEARANCE))
    return _Cutters(PageLayoutType.TWO_PAGES, (_vert_line(x),))
