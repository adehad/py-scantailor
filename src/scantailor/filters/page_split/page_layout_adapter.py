"""Adapting a hand-placed page layout to a new outline instead of detecting it again.

Port of the C++ PageLayoutAdapter. The cutter lines are carried over to the new
outline by extending them to its top and bottom edges, which keeps a manual
split line when the image size or orientation changes.
"""

import math

from scantailor.dewarping.line_geometry import projection_dist

from .page_layout import PageLayout, PageLayoutType

Point = tuple[float, float]
Line = tuple[Point, Point]
Rect = tuple[float, float, float, float]

_COINCIDENT_PX = 1.0
"""Cutters closer than this everywhere bound a page under one pixel wide."""


def adapt_page_layout(layout: PageLayout, outline: Rect) -> PageLayout:
    """Create the layout for another outline, adapting the cutters to it.

    Args:
        layout (PageLayout): The layout to carry over.
        outline (scantailor.core.transformation.Rect):
            New outline as ``(x, y, width, height)``.

    Returns:
        PageLayout: ``layout`` itself when the outline is the same. Otherwise a
            layout of the same type, or a single uncut page when the adapted
            cutters leave nothing to cut (see :func:`correct_page_layout_type`).
    """
    if all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(layout.outline, outline)):
        return layout

    if layout.layout_type == PageLayoutType.SINGLE_PAGE_CUT:
        cutter1, cutter2 = _adapt_cutters(
            [_cutter(layout, 0), _cutter(layout, 1)], outline
        )
        adapted = PageLayout(
            outline=outline,
            layout_type=PageLayoutType.SINGLE_PAGE_CUT,
            cutter1=cutter1,
            cutter2=cutter2,
        )
        return correct_page_layout_type(adapted)
    if layout.layout_type == PageLayoutType.TWO_PAGES:
        cutter = _adapt_cutter(_cutter(layout, 0), outline)
        adapted = PageLayout(
            outline=outline,
            layout_type=PageLayoutType.TWO_PAGES,
            cutter1=cutter,
            cutter2=cutter,
        )
        return correct_page_layout_type(adapted)
    return PageLayout(outline=outline, layout_type=PageLayoutType.SINGLE_PAGE_UNCUT)


def correct_page_layout_type(layout: PageLayout) -> PageLayout:
    """Turn a layout whose cutters cut nothing into a single uncut page.

    Cutters are compared on the whole-pixel grid, like the outline: a cutter
    lying on the left or right edge cuts nothing, and neither do two cutters
    that cross inside the outline or lie on one line, whatever their direction
    or start point.

    Args:
        layout (PageLayout): The layout to check.

    Returns:
        PageLayout: ``layout``, or its single uncut version with the cutters kept.
    """
    x, y, width, height = layout.outline
    left, top = _round(x), _round(y)
    right, bottom = _round(x + width), _round(y + height)

    def on_side_edge(line: Line) -> bool:
        (x1, _), (x2, _) = line
        return x1 == x2 and x1 in (left, right)

    uncut = False
    if layout.layout_type == PageLayoutType.SINGLE_PAGE_CUT:
        line1 = _rounded(_cutter(layout, 0))
        line2 = _rounded(_cutter(layout, 1))
        uncut = on_side_edge(line1) and on_side_edge(line2)
        crossing = _intersection(line1, line2)
        crossing_inside = crossing is not None and top < crossing[1] < bottom
        uncut = uncut or crossing_inside or _coincide(line1, line2)
    elif layout.layout_type == PageLayoutType.TWO_PAGES:
        uncut = on_side_edge(_rounded(_cutter(layout, 0)))

    if not uncut:
        return layout
    return layout.with_type(PageLayoutType.SINGLE_PAGE_UNCUT)


def _coincide(line1: Line, line2: Line) -> bool:
    """Whether two cutters lie on one line, whatever their direction or extent.

    Each endpoint of either cutter must be within :data:`_COINCIDENT_PX` of the
    infinite line through the other. Both ways are tested because a slight
    angle that keeps a short cutter's ends on a long cutter's line can still
    carry the long cutter's ends off the short one's. A zero-length cutter has
    no line of its own, so only its point is tested against the other's line.
    """
    with_a_line = [
        (line, other)
        for line, other in ((line1, line2), (line2, line1))
        if line[0] != line[1]
    ]
    if not with_a_line:
        return math.dist(line1[0], line2[0]) < _COINCIDENT_PX
    return all(
        projection_dist(line, point) < _COINCIDENT_PX
        for line, other in with_a_line
        for point in other
    )


def _cutter(layout: PageLayout, index: int) -> Line:
    """Return a cutter, defaulting to the left or right edge of the outline."""
    x, y, width, height = layout.outline
    if index == 0:
        return layout.cutter1 or ((x, y), (x, y + height))
    return layout.cutter2 or ((x + width, y), (x + width, y + height))


def _adapt_cutter(cutter: Line, outline: Rect) -> Line:
    """Extend a cutter to the top and bottom edges of the outline.

    The result starts on the top edge and ends on the bottom edge, each point
    clamped to the outline. A cutter parallel to those edges, or a degenerate
    one, is returned unchanged.
    """
    x, y, width, height = outline
    if width <= 0 or height <= 0 or cutter[0] == cutter[1]:
        return cutter

    upper = _intersection(((x, y), (x + width, y)), cutter)
    lower = _intersection(((x, y + height), (x + width, y + height)), cutter)
    if upper is None or lower is None:
        return cutter
    return (
        (min(max(upper[0], x), x + width), upper[1]),
        (min(max(lower[0], x), x + width), lower[1]),
    )


def _adapt_cutters(cutters: list[Line], outline: Rect) -> list[Line]:
    """Adapt cutters and keep them from crossing each other inside the outline.

    Where two neighbouring cutters meet inside the outline, both are cut short
    at the meeting point on the side where they are closer to the outline edge.
    """
    adapted = sorted(
        (_adapt_cutter(cutter, outline) for cutter in cutters),
        key=lambda line: line[0][0],
    )
    upper_bound = outline[1]
    lower_bound = outline[1] + outline[3]
    for index in range(1, len(adapted)):
        left, right = adapted[index - 1], adapted[index]
        crossing = _intersection(left, right)
        if crossing is None or not upper_bound < crossing[1] < lower_bound:
            continue
        cross_x, cross_y = crossing
        if lower_bound - cross_y <= (lower_bound - upper_bound) / 2:
            left = (left[0], (cross_x, lower_bound))
            right = (right[0], (cross_x, lower_bound))
        else:
            left = ((cross_x, upper_bound), left[1])
            right = ((cross_x, upper_bound), right[1])
        adapted[index - 1], adapted[index] = left, right
    return adapted


def _intersection(line1: Line, line2: Line) -> Point | None:
    """Intersect the infinite lines through two segments; None if parallel."""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    direction_x, direction_y = x2 - x1, y2 - y1
    other_x, other_y = x3 - x4, y3 - y4
    offset_x, offset_y = x1 - x3, y1 - y3
    denominator = direction_y * other_x - direction_x * other_y
    if denominator == 0 or not math.isfinite(denominator):
        return None
    along = (other_y * offset_x - other_x * offset_y) / denominator
    return (x1 + direction_x * along, y1 + direction_y * along)


def _round(value: float) -> int:
    """Round half away from zero, as Qt does."""
    return math.floor(abs(value) + 0.5) * (1 if value >= 0 else -1)


def _rounded(line: Line) -> Line:
    (x1, y1), (x2, y2) = line
    return (
        (float(_round(x1)), float(_round(y1))),
        (float(_round(x2)), float(_round(y2))),
    )
