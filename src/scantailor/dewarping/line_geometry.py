"""Planar line helpers with the conventions of the C++ ``QLineF`` code they replace.

The tracers decide on which side of a bound a point lies, where two lines cross
and how a line is clipped by a rectangle. Those decisions depend on small
conventions (the direction of a normal, what counts as parallel, how a
traversal rounds), so they are kept here in one place and in plain floats.

A line is a pair of points ``((x1, y1), (x2, y2))``; it is treated as infinite
unless a function says otherwise.
"""

import math
import sys
from collections.abc import Iterable

Point = tuple[float, float]
Line = tuple[Point, Point]
Rect = tuple[float, float, float, float]

_EPSILON = sys.float_info.epsilon
_DOUBLE_MAX = sys.float_info.max


def q_round(value: float) -> int:
    """Round half up, like Qt's ``qRound``.

    Args:
        value (float): Number to round.

    Returns:
        int: ``floor(value + 0.5)``.
    """
    return math.floor(value + 0.5)


def normal_vector(line: Line) -> Point:
    """Vector from the start of ``line`` to the end of its normal.

    Args:
        line (Line): The line.

    Returns:
        Point: The line direction turned by (dy, -dx); same length as the line.
    """
    (x1, y1), (x2, y2) = line
    return (y2 - y1, x1 - x2)


def sides_of_line(line: Line, point1: Point, point2: Point) -> float:
    """Tell whether two points are on the same side of a line.

    Args:
        line (Line): The line.
        point1 (Point): First point.
        point2 (Point): Second point.

    Returns:
        float: Positive when both are on the same side, negative when on
            opposite sides, zero when either lies on the line.
    """
    normal = normal_vector(line)
    (x1, y1), _ = line
    dot1 = normal[0] * (point1[0] - x1) + normal[1] * (point1[1] - y1)
    dot2 = normal[0] * (point2[0] - x1) + normal[1] * (point2[1] - y1)
    return dot1 * dot2


def projection_scalar(line: Line, point: Point) -> float:
    """Position of the projection of ``point`` on ``line``, with 0 at its start.

    Args:
        line (Line): The line; a degenerate line projects everything to 0.
        point (Point): The point to project.

    Returns:
        float: Parameter of the foot point along the line, 1 being its end.
    """
    (x1, y1), (x2, y2) = line
    vx, vy = x2 - x1, y2 - y1
    norm = vx * vx + vy * vy
    if abs(norm) > _EPSILON:
        mx, my = vx / norm, vy / norm
    else:
        mx = my = 0.0
    return mx * (point[0] - x1) + my * (point[1] - y1)


def projection_point(line: Line, point: Point) -> Point:
    """Foot of the perpendicular from ``point`` to ``line``.

    Args:
        line (Line): The line.
        point (Point): The point to project.

    Returns:
        Point: The projection.
    """
    scalar = projection_scalar(line, point)
    (x1, y1), (x2, y2) = line
    return (x1 + (x2 - x1) * scalar, y1 + (y2 - y1) * scalar)


def projection_dist(line: Line, point: Point) -> float:
    """Distance from ``point`` to its projection on ``line``.

    Args:
        line (Line): The line.
        point (Point): The point.

    Returns:
        float: The distance.
    """
    foot = projection_point(line, point)
    return math.hypot(foot[0] - point[0], foot[1] - point[1])


def line_intersection(line1: Line, line2: Line) -> Point | None:
    """Intersection of the infinite lines through two segments.

    Args:
        line1 (Line): First line; the point is computed from its start.
        line2 (Line): Second line.

    Returns:
        Point | None: The crossing, or None when the lines are parallel.
    """
    (p1x, p1y), (p2x, p2y) = line1
    (q1x, q1y), (q2x, q2y) = line2
    ax, ay = p2x - p1x, p2y - p1y
    bx, by = q1x - q2x, q1y - q2y
    cx, cy = p1x - q1x, p1y - q1y
    denominator = ay * bx - ax * by
    if denominator == 0 or not math.isfinite(denominator):
        return None
    reciprocal = 1 / denominator
    along = (by * cx - bx * cy) * reciprocal
    return (p1x + ax * along, p1y + ay * along)


def line_intersection_scalars(line1: Line, line2: Line) -> tuple[float, float] | None:
    """Parameters at which two infinite lines cross.

    Args:
        line1 (Line): First line.
        line2 (Line): Second line.

    Returns:
        tuple[float, float] | None: ``(s1, s2)`` such that
            ``line1.start + s1 * v1 == line2.start + s2 * v2``, or None when the
            lines are parallel.
    """
    (p1x, p1y), (e1x, e1y) = line1
    (p2x, p2y), (e2x, e2y) = line2
    v1x, v1y = e1x - p1x, e1y - p1y
    v2x, v2y = e2x - p2x, e2y - p2y
    det = v2x * v1y - v1x * v2y
    if abs(det) < _EPSILON:
        return None
    reciprocal = 1.0 / det
    bx, by = p2x - p1x, p2y - p1y
    s1 = (v2x * by - bx * v2y) * reciprocal
    s2 = (v1x * by - bx * v1y) * reciprocal
    return s1, s2


def point_at(line: Line, scalar: float) -> Point:
    """Point of ``line`` at a parameter, 0 being its start and 1 its end.

    Args:
        line (Line): The line.
        scalar (float): Parameter.

    Returns:
        Point: The point.
    """
    (x1, y1), (x2, y2) = line
    return (x1 + (x2 - x1) * scalar, y1 + (y2 - y1) * scalar)


def line_bounded_by_rect(line: Line, rect: Rect) -> Line | None:
    """Clip a line to the part that lies inside a rectangle.

    Args:
        line (Line): The line, treated as infinite.
        rect (Rect): ``(x, y, width, height)``.

    Returns:
        Line | None: The clipped segment, running in the direction of ``line``.
            None when the line does not cross the rectangle.
    """
    x, y, width, height = rect
    top_left, top_right = (x, y), (x + width, y)
    bottom_left, bottom_right = (x, y + height), (x + width, y + height)
    sides = (
        (top_left, top_right),
        (bottom_left, bottom_right),
        (top_left, bottom_left),
        (top_right, bottom_right),
    )
    largest = -_DOUBLE_MAX
    smallest = _DOUBLE_MAX
    for side in sides:
        scalars = line_intersection_scalars(side, line)
        if scalars is None:
            continue
        on_side, on_line = scalars
        if on_side < 0 or on_side > 1:
            continue
        largest = max(largest, on_line)
        smallest = min(smallest, on_line)
    if largest > smallest:
        return (point_at(line, smallest), point_at(line, largest))
    return None


def lines_within_image(lines: Iterable[Line], shape: tuple[int, ...]) -> list[Line]:
    """Clip infinite lines to an image, dropping those that miss it.

    Args:
        lines (Iterable[Line]): The lines.
        shape (tuple[int, ...]): The image's shape, height first.

    Returns:
        list[Line]: The visible segments, in the order of ``lines``.
    """
    rect = (0.0, 0.0, float(shape[1]), float(shape[0]))
    clipped = (line_bounded_by_rect(line, rect) for line in lines)
    return [line for line in clipped if line is not None]


def grid_line_cells(line: Line) -> list[tuple[int, int]]:
    """Cells visited when drawing a segment on a pixel grid.

    Args:
        line (Line): The segment.

    Returns:
        list[tuple[int, int]]: ``(x, y)`` cells from the start to the end, one
            per step along the longer axis, ends included.
    """
    (x1, y1), (x2, y2) = line
    start = (q_round(x1), q_round(y1))
    end = (q_round(x2), q_round(y2))
    horizontal_spans = abs(start[0] - end[0])
    vertical_spans = abs(start[1] - end[1])
    if horizontal_spans > vertical_spans:
        num_spans = horizontal_spans
        along_start = line_intersection_scalars(line, (start, (start[0], start[1] + 1)))
        along_end = line_intersection_scalars(line, (end, (end[0], end[1] + 1)))
    else:
        num_spans = vertical_spans
        along_start = line_intersection_scalars(line, (start, (start[0] + 1, start[1])))
        along_end = line_intersection_scalars(line, (end, (end[0] + 1, end[1])))
    first = point_at(line, along_start[0] if along_start else 0.0)
    last = point_at(line, along_end[0] if along_end else 0.0)
    step = 0.0 if num_spans == 0 else 1.0 / num_spans
    cells = []
    for stop in range(num_spans + 1):
        px, py = point_at((first, last), stop * step)
        cells.append((q_round(px), q_round(py)))
    return cells
