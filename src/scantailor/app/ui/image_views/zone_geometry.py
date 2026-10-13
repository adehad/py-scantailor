"""Hit testing and vertex editing of zone polygons.

Everything here works on plain lists of ``QPointF`` so that it does not depend
on a view. Hit tests take widget coordinates, so that the pick radius is in
screen pixels whatever the zoom; edits take whatever coordinates the points are
in.
"""

import math
from collections.abc import Sequence

from PySide6.QtCore import QPointF

MIN_VERTICES = 3


def distance(a: QPointF, b: QPointF) -> float:
    """Return the distance between two points.

    Args:
        a (QPointF): First point.
        b (QPointF): Second point.

    Returns:
        float: Euclidean distance.
    """
    return math.hypot(a.x() - b.x(), a.y() - b.y())


def nearest_on_segment(
    point: QPointF, start: QPointF, end: QPointF
) -> tuple[QPointF, float]:
    """Return the point of a segment closest to a point, and its distance.

    Args:
        point (QPointF): The point to measure from.
        start (QPointF): One end of the segment.
        end (QPointF): The other end.

    Returns:
        tuple[QPointF, float]: The closest point on the segment and its distance
            to ``point``.
    """
    edge = end - start
    length_squared = edge.x() ** 2 + edge.y() ** 2
    if length_squared == 0:
        nearest = QPointF(start)
    else:
        along = (point.x() - start.x()) * edge.x() + (point.y() - start.y()) * edge.y()
        nearest = start + edge * min(1.0, max(0.0, along / length_squared))
    return nearest, distance(point, nearest)


def vertex_near(corners: Sequence[QPointF], pos: QPointF, radius: float) -> int:
    """Return the vertex closest to a position within a radius, or -1.

    Args:
        corners (Sequence[QPointF]): Polygon vertices in widget coordinates.
        pos (QPointF): Position in widget coordinates.
        radius (float): Largest distance that still counts as a hit.

    Returns:
        int: Index of the closest vertex, -1 when none is within ``radius``.
    """
    best, best_distance = -1, radius
    for index, corner in enumerate(corners):
        gap = distance(corner, pos)
        if gap <= best_distance:
            best, best_distance = index, gap
    return best


def segment_near(
    corners: Sequence[QPointF], pos: QPointF, radius: float
) -> tuple[int, QPointF] | None:
    """Return the polygon edge closest to a position within a radius.

    Args:
        corners (Sequence[QPointF]): Polygon vertices in widget coordinates.
        pos (QPointF): Position in widget coordinates.
        radius (float): Largest distance that still counts as a hit.

    Returns:
        tuple[int, QPointF] | None: Index of the edge's first vertex (the edge
            runs to the next vertex, wrapping round) and the closest point on
            it; None when no edge is within ``radius``.
    """
    best: tuple[int, QPointF] | None = None
    best_distance = radius
    for index, start in enumerate(corners):
        end = corners[(index + 1) % len(corners)]
        nearest, gap = nearest_on_segment(pos, start, end)
        if gap <= best_distance:
            best, best_distance = (index, nearest), gap
    return best


def insert_vertex(corners: list[QPointF], segment: int, point: QPointF) -> int:
    """Split an edge by adding a vertex on it.

    Args:
        corners (list[QPointF]): Polygon vertices; changed in place.
        segment (int): Index of the edge's first vertex.
        point (QPointF): Where the new vertex goes.

    Returns:
        int: Index of the new vertex.
    """
    corners.insert(segment + 1, QPointF(point))
    return segment + 1


def can_remove_vertex(count: int) -> bool:
    """Return whether a polygon with ``count`` vertices can lose one.

    Args:
        count (int): Number of vertices.

    Returns:
        bool: True when it would still have at least three.
    """
    return count > MIN_VERTICES


def remove_vertex(corners: list[QPointF], index: int) -> bool:
    """Remove a vertex unless the polygon would drop below three.

    Args:
        corners (list[QPointF]): Polygon vertices; changed in place.
        index (int): Vertex to remove.

    Returns:
        bool: True when the vertex was removed.
    """
    if not can_remove_vertex(len(corners)):
        return False
    del corners[index]
    return True


def merge_target(
    corners: Sequence[QPointF], index: int, threshold: float
) -> int | None:
    """Return the neighbour a dragged vertex is close enough to merge with.

    Args:
        corners (Sequence[QPointF]): Polygon vertices in widget coordinates.
        index (int): The dragged vertex.
        threshold (float): Distance at which two vertices merge.

    Returns:
        int | None: Index of the nearer neighbour within ``threshold``, or None.
            Always None when the polygon could not lose a vertex.
    """
    if not can_remove_vertex(len(corners)):
        return None
    here = corners[index]
    previous = (index - 1) % len(corners)
    following = (index + 1) % len(corners)
    to_previous = distance(here, corners[previous])
    to_next = distance(here, corners[following])
    if to_previous <= threshold and to_previous < to_next:
        return previous
    if to_next <= threshold:
        return following
    return None


def right_angle_neighbours(
    current: QPointF, previous: QPointF, following: QPointF
) -> tuple[QPointF, QPointF]:
    """Move the neighbours of a vertex so that its two edges meet at a right angle.

    The edge that is closer to horizontal keeps its neighbour's row and the
    other keeps its column, so the corner turns into an axis-aligned one.

    Args:
        current (QPointF): The dragged vertex.
        previous (QPointF): Its neighbour before it.
        following (QPointF): Its neighbour after it.

    Returns:
        tuple[QPointF, QPointF]: New positions of ``previous`` and ``following``;
            the inputs come back unchanged when all three coincide.
    """
    if current == previous and current == following:
        return previous, following

    def horizontality(other: QPointF) -> float:
        gap = distance(current, other)
        return math.nan if gap == 0 else abs(other.x() - current.x()) / gap

    previous_cos, next_cos = horizontality(previous), horizontality(following)
    previous_flatter = (
        previous_cos < next_cos
        or (math.isnan(previous_cos) and next_cos > 1 / math.sqrt(2))
        or (math.isnan(next_cos) and previous_cos < 1 / math.sqrt(2))
    )
    if previous_flatter:
        return (
            QPointF(current.x(), previous.y()),
            QPointF(following.x(), current.y()),
        )
    return QPointF(previous.x(), current.y()), QPointF(current.x(), following.y())
