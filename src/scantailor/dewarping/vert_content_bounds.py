"""Left and right content boundaries of a cleaned binary page.

Each boundary starts as the convex envelope of the ink seen from one side, kept
as a path from top to bottom. A line fitted to the path's steep segments with
RANSAC then stands for the boundary, and is extended to the top and the bottom
of the image.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scantailor.dewarping.line_geometry import Line, Point, line_intersection
from scantailor.dewarping.qt_random import QtRandomGenerator

# Segments whose direction differs less than this from a seed's are inliers.
_COS_THRESHOLD = math.cos(math.radians(4.0))
_RANSAC_ITERATIONS = 200
_NUM_BEST_SEEDS = 6
# A path point is kept when dropping it needs a segment longer than this share of
# the image diagonal.
_MAX_SEGMENT_SQLEN_DIVISOR = 3


@dataclass(frozen=True)
class _Segment:
    """A steep step of the envelope path, from ``top`` down to ``bottom``."""

    top: tuple[int, int]
    bottom: tuple[int, int]
    unit: tuple[float, float]
    vertical_extent: int


def detect_vert_content_bounds(binary: NDArray[np.bool_]) -> tuple[Line, Line]:
    """Find the left and right boundaries of the content.

    Args:
        binary (NDArray[np.bool_]): 2-D page, True for ink. Clutter and specks
            outside the content should be gone already.

    Returns:
        tuple[Line, Line]: The left and the right boundary as lines running from
            the top to the bottom of the image; they may lie partly or wholly
            outside it. A page without ink gives two vertical lines at x = -1
            and x = 1.
    """
    height, width = binary.shape
    tops, bottoms, has_ink = _vertical_ranges(binary)

    left = _SideProcessor(width, height, is_left=True)
    for x in range(width):
        if has_ink[x]:
            left.process(x, int(tops[x]), int(bottoms[x]))
    right = _SideProcessor(width, height, is_left=False)
    for x in range(width - 1, -1, -1):
        if has_ink[x]:
            right.process(x, int(tops[x]), int(bottoms[x]))

    left_line = _translate(left.approximate_with_line(), -1.0)
    right_line = _translate(right.approximate_with_line(), 1.0)
    return _extend_line(left_line, height), _extend_line(right_line, height)


def _vertical_ranges(
    binary: NDArray[np.bool_],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
    """First and last ink row of every column, and whether the column has ink."""
    height = binary.shape[0]
    has_ink = binary.any(axis=0)
    tops = binary.argmax(axis=0)
    bottoms = height - 1 - binary[::-1].argmax(axis=0)
    return (
        np.asarray(tops, dtype=np.intp),
        np.asarray(bottoms, dtype=np.intp),
        np.asarray(has_ink, dtype=np.bool_),
    )


def _translate(line: Line, dx: float) -> Line:
    (x1, y1), (x2, y2) = line
    return ((x1 + dx, y1), (x2 + dx, y2))


def _extend_line(line: Line, height: int) -> Line:
    """Extend a line to the top (y = 0) and bottom (y = height) of the image."""
    top = line_intersection(line, ((0.0, 0.0), (1.0, 0.0)))
    bottom = line_intersection(line, ((0.0, float(height)), (1.0, float(height))))
    # A horizontal line never reaches either edge; the origin stands in for both.
    return (top or (0.0, 0.0), bottom or (0.0, 0.0))


class _SideProcessor:
    """Builds the convex envelope of the ink seen from the left or the right.

    Columns are fed from the outside in. The path holds the envelope's vertices
    from the top to the bottom.
    """

    def __init__(self, width: int, height: int, is_left: bool) -> None:
        self._is_left = is_left
        # The sign of the cross product that marks a concave turn.
        self._side_sign = -1 if is_left else 1
        self._max_segment_sqlen = (width * width + height * height) // (
            _MAX_SEGMENT_SQLEN_DIVISOR
        )
        self._leading_top: tuple[int, int] = (0, 0)
        self._path: list[tuple[int, int]] = []

    def process(self, x: int, top_y: int, bottom_y: int) -> None:
        """Add a column that has ink, from its first to its last ink row."""
        path = self._path
        if not path:
            self._leading_top = (x, top_y)
            path.append((x, top_y))
            if top_y != bottom_y:
                path.append((x, bottom_y))
            return

        if top_y < path[0][1]:
            top = (x, top_y)
            mid_idx = 0
            bottom_idx = 1
            while bottom_idx < len(path):
                if not self._is_concave(top, path[mid_idx], path[bottom_idx]):
                    break
                mid_idx += 1
                bottom_idx += 1
            # Trimming too much would let a heading wider than the text pull the
            # boundary outwards for the whole page.
            if not self._is_too_long(top, path[mid_idx]):
                del path[:mid_idx]
            path.insert(0, top)

        if bottom_y > path[-1][1]:
            bottom = (x, bottom_y)
            mid_idx = len(path) - 1
            top_idx = mid_idx - 1
            while top_idx >= 0:
                if not self._is_concave(path[top_idx], path[mid_idx], bottom):
                    break
                top_idx -= 1
                mid_idx -= 1
            if not self._is_too_long(bottom, path[mid_idx]):
                del path[mid_idx + 1 :]
            path.append(bottom)

    def _is_concave(
        self, top: tuple[int, int], mid: tuple[int, int], bottom: tuple[int, int]
    ) -> bool:
        cross = (mid[0] - top[0]) * (bottom[1] - mid[1]) - (bottom[0] - mid[0]) * (
            mid[1] - top[1]
        )
        return cross * self._side_sign < 0

    def _is_too_long(self, p1: tuple[int, int], p2: tuple[int, int]) -> bool:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return dx * dx + dy * dy > self._max_segment_sqlen

    def approximate_with_line(self) -> Line:
        """Fit a line to the steep part of the envelope, with RANSAC.

        Returns:
            Line: The fitted line, or a vertical line through the outermost ink
                column when the envelope has no steep segment.
        """
        segments = self._steep_segments()
        lead_x = self._leading_top[0]
        if not segments:
            return (
                (float(lead_x), float(self._leading_top[1])),
                (float(lead_x), float(self._leading_top[1]) + 1.0),
            )

        units = np.array([segment.unit for segment in segments])
        extents = np.array([segment.vertical_extent for segment in segments])

        def inliers_of(seed: int) -> tuple[NDArray[np.intp], int]:
            close = np.flatnonzero(units @ units[seed] > _COS_THRESHOLD)
            members = np.concatenate(([seed], close))
            return members, int(extents[members].sum())

        best_members: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        best_total = 0

        # Segments next to the outermost column are tried first, then random ones.
        distance_to_lead = np.array(
            [
                min(abs(segment.top[0] - lead_x), abs(segment.bottom[0] - lead_x))
                for segment in segments
            ]
        )
        nearest = np.argsort(distance_to_lead, kind="stable")[:_NUM_BEST_SEEDS]
        for seed in nearest:
            members, total = inliers_of(int(seed))
            if total > best_total:
                best_members, best_total = members, total
        generator = QtRandomGenerator(0)
        for _ in range(_RANSAC_ITERATIONS):
            seed = generator.generate() % len(segments)
            members, total = inliers_of(seed)
            if total > best_total:
                best_members, best_total = members, total

        return self._interpolate(
            [segments[i] for i in best_members],
        )

    def _steep_segments(self) -> list[_Segment]:
        segments = []
        for first, second in zip(self._path, self._path[1:], strict=False):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            if abs(dx) > abs(dy):
                continue
            length = math.hypot(dx, dy)
            segments.append(_Segment(first, second, (dx / length, dy / length), dy))
        return segments

    def _interpolate(self, segments: list[_Segment]) -> Line:
        """Average the segments' directions and place the line on the envelope."""
        accum_x = 0.0
        accum_y = 0.0
        accum_weight = 0.0
        for segment in segments:
            weight = math.sqrt(segment.vertical_extent)
            accum_x += weight * segment.unit[0]
            accum_y += weight * segment.unit[1]
            accum_weight += weight
        accum_x /= accum_weight
        accum_y /= accum_weight

        origin: Point = (float(self._path[0][0]), float(self._path[0][1]))
        normal_x, normal_y = -accum_y, accum_x
        if (not self._is_left) != (normal_x < 0):
            normal_x, normal_y = -normal_x, -normal_y
        # The normal now points from the boundary into the page. The line is
        # moved to the path vertex that lies furthest outside it.
        for vx, vy in self._path:
            if normal_x * (vx - origin[0]) + normal_y * (vy - origin[1]) < 0:
                origin = (float(vx), float(vy))
        return (origin, (origin[0] + accum_x, origin[1] + accum_y))
