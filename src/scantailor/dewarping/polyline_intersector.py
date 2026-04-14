"""Polyline intersection utilities for dewarping.

This module provides the PolylineIntersector class that efficiently finds
intersections between lines and polylines, used for mapping between
warped and dewarped coordinate systems.

SIMPLIFICATIONS FROM C++ IMPLEMENTATION
========================================

The original C++ PolylineIntersector (src/math/PolylineIntersector.cpp) uses:
- A Hint class that tracks last segment and search direction for O(1) sequential access
- Binary search fallback for random access
- ToLineProjector for projecting points onto lines

PYTHON SIMPLIFICATION:
- Uses numpy for vectorized operations
- Simpler approach: checks all segments when hint misses
- Uses numpy line intersection formulas directly
- Performance is adequate for typical polylines (< 1000 points)

The intersector finds where a line crosses a polyline, with optimization
for sequential queries where intersections move monotonically along the polyline.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class PolylineIntersector:
    """Finds intersections between lines and a polyline.

    Efficiently computes where an infinite line intersects a polyline,
    with optimization for sequential queries.

    Attributes:
        polyline: Array of shape (N, 2) with polyline vertices.
    """

    polyline: NDArray[np.float64]
    _last_segment: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate polyline and convert to numpy array."""
        self.polyline = np.asarray(self.polyline, dtype=np.float64)
        if self.polyline.ndim != 2 or self.polyline.shape[1] != 2:
            msg = "Polyline must have shape (N, 2)"
            raise ValueError(msg)
        if len(self.polyline) < 2:
            msg = "Polyline must have at least 2 points"
            raise ValueError(msg)

    @property
    def num_segments(self) -> int:
        """Return the number of segments in the polyline."""
        return len(self.polyline) - 1

    def intersect(
        self,
        line_p1: NDArray[np.floating],
        line_p2: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Find intersection of a line with the polyline.

        The line is defined by two points (infinite line through both).

        Args:
            line_p1: First point on the line [x, y].
            line_p2: Second point on the line [x, y].

        Returns:
            Intersection point [x, y]. If no intersection exists,
            returns the closest polyline endpoint projected onto the line.
        """
        line_p1 = np.asarray(line_p1, dtype=np.float64)
        line_p2 = np.asarray(line_p2, dtype=np.float64)

        # Compute line normal vector
        line_dir = line_p2 - line_p1
        normal = np.array([-line_dir[1], line_dir[0]], dtype=np.float64)

        # Check the last segment first (optimization for sequential queries)
        if self._segment_intersects_line(self._last_segment, line_p1, normal):
            return self._intersect_with_segment(self._last_segment, line_p1, line_p2)

        # Check adjacent segments
        next_seg = self._last_segment + 1
        if next_seg < self.num_segments and self._segment_intersects_line(
            next_seg, line_p1, normal
        ):
            self._last_segment = next_seg
            return self._intersect_with_segment(next_seg, line_p1, line_p2)

        prev_seg = self._last_segment - 1
        if prev_seg >= 0 and self._segment_intersects_line(prev_seg, line_p1, normal):
            self._last_segment = prev_seg
            return self._intersect_with_segment(prev_seg, line_p1, line_p2)

        # Check if line doesn't intersect polyline at all
        intersection = self._try_intersecting_outside_polyline(line_p1, line_p2, normal)
        if intersection is not None:
            return intersection

        # Binary search for the intersecting segment
        segment = self._find_intersecting_segment(line_p1, normal)
        if segment >= 0:
            self._last_segment = segment
            return self._intersect_with_segment(segment, line_p1, line_p2)

        # Fallback: return projection of nearest endpoint
        return self._project_point_to_line(self.polyline[0], line_p1, line_p2)

    def _segment_intersects_line(
        self,
        segment: int,
        line_origin: NDArray[np.floating],
        line_normal: NDArray[np.floating],
    ) -> bool:
        """Check if a segment intersects with the line.

        Args:
            segment: Segment index.
            line_origin: A point on the line.
            line_normal: Normal vector of the line.

        Returns:
            True if the segment crosses the line.
        """
        if segment < 0 or segment >= self.num_segments:
            return False

        p1 = self.polyline[segment]
        p2 = self.polyline[segment + 1]

        # Check if endpoints are on opposite sides of the line
        dot1 = np.dot(p1 - line_origin, line_normal)
        dot2 = np.dot(p2 - line_origin, line_normal)

        return dot1 * dot2 <= 0

    def _intersect_with_segment(
        self,
        segment: int,
        line_p1: NDArray[np.floating],
        line_p2: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Compute intersection point with a specific segment.

        Args:
            segment: Segment index.
            line_p1: First point on the line.
            line_p2: Second point on the line.

        Returns:
            Intersection point.
        """
        seg_p1 = self.polyline[segment]
        seg_p2 = self.polyline[segment + 1]

        intersection = self._line_line_intersection(line_p1, line_p2, seg_p1, seg_p2)
        if intersection is not None:
            return intersection

        # Lines are parallel - return segment midpoint
        return (seg_p1 + seg_p2) / 2

    def _line_line_intersection(
        self,
        p1: NDArray[np.floating],
        p2: NDArray[np.floating],
        p3: NDArray[np.floating],
        p4: NDArray[np.floating],
    ) -> NDArray[np.float64] | None:
        """Compute intersection of two lines.

        Line 1 passes through p1 and p2.
        Line 2 passes through p3 and p4.

        Args:
            p1: First point defining line 1.
            p2: Second point defining line 1.
            p3: First point defining line 2.
            p4: Second point defining line 2.

        Returns:
            Intersection point, or None if lines are parallel.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return np.array([x, y], dtype=np.float64)

    def _try_intersecting_outside_polyline(
        self,
        line_p1: NDArray[np.floating],
        line_p2: NDArray[np.floating],
        line_normal: NDArray[np.floating],
    ) -> NDArray[np.float64] | None:
        """Check if line misses polyline entirely, return closest projection.

        Args:
            line_p1: First point on the line.
            line_p2: Second point on the line.
            line_normal: Normal vector of the line.

        Returns:
            Projection of nearest endpoint if line misses polyline, else None.
        """
        front_vec = self.polyline[0] - line_p1
        back_vec = self.polyline[-1] - line_p1

        front_dot = np.dot(front_vec, line_normal)
        back_dot = np.dot(back_vec, line_normal)

        # If same sign, line doesn't intersect
        if front_dot * back_dot > 0:
            if abs(front_dot) < abs(back_dot):
                self._last_segment = 0
                return self._project_point_to_line(self.polyline[0], line_p1, line_p2)
            else:
                self._last_segment = self.num_segments - 1
                return self._project_point_to_line(self.polyline[-1], line_p1, line_p2)

        return None

    def _find_intersecting_segment(
        self,
        line_origin: NDArray[np.floating],
        line_normal: NDArray[np.floating],
    ) -> int:
        """Binary search for the segment that intersects the line.

        Args:
            line_origin: A point on the line.
            line_normal: Normal vector of the line.

        Returns:
            Segment index, or -1 if not found.
        """
        left = 0
        right = len(self.polyline) - 1

        left_dot = np.dot(self.polyline[left] - line_origin, line_normal)

        while left + 1 < right:
            mid = (left + right) >> 1
            mid_dot = np.dot(self.polyline[mid] - line_origin, line_normal)

            if mid_dot * left_dot <= 0:
                right = mid
            else:
                left = mid
                left_dot = mid_dot

        return left

    def _project_point_to_line(
        self,
        point: NDArray[np.floating],
        line_p1: NDArray[np.floating],
        line_p2: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Project a point onto a line.

        Args:
            point: Point to project.
            line_p1: First point on the line.
            line_p2: Second point on the line.

        Returns:
            Projection of point onto line.
        """
        line_vec = line_p2 - line_p1
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq < 1e-10:
            return np.asarray(line_p1, dtype=np.float64).copy()

        t = np.dot(point - line_p1, line_vec) / line_len_sq
        return np.asarray(line_p1 + t * line_vec, dtype=np.float64)


def project_point_to_line(
    point: NDArray[np.floating],
    line_p1: NDArray[np.floating],
    line_p2: NDArray[np.floating],
) -> tuple[NDArray[np.float64], float]:
    """Project a point onto a line and return projection and scalar.

    Args:
        point: Point to project [x, y].
        line_p1: First point defining the line.
        line_p2: Second point defining the line.

    Returns:
        Tuple of (projection_point, projection_scalar).
        The scalar is 0 at line_p1, 1 at line_p2.
    """
    pt: NDArray[np.float64] = np.asarray(point, dtype=np.float64)
    p1: NDArray[np.float64] = np.asarray(line_p1, dtype=np.float64)
    p2: NDArray[np.float64] = np.asarray(line_p2, dtype=np.float64)

    line_vec = p2 - p1
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-10:
        return p1.copy(), 0.0

    t = float(np.dot(pt - p1, line_vec) / line_len_sq)
    projection: NDArray[np.float64] = p1 + t * line_vec
    return projection, t
