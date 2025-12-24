"""Line and line segment utilities.

This module provides classes and functions for working with lines and line segments
in 2D space, including distance calculations and projections.
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator


class Line2D(BaseModel):
    """Represents a 2D line in the form ax + by + c = 0.

    This is the implicit form of a line equation, where any point (x, y)
    on the line satisfies ax + by + c = 0.

    Attributes:
        a: Coefficient for x.
        b: Coefficient for y.
        c: Constant term.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    a: float
    b: float
    c: float

    @field_validator("a", "b")
    @classmethod
    def validate_not_both_zero(cls, v: float, info) -> float:
        """Validate that a and b are not both zero."""
        if info.field_name == "b":
            # Only check when b is being validated (after a is set)
            data = info.data
            if "a" in data and abs(data["a"]) < 1e-10 and abs(v) < 1e-10:
                msg = "Coefficients a and b cannot both be zero"
                raise ValueError(msg)
        return v

    @classmethod
    def from_points(
        cls, p1: NDArray[np.floating], p2: NDArray[np.floating]
    ) -> "Line2D":
        """Create a line from two points.

        Args:
            p1: First point as array [x, y].
            p2: Second point as array [x, y].

        Returns:
            Line2D: Line passing through both points.

        Raises:
            ValueError: If points are identical.
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)

        if p1.shape != (2,) or p2.shape != (2,):
            msg = "Points must be 2D arrays with shape (2,)"
            raise ValueError(msg)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            msg = "Points must be distinct"
            raise ValueError(msg)

        # Normal vector to the line is perpendicular to direction vector
        # Direction: (dx, dy), Normal: (-dy, dx)
        a = -dy
        b = dx
        c = -(a * p1[0] + b * p1[1])

        return cls(a=a, b=b, c=c)

    @classmethod
    def from_point_and_normal(
        cls, point: NDArray[np.floating], normal: NDArray[np.floating]
    ) -> "Line2D":
        """Create a line from a point and normal vector.

        Args:
            point: Point on the line as array [x, y].
            normal: Normal vector as array [nx, ny].

        Returns:
            Line2D: Line passing through point with given normal.

        Raises:
            ValueError: If normal is zero vector.
        """
        point = np.asarray(point, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)

        if point.shape != (2,) or normal.shape != (2,):
            msg = "Point and normal must be 2D arrays with shape (2,)"
            raise ValueError(msg)

        a = normal[0]
        b = normal[1]

        if abs(a) < 1e-10 and abs(b) < 1e-10:
            msg = "Normal vector cannot be zero"
            raise ValueError(msg)

        c = -(a * point[0] + b * point[1])

        return cls(a=a, b=b, c=c)

    def distance_to_point(self, point: NDArray[np.floating]) -> float:
        """Calculate perpendicular distance from a point to the line.

        Args:
            point: Point as array [x, y].

        Returns:
            float: Perpendicular distance from point to line.
        """
        point = np.asarray(point, dtype=np.float64)

        if point.shape != (2,):
            msg = "Point must be a 2D array with shape (2,)"
            raise ValueError(msg)

        numerator = abs(self.a * point[0] + self.b * point[1] + self.c)
        denominator = np.sqrt(self.a**2 + self.b**2)

        return numerator / denominator

    def signed_distance_to_point(self, point: NDArray[np.floating]) -> float:
        """Calculate signed distance from a point to the line.

        The sign indicates which side of the line the point is on.

        Args:
            point: Point as array [x, y].

        Returns:
            float: Signed distance from point to line.
        """
        point = np.asarray(point, dtype=np.float64)

        if point.shape != (2,):
            msg = "Point must be a 2D array with shape (2,)"
            raise ValueError(msg)

        numerator = self.a * point[0] + self.b * point[1] + self.c
        denominator = np.sqrt(self.a**2 + self.b**2)

        return numerator / denominator

    def project_point(self, point: NDArray[np.floating]) -> NDArray[np.floating]:
        """Project a point onto the line.

        Args:
            point: Point to project as array [x, y].

        Returns:
            NDArray[np.floating]: Projection of point onto line.
        """
        point = np.asarray(point, dtype=np.float64)

        if point.shape != (2,):
            msg = "Point must be a 2D array with shape (2,)"
            raise ValueError(msg)

        # Project along normal direction
        signed_dist = self.signed_distance_to_point(point)
        norm_sq = self.a**2 + self.b**2
        normal = np.array([self.a, self.b], dtype=np.float64)

        return point - (signed_dist * normal / np.sqrt(norm_sq))

    def normalize(self) -> "Line2D":
        """Return a normalized version of the line.

        Returns a line with the same geometry but normalized such that
        a^2 + b^2 = 1.

        Returns:
            Line2D: Normalized line.
        """
        norm = np.sqrt(self.a**2 + self.b**2)
        return Line2D(a=self.a / norm, b=self.b / norm, c=self.c / norm)


class LineSegment(BaseModel):
    """Represents a line segment defined by two endpoints.

    Attributes:
        p1: First endpoint as array [x, y].
        p2: Second endpoint as array [x, y].
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    p1: NDArray[np.floating]
    p2: NDArray[np.floating]

    @field_validator("p1", "p2")
    @classmethod
    def validate_point_shape(cls, v: NDArray[np.floating]) -> NDArray[np.floating]:
        """Validate that points are 2D arrays."""
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (2,):
            msg = "Points must be 2D arrays with shape (2,)"
            raise ValueError(msg)
        return v

    def __init__(self, **data):
        """Initialize line segment."""
        super().__init__(**data)
        # Validate that points are distinct
        if np.allclose(self.p1, self.p2, atol=1e-10):
            msg = "Line segment endpoints must be distinct"
            raise ValueError(msg)

    def length(self) -> float:
        """Calculate the length of the line segment.

        Returns:
            float: Length of the segment.
        """
        return float(np.linalg.norm(self.p2 - self.p1))

    def direction(self) -> NDArray[np.floating]:
        """Get the direction vector of the line segment.

        Returns:
            NDArray[np.floating]: Unit direction vector from p1 to p2.
        """
        vec = self.p2 - self.p1
        return vec / np.linalg.norm(vec)

    def to_line(self) -> "Line2D":
        """Convert the line segment to an infinite line.

        Returns:
            Line2D: Line containing this segment.
        """
        return Line2D.from_points(self.p1, self.p2)

    def point_at(self, t: float) -> NDArray[np.floating]:
        """Get a point on the line segment at parameter t.

        Args:
            t: Parameter value. t=0 gives p1, t=1 gives p2.

        Returns:
            NDArray[np.floating]: Point at parameter t.
        """
        return (1 - t) * self.p1 + t * self.p2

    def distance_to_point(self, point: NDArray[np.floating]) -> float:
        """Calculate distance from a point to the line segment.

        Unlike Line2D.distance_to_point, this returns the distance to the
        nearest point on the segment (not the infinite line).

        Args:
            point: Point as array [x, y].

        Returns:
            float: Distance from point to nearest point on segment.
        """
        point = np.asarray(point, dtype=np.float64)

        if point.shape != (2,):
            msg = "Point must be a 2D array with shape (2,)"
            raise ValueError(msg)

        # Vector from p1 to point
        v = point - self.p1
        # Vector from p1 to p2
        w = self.p2 - self.p1

        # Project v onto w
        c1 = np.dot(v, w)
        if c1 <= 0:
            # Point is closest to p1
            return float(np.linalg.norm(point - self.p1))

        c2 = np.dot(w, w)
        if c1 >= c2:
            # Point is closest to p2
            return float(np.linalg.norm(point - self.p2))

        # Point projects onto the segment
        t = c1 / c2
        projection = self.point_at(t)
        return float(np.linalg.norm(point - projection))

    def project_point(self, point: NDArray[np.floating]) -> NDArray[np.floating]:
        """Project a point onto the line segment.

        The projection is clamped to the segment endpoints.

        Args:
            point: Point to project as array [x, y].

        Returns:
            NDArray[np.floating]: Projection of point onto segment.
        """
        point = np.asarray(point, dtype=np.float64)

        if point.shape != (2,):
            msg = "Point must be a 2D array with shape (2,)"
            raise ValueError(msg)

        # Vector from p1 to point
        v = point - self.p1
        # Vector from p1 to p2
        w = self.p2 - self.p1

        # Project v onto w
        c1 = np.dot(v, w)
        if c1 <= 0:
            return self.p1.copy()

        c2 = np.dot(w, w)
        if c1 >= c2:
            return self.p2.copy()

        t = c1 / c2
        return self.point_at(t)


def point_to_line_distance(point: NDArray[np.floating], line: Line2D) -> float:
    """Calculate perpendicular distance from a point to a line.

    Args:
        point: Point as array [x, y].
        line: Line to measure distance to.

    Returns:
        float: Perpendicular distance from point to line.
    """
    return line.distance_to_point(point)


def point_to_segment_distance(
    point: NDArray[np.floating], segment: LineSegment
) -> float:
    """Calculate distance from a point to a line segment.

    Args:
        point: Point as array [x, y].
        segment: Line segment to measure distance to.

    Returns:
        float: Distance from point to nearest point on segment.
    """
    return segment.distance_to_point(point)


def sides_of_line(
    line: LineSegment,
    p1: NDArray[np.floating],
    p2: NDArray[np.floating],
) -> float:
    """Check if two points are on the same or different sides of a line.

    This function determines the relative position of two points with
    respect to a line (not a line segment - the endpoints don't matter).

    Args:
        line: A line segment defining the line direction.
        p1: First point as array [x, y].
        p2: Second point as array [x, y].

    Returns:
        - Negative value if points are on different sides of the line.
        - Positive value if points are on the same side of the line.
        - Zero if one or both points are on the line.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    # Get normal vector (perpendicular to line direction)
    direction = line.p2 - line.p1
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    # Vectors from line start to each point
    vec1 = p1 - line.p1
    vec2 = p2 - line.p1

    # Dot products with normal give signed distances
    dot1 = np.dot(normal, vec1)
    dot2 = np.dot(normal, vec2)

    # Product: positive if same side, negative if different sides
    return float(dot1 * dot2)


def line_intersection_scalar(
    line1: LineSegment,
    line2: LineSegment,
) -> tuple[float, float] | None:
    """Find intersection scalars for two line segments.

    Computes scalars s1 and s2 such that:
    - line1.point_at(s1) is the intersection point
    - line2.point_at(s2) is the intersection point

    Note that the actual intersection may be outside the segment
    bounds (s1 or s2 outside [0, 1]).

    Args:
        line1: First line segment.
        line2: Second line segment.

    Returns:
        Tuple (s1, s2) of scalars, or None if lines are parallel
        or either line has zero length.
    """
    p1 = line1.p1
    p2 = line2.p1
    v1 = line1.p2 - line1.p1
    v2 = line2.p2 - line2.p1

    # Solve: p1 + s1 * v1 = p2 + s2 * v2
    # Using Cramer's rule on the 2x2 system
    det_A = v2[0] * v1[1] - v1[0] * v2[1]

    if abs(det_A) < np.finfo(float).eps:
        return None

    b = p2 - p1
    r_det_A = 1.0 / det_A

    s1 = (v2[0] * b[1] - b[0] * v2[1]) * r_det_A
    s2 = (v1[0] * b[1] - b[0] * v1[1]) * r_det_A

    return (float(s1), float(s2))


def line_bounded_by_rect(
    line: LineSegment,
    rect: tuple[float, float, float, float],
) -> LineSegment | None:
    """Clip a line to the bounds of a rectangle.

    Given a line (extended infinitely in both directions) and a rectangle,
    returns the portion of the line that lies within the rectangle.

    Args:
        line: Line segment defining the line direction (extended infinitely).
        rect: Rectangle as (x, y, width, height).

    Returns:
        New LineSegment clipped to rectangle bounds, or None if the line
        doesn't intersect the rectangle.
    """
    x, y, w, h = rect

    # Four sides of rectangle as line segments
    rect_lines = [
        LineSegment(
            p1=np.array([x, y], dtype=np.float64),
            p2=np.array([x + w, y], dtype=np.float64),
        ),  # top
        LineSegment(
            p1=np.array([x, y + h], dtype=np.float64),
            p2=np.array([x + w, y + h], dtype=np.float64),
        ),  # bottom
        LineSegment(
            p1=np.array([x, y], dtype=np.float64),
            p2=np.array([x, y + h], dtype=np.float64),
        ),  # left
        LineSegment(
            p1=np.array([x + w, y], dtype=np.float64),
            p2=np.array([x + w, y + h], dtype=np.float64),
        ),  # right
    ]

    min_s = float("inf")
    max_s = float("-inf")

    for rect_line in rect_lines:
        result = line_intersection_scalar(rect_line, line)
        if result is None:
            # Line is parallel to this rect edge
            continue

        s1, s2 = result

        # s1 is the scalar on rect_line, must be in [0, 1] to be on rect edge
        if s1 < 0 or s1 > 1:
            continue

        # s2 is the scalar on our line
        if s2 > max_s:
            max_s = s2
        if s2 < min_s:
            min_s = s2

    if max_s > min_s:
        return LineSegment(
            p1=line.point_at(min_s),
            p2=line.point_at(max_s),
        )
    return None
