"""Polygon utilities for geometric operations.

This module provides the Polygon class for working with polygons in 2D space,
including area calculation, point containment tests, and bounding box computation.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator


class Polygon(BaseModel):
    """Represents a 2D polygon defined by vertices.

    The polygon is defined by a sequence of vertices in counter-clockwise order.
    The vertices form a closed polygon (last vertex implicitly connects to first).

    Attributes:
        vertices: Array of vertices with shape (N, 2) where N >= 3.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    vertices: NDArray[np.floating]

    @field_validator("vertices")
    @classmethod
    def validate_vertices(cls, v: NDArray[np.floating]) -> NDArray[np.floating]:
        """Validate that vertices form a valid polygon."""
        v = np.asarray(v, dtype=np.float64)

        if v.ndim != 2 or v.shape[1] != 2:
            msg = "Vertices must be a 2D array with shape (N, 2)"
            raise ValueError(msg)

        if v.shape[0] < 3:
            msg = "Polygon must have at least 3 vertices"
            raise ValueError(msg)

        return v

    @classmethod
    def from_rectangle(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> "Polygon":
        """Create a rectangular polygon.

        Args:
            x: X coordinate of top-left corner.
            y: Y coordinate of top-left corner.
            width: Width of rectangle.
            height: Height of rectangle.

        Returns:
            Polygon: Rectangular polygon.

        Raises:
            ValueError: If width or height is non-positive.
        """
        if width <= 0 or height <= 0:
            msg = "Width and height must be positive"
            raise ValueError(msg)

        vertices = np.array(
            [
                [x, y],  # Top-left
                [x + width, y],  # Top-right
                [x + width, y + height],  # Bottom-right
                [x, y + height],  # Bottom-left
            ],
            dtype=np.float64,
        )

        return cls(vertices=vertices)

    def area(self) -> float:
        """Calculate the signed area of the polygon.

        Uses the shoelace formula. Positive area indicates counter-clockwise
        vertex ordering, negative indicates clockwise.

        Returns:
            float: Signed area of the polygon.
        """
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]

        # Shoelace formula: A = 0.5 * |sum(x[i] * y[i+1] - x[i+1] * y[i])|
        return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def unsigned_area(self) -> float:
        """Calculate the unsigned area of the polygon.

        Returns:
            float: Absolute area of the polygon.
        """
        return abs(self.area())

    def centroid(self) -> NDArray[np.floating]:
        """Calculate the centroid (center of mass) of the polygon.

        Returns:
            NDArray[np.floating]: Centroid as array [x, y].
        """
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]

        # Calculate signed area
        a = self.area()

        if abs(a) < 1e-10:
            # Degenerate polygon, return center of vertices
            return np.mean(self.vertices, axis=0)

        # Calculate centroid using shoelace-based formula
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * a)
        cy = np.sum((y + np.roll(y, -1)) * cross) / (6 * a)

        return np.array([cx, cy], dtype=np.float64)

    def contains(self, point: NDArray[np.floating]) -> bool:
        """Test if a point is inside the polygon.

        Uses the even-odd rule (ray casting algorithm).

        Args:
            point: Point to test as array [x, y].

        Returns:
            bool: True if point is inside or on the boundary, False otherwise.
        """
        point = np.asarray(point, dtype=np.float64)

        if point.shape != (2,):
            msg = "Point must be a 2D array with shape (2,)"
            raise ValueError(msg)

        # Use OpenCV's pointPolygonTest
        # Returns +1 (inside), 0 (on edge), -1 (outside)
        result = cv2.pointPolygonTest(
            self.vertices.astype(np.float32), tuple(point), False
        )
        return result >= 0

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Calculate the axis-aligned bounding box of the polygon.

        Returns:
            tuple[float, float, float, float]: Bounding box as (x, y, width, height)
                where (x, y) is the top-left corner.
        """
        min_x = np.min(self.vertices[:, 0])
        max_x = np.max(self.vertices[:, 0])
        min_y = np.min(self.vertices[:, 1])
        max_y = np.max(self.vertices[:, 1])

        return (
            float(min_x),
            float(min_y),
            float(max_x - min_x),
            float(max_y - min_y),
        )

    def intersects(self, other: "Polygon") -> bool:
        """Test if this polygon intersects with another polygon.

        Two polygons intersect if they share any common area or if one
        is contained within the other.

        Args:
            other: Another polygon to test intersection with.

        Returns:
            bool: True if polygons intersect, False otherwise.
        """
        # Check if any vertices of one polygon are inside the other
        for vertex in self.vertices:
            if other.contains(vertex):
                return True

        for vertex in other.vertices:
            if self.contains(vertex):
                return True

        # Check for edge intersections
        for i in range(len(self.vertices)):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % len(self.vertices)]

            for j in range(len(other.vertices)):
                p3 = other.vertices[j]
                p4 = other.vertices[(j + 1) % len(other.vertices)]

                if _segments_intersect(p1, p2, p3, p4):
                    return True

        return False

    def translate(self, dx: float, dy: float) -> "Polygon":
        """Translate the polygon by a given offset.

        Args:
            dx: Translation in x direction.
            dy: Translation in y direction.

        Returns:
            Polygon: Translated polygon.
        """
        offset = np.array([dx, dy], dtype=np.float64)
        new_vertices = self.vertices + offset
        return Polygon(vertices=new_vertices)

    def scale(
        self,
        scale_x: float,
        scale_y: float | None = None,
        center: NDArray[np.floating] | None = None,
    ) -> "Polygon":
        """Scale the polygon by given factors.

        Args:
            scale_x: Scale factor in x direction.
            scale_y: Scale factor in y direction. If None, uses scale_x.
            center: Center point for scaling. If None, uses origin (0, 0).

        Returns:
            Polygon: Scaled polygon.
        """
        if scale_y is None:
            scale_y = scale_x

        if center is None:
            center = np.array([0.0, 0.0], dtype=np.float64)
        else:
            center = np.asarray(center, dtype=np.float64)

        # Translate to origin, scale, translate back
        translated = self.vertices - center
        scaled = translated * np.array([scale_x, scale_y], dtype=np.float64)
        new_vertices = scaled + center

        return Polygon(vertices=new_vertices)

    def to_cv2_contour(self) -> NDArray[np.int32]:
        """Convert polygon to OpenCV contour format.

        Returns:
            NDArray[np.int32]: Contour array with shape (N, 1, 2).
        """
        return self.vertices.astype(np.int32).reshape((-1, 1, 2))


def _segments_intersect(
    p1: NDArray[np.floating],
    p2: NDArray[np.floating],
    p3: NDArray[np.floating],
    p4: NDArray[np.floating],
) -> bool:
    """Test if two line segments intersect.

    Args:
        p1: First point of first segment.
        p2: Second point of first segment.
        p3: First point of second segment.
        p4: Second point of second segment.

    Returns:
        bool: True if segments intersect, False otherwise.
    """
    # Using the orientation method
    def orientation(
        a: NDArray[np.floating], b: NDArray[np.floating], c: NDArray[np.floating]
    ) -> int:
        """Calculate orientation of ordered triplet (a, b, c).

        Returns:
            0 if collinear, 1 if clockwise, 2 if counterclockwise.
        """
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    def on_segment(
        a: NDArray[np.floating], b: NDArray[np.floating], c: NDArray[np.floating]
    ) -> bool:
        """Check if point b lies on segment ac (assumes collinear)."""
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases - collinear points
    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return False


def polygon_area(vertices: NDArray[np.floating]) -> float:
    """Calculate the signed area of a polygon.

    Args:
        vertices: Array of vertices with shape (N, 2).

    Returns:
        float: Signed area of the polygon.
    """
    return Polygon(vertices=vertices).area()


def polygon_contains_point(
    vertices: NDArray[np.floating], point: NDArray[np.floating]
) -> bool:
    """Test if a point is inside a polygon.

    Args:
        vertices: Array of polygon vertices with shape (N, 2).
        point: Point to test as array [x, y].

    Returns:
        bool: True if point is inside or on the boundary, False otherwise.
    """
    return Polygon(vertices=vertices).contains(point)
