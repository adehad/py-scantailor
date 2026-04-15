"""Distortion model for dewarping.

This module provides the DistortionModel class that stores the top and bottom
curves defining page distortion, used for cylindrical surface dewarping.

SIMPLIFICATIONS FROM C++ IMPLEMENTATION
========================================

The original C++ DistortionModel (src/dewarping/DistortionModel.cpp):
- Uses XML serialization via QDomDocument
- Has a Curve class wrapping XSpline with polyline caching
- Validates convexity of the corner quadrilateral

PYTHON SIMPLIFICATION:
- Uses numpy arrays for polylines
- Supports our XSpline class directly
- Simple convexity validation
- No XML serialization (can add later if needed)
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from scantailor.math.spline import XSpline


@dataclass
class Curve:
    """A curve representing a page edge (top or bottom).

    Can be constructed from either a polyline (array of points) or an XSpline.
    If constructed from a spline, the polyline is generated automatically.

    Attributes:
        polyline: Array of points representing the curve, shape (N, 2).
        spline: Optional XSpline for smooth curve representation.
    """

    polyline: NDArray[np.float64]
    spline: XSpline | None = None

    def __post_init__(self) -> None:
        """Validate and convert polyline."""
        self.polyline = np.asarray(self.polyline, dtype=np.float64)
        if self.polyline.ndim != 2 or self.polyline.shape[1] != 2:
            msg = "Polyline must have shape (N, 2)"
            raise ValueError(msg)

    @classmethod
    def from_spline(cls, spline: XSpline, max_dist: float = 1.0) -> Self:
        """Create a curve from an XSpline.

        Args:
            spline: The XSpline to use.
            max_dist: Maximum distance from spline to polyline approximation.

        Returns:
            Curve with both spline and polyline representation.
        """
        polyline = spline.to_polyline(max_dist_from_spline=max_dist)
        return cls(polyline=polyline, spline=spline)

    @classmethod
    def from_polyline(cls, polyline: NDArray[np.float64]) -> Self:
        """Create a curve from a polyline.

        Args:
            polyline: Array of points, shape (N, 2).

        Returns:
            Curve with polyline representation only.
        """
        return cls(polyline=polyline)

    def is_valid(self) -> bool:
        """Check if the curve is valid.

        A valid curve has at least 2 points and endpoints are distinct.
        """
        if len(self.polyline) < 2:
            return False
        # Check that endpoints are distinct
        return not np.allclose(self.polyline[0], self.polyline[-1], atol=1e-6)

    def matches(self, other: "Curve", tolerance: float = 0.01) -> bool:
        """Check if this curve approximately matches another.

        Args:
            other: Curve to compare with.
            tolerance: Maximum allowed distance between corresponding points.

        Returns:
            True if curves match within tolerance.
        """
        if len(self.polyline) != len(other.polyline):
            return False

        diffs = self.polyline - other.polyline
        distances_sq = np.sum(diffs * diffs, axis=1)
        return bool(np.all(distances_sq <= tolerance * tolerance))

    @property
    def start_point(self) -> NDArray[np.float64]:
        """Return the first point of the curve."""
        return self.polyline[0].copy()

    @property
    def end_point(self) -> NDArray[np.float64]:
        """Return the last point of the curve."""
        return self.polyline[-1].copy()


@dataclass
class DistortionModel:
    """Model of page distortion for dewarping.

    The distortion is defined by two curves: the top edge and bottom edge
    of the page. These curves, along with the vertical lines connecting
    their endpoints, form a curved quadrilateral that represents the
    visible page surface.

    Attributes:
        top_curve: Curve defining the top edge of the page.
        bottom_curve: Curve defining the bottom edge of the page.
    """

    top_curve: Curve | None = field(default=None)
    bottom_curve: Curve | None = field(default=None)

    def is_valid(self) -> bool:
        """Check if the distortion model is valid.

        A valid model has valid top and bottom curves, and their endpoints
        form a convex quadrilateral.
        """
        if self.top_curve is None or self.bottom_curve is None:
            return False

        if not self.top_curve.is_valid() or not self.bottom_curve.is_valid():
            return False

        # Check that corner points form a convex quadrilateral
        return self._corners_are_convex()

    def _corners_are_convex(self) -> bool:
        """Check if the corner quadrilateral is convex.

        The corners are: top-left, top-right, bottom-right, bottom-left.
        """
        if self.top_curve is None or self.bottom_curve is None:
            return False

        corners = np.array(
            [
                self.top_curve.start_point,
                self.top_curve.end_point,
                self.bottom_curve.end_point,
                self.bottom_curve.start_point,
            ]
        )

        # Check convexity by ensuring all cross products have same sign
        min_dot = np.inf
        max_dot = -np.inf

        for i in range(4):
            cur = corners[i]
            prev = corners[(i + 3) % 4]
            next_pt = corners[(i + 1) % 4]

            # Normal to edge from prev to cur
            edge = cur - prev
            normal = np.array([-edge[1], edge[0]])

            # Dot product with edge from cur to next
            dot = np.dot(normal, next_pt - cur)

            min_dot = min(min_dot, dot)
            max_dot = max(max_dot, dot)

        # Convex if all dots have same sign
        if min_dot * max_dot <= 0:
            return False

        # Check for degenerate cases (points too close together)
        if abs(min_dot) < 0.01 or abs(max_dot) < 0.01:
            return False

        return True

    def matches(self, other: "DistortionModel") -> bool:
        """Check if this model approximately matches another.

        Args:
            other: DistortionModel to compare with.

        Returns:
            True if models match.
        """
        this_valid = self.is_valid()
        other_valid = other.is_valid()

        if not this_valid and not other_valid:
            return True
        if this_valid != other_valid:
            return False

        if self.top_curve is None or other.top_curve is None:
            return False
        if self.bottom_curve is None or other.bottom_curve is None:
            return False

        if not self.top_curve.matches(other.top_curve):
            return False
        if not self.bottom_curve.matches(other.bottom_curve):
            return False

        return True

    def bounding_box(
        self,
        transform: NDArray[np.float64] | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Compute the bounding box of the distorted region.

        Args:
            transform: Optional 3x3 transformation matrix to apply to points.

        Returns:
            Tuple of (left, top, right, bottom), or None if model is invalid.
        """
        if self.top_curve is None or self.bottom_curve is None:
            return None

        # Collect all points
        points = np.vstack([self.top_curve.polyline, self.bottom_curve.polyline])

        # Apply transform if provided
        if transform is not None:
            # Convert to homogeneous coordinates
            ones = np.ones((len(points), 1))
            points_h = np.hstack([points, ones])
            transformed = points_h @ transform.T
            # Normalize
            points = transformed[:, :2] / transformed[:, 2:3]

        left = float(np.min(points[:, 0]))
        right = float(np.max(points[:, 0]))
        top = float(np.min(points[:, 1]))
        bottom = float(np.max(points[:, 1]))

        return left, top, right, bottom


def create_distortion_model(
    top_polyline: NDArray[np.float64],
    bottom_polyline: NDArray[np.float64],
) -> DistortionModel:
    """Create a distortion model from two polylines.

    Args:
        top_polyline: Points defining the top edge, shape (N, 2).
        bottom_polyline: Points defining the bottom edge, shape (M, 2).

    Returns:
        DistortionModel with the given curves.
    """
    return DistortionModel(
        top_curve=Curve.from_polyline(top_polyline),
        bottom_curve=Curve.from_polyline(bottom_polyline),
    )


def create_distortion_model_from_splines(
    top_spline: XSpline,
    bottom_spline: XSpline,
    max_dist: float = 1.0,
) -> DistortionModel:
    """Create a distortion model from two XSplines.

    Args:
        top_spline: Spline defining the top edge.
        bottom_spline: Spline defining the bottom edge.
        max_dist: Maximum distance from spline to polyline approximation.

    Returns:
        DistortionModel with the given curves.
    """
    return DistortionModel(
        top_curve=Curve.from_spline(top_spline, max_dist),
        bottom_curve=Curve.from_spline(bottom_spline, max_dist),
    )
