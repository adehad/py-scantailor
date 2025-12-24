"""Math utilities for geometric operations.

This module provides classes and functions for working with 2D geometry:
- Lines and line segments
- Polygons
- Affine transformations
- Perspective (homography) transformations
- X-Splines for smooth curve fitting
- Linear and quadratic functions for optimization
"""

from scantailor.math.functions import (
    LinearFunction,
    QuadraticFunction,
    QuadraticGradient,
)
from scantailor.math.homography import Homography, warp_perspective
from scantailor.math.line import (
    Line2D,
    LineSegment,
    line_bounded_by_rect,
    line_intersection_scalar,
    point_to_line_distance,
    point_to_segment_distance,
    sides_of_line,
)
from scantailor.math.polygon import Polygon, polygon_area, polygon_contains_point
from scantailor.math.spline import (
    ControlPoint,
    LinearCoefficient,
    PointAndDerivatives,
    XSpline,
)
from scantailor.math.spline_fitter import (
    FitResult,
    fit_spline_initial_guess,
    fit_spline_to_points,
    fit_spline_to_polyline,
    sample_spline_by_arc_length,
    sample_spline_uniformly,
)
from scantailor.math.transform import AffineTransform, compose_transforms

__all__ = [
    # Transform classes
    AffineTransform.__name__,
    Homography.__name__,
    # Spline classes
    ControlPoint.__name__,
    LinearCoefficient.__name__,
    PointAndDerivatives.__name__,
    XSpline.__name__,
    FitResult.__name__,
    # Geometry classes
    Line2D.__name__,
    LineSegment.__name__,
    Polygon.__name__,
    # Function classes
    LinearFunction.__name__,
    QuadraticFunction.__name__,
    QuadraticGradient.__name__,
    # Transform functions
    compose_transforms.__name__,
    warp_perspective.__name__,
    # Spline functions
    fit_spline_initial_guess.__name__,
    fit_spline_to_points.__name__,
    fit_spline_to_polyline.__name__,
    sample_spline_by_arc_length.__name__,
    sample_spline_uniformly.__name__,
    # Line functions
    line_bounded_by_rect.__name__,
    line_intersection_scalar.__name__,
    point_to_line_distance.__name__,
    point_to_segment_distance.__name__,
    sides_of_line.__name__,
    # Polygon functions
    polygon_area.__name__,
    polygon_contains_point.__name__,
]
