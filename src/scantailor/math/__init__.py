"""Math utilities for geometric operations.

This module provides classes and functions for working with 2D geometry:
- Lines and line segments
- Polygons
- Affine transformations
- Perspective (homography) transformations
"""

from scantailor.math.homography import Homography, warp_perspective
from scantailor.math.line import (
    Line2D,
    LineSegment,
    point_to_line_distance,
    point_to_segment_distance,
)
from scantailor.math.polygon import Polygon, polygon_area, polygon_contains_point
from scantailor.math.transform import AffineTransform, compose_transforms

__all__ = [
    AffineTransform.__name__,
    Homography.__name__,
    Line2D.__name__,
    LineSegment.__name__,
    Polygon.__name__,
    compose_transforms.__name__,
    point_to_line_distance.__name__,
    point_to_segment_distance.__name__,
    polygon_area.__name__,
    polygon_contains_point.__name__,
    warp_perspective.__name__,
]
