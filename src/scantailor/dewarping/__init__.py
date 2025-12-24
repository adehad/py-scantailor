"""Dewarping module for correcting page curvature.

This module provides tools for correcting perspective distortion from
curved page surfaces (like book spines) using a cylindrical surface model.

The dewarping process involves:
1. Defining the page edges as curves (top and bottom)
2. Creating a distortion model from these curves
3. Using the cylindrical surface dewarper to map between
   warped (curved) and dewarped (flat) coordinates

Example:
    >>> from scantailor.dewarping import (
    ...     CylindricalSurfaceDewarper,
    ...     create_distortion_model,
    ... )
    >>> import numpy as np
    >>>
    >>> # Define page edges as polylines
    >>> top_curve = np.array([[0, 10], [100, 5], [200, 15]])
    >>> bottom_curve = np.array([[0, 290], [100, 295], [200, 285]])
    >>>
    >>> # Create dewarper
    >>> dewarper = CylindricalSurfaceDewarper.from_directrices(
    ...     top_curve, bottom_curve, depth_perception=2.0
    ... )
    >>>
    >>> # Map points between coordinate systems
    >>> warped_pt = np.array([100, 150])
    >>> dewarped_pt = dewarper.map_to_dewarped_space(warped_pt)
"""

from scantailor.dewarping.arc_length_mapper import ArcLengthMapper
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper, Generatrix
from scantailor.dewarping.distortion_model import (
    Curve,
    DistortionModel,
    create_distortion_model,
    create_distortion_model_from_splines,
)
from scantailor.dewarping.polyline_intersector import (
    PolylineIntersector,
    project_point_to_line,
)
from scantailor.dewarping.raster_dewarper import (
    InterpolationMethod,
    compute_dewarped_size,
    dewarp_image,
)

__all__ = [
    "ArcLengthMapper",
    "compute_dewarped_size",
    "create_distortion_model",
    "create_distortion_model_from_splines",
    "Curve",
    "CylindricalSurfaceDewarper",
    "dewarp_image",
    "DistortionModel",
    "Generatrix",
    "InterpolationMethod",
    "PolylineIntersector",
    "project_point_to_line",
]
