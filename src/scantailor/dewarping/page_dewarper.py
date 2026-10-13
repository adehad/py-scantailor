"""Dewarpers for a page, from a manual model or detected from the page itself.

Every function works in the pixel space of the gray page it is given and
returns None, rather than raising, when no usable dewarper can be made, so that
the caller can leave the page as it is.
"""

from collections.abc import Sequence

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.models import Dpi
from scantailor.dewarping.auto_model import build_auto_distortion_model
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.dewarping.distortion_model import (
    DistortionModel,
    create_distortion_model,
)
from scantailor.dewarping.marginal_model import build_marginal_model

ContentRect = tuple[float, float, float, float]

_DEWARPER_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError)
_MIN_CURVE_POINTS = 2


def dewarper_from_polylines(
    top: Sequence[tuple[float, float]] | NDArray[np.float64],
    bottom: Sequence[tuple[float, float]] | NDArray[np.float64],
    depth_perception: float,
) -> CylindricalSurfaceDewarper | None:
    """Create a dewarper from the page's top and bottom edges.

    Args:
        top (Sequence[tuple[float, float]]): Top edge as (x, y) points, left to right.
        bottom (Sequence[tuple[float, float]]): Bottom edge, left to right.
        depth_perception (float): Distance from the camera to the plane through
            the outer generatrices; 1 to 3 work well.

    Returns:
        CylindricalSurfaceDewarper | None: The dewarper, or None if an edge has
            fewer than two points or the edges do not form a valid surface.
    """
    if len(top) < _MIN_CURVE_POINTS or len(bottom) < _MIN_CURVE_POINTS:
        return None
    top_edge = np.asarray(top, dtype=np.float64)
    bottom_edge = np.asarray(bottom, dtype=np.float64)
    model = create_distortion_model(top_edge, bottom_edge)
    if not model.is_valid():
        return None
    try:
        return CylindricalSurfaceDewarper.from_directrices(
            top_edge, bottom_edge, depth_perception
        )
    except _DEWARPER_ERRORS as error:
        logger.warning("Distortion model rejected by the dewarper: {}", error)
        return None


def dewarper_from_model(
    model: DistortionModel | None, depth_perception: float
) -> CylindricalSurfaceDewarper | None:
    """Create a dewarper from a distortion model; None when it is missing or invalid."""
    if model is None or model.top_curve is None or model.bottom_curve is None:
        return None
    return dewarper_from_polylines(
        model.top_curve.polyline, model.bottom_curve.polyline, depth_perception
    )


def auto_dewarper(
    gray: NDArray[np.uint8],
    dpi: Dpi | tuple[float, float] | float,
    content_rect: ContentRect,
    depth_perception: float,
    spine_on_right: bool = True,
) -> CylindricalSurfaceDewarper | None:
    """Detect the page curvature from its text lines and page edges.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page with dark text on a light
            background, illumination already normalised.
        dpi (Dpi | tuple[float, float] | float): Horizontal and vertical
            resolution of ``gray``.
        content_rect (ContentRect): (x, y, width, height) of the content.
        depth_perception (float): See ``dewarper_from_polylines``.
        spine_on_right (bool): True for a single or left page, False for a
            right page.

    Returns:
        CylindricalSurfaceDewarper | None: The dewarper, or None when the curves
            found give no valid distortion model.
    """
    model = build_auto_distortion_model(gray, dpi, content_rect, spine_on_right)
    return dewarper_from_model(model, depth_perception)


def marginal_dewarper(
    gray: NDArray[np.uint8],
    content_rect: ContentRect,
    spine_on_right: bool,
    depth_perception: float,
) -> CylindricalSurfaceDewarper | None:
    """Dewarp only the strip of page next to the binding.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page.
        content_rect (ContentRect): (x, y, width, height) of the content.
        spine_on_right (bool): True for a single or left page.
        depth_perception (float): See ``dewarper_from_polylines``.

    Returns:
        CylindricalSurfaceDewarper | None: The dewarper, or None if no model fits.
    """
    model = build_marginal_model(gray, content_rect, spine_on_right)
    return dewarper_from_model(model, depth_perception)
