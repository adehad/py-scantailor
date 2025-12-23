"""Geometric transformations for images.

Functions for rotating, scaling, and transforming images using OpenCV.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from scantailor.core import OrthogonalDegrees

InterpolationMethod = Literal["nearest", "linear", "cubic", "area", "lanczos"]


def _get_interpolation_flag(method: InterpolationMethod) -> int:
    """Convert interpolation method name to OpenCV flag.

    Args:
        method (InterpolationMethod): Interpolation method name.

    Returns:
        int: OpenCV interpolation flag.
    """
    method_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return method_map[method]


def rotate_orthogonal(
    image: NDArray[np.uint8],
    degrees: OrthogonalDegrees,
) -> NDArray[np.uint8]:
    """Rotate an image by 0, 90, 180, or 270 degrees clockwise.

    Args:
        image (NDArray[np.uint8]): Input image.
        degrees (OrthogonalDegrees): Rotation angle (0, 90, 180, or 270).

    Returns:
        NDArray[np.uint8]: Rotated image.
    """
    if degrees == 0:
        return image.copy()
    if degrees == 90:
        return np.asarray(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), dtype=np.uint8)
    if degrees == 180:
        return np.asarray(cv2.rotate(image, cv2.ROTATE_180), dtype=np.uint8)
    # 270 degrees
    return np.asarray(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), dtype=np.uint8)


def scale(
    image: NDArray[np.uint8],
    scale_x: float,
    scale_y: float | None = None,
    interpolation: InterpolationMethod = "linear",
) -> NDArray[np.uint8]:
    """Scale an image by the given factors.

    Args:
        image (NDArray[np.uint8]): Input image.
        scale_x (float): Horizontal scale factor.
        scale_y (float | None): Vertical scale factor. If None, uses scale_x.
        interpolation (InterpolationMethod): Interpolation method to use.

    Returns:
        NDArray[np.uint8]: Scaled image.
    """
    if scale_y is None:
        scale_y = scale_x

    if scale_x == 1.0 and scale_y == 1.0:
        return image.copy()

    interp = _get_interpolation_flag(interpolation)
    result = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=interp)
    return np.asarray(result, dtype=np.uint8)
