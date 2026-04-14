"""Geometric transformations for images.

Functions for rotating, scaling, and transforming images using OpenCV.
"""

from typing import Literal

import cv2
import numpy as np
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


def transform_affine(
    image: NDArray[np.uint8],
    matrix: NDArray,
    output_size: tuple[int, int] | None = None,
    interpolation: InterpolationMethod = "linear",
    border_value: int = 0,
) -> NDArray[np.uint8]:
    """Apply an affine transformation to an image.

    Args:
        image: Input image.
        matrix: 2x3 affine transformation matrix.
        output_size: Output image size (width, height). If None, uses input size.
        interpolation: Interpolation method to use.
        border_value: Value for pixels outside the image boundary.

    Returns:
        Transformed image.
    """
    h, w = image.shape[:2]
    if output_size is None:
        output_size = (w, h)

    interp = _get_interpolation_flag(interpolation)
    result = cv2.warpAffine(
        image,
        matrix,
        output_size,
        flags=interp,
        borderValue=border_value,
    )
    return np.asarray(result, dtype=np.uint8)


def transform_perspective(
    image: NDArray[np.uint8],
    matrix: NDArray,
    output_size: tuple[int, int] | None = None,
    interpolation: InterpolationMethod = "linear",
    border_value: int = 0,
) -> NDArray[np.uint8]:
    """Apply a perspective transformation to an image.

    Args:
        image: Input image.
        matrix: 3x3 perspective transformation matrix.
        output_size: Output image size (width, height). If None, uses input size.
        interpolation: Interpolation method to use.
        border_value: Value for pixels outside the image boundary.

    Returns:
        Transformed image.
    """
    h, w = image.shape[:2]
    if output_size is None:
        output_size = (w, h)

    interp = _get_interpolation_flag(interpolation)
    result = cv2.warpPerspective(
        image,
        matrix,
        output_size,
        flags=interp,
        borderValue=border_value,
    )
    return np.asarray(result, dtype=np.uint8)


def shear(
    image: NDArray[np.uint8],
    shear_x: float = 0.0,
    shear_y: float = 0.0,
    interpolation: InterpolationMethod = "linear",
    border_value: int = 0,
) -> NDArray[np.uint8]:
    """Apply a shear transformation to an image.

    Args:
        image: Input image.
        shear_x: Shear factor in x direction (horizontal shear).
        shear_y: Shear factor in y direction (vertical shear).
        interpolation: Interpolation method to use.
        border_value: Value for pixels outside the image boundary.

    Returns:
        Sheared image (same size as input).
    """
    if shear_x == 0.0 and shear_y == 0.0:
        return image.copy()

    h, w = image.shape[:2]

    # Build shear matrix:
    # [1, shear_x, 0]
    # [shear_y, 1, 0]
    matrix = np.array(
        [
            [1.0, shear_x, 0.0],
            [shear_y, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    return transform_affine(
        image,
        matrix,
        output_size=(w, h),
        interpolation=interpolation,
        border_value=border_value,
    )


def rotate(
    image: NDArray[np.uint8],
    angle: float,
    center: tuple[float, float] | None = None,
    scale_factor: float = 1.0,
    interpolation: InterpolationMethod = "linear",
    border_value: int = 0,
) -> NDArray[np.uint8]:
    """Rotate an image by an arbitrary angle.

    Args:
        image: Input image.
        angle: Rotation angle in degrees (positive = counter-clockwise).
        center: Center of rotation (x, y). If None, uses image center.
        scale_factor: Optional scaling factor applied during rotation.
        interpolation: Interpolation method to use.
        border_value: Value for pixels outside the image boundary.

    Returns:
        Rotated image (same size as input).
    """
    h, w = image.shape[:2]

    if center is None:
        center = (w / 2.0, h / 2.0)

    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)

    return transform_affine(
        image,
        matrix,
        output_size=(w, h),
        interpolation=interpolation,
        border_value=border_value,
    )
