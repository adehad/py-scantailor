"""Output image generator for the output filter.

This module provides functions for generating final output images
with binarization, despeckling, dewarping, and other processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.imageproc import (
    binarize_bradley,
    binarize_edge_div,
    binarize_otsu,
    binarize_sauvola,
    binarize_wolf,
    remove_small_components,
)

from .binarization import BinarizationMethod, BinarizationOptions
from .color_mode import ColorMode
from .despeckle import DespeckleLevel
from .params import Params

if TYPE_CHECKING:
    from scantailor.dewarping import CylindricalSurfaceDewarper


@dataclass
class OutputResult:
    """Result of output generation."""

    image: NDArray[np.uint8]
    is_binary: bool = False
    was_dewarped: bool = False


def generate_output(
    image: NDArray[np.uint8],
    params: Params,
    dewarper: CylindricalSurfaceDewarper | None = None,
) -> OutputResult:
    """Generate the output image according to parameters.

    Args:
        image: Input image (grayscale or color).
        params: Output parameters.
        dewarper: Optional dewarper for curved page correction.
            Required if dewarping mode is AUTO or MANUAL.

    Returns:
        OutputResult with the processed image.
    """
    processed = image
    was_dewarped = False

    # Apply dewarping if enabled and dewarper provided
    if params.needs_dewarping() and dewarper is not None:
        processed = _apply_dewarping(processed, params, dewarper)
        was_dewarped = True

    # Convert based on color mode
    if params.color_mode == ColorMode.BLACK_AND_WHITE:
        result = _generate_binary_output(processed, params)
    elif params.color_mode == ColorMode.COLOR_GRAYSCALE:
        result = _generate_grayscale_output(processed, params)
    else:
        result = _generate_mixed_output(processed, params)

    result.was_dewarped = was_dewarped
    return result


def _apply_dewarping(
    image: NDArray[np.uint8],
    params: Params,
    dewarper: CylindricalSurfaceDewarper,
) -> NDArray[np.uint8]:
    """Apply dewarping to an image.

    Args:
        image: Input image.
        params: Output parameters with dewarping options.
        dewarper: The dewarper model.

    Returns:
        Dewarped image.
    """
    from scantailor.dewarping import compute_dewarped_size, dewarp_image

    # Compute output size based on source size and dewarper
    src_size = (image.shape[1], image.shape[0])  # (width, height)
    dst_size = compute_dewarped_size(src_size, dewarper)

    # Determine background color based on image type
    if len(image.shape) == 2:
        background = 255  # White for grayscale
    elif image.shape[2] == 3:
        background = (255, 255, 255)  # White for RGB
    else:
        background = (255, 255, 255, 255)  # White for RGBA

    # Apply dewarping with full model domain
    dewarped = dewarp_image(
        image,
        dewarper,
        dst_size=dst_size,
        model_domain=(0.0, 0.0, 1.0, 1.0),
        background_color=background,
    )

    # Apply post-deskew if requested
    if params.dewarping.post_deskew and params.dewarping.post_deskew_angle != 0.0:
        dewarped = _apply_post_deskew(dewarped, params.dewarping.post_deskew_angle)

    return dewarped


def _apply_post_deskew(
    image: NDArray[np.uint8],
    angle: float,
) -> NDArray[np.uint8]:
    """Apply post-dewarping deskew rotation.

    Args:
        image: Dewarped image.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding box size
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix for the new size
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Determine background color
    if len(image.shape) == 2:
        border_value = 255
    else:
        border_value = (255,) * image.shape[2]

    # Apply rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return np.asarray(rotated, dtype=np.uint8)


def _generate_binary_output(
    image: NDArray[np.uint8],
    params: Params,
) -> OutputResult:
    """Generate black and white output.

    Args:
        image: Input image.
        params: Output parameters.

    Returns:
        OutputResult with binary image.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    else:
        gray = image

    # Apply illumination normalization if requested
    if params.binarization.normalize_illumination:
        gray = _normalize_illumination(gray)

    # Apply binarization
    binary = _binarize(gray, params.binarization)

    # Invert if not black on white
    if not params.black_on_white:
        binary = 255 - binary

    # Apply despeckle
    if params.despeckle_level.is_enabled():
        binary = _despeckle(binary, params.despeckle_level)

    # Apply morphological smoothing if requested
    if params.binarization.morphological_smoothing:
        binary = _morphological_smooth(binary)

    return OutputResult(image=binary, is_binary=True)


def _generate_grayscale_output(
    image: NDArray[np.uint8],
    params: Params,
) -> OutputResult:
    """Generate grayscale or color output.

    Args:
        image: Input image.
        params: Output parameters.

    Returns:
        OutputResult with grayscale/color image.
    """
    # For grayscale mode, we preserve the image as-is
    # but may need to resize based on DPI
    result = image.copy()

    return OutputResult(image=result, is_binary=False)


def _generate_mixed_output(
    image: NDArray[np.uint8],
    params: Params,
) -> OutputResult:
    """Generate mixed mode output.

    In mixed mode, the background is binarized while foreground
    elements (like photos) are preserved in grayscale/color.

    For now, this is a simplified implementation that just
    does binarization - full mixed mode would require picture
    zone detection.

    Args:
        image: Input image.
        params: Output parameters.

    Returns:
        OutputResult with mixed image.
    """
    # Simplified: treat as binary for now
    # Full implementation would detect picture zones and preserve them
    return _generate_binary_output(image, params)


def _binarize(
    gray: NDArray[np.uint8],
    options: BinarizationOptions,
) -> NDArray[np.uint8]:
    """Apply binarization to a grayscale image.

    Args:
        gray: Grayscale input image.
        options: Binarization options.

    Returns:
        Binary image (0 or 255).
    """
    # Apply threshold adjustment
    if options.threshold_adjustment != 0:
        # Adjust by shifting pixel values before binarization
        adjusted = gray.astype(np.int16) - options.threshold_adjustment
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    else:
        adjusted = gray

    if options.method == BinarizationMethod.OTSU:
        return binarize_otsu(adjusted)
    if options.method == BinarizationMethod.SAUVOLA:
        return binarize_sauvola(
            adjusted,
            window_size=options.window_size,
            k=options.sauvola_coef,
        )
    if options.method == BinarizationMethod.WOLF:
        return binarize_wolf(
            adjusted,
            window_size=options.window_size,
            k=options.wolf_coef,
        )
    if options.method == BinarizationMethod.BRADLEY:
        return binarize_bradley(
            adjusted,
            window_size=options.window_size,
            k=options.bradley_coef,
        )
    # EdgeDiv method
    return binarize_edge_div(
        adjusted,
        window_size=options.window_size,
        kep=options.edge_div_kep,
        kbd=options.edge_div_kbd,
    )


def _normalize_illumination(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Normalize illumination in a grayscale image.

    This helps with uneven lighting across the page.

    Args:
        gray: Grayscale input image.

    Returns:
        Illumination-normalized image.
    """
    # Use morphological opening to estimate background
    kernel_size = max(gray.shape) // 10
    kernel_size = max(kernel_size, 3)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Divide original by background
    # This normalizes the illumination
    normalized = gray.astype(np.float32) / (background.astype(np.float32) + 1)
    normalized = normalized * 255
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return normalized


def _despeckle(
    binary: NDArray[np.uint8],
    level: DespeckleLevel,
) -> NDArray[np.uint8]:
    """Remove small noise components from a binary image.

    Args:
        binary: Binary input image.
        level: Despeckle level.

    Returns:
        Despeckled binary image.
    """
    min_size = level.to_component_size()
    if min_size <= 0:
        return binary

    # Remove small black components (noise in white areas)
    result = remove_small_components(binary, min_size, connectivity=8)

    # Also remove small white components (noise in black areas)
    inverted = 255 - result
    inverted = remove_small_components(inverted, min_size, connectivity=8)
    result = 255 - inverted

    return result


def _morphological_smooth(binary: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply morphological smoothing to a binary image.

    This helps smooth jagged edges.

    Args:
        binary: Binary input image.

    Returns:
        Smoothed binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Opening removes small protrusions
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Closing fills small holes
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return np.asarray(result, dtype=np.uint8)
