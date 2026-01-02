"""Image type aliases and utilities for the imageproc module.

This module defines type aliases for different image types used throughout
the codebase. Instead of creating custom image classes (like the C++ BinaryImage
and GrayImage classes), we use NumPy arrays with type annotations for clarity
and simplicity.

Type Conventions:
    - BinaryImage: uint8 array with values 0 (black/background) or 255 (white/foreground)
    - GrayImage: uint8 array with grayscale values 0-255
    - ColorImage: uint8 array with shape (H, W, 3) for BGR color images (OpenCV convention)
    - FloatImage: float32 or float64 array for intermediate computations

Notes:
    - Binary images use 0=black, 255=white (opposite of C++ which uses 0=white, 1=black)
    - OpenCV uses BGR color order by default, not RGB
    - All pixel coordinates use (x, y) convention where x=column, y=row
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type aliases for image arrays
BinaryImage = NDArray[np.uint8]  # Binary image: 0 or 255
GrayImage = NDArray[np.uint8]  # Grayscale image: 0-255
ColorImage = NDArray[np.uint8]  # Color image: (H, W, 3) BGR
FloatImage = NDArray[np.floating]  # Float image for computation


def is_binary(image: NDArray[np.uint8]) -> bool:
    """Check if an image is binary (contains only 0 and 255).

    Args:
        image: Image to check.

    Returns:
        True if image contains only 0 and 255, False otherwise.
    """
    unique = np.unique(image)
    return len(unique) <= 2 and np.all((unique == 0) | (unique == 255))


def is_grayscale(image: NDArray) -> bool:
    """Check if an image is grayscale (single channel uint8).

    Args:
        image: Image to check.

    Returns:
        True if image is grayscale (H, W) shape with uint8 dtype.
    """
    return image.ndim == 2 and image.dtype == np.uint8


def is_color(image: NDArray) -> bool:
    """Check if an image is a color image (3-channel uint8).

    Args:
        image: Image to check.

    Returns:
        True if image is color (H, W, 3) shape with uint8 dtype.
    """
    return image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8


def ensure_binary(image: NDArray[np.uint8], threshold: int = 127) -> BinaryImage:
    """Ensure an image is binary, thresholding if necessary.

    Args:
        image: Input image (grayscale or binary).
        threshold: Threshold value if image needs binarization.

    Returns:
        Binary image with values 0 or 255.
    """
    if is_binary(image):
        return image

    # Threshold grayscale image
    return np.where(image > threshold, 255, 0).astype(np.uint8)


def ensure_grayscale(image: NDArray[np.uint8]) -> GrayImage:
    """Ensure an image is grayscale, converting if necessary.

    Args:
        image: Input image (grayscale, binary, or color).

    Returns:
        Grayscale image.

    Raises:
        ValueError: If image cannot be converted to grayscale.
    """
    if is_grayscale(image):
        return image

    if is_color(image):
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Cannot convert image with shape {image.shape} to grayscale")


def ensure_color(image: NDArray[np.uint8]) -> ColorImage:
    """Ensure an image is color (BGR), converting if necessary.

    Args:
        image: Input image (grayscale, binary, or color).

    Returns:
        Color (BGR) image.

    Raises:
        ValueError: If image cannot be converted to color.
    """
    if is_color(image):
        return image

    if is_grayscale(image):
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Cannot convert image with shape {image.shape} to color")


def invert_binary(image: BinaryImage) -> BinaryImage:
    """Invert a binary image (swap black and white).

    Args:
        image: Binary input image.

    Returns:
        Inverted binary image.
    """
    return np.asarray(255 - image, dtype=np.uint8)


__all__ = [
    "BinaryImage",
    "GrayImage",
    "ColorImage",
    "FloatImage",
    "is_binary",
    "is_grayscale",
    "is_color",
    "ensure_binary",
    "ensure_grayscale",
    "ensure_color",
    "invert_binary",
]
