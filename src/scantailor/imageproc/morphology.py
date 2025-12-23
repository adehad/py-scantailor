"""Morphological operations for binary and grayscale images.

All functions use OpenCV's morphology operations with configurable
structuring elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

StructuringElementShape = Literal["rect", "ellipse", "cross"]


def _get_structuring_element(
    shape: StructuringElementShape,
    size: int | tuple[int, int],
) -> NDArray[np.uint8]:
    """Create a structuring element for morphological operations.

    Args:
        shape (StructuringElementShape): Shape of the element.
        size (int | tuple[int, int]): Size as single int or (width, height).

    Returns:
        NDArray[np.uint8]: The structuring element.
    """
    if isinstance(size, int):
        size = (size, size)

    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return np.asarray(cv2.getStructuringElement(shape_map[shape], size), dtype=np.uint8)


def dilate(
    image: NDArray[np.uint8],
    kernel_size: int = 3,
    shape: StructuringElementShape = "rect",
    iterations: int = 1,
) -> NDArray[np.uint8]:
    """Dilate an image, expanding white regions.

    Args:
        image (NDArray[np.uint8]): Input image (binary or grayscale).
        kernel_size (int): Size of the structuring element.
        shape (StructuringElementShape): Shape of the structuring element.
        iterations (int): Number of times to apply the operation.

    Returns:
        NDArray[np.uint8]: Dilated image.
    """
    kernel = _get_structuring_element(shape, kernel_size)
    result = cv2.dilate(image, kernel, iterations=iterations)
    return np.asarray(result, dtype=np.uint8)


def erode(
    image: NDArray[np.uint8],
    kernel_size: int = 3,
    shape: StructuringElementShape = "rect",
    iterations: int = 1,
) -> NDArray[np.uint8]:
    """Erode an image, shrinking white regions.

    Args:
        image (NDArray[np.uint8]): Input image (binary or grayscale).
        kernel_size (int): Size of the structuring element.
        shape (StructuringElementShape): Shape of the structuring element.
        iterations (int): Number of times to apply the operation.

    Returns:
        NDArray[np.uint8]: Eroded image.
    """
    kernel = _get_structuring_element(shape, kernel_size)
    result = cv2.erode(image, kernel, iterations=iterations)
    return np.asarray(result, dtype=np.uint8)


def open_morph(
    image: NDArray[np.uint8],
    kernel_size: int = 3,
    shape: StructuringElementShape = "rect",
) -> NDArray[np.uint8]:
    """Apply morphological opening (erosion followed by dilation).

    Opening removes small white spots and thin white protrusions.

    Args:
        image (NDArray[np.uint8]): Input image (binary or grayscale).
        kernel_size (int): Size of the structuring element.
        shape (StructuringElementShape): Shape of the structuring element.

    Returns:
        NDArray[np.uint8]: Opened image.
    """
    kernel = _get_structuring_element(shape, kernel_size)
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return np.asarray(result, dtype=np.uint8)


def close_morph(
    image: NDArray[np.uint8],
    kernel_size: int = 3,
    shape: StructuringElementShape = "rect",
) -> NDArray[np.uint8]:
    """Apply morphological closing (dilation followed by erosion).

    Closing fills small black holes and thin black gaps.

    Args:
        image (NDArray[np.uint8]): Input image (binary or grayscale).
        kernel_size (int): Size of the structuring element.
        shape (StructuringElementShape): Shape of the structuring element.

    Returns:
        NDArray[np.uint8]: Closed image.
    """
    kernel = _get_structuring_element(shape, kernel_size)
    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return np.asarray(result, dtype=np.uint8)


def remove_small_components(
    image: NDArray[np.uint8],
    min_size: int,
    connectivity: int = 8,
) -> NDArray[np.uint8]:
    """Remove small connected components from a binary image.

    Components (connected regions of white pixels) smaller than min_size
    will be removed (set to black).

    Args:
        image (NDArray[np.uint8]): Binary input image (0 or 255).
        min_size (int): Minimum component size to keep.
        connectivity (int): 4 or 8 connectivity.

    Returns:
        NDArray[np.uint8]: Filtered binary image.
    """
    if min_size <= 0:
        return image.copy()

    # Find connected components
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        image, connectivity=connectivity
    )

    # Create output image
    result = np.zeros_like(image)

    # Keep only components larger than min_size
    # Start from 1 to skip background (label 0)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            result[labels == label] = 255

    return result
