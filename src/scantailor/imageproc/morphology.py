"""Morphological operations for binary and grayscale images.

All functions use OpenCV's morphology operations with configurable
structuring elements.
"""

from typing import Literal

import cv2
import numpy as np
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


def white_top_hat(
    image: NDArray[np.uint8],
    kernel_size: int = 3,
    shape: StructuringElementShape = "rect",
) -> NDArray[np.uint8]:
    """Apply white top-hat transform (image - opening).

    Extracts bright spots smaller than the structuring element.
    Useful for finding small bright features on a varying background.

    Args:
        image: Input image (grayscale).
        kernel_size: Size of the structuring element.
        shape: Shape of the structuring element.

    Returns:
        Top-hat transformed image.
    """
    kernel = _get_structuring_element(shape, kernel_size)
    result = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return np.asarray(result, dtype=np.uint8)


def black_top_hat(
    image: NDArray[np.uint8],
    kernel_size: int = 3,
    shape: StructuringElementShape = "rect",
) -> NDArray[np.uint8]:
    """Apply black top-hat transform (closing - image).

    Extracts dark spots smaller than the structuring element.
    Useful for finding small dark features on a varying background.

    Args:
        image: Input image (grayscale).
        kernel_size: Size of the structuring element.
        shape: Shape of the structuring element.

    Returns:
        Black top-hat transformed image.
    """
    kernel = _get_structuring_element(shape, kernel_size)
    result = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return np.asarray(result, dtype=np.uint8)


def hit_miss(
    image: NDArray[np.uint8],
    kernel: NDArray[np.int8],
) -> NDArray[np.uint8]:
    """Apply hit-or-miss transform.

    The hit-or-miss transform detects specific patterns in binary images.
    The kernel uses: 1 for foreground, -1 for background, 0 for don't care.

    Args:
        image: Binary input image (0 or 255).
        kernel: Hit-miss kernel with values -1, 0, or 1.

    Returns:
        Binary image with detected patterns.
    """
    result = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    return np.asarray(result, dtype=np.uint8)


def remove_small_components(
    image: NDArray[np.uint8],
    min_size: int,
    connectivity: int = 8,
) -> NDArray[np.uint8]:
    """Remove small connected components from a binary image.

    Any non-zero pixel counts as foreground. Components with fewer than
    min_size pixels become 0; the rest become 255.

    Args:
        image (NDArray[np.uint8]): Binary input image, foreground non-zero.
        min_size (int): Smallest component area, in pixels, that survives.
        connectivity (int): 4 or 8 neighbour connectivity.

    Returns:
        NDArray[np.uint8]: New image with values 0 or 255.
    """
    if min_size <= 0:
        return image.copy()

    _num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        image, connectivity=connectivity
    )
    # A 300 dpi page has tens of thousands of specks, so the filter must be one
    # gather over the label image; a pass per component takes about 20 s.
    keep = np.where(stats[:, cv2.CC_STAT_AREA] >= min_size, 255, 0).astype(np.uint8)
    keep[0] = 0
    return keep[np.asarray(labels, dtype=np.int32)]
