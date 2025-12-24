"""Binarization algorithms for converting grayscale images to binary.

All functions take grayscale images as numpy arrays and return binary images.
Binary images use 0 for black (background) and 255 for white (foreground).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

if TYPE_CHECKING:
    from numpy.typing import NDArray


def binarize_otsu(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Binarize an image using Otsu's method.

    Otsu's method automatically determines the optimal threshold by minimizing
    intra-class variance. Works well for images with bimodal histograms.

    Args:
        image (NDArray[np.uint8]): Grayscale input image.

    Returns:
        NDArray[np.uint8]: Binary image (0 or 255).
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.asarray(binary, dtype=np.uint8)


def binarize_sauvola(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.2,
) -> NDArray[np.uint8]:
    """Binarize an image using Sauvola's adaptive thresholding.

    Sauvola's method computes a local threshold based on the mean and standard
    deviation within a window. Good for documents with uneven illumination.

    Args:
        image (NDArray[np.uint8]): Grayscale input image.
        window_size (int): Size of the local window (must be odd).
        k (float): Sensitivity parameter (typically 0.1-0.5).

    Returns:
        NDArray[np.uint8]: Binary image (0 or 255).
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    threshold = threshold_sauvola(image, window_size=window_size, k=k)
    binary = (image > threshold).astype(np.uint8) * 255
    return binary


def binarize_wolf(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.5,
) -> NDArray[np.uint8]:
    """Binarize an image using Wolf's adaptive thresholding.

    Wolf's method is a modification of Sauvola's that uses the minimum
    and maximum grayscale values in addition to mean and standard deviation.
    It's more robust for low-contrast images.

    Args:
        image (NDArray[np.uint8]): Grayscale input image.
        window_size (int): Size of the local window (must be odd).
        k (float): Sensitivity parameter (typically 0.3-0.5).

    Returns:
        NDArray[np.uint8]: Binary image (0 or 255).
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Compute local mean using box filter
    mean = cv2.blur(image.astype(np.float64), (window_size, window_size))

    # Compute local standard deviation
    sq_mean = cv2.blur((image.astype(np.float64)) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(sq_mean - mean**2, 0))

    # Get min value of image
    min_val = float(np.min(image))

    # Get max standard deviation
    max_std = float(np.max(std))
    if max_std < 1e-6:
        max_std = 1.0

    # Wolf's formula: T = (1-k)*mean + k*min + k*(std/max_std)*(mean - min)
    threshold = (1 - k) * mean + k * min_val + k * (std / max_std) * (mean - min_val)

    binary = (image > threshold).astype(np.uint8) * 255
    return binary


def binarize_bradley(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.15,
) -> NDArray[np.uint8]:
    """Binarize an image using Bradley's adaptive thresholding.

    Bradley's method computes a local threshold based only on the mean
    within a window. A pixel is set to black if it's sufficiently darker
    than the local mean. Simple and fast, works well for clean documents.

    The threshold at each pixel is: mean * (1 - k)
    Pixels below this threshold are classified as foreground (black).

    Args:
        image: Grayscale input image.
        window_size: Size of the local window (must be odd).
        k: Sensitivity parameter (0-1). Higher values make it
            easier for pixels to be classified as foreground (black).
            Typically 0.1-0.2.

    Returns:
        Binary image (0 or 255).
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Compute local mean using box filter (efficient via integral image internally)
    local_mean = cv2.blur(image.astype(np.float64), (window_size, window_size))

    # Bradley threshold: pixel < mean * (1 - k) -> black
    # For k >= 1, threshold is 0 (everything white except pure black)
    if k >= 1.0:
        threshold = np.zeros_like(local_mean)
    else:
        threshold = local_mean * (1.0 - k)

    binary = (image >= threshold).astype(np.uint8) * 255
    return binary
