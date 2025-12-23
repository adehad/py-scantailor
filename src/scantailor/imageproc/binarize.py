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
    return binary


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
