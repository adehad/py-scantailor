"""Image filtering operations.

Provides common image filters including blur, edge detection,
and noise reduction.
"""

from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from numpy.typing import NDArray


def gaussian_blur(
    image: NDArray[np.uint8],
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> NDArray[np.uint8]:
    """Apply Gaussian blur to an image.

    Args:
        image: Input image (grayscale or color).
        kernel_size: Size of the Gaussian kernel (must be odd and positive).
        sigma: Gaussian kernel standard deviation. If 0, computed from kernel_size.

    Returns:
        Blurred image.
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 1:
        kernel_size = 1

    result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return np.asarray(result, dtype=image.dtype)


def sobel(
    image: NDArray[np.uint8],
    dx: int = 1,
    dy: int = 0,
    kernel_size: int = 3,
) -> NDArray[np.float64]:
    """Apply Sobel edge detection filter.

    Computes the image derivative using the Sobel operator.

    Args:
        image: Input grayscale image.
        dx: Order of the derivative in x direction.
        dy: Order of the derivative in y direction.
        kernel_size: Size of the extended Sobel kernel (1, 3, 5, or 7).

    Returns:
        Filtered image as float64 (can contain negative values).
    """
    result = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=kernel_size)
    return np.asarray(result, dtype=np.float64)


def sobel_magnitude(
    image: NDArray[np.uint8],
    kernel_size: int = 3,
) -> NDArray[np.float64]:
    """Compute Sobel edge magnitude (gradient magnitude).

    Args:
        image: Input grayscale image.
        kernel_size: Size of the extended Sobel kernel (1, 3, 5, or 7).

    Returns:
        Gradient magnitude image as float64.
    """
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude


def wiener_filter(
    image: NDArray[np.uint8],
    noise_variance: float | None = None,
) -> NDArray[np.uint8]:
    """Apply Wiener filter for noise reduction.

    The Wiener filter is optimal for reducing additive noise while
    preserving edges better than simple smoothing filters.

    Args:
        image: Input grayscale image.
        noise_variance: Noise variance. If None, estimated from the image.

    Returns:
        Filtered image.
    """
    # Convert to float for processing
    img_float = image.astype(np.float64)

    # Apply Wiener filter
    # scipy.signal.wiener expects a 2D array and optional noise parameter
    if noise_variance is not None:
        filtered = signal.wiener(img_float, noise=noise_variance)
    else:
        filtered = signal.wiener(img_float)

    # Clip and convert back to uint8
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered


def savgol_filter_2d(
    image: NDArray[np.uint8],
    window_size: int = 5,
    poly_order: int = 2,
) -> NDArray[np.uint8]:
    """Apply 2D Savitzky-Golay filter for smoothing while preserving edges.

    The Savitzky-Golay filter fits a polynomial to a window of data points
    and uses the polynomial value at the center. This preserves higher moments
    of the data better than simple moving average filters.

    Args:
        image: Input grayscale image.
        window_size: Size of the filter window (must be odd).
        poly_order: Order of the polynomial (must be less than window_size).

    Returns:
        Filtered image.
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Ensure poly_order is valid
    if poly_order >= window_size:
        poly_order = window_size - 1

    # Convert to float for processing
    img_float = image.astype(np.float64)

    # Apply 1D Savitzky-Golay filter to rows then columns
    # This is a separable approximation of 2D Savitzky-Golay
    filtered = signal.savgol_filter(img_float, window_size, poly_order, axis=0)
    filtered = signal.savgol_filter(filtered, window_size, poly_order, axis=1)

    # Clip and convert back to uint8
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered
