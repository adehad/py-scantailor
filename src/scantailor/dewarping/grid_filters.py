"""Float-grid filters shared by the text line and page edge tracers.

The blur is a true Gaussian from OpenCV. The C++ code uses a recursive
approximation of it, which differs from a true Gaussian by up to about 0.15% of
the value range, and the ridge and gradient maps built on it follow suit.
"""

import math

import cv2
import numpy as np
from numpy.typing import NDArray

# Sobel responses are eight times the gradient; the scale keeps partial
# derivatives of an 8-bit image inside [-1, 1].
SOBEL_GRAY_SCALE = 1.0 / (255.0 * 8.0)
# The Gaussian kernel reaches this many standard deviations to each side.
_KERNEL_RADIUS_SIGMAS = 5.0


def sobel_xy(
    grid: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Sobel gradient of a grid, eight times the true derivative, edges replicated.

    Args:
        grid (NDArray[np.float32]): 2-D float32 grid.

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32]]: Derivatives with
            respect to x (right minus left) and y (down minus up).
    """
    grad_x = cv2.Sobel(grid, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    grad_y = cv2.Sobel(grid, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return (
        np.asarray(grad_x, dtype=np.float32),
        np.asarray(grad_y, dtype=np.float32),
    )


def directional_gradient(
    gray: NDArray[np.uint8], direction: tuple[float, float]
) -> NDArray[np.float32]:
    """Derivative of an 8-bit image along a direction, in the range [-1, 1].

    Args:
        gray (NDArray[np.uint8]): 2-D gray image.
        direction (tuple[float, float]): Unit vector ``(dx, dy)``.

    Returns:
        NDArray[np.float32]: ``sobel_x * dx + sobel_y * dy`` of the image
            scaled by ``SOBEL_GRAY_SCALE``.
    """
    scaled = gray.astype(np.float32) * np.float32(SOBEL_GRAY_SCALE)
    grad_x, grad_y = sobel_xy(scaled)
    return grad_x * np.float32(direction[0]) + grad_y * np.float32(direction[1])


def gauss_blur(
    grid: NDArray[np.float32], sigma_h: float, sigma_v: float
) -> NDArray[np.float32]:
    """Gaussian blur with edges replicated.

    Args:
        grid (NDArray[np.float32]): 2-D float32 grid.
        sigma_h (float): Standard deviation along x, in pixels.
        sigma_v (float): Standard deviation along y, in pixels.

    Returns:
        NDArray[np.float32]: The blurred grid.
    """
    kernel_w = 2 * math.ceil(_KERNEL_RADIUS_SIGMAS * sigma_h) + 1
    kernel_h = 2 * math.ceil(_KERNEL_RADIUS_SIGMAS * sigma_v) + 1
    blurred = cv2.GaussianBlur(
        grid,
        (kernel_w, kernel_h),
        sigmaX=sigma_h,
        sigmaY=sigma_v,
        borderType=cv2.BORDER_REPLICATE,
    )
    return np.asarray(blurred, dtype=np.float32)
