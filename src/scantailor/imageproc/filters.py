"""Image filtering operations.

Provides common image filters including blur, edge detection,
and noise reduction.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

_VARIANCE_EPSILON = 1e-6
# Added before truncating to 8 bits so that a value that is an integer up to
# floating-point error (99.99999999999999) does not drop to the level below.
_TRUNCATION_GUARD = 1e-6
# Luma weights (R, G, B) in 1/32 units.
_LUMA_R = 11
_LUMA_G = 16
_LUMA_B = 5
_LUMA_SCALE = 32
# floor(S / 32) of the weighted sum S, as cv2.transform's rounding of
# (S - 15.5) / 32: S is an integer, so the fraction never reaches 0.5 and every
# term is exact in float32. The last column is the offset; the BGRA matrix gives
# alpha a zero weight.
_GRAY_FROM_BGR = (
    np.array([[_LUMA_B, _LUMA_G, _LUMA_R, -15.5]], dtype=np.float32) / _LUMA_SCALE
)
_GRAY_FROM_BGRA = (
    np.array([[_LUMA_B, _LUMA_G, _LUMA_R, 0, -15.5]], dtype=np.float32) / _LUMA_SCALE
)


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


def bgr_to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR(A) to gray with integer weights 11/16/5 over 32, as ``qGray``.

    Each pixel is ``(11 R + 16 G + 5 B) // 32``, from one ``cv2.transform``
    pass that allocates only the result.

    Args:
        image (NDArray[np.uint8]): 2-D gray or (H, W, 3|4) BGR(A) image.

    Returns:
        NDArray[np.uint8]: 2-D gray image. Aliases ``image`` for gray input.
    """
    if image.ndim == 2:
        return image
    return qgray(image)


def qgray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Gray of a BGR(A) uint8 image: ``(11 R + 16 G + 5 B) // 32`` per pixel.

    Args:
        image (NDArray[np.uint8]): (H, W, 3|4) BGR(A) image; alpha is ignored.
            Any strides; read-only views are fine.

    Returns:
        NDArray[np.uint8]: (H, W) gray image, new.
    """
    if image.size == 0:
        # cv2.transform returns None for an empty image.
        return np.zeros(image.shape[:2], dtype=np.uint8)
    matrix = _GRAY_FROM_BGRA if image.shape[2] == 4 else _GRAY_FROM_BGR
    gray = cv2.transform(image, matrix).reshape(image.shape[:2])
    return np.asarray(gray, dtype=np.uint8)


def _window_shape(window_size: int | tuple[int, int]) -> tuple[int, int]:
    """Return (width, height) for a square side or a (width, height) pair."""
    width, height = (
        (window_size, window_size) if isinstance(window_size, int) else window_size
    )
    if width < 1 or height < 1:
        msg = f"window size must be positive, got {width}x{height}"
        raise ValueError(msg)
    return width, height


def _samples_in_window(length: int, size: int) -> NDArray[np.float64]:
    """Count of positions inside ``[0, length)`` covered by each window.

    The window around position ``i`` spans ``i - size // 2`` up to, excluding,
    ``i + size - size // 2``.
    """
    index = np.arange(length)
    low = np.clip(index - size // 2, 0, length)
    high = np.clip(index + size - size // 2, 0, length)
    return (high - low).astype(np.float64)


def wiener_filter(
    image: NDArray[np.uint8],
    window_size: int | tuple[int, int] = 5,
    noise_sigma: float = 0.0,
) -> NDArray[np.uint8]:
    """Adaptive Wiener filter over a window that is clipped at the image border.

    Each pixel moves towards its local mean by the share of the local variance
    that exceeds the noise variance. Flat areas (variance at most 1e-6) are
    left alone.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int | tuple[int, int]): Window as one side or (width, height).
        noise_sigma (float): Noise standard deviation in gray levels.

    Returns:
        NDArray[np.uint8]: New filtered image of the same shape.

    Raises:
        ValueError: If the window is empty or ``noise_sigma`` is negative.
    """
    width, height = _window_shape(window_size)
    if noise_sigma < 0:
        msg = "noise_sigma must not be negative"
        raise ValueError(msg)
    if image.size == 0:
        return image.copy()

    pixels = image.astype(np.float64)
    box = {
        "ksize": (width, height),
        "anchor": (width // 2, height // 2),
        "normalize": False,
        "borderType": cv2.BORDER_CONSTANT,
    }
    area = _samples_in_window(image.shape[0], height)[:, None] * _samples_in_window(
        image.shape[1], width
    )
    mean = cv2.boxFilter(pixels, cv2.CV_64F, **box) / area
    variance = cv2.boxFilter(pixels * pixels, cv2.CV_64F, **box) / area - mean * mean

    active = variance > _VARIANCE_EPSILON
    gain = np.maximum(variance - noise_sigma * noise_sigma, 0.0) / np.where(
        active, variance, 1.0
    )
    filtered = np.clip(mean + (pixels - mean) * gain + _TRUNCATION_GUARD, 0.0, 255.0)
    # npd: ok the Wiener gain needs float64
    return np.where(active, filtered, pixels).astype(np.uint8)


def wiener_color_filter(
    image: NDArray[np.uint8],
    window_size: int | tuple[int, int] = 5,
    coef: float = 0.0,
) -> NDArray[np.uint8]:
    """Denoise the brightness of a gray or colour image and keep its hues.

    The brightness is Wiener filtered and every colour channel is rescaled by
    the ratio of filtered to original brightness.

    Args:
        image (NDArray[np.uint8]): 2-D gray or (H, W, 3|4) BGR(A) image. Not
            modified; an alpha channel is passed through.
        window_size (int | tuple[int, int]): Window as one side or (width, height).
        coef (float): Noise strength from 0 to 1 (sigma is 255 * coef). Zero or
            less returns an unchanged copy.

    Returns:
        NDArray[np.uint8]: New image of the same shape.
    """
    width, height = _window_shape(window_size)
    if coef <= 0.0 or image.size == 0:
        return image.copy()

    gray = bgr_to_gray(image)
    smooth = wiener_filter(gray, (width, height), 255.0 * coef)
    scale = (smooth.astype(np.float32) + 1.0) / (gray.astype(np.float32) + 1.0)
    delta = smooth.astype(np.float32) - gray.astype(np.float32) * scale

    result = image.copy()
    if image.ndim == 2:
        result[:] = _apply_scale(image, scale, delta)
    else:
        result[..., :3] = _apply_scale(
            image[..., :3], scale[..., None], delta[..., None]
        )
    return result


def _apply_scale(
    channels: NDArray[np.uint8],
    scale: NDArray[np.float32],
    delta: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Return ``channels * scale + delta`` clipped to 0..255 and rounded."""
    value = channels.astype(np.float32) * scale + delta
    # npd: ok the per-pixel scale is fractional
    return (np.clip(value, 0.0, 255.0) + 0.5).astype(np.uint8)


def _savgol_hat_matrix(window: int, degree: int) -> NDArray[np.float64]:
    """Least-squares polynomial fit operator for one window.

    Args:
        window (int): Number of samples.
        degree (int): Polynomial degree; below ``window``.

    Returns:
        NDArray[np.float64]: (window, window) matrix whose row ``o`` holds the
            weights that evaluate the fitted polynomial at sample ``o``.
    """
    positions = np.linspace(-1.0, 1.0, window) if window > 1 else np.zeros(1)
    design = np.vander(positions, degree + 1, increasing=True)
    return np.asarray(design @ np.linalg.pinv(design), dtype=np.float64)


# OpenCV counts into float32, which holds whole numbers exactly up to 2**24.
_HISTOGRAM_CHUNK_PIXELS = 1 << 23
_HISTOGRAM_LEVELS = 256


def gray_histogram(image: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Count the pixels at each gray level.

    Args:
        image (NDArray[np.uint8]): Gray image, or any array of 8-bit values.

    Returns:
        NDArray[np.int64]: 256 counts that add up to ``image.size``.
    """
    if image.dtype != np.uint8 or image.ndim != 2 or image.size == 0:
        return np.bincount(image.ravel(), minlength=_HISTOGRAM_LEVELS).astype(np.int64)
    contiguous = np.ascontiguousarray(image)
    rows = max(1, _HISTOGRAM_CHUNK_PIXELS // contiguous.shape[1])
    counts = np.zeros(_HISTOGRAM_LEVELS, dtype=np.int64)
    for first in range(0, contiguous.shape[0], rows):
        chunk = contiguous[first : first + rows]
        histogram = cv2.calcHist([chunk], [0], None, [_HISTOGRAM_LEVELS], [0, 256])
        counts += histogram.ravel().astype(np.int64)
    return counts


def _correlate_along(
    data: NDArray[np.float32],
    weights: NDArray[np.float32],
    axis: int,
) -> NDArray[np.float32]:
    """Correlate every line along ``axis`` with ``weights``, mirroring at the ends.

    Sums run in float64 and are rounded once to float32; float32 sums move the
    last bits and flip the final rounding of a few pixels per page.
    """
    taps = weights.astype(np.float64)
    one_tap = np.ones(1, dtype=np.float64)
    along_x, along_y = (taps, one_tap) if axis == 1 else (one_tap, taps)
    result = cv2.sepFilter2D(
        data.astype(np.float64),
        cv2.CV_64F,
        along_x,
        along_y,
        borderType=cv2.BORDER_REFLECT,
    )
    return np.asarray(result, dtype=np.float32)


def _savgol_axis(
    data: NDArray[np.float32],
    window: int,
    degree: int,
    axis: int,
) -> NDArray[np.float32]:
    """Savitzky-Golay smooth along one axis.

    Positions within half a window of either end use the first or last full
    window with the evaluation point moved, so nothing outside the image is read.
    """
    length = data.shape[axis]
    hat = _savgol_hat_matrix(window, degree)
    left = window // 2
    right = window - left - 1

    result = _correlate_along(data, hat[left].astype(np.float32), axis)
    source = np.moveaxis(data, axis, 0)
    target = np.moveaxis(result, axis, 0)
    target[:left] = np.tensordot(hat[:left], source[:window], axes=(1, 0))
    if right:
        target[length - right :] = np.tensordot(
            hat[window - right :], source[length - window :], axes=(1, 0)
        )
    return result


def savgol_filter_2d(
    image: NDArray[np.uint8],
    window_size: int | tuple[int, int] = 5,
    hor_degree: int = 2,
    vert_degree: int = 2,
) -> NDArray[np.uint8]:
    """Smooth a gray image by local least-squares polynomial fitting.

    Each pixel becomes the value of the polynomial that best fits its window.
    The fit is separable, so it runs as one pass per axis. Near the border the
    window stays inside the image and the polynomial is evaluated off centre.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int | tuple[int, int]): Window as one side or (width, height).
        hor_degree (int): Polynomial degree along x; below the window width.
        vert_degree (int): Polynomial degree along y; below the window height.

    Returns:
        NDArray[np.uint8]: New smoothed image. An image smaller than the window
            in either direction is returned unchanged (as a copy).

    Raises:
        ValueError: If a degree is negative or not below its window side.
    """
    width, height = _window_shape(window_size)
    if hor_degree < 0 or vert_degree < 0:
        msg = "polynomial degree must not be negative"
        raise ValueError(msg)
    if hor_degree >= width or vert_degree >= height:
        msg = "polynomial degree must be below the window size"
        raise ValueError(msg)
    if width > image.shape[1] or height > image.shape[0]:
        return image.copy()

    smooth = _savgol_axis(image.astype(np.float32), width, hor_degree, axis=1)
    smooth = _savgol_axis(smooth, height, vert_degree, axis=0)
    return np.clip(np.floor(smooth + 0.5), 0, 255).astype(np.uint8)


def stretch_gray_range(
    image: NDArray[np.uint8],
    black_clip: float,
    white_clip: float,
) -> NDArray[np.uint8]:
    """Stretch the used gray range to 0..255, optionally clipping the tails.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        black_clip (float): Fraction of pixels (below 1.0) clipped to black, or
            an absolute gray level when at least 1.0.
        white_clip (float): Fraction of pixels (below 1.0) clipped to white, or
            an absolute gray level when at least 1.0.

    Returns:
        NDArray[np.uint8]: New stretched image.
    """
    if image.size == 0:
        return image.copy()
    hist = gray_histogram(image)
    count = image.size
    if black_clip >= 1.0:
        low = int(black_clip + 0.5)
    else:
        low = _tail_level(hist, int(black_clip * count + 0.5))
    if white_clip >= 1.0:
        high = int(white_clip + 0.5)
    else:
        high = 255 - _tail_level(hist[::-1], int(white_clip * count + 0.5))

    levels = np.arange(256)
    if low >= high:
        mapping = np.where(levels <= (low + high) // 2, np.uint8(0), np.uint8(255))
    else:
        span = high - low
        mapping = (255 * (np.clip(levels, low, high) - low) + span // 2) // span
    return mapping.astype(np.uint8)[image]


def _tail_level(hist: NDArray[np.int64], clip_pixels: int) -> int:
    """Index of the first histogram bin that remains after clipping the tail.

    Args:
        hist (NDArray[np.int64]): Histogram read from the end being clipped.
        clip_pixels (int): Number of pixels to discard from that end.

    Returns:
        int: Bin index; one past the last bin when everything is clipped.
    """
    survivors = np.flatnonzero(np.cumsum(hist) > clip_pixels)
    return int(survivors[0]) if survivors.size else len(hist)
