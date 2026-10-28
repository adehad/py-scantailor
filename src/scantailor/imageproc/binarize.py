"""Binarization algorithms for converting grayscale images to binary.

All functions take grayscale images as numpy arrays and return binary images.
Binary images use 0 for black (ink, the foreground) and 255 for white (the
background).

The local methods take ``delta``, the threshold adjustment of the output
filter; a positive value turns more pixels black. How it enters the threshold
differs per method and follows ScanTailor Advanced 1.2.1: Sauvola, Wolf and Fox
add ``delta / 128`` to the local contrast term, Window shifts the local mean,
Grad shifts the local mean the gradient weights are taken against, and Bradley
adds it to the threshold. A pixel is black when its gray level is strictly below
the threshold.

Window sums are taken over the part of the window inside the image, and a
mean is ``sum * (1 / area)`` like the C++ integral image code, so thresholds
agree with it to the last bit.
"""

from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from .background_color import otsu_threshold
from .filters import gray_histogram
from .row_bands import RowBand, map_row_bands

_GRAY_LEVELS = 256
# Divisor of the local deviation in Sauvola's formula.
_SAUVOLA_RANGE = 128.0
# The Window method states its coefficient on the Wolf scale.
_WINDOW_COEF_SCALE = 3.0
# Added to the local mean before it is cut to a gray level.
_GRAD_ROUNDING = 0.5
# Page-wide level of Grad when no pixel differs from its local mean.
_GRAD_FLAT_PAGE_LEVEL = 127.5

# Bytes of float64 temporaries per pixel of a band of window statistics: the
# pixels, their squares, the two window sums, the reciprocal areas, the mean,
# the variance and the threshold.
_WINDOW_BYTES_PER_PIXEL = 64

# Side of a square window, or (width, height) of a rectangular one.
WindowSize = int | tuple[int, int]


def peak_threshold(image: NDArray[np.uint8]) -> int:
    """Find a threshold value using peak detection in the histogram.

    This algorithm finds the dominant peaks from both ends of the histogram,
    then sets the threshold at 75% between the left and right peaks.
    Good for documents where foreground and background are clearly separated.

    Args:
        image: Grayscale input image.

    Returns:
        Threshold value (0-255).
    """
    # Compute histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten().astype(int)

    # Find right peak (starting from bright end)
    ri = 255
    right_peak = hist[ri]
    for i in range(254, -1, -1):
        if hist[i] <= right_peak:
            if float(hist[i]) < float(right_peak) * 0.66:
                break
            continue
        if hist[i] > right_peak:
            right_peak = hist[i]
            ri = i

    # Find left peak (starting from dark end)
    li = 0
    left_peak = hist[li]
    for i in range(1, 256):
        if hist[i] <= left_peak:
            if float(hist[i]) < float(left_peak) * 0.66:
                break
            continue
        if hist[i] > left_peak:
            left_peak = hist[i]
            li = i

    # Set threshold at 75% between peaks
    threshold = int(li + (ri - li) * 0.75)
    return threshold


def binarize_peak(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Binarize an image using peak threshold detection.

    Args:
        image: Grayscale input image.

    Returns:
        Binary image (0 or 255).
    """
    threshold = peak_threshold(image)
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return np.asarray(binary, dtype=np.uint8)


def mokji_threshold(
    image: NDArray[np.uint8],
    max_edge_width: int = 3,
    min_edge_magnitude: int = 20,
) -> int:
    """Global threshold from edge pixels, after Mokji and Abu-Bakar.

    Every pixel is paired with the darkest pixel within ``max_edge_width`` of
    it. Pairs whose difference reaches ``min_edge_magnitude`` are edges; the
    threshold is the mean of their midpoints. A margin of ``max_edge_width``
    pixels is skipped so that every window lies inside the image.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        max_edge_width (int): Widest edge to consider, in pixels; at least 1.
        min_edge_magnitude (int): Smallest gray-level step that counts as an
            edge; at least 1.

    Returns:
        int: Threshold; 128 when the image has no edges.

    Raises:
        ValueError: If ``max_edge_width`` or ``min_edge_magnitude`` is below 1.
    """
    if max_edge_width < 1:
        raise ValueError("max_edge_width must be >= 1")
    if min_edge_magnitude < 1:
        raise ValueError("min_edge_magnitude must be >= 1")

    window = 2 * max_edge_width + 1
    darkest = cv2.erode(image, np.ones((window, window), dtype=np.uint8))
    inner = (slice(max_edge_width, -max_edge_width),) * 2
    pairs = darkest[inner].astype(np.int64) * _GRAY_LEVELS + image[inner]
    counts = np.bincount(pairs.ravel(), minlength=_GRAY_LEVELS**2).reshape(
        _GRAY_LEVELS, _GRAY_LEVELS
    )

    dark_level, pixel_level = np.indices(counts.shape)
    is_edge = pixel_level - dark_level >= min_edge_magnitude
    edge_counts = np.where(is_edge, counts, 0)
    edge_total = int(edge_counts.sum())
    if edge_total == 0:
        return 128
    midpoint_sum = int(((dark_level + pixel_level) * edge_counts).sum())
    return int(0.5 * midpoint_sum / edge_total + 0.5)


def binarize_mokji(
    image: NDArray[np.uint8],
    max_edge_width: int = 3,
    min_edge_magnitude: int = 20,
) -> NDArray[np.uint8]:
    """Binarize an image with Mokji's edge-based global threshold.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        max_edge_width (int): See ``mokji_threshold``.
        min_edge_magnitude (int): See ``mokji_threshold``.

    Returns:
        NDArray[np.uint8]: New array, 0 for pixels below the threshold and 255
            elsewhere.
    """
    threshold = mokji_threshold(image, max_edge_width, min_edge_magnitude)
    return np.where(image < threshold, np.uint8(0), np.uint8(255))


def binarize_otsu(
    image: NDArray[np.uint8],
    adjustment: int = 0,
    bounds: tuple[int, int] = (0, _GRAY_LEVELS),
) -> NDArray[np.uint8]:
    """Binarize an image with Otsu's global threshold.

    The threshold minimises the intra-class variance of the histogram. It is
    shifted by ``adjustment`` and then clamped to ``bounds``; the clamp keeps a
    page whose content is all masked out from turning entirely black or white.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        adjustment (int): Added to the Otsu threshold; positive means more ink.
        bounds (tuple[int, int]): Lowest and highest threshold allowed after
            the adjustment. A pixel is black when it is below the threshold.

    Returns:
        NDArray[np.uint8]: New array of the same shape, 0 for ink and 255 for
            background.
    """
    threshold = _adjusted_otsu_threshold(image, adjustment, bounds)
    return _ink_below_level(image, threshold)


def _ink_below_level(image: NDArray[np.uint8], level: int) -> NDArray[np.uint8]:
    """Return 0 where ``image`` is below the gray ``level`` and 255 elsewhere."""
    if level <= 0:
        return np.full(image.shape, 255, dtype=np.uint8)
    # OpenCV's binary threshold is strict, so "below level" is "not above level - 1".
    _, ink = cv2.threshold(
        np.ascontiguousarray(image),
        min(level, _GRAY_LEVELS) - 1,
        255,
        cv2.THRESH_BINARY,
    )
    return np.asarray(ink, dtype=np.uint8)


def _adjusted_otsu_threshold(
    image: NDArray[np.uint8],
    adjustment: int,
    bounds: tuple[int, int],
) -> int:
    """Otsu threshold of ``image`` plus ``adjustment``, clamped to ``bounds``."""
    histogram = gray_histogram(image)
    low, high = bounds
    return int(np.clip(otsu_threshold(histogram) + adjustment, low, high))


def binarize_sauvola(
    image: NDArray[np.uint8],
    window_size: WindowSize = 25,
    k: float = 0.2,
    delta: float = 0.0,
) -> NDArray[np.uint8]:
    """Binarize an image using Sauvola's adaptive thresholding.

    The local threshold is ``mean * (1 - k * (1 - (deviation / 128 + delta / 128)))``
    over a local window. Good for uneven illumination.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int | tuple[int, int]): Side of a square local window, or
            its ``(width, height)``, clipped at the image border. An even size
            reaches one pixel further after the centre than before.
        k (float): Sensitivity, typically 0.1 to 0.5.
        delta (float): Threshold adjustment; positive means more ink.

    Returns:
        NDArray[np.uint8]: New array, 0 for ink and 255 for background.
    """
    ink = np.empty(image.shape, dtype=np.uint8)

    def threshold(
        rows: RowBand, mean: NDArray[np.float64], deviation: NDArray[np.float64]
    ) -> None:
        contrast = deviation / _SAUVOLA_RANGE + delta / _SAUVOLA_RANGE
        ink[rows.rows] = _ink_below(
            image[rows.rows], mean * (1.0 - k * (1.0 - contrast))
        )

    _map_window_statistics(image, window_size, threshold)
    return ink


def binarize_wolf(
    image: NDArray[np.uint8],
    window_size: WindowSize = 25,
    k: float = 0.5,
    delta: float = 0.0,
    lower_bound: int = 1,
    upper_bound: int = 254,
) -> NDArray[np.uint8]:
    """Binarize an image using Wolf's adaptive thresholding.

    The local threshold is ``min + (mean - min) * (1 - k * a)`` where ``min`` is
    the darkest level of the whole image and
    ``a = 1 - (deviation / max_deviation + delta / 128)``. The local mean and
    deviation are held in single precision, as C++ stores them. Pixels darker
    than ``lower_bound`` are always ink and pixels brighter than ``upper_bound``
    are always background.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int | tuple[int, int]): Side of a square local window, or
            its ``(width, height)``, clipped at the image border. An even size
            reaches one pixel further after the centre than before.
        k (float): Sensitivity, typically 0.3 to 0.5.
        delta (float): Threshold adjustment; positive means more ink.
        lower_bound (int): Gray levels below this are ink whatever the threshold.
        upper_bound (int): Gray levels above this are background.

    Returns:
        NDArray[np.uint8]: New array, 0 for ink and 255 for background.
    """
    # C++ keeps the statistics in float32, so storing them that way loses
    # nothing; only the page's largest deviation needs float64.
    mean32 = np.empty(image.shape, dtype=np.float32)
    deviation32 = np.empty(image.shape, dtype=np.float32)

    def store(
        rows: RowBand, mean: NDArray[np.float64], deviation: NDArray[np.float64]
    ) -> float:
        mean32[rows.rows] = mean
        deviation32[rows.rows] = deviation
        return float(deviation.max(initial=0.0))

    max_deviation = max(_map_window_statistics(image, window_size, store))
    if max_deviation <= 0.0:
        max_deviation = 1.0
    darkest = float(image.min())
    ink = np.empty(image.shape, dtype=np.uint8)

    def threshold(rows: RowBand) -> None:
        base = mean32[rows.rows].astype(np.float64) - darkest
        contrast = (
            deviation32[rows.rows].astype(np.float64) / max_deviation
            + delta / _SAUVOLA_RANGE
        )
        level = base * (1.0 - k * (1.0 - contrast)) + darkest
        ink[rows.rows] = _ink_within_bounds(
            image[rows.rows], level, lower_bound, upper_bound
        )

    map_row_bands(
        image.shape[0], 0, image.shape[1] * _WINDOW_BYTES_PER_PIXEL, threshold
    )
    return ink


def binarize_edge_div(
    image: NDArray[np.uint8],
    window_size: int = 25,
    kep: float = 0.5,
    kbd: float = 0.5,
    delta: float = 0.0,
) -> NDArray[np.uint8]:
    """Binarize an image using edge division preprocessing with Otsu.

    This method enhances edges by dividing each pixel by its local mean,
    then applies Otsu thresholding. The result is a combination of:
    - EdgePlus: Enhances pixels that differ from local mean
    - BlurDiv: Normalizes by the local mean to handle uneven illumination

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int): Side of the local window, clipped at the image
            border. An even size reaches one pixel further after the centre than
            before.
        kep (float): Edge enhancement weight (0-1). Higher = more edge enhancement.
        kbd (float): Blur division weight (0-1). Higher = more illumination
            normalization.
        delta (float): Added to the Otsu threshold of the normalised image
            without clamping; positive means more ink. The sum is truncated
            to an integer.

    Returns:
        NDArray[np.uint8]: New array, 0 for ink and 255 for background.
    """
    img_float = image.astype(np.float64)

    # Compute local mean using box filter
    local_mean = _window_mean(image, window_size)

    # Start with original values
    result = img_float.copy()

    if kep > 0.0:
        # EdgePlus transformation
        # edge = I / blur (shift = -0.5), mean value = 0.5
        edge = (result + 1) / (local_mean + 1) - 0.5
        # edgeplus = I * edge, mean value = 0.5 * mean(I)
        edgeplus = img_float * edge
        # Blend: k * edgeplus + (1 - k) * I
        result = kep * edgeplus + (1.0 - kep) * img_float

    if kbd > 0.0:
        # BlurDiv transformation
        # edgeinv = blur / I (shift = -0.5)
        edgeinv = (local_mean + 1) / (result + 1) - 0.5
        # edgenorm = edge * k + 1 * (1 - k)
        edgenorm = kbd * edgeinv + (1.0 - kbd)
        # Normalize: I / edgenorm
        result = np.where(edgenorm > 0, img_float / edgenorm, img_float)

    # Clip to valid range and convert back to uint8
    # npd: ok the edge division needs float
    result = np.clip(result, 0, 255).astype(np.uint8)

    histogram = gray_histogram(result)
    threshold = int(np.clip(int(otsu_threshold(histogram) + delta), 0, _GRAY_LEVELS))
    return np.where(result < threshold, np.uint8(0), np.uint8(255))


def binarize_bradley(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.15,
    delta: float = 0.0,
) -> NDArray[np.uint8]:
    """Binarize an image using Bradley's adaptive thresholding.

    The local threshold is ``mean * (1 - k)`` over a square window, plus
    ``delta``; for ``k >= 1`` it is ``delta`` alone. Simple and fast, works
    well for clean documents.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int): Side of the local window, clipped at the image
            border. An even size reaches one pixel further after the centre than
            before.
        k (float): Sensitivity (0-1). Higher values make it easier for pixels to
            be classified as ink. Typically 0.1-0.2.
        delta (float): Threshold adjustment; positive means more ink.

    Returns:
        NDArray[np.uint8]: New array, 0 for ink and 255 for background.
    """
    local_mean = _window_mean(image, window_size)
    threshold = local_mean * (1.0 - k) if k < 1.0 else np.zeros_like(local_mean)
    return _ink_below(image, threshold + delta)


def binarize_fox(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.3,
    delta: float = 0.0,
    lower_bound: int = 1,
    upper_bound: int = 254,
) -> NDArray[np.uint8]:
    """Binarize an image with Fox's local threshold, Wolf with Singh's contrast.

    A Wolf variant that measures local contrast as ``d / (256 - d)`` with
    ``d = pixel - mean`` (Singh et al., 2012) instead of the window deviation.
    With ``m`` the darkest level of the image and ``c`` that contrast divided by
    its largest value over the page, the threshold is
    ``m + (mean - m) * (1 - k / 2 * (1 - (c + delta / 128)))``. Pixels darker
    than ``lower_bound`` are always ink and pixels brighter than ``upper_bound``
    are always background.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int): Side of the local window, clipped at the image
            border. An even size reaches one pixel further before the centre
            than after.
        k (float): Sensitivity, typically 0.3.
        delta (float): Threshold adjustment; positive means more ink.
        lower_bound (int): Gray levels below this are ink whatever the threshold.
        upper_bound (int): Gray levels above this are background.

    Returns:
        NDArray[np.uint8]: New array, 0 for ink and 255 for background.

    Raises:
        ValueError: If ``window_size`` is below 1.
    """
    _require_window(window_size)
    if image.size == 0:
        return image.copy()
    gray = image.astype(np.float64)
    mean = _window_sum(gray, window_size) * _reciprocal_window_area(
        image.shape, window_size
    )
    difference = gray - mean
    deviation = difference / (_GRAY_LEVELS - difference)
    max_deviation = max(float(deviation.max()), 0.0)
    if max_deviation <= 0.0:
        max_deviation = 1.0
    darkest = float(image.min())
    base = mean - darkest
    contrast = deviation / max_deviation
    threshold = base * (1.0 - k * 0.5 * (1.0 - (contrast + delta / _SAUVOLA_RANGE)))
    return _ink_within_bounds(image, threshold + darkest, lower_bound, upper_bound)


def binarize_window(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.33,
    delta: float = 0.0,
    lower_bound: int = 1,
    upper_bound: int = 254,
) -> NDArray[np.uint8]:
    """Binarize an image with the dynamic window method of Bataineh et al.

    With ``mean`` and ``deviation`` of the local window, ``mean_full`` the
    integer mean of the whole page and ``dmin`` and ``dmax`` the extremes of the
    deviation over the page, the threshold is ``mean * (1 - 3 * k * md / kd)``
    where ``md = (mean + 1 - delta) / (mean_full + deviation + 1)``,
    ``kd = 1 + kdm * kds``, ``kdm = (2 * mean_full + 1) / (deviation + 1)`` and
    ``kds = (deviation - dmin) / (dmax - dmin)``, or 1 on a page of uniform
    deviation. Pixels darker than ``lower_bound`` are always ink and pixels
    brighter than ``upper_bound`` are always background.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int): Side of the local window, clipped at the image
            border. An even size reaches one pixel further before the centre
            than after.
        k (float): Sensitivity, on the Wolf scale; typically 0.3.
        delta (float): Threshold adjustment; positive means more ink.
        lower_bound (int): Gray levels below this are ink whatever the threshold.
        upper_bound (int): Gray levels above this are background.

    Returns:
        NDArray[np.uint8]: New array, 0 for ink and 255 for background.

    Raises:
        ValueError: If ``window_size`` is below 1.
    """
    _require_window(window_size)
    if image.size == 0:
        return image.copy()
    mean = np.empty(image.shape, dtype=np.float64)
    deviation = np.empty(image.shape, dtype=np.float64)

    def store(
        rows: RowBand,
        band_mean: NDArray[np.float64],
        band_deviation: NDArray[np.float64],
    ) -> None:
        mean[rows.rows] = band_mean
        deviation[rows.rows] = band_deviation

    _map_window_statistics(image, window_size, store)
    # Summed exactly: a 32-bit sum wraps on pages beyond about 16 megapixels.
    mean_full = float(int(image.sum(dtype=np.uint64)) // image.size)
    deviation_min = float(deviation.min())
    deviation_range = float(deviation.max()) - deviation_min

    ink = np.empty(image.shape, dtype=np.uint8)

    def threshold(rows: RowBand) -> None:
        band_mean = mean[rows.rows]
        band_deviation = deviation[rows.rows]
        md = (band_mean + 1.0 - delta) / (mean_full + band_deviation + 1.0)
        kdm = (mean_full + mean_full + 1.0) / (band_deviation + 1.0)
        if deviation_range > 0.0:
            kds = (band_deviation - deviation_min) / deviation_range
        else:
            kds = np.ones_like(band_deviation)
        kd = 1.0 + kdm * kds
        level = band_mean * (1.0 - k * _WINDOW_COEF_SCALE * md / kd)
        ink[rows.rows] = _ink_within_bounds(
            image[rows.rows], level, lower_bound, upper_bound
        )

    map_row_bands(
        image.shape[0], 0, image.shape[1] * _WINDOW_BYTES_PER_PIXEL, threshold
    )
    return ink


def binarize_grad(
    image: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.3,
    delta: float = 0.0,
    lower_bound: int = 1,
    upper_bound: int = 254,
) -> NDArray[np.uint8]:
    """Binarize an image with the gradient-snip threshold.

    The local mean, shifted by ``delta`` and held as a whole gray level, is
    weighted with ``g = |pixel - local mean|``. The page-wide level
    ``mean_g = sum(pixel * g) / sum(g)`` sits where the contrast is, and the
    threshold blends it with the local mean:
    ``mean_g * (1 - k) + local_mean * k``. Pixels darker than ``lower_bound``
    are always ink and pixels brighter than ``upper_bound`` are always
    background.

    Args:
        image (NDArray[np.uint8]): 2-D gray image. Not modified.
        window_size (int): Side of the local window, clipped at the image
            border. An even size reaches one pixel further before the centre
            than after.
        k (float): Weight of the local mean against the page-wide level,
            typically 0.3.
        delta (float): Shift of the local mean; positive means more ink.
        lower_bound (int): Gray levels below this are ink whatever the threshold.
        upper_bound (int): Gray levels above this are background.

    Returns:
        NDArray[np.uint8]: New array, 0 for ink and 255 for background.

    Raises:
        ValueError: If ``window_size`` is below 1.
    """
    _require_window(window_size)
    if image.size == 0:
        return image.copy()
    mean = _window_mean(image, window_size) + _GRAD_ROUNDING + delta
    # The cast to an integer level truncates toward zero.
    local_mean = np.floor(np.clip(mean, 0.0, 255.0)).astype(np.uint8)

    gradient = np.abs(local_mean.astype(np.int64) - image)
    gradient_total = int(gradient.sum())
    if gradient_total > 0:
        weighted_total = int((gradient * image.astype(np.int64)).sum())
        page_level = weighted_total / gradient_total
    else:
        page_level = _GRAD_FLAT_PAGE_LEVEL

    threshold = page_level * (1.0 - k) + local_mean * k
    return _ink_within_bounds(image, threshold, lower_bound, upper_bound)


def _require_window(window_size: int) -> None:
    """Raise ValueError for a window with no pixels."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")


def _ink_within_bounds(
    image: NDArray[np.uint8],
    threshold: NDArray[np.floating],
    lower_bound: int,
    upper_bound: int,
) -> NDArray[np.uint8]:
    """Ink below ``lower_bound``, or at most ``upper_bound`` and below ``threshold``."""
    ink = (image < lower_bound) | ((image <= upper_bound) & (image < threshold))
    return np.where(ink, np.uint8(0), np.uint8(255))


def _ink_below(
    image: NDArray[np.uint8],
    threshold: NDArray[np.floating],
) -> NDArray[np.uint8]:
    """Return 0 where ``image`` is strictly below ``threshold`` and 255 elsewhere."""
    return np.where(image < threshold, np.uint8(0), np.uint8(255))


def _map_window_statistics[T](
    image: NDArray[np.uint8],
    window_size: WindowSize,
    kernel: Callable[[RowBand, NDArray[np.float64], NDArray[np.float64]], T],
) -> list[T]:
    """Hand each band's window mean and deviation to ``kernel``.

    The window is clipped at the image border. Sums of whole numbers are exact
    in any order, so a band that reads a window's height of rows beyond its own
    gets the values of the whole image.

    Args:
        image (NDArray[np.uint8]): 2-D gray image.
        window_size (int | tuple[int, int]): Window side, or ``(width,
            height)``, in pixels.
        kernel (Callable): Takes the band and its float64 mean and standard
            deviation, both of the band's rows by the image's columns.

    Returns:
        list[T]: The kernel's results in band order.
    """
    rows, columns = image.shape
    window_width, window_height = _window_shape(window_size)
    row_extent = _window_extent(rows, window_height)
    column_extent = _window_extent(columns, window_width)

    def band(band_rows: RowBand) -> T:
        gray = image[band_rows.padded].astype(np.float64)
        inside = band_rows.inside
        reciprocal = 1.0 / np.multiply.outer(row_extent[band_rows.rows], column_extent)
        mean = _window_sum(gray, window_size)[inside] * reciprocal
        squares = _window_sum(gray * gray, window_size)[inside]
        variance = squares * reciprocal - mean * mean
        return kernel(band_rows, mean, np.sqrt(np.abs(variance)))

    return map_row_bands(rows, window_height, columns * _WINDOW_BYTES_PER_PIXEL, band)


def _window_mean(image: NDArray[np.uint8], window_size: int) -> NDArray[np.float64]:
    """Mean over a window clipped at the image border."""
    gray = image.astype(np.float64)
    return _window_sum(gray, window_size) * _reciprocal_window_area(
        image.shape, window_size
    )


def _reciprocal_window_area(
    shape: tuple[int, ...], window_size: int
) -> NDArray[np.float64]:
    """One over the number of image pixels in the window around each pixel."""
    return 1.0 / np.multiply.outer(
        _window_extent(shape[0], window_size), _window_extent(shape[1], window_size)
    )


def _window_extent(length: int, window_size: int) -> NDArray[np.float64]:
    """Number of the ``length`` positions that the window around each one covers."""
    before = window_size >> 1
    after = window_size - before
    centre = np.arange(length)
    return (np.minimum(length, centre + after) - np.maximum(0, centre - before)).astype(
        np.float64
    )


def _window_shape(window_size: WindowSize) -> tuple[int, int]:
    """Return ``(width, height)`` of a square or rectangular window."""
    if isinstance(window_size, int):
        return window_size, window_size
    return window_size


def _window_sum(
    values: NDArray[np.float64], window_size: WindowSize
) -> NDArray[np.float64]:
    """Sum over a window around each pixel, zero outside the image.

    An even window reaches one pixel further after the centre than before.
    """
    summed = cv2.boxFilter(
        values,
        cv2.CV_64F,
        _window_shape(window_size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.asarray(summed, dtype=np.float64)
