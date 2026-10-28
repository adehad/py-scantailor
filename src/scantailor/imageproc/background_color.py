"""Dominant background colour of a page."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .filters import bgr_to_gray

_LEVELS = 256
# Width of the histogram window that is searched for the densest cluster of levels.
_CLUSTER_WIDTH = 10
# Each cv2.calcHist call counts at most this many pixels.
_MAX_CHUNK_PIXELS = 1 << 23


def otsu_threshold(hist: NDArray[np.int64]) -> int:
    """Otsu threshold of a 256-bin histogram.

    Pixels below the returned level are the dark class. When several levels
    tie for the best split, the middle one is used.

    Args:
        hist (NDArray[np.int64]): Pixel counts per gray level, length 256.

    Returns:
        int: Threshold from 0 to 256; 0 when the histogram has a single level.
    """
    levels = np.arange(_LEVELS)
    below = np.cumsum(hist).astype(np.float64)
    moment_below = np.cumsum(hist * levels).astype(np.float64)
    above = below[-1] - below
    moment_above = moment_below[-1] - moment_below

    valid = (below > 0) & (above > 0)
    if not valid.any():
        return 0
    mean_diff = np.zeros(_LEVELS)
    mean_diff[valid] = (
        moment_below[valid] / below[valid] - moment_above[valid] / above[valid]
    )
    variance = np.where(valid, mean_diff * mean_diff * below * above, -1.0)
    best = np.flatnonzero(variance == variance.max())
    return (int(best[0]) + 1 + int(best[-1]) + 1) >> 1


def dominant_level(hist: NDArray[np.int64]) -> int:
    """Median level of the densest ten-level window of a histogram.

    Args:
        hist (NDArray[np.int64]): Pixel counts per level, length 256.

    Returns:
        int: A level from 0 to 255.
    """
    window_sums = np.convolve(hist, np.ones(_CLUSTER_WIDTH, dtype=np.int64), "valid")
    start = int(np.argmax(window_sums))
    best_sum = int(window_sums[start])
    cumulative = np.cumsum(hist[start : start + _CLUSTER_WIDTH])
    return start + int(np.argmax(cumulative >= best_sum // 2))


def dominant_background_color(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_] | None = None,
) -> int | tuple[int, ...]:
    """Most common colour among the light (background) pixels of an image.

    Pixels at or above the Otsu threshold of the gray image inside ``mask`` are
    background; each channel's dominant level is taken separately.

    Args:
        image (NDArray[np.uint8]): 2-D gray or (H, W, 3|4) BGR(A) image.
        mask (NDArray[np.bool_] | None): True for the pixels to consider; None
            considers all. A mask with no True pixel counts as None.

    Returns:
        int | tuple[int, ...]: A gray level for 2-D input; one level per channel
            of ``image`` otherwise (alpha, if any, is 255).

    Raises:
        ValueError: If ``image`` is empty or ``mask`` differs in height and width.
    """
    if image.size == 0:
        msg = "image is empty"
        raise ValueError(msg)
    if mask is not None and mask.shape != image.shape[:2]:
        msg = "image and mask have different sizes"
        raise ValueError(msg)
    if mask is not None and not mask.any():
        mask = None

    gray = bgr_to_gray(image)
    background = gray >= otsu_threshold(_histogram(gray, 0, mask))
    if mask is not None:
        background &= mask
    if not background.any():
        background = mask

    if image.ndim == 2:
        return dominant_level(_histogram(image, 0, background))
    channels = [
        dominant_level(_histogram(image, channel, background))
        for channel in range(min(3, image.shape[2]))
    ]
    if image.shape[2] == 4:
        channels.append(255)
    return tuple(channels)


def _histogram(
    image: NDArray[np.uint8], channel: int, mask: NDArray[np.bool_] | None
) -> NDArray[np.int64]:
    """Count the levels of one channel of the pixels inside ``mask``.

    ``cv2.calcHist`` counts in float32, which is exact only up to 2^24, so the
    image is counted in chunks of rows below that and the counts summed as
    integers.

    Args:
        image (NDArray[np.uint8]): 2-D gray or (H, W, C) image.
        channel (int): Channel to count; 0 for gray.
        mask (NDArray[np.bool_] | None): Pixels to count; None counts all.

    Returns:
        NDArray[np.int64]: Pixel count per level, length 256.
    """
    hist = np.zeros(_LEVELS, dtype=np.int64)
    rows_per_chunk = max(1, _MAX_CHUNK_PIXELS // max(1, image.shape[1]))
    for top in range(0, image.shape[0], rows_per_chunk):
        rows = slice(top, top + rows_per_chunk)
        chunk_mask = None if mask is None else mask[rows].view(np.uint8)
        counts = cv2.calcHist(
            [image[rows]], [channel], chunk_mask, [_LEVELS], [0, _LEVELS]
        )
        hist += counts.ravel().astype(np.int64)
    return hist
