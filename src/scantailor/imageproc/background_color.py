"""Dominant background colour of a page."""

import numpy as np
from numpy.typing import NDArray

from .filters import bgr_to_gray

_LEVELS = 256
# Width of the histogram window that is searched for the densest cluster of levels.
_CLUSTER_WIDTH = 10


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
    if mask is None or not mask.any():
        mask = np.ones(image.shape[:2], dtype=bool)

    gray = bgr_to_gray(image)
    gray_hist = np.bincount(gray[mask], minlength=_LEVELS)
    background = mask & (gray >= otsu_threshold(gray_hist))
    if not background.any():
        background = mask

    if image.ndim == 2:
        return dominant_level(np.bincount(image[background], minlength=_LEVELS))
    channels = [
        dominant_level(np.bincount(image[..., c][background], minlength=_LEVELS))
        for c in range(min(3, image.shape[2]))
    ]
    if image.shape[2] == 4:
        channels.append(255)
    return tuple(channels)
