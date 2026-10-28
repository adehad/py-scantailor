"""Gray conversion and the global threshold used by page-split detection.

Both follow the arithmetic of the upstream C++ code so that detection on the
same scan lands on the same pixels.
"""

import numpy as np
from numpy.typing import NDArray

from scantailor.imageproc.filters import qgray

_GRAY_LEVELS = 256


def to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert an image to gray the way Qt's ``qGray`` does.

    Args:
        image (NDArray[np.uint8]): (H, W) gray, or (H, W, 3 or 4) in BGR(A) order.
            An alpha channel is ignored.

    Returns:
        NDArray[np.uint8]: (H, W) gray image; a gray input is returned as is.
    """
    if image.ndim == 2:
        return image
    if image.dtype == np.uint8 and image.shape[2] in (3, 4):
        return qgray(image)
    blue = image[..., 0].astype(np.int32)
    green = image[..., 1].astype(np.int32)
    red = image[..., 2].astype(np.int32)
    # npd: ok input wider than uint8
    return ((red * 11 + green * 16 + blue * 5) // 32).astype(np.uint8)


def otsu_threshold(gray: NDArray[np.uint8]) -> int:
    """Pick Otsu's global threshold; pixels below it are black.

    Ties between equally good thresholds resolve to the middle of the tied
    range, and a single-tone image yields 0.

    Args:
        gray (NDArray[np.uint8]): (H, W) gray image.

    Returns:
        int: The threshold; a pixel is black when its level is below it.
    """
    histogram = np.bincount(gray.ravel(), minlength=_GRAY_LEVELS)
    pixels_by_threshold = np.cumsum(histogram)
    moment_by_threshold = np.cumsum(histogram * np.arange(_GRAY_LEVELS, dtype=np.int64))
    total_pixels = int(pixels_by_threshold[-1])
    total_moment = int(moment_by_threshold[-1])

    max_variance = 0.0
    first_best = -1
    last_best = -1
    for level in range(_GRAY_LEVELS):
        pixels_below = int(pixels_by_threshold[level])
        pixels_above = total_pixels - pixels_below
        if pixels_below <= 0 or pixels_above <= 0:
            continue
        moment_below = int(moment_by_threshold[level])
        mean_below = moment_below / pixels_below
        mean_above = (total_moment - moment_below) / pixels_above
        mean_diff = mean_below - mean_above
        variance = mean_diff * mean_diff * pixels_below * pixels_above
        if variance > max_variance:
            max_variance = variance
            first_best = level
            last_best = level
        elif variance == max_variance:
            last_best = level

    # The search treats "<= level" as black while the result is used as "< threshold".
    return ((first_best + 1) + (last_best + 1)) >> 1
