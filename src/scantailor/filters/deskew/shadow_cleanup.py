"""Removal of horizontal shadows before the skew is measured.

Scanner shadows along a page edge are long dark bands that would otherwise
pull the detected angle towards zero or towards their own edge. Only those are
removed; other garbage is left in place.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi

# Resolution below which no further halving is done; the brick is defined at
# 150 DPI and scaled to the reduced resolution.
_REDUCE_WHILE_AT_LEAST_DPI = 200
_BRICK_DPI = 150
_BRICK_WIDTH_AT_BRICK_DPI = 200
_BRICK_HEIGHT_AT_BRICK_DPI = 14
_MIN_BRICK_PX = 1
_INK = 0
_WHITE = 255


def remove_horizontal_shadows(binary: NDArray[np.uint8], dpi: Dpi) -> NDArray[np.uint8]:
    """Remove every dark component that holds a long, thick horizontal run.

    The page is reduced to about 150 DPI and opened with a brick 200 pixels
    wide and 14 tall at that resolution. Whatever survives the opening marks a
    shadow, and the whole connected component of ink it touches is removed.

    Args:
        binary (numpy.ndarray of numpy.uint8): 2-D page with 0 for ink and 255 for
            background. Not modified.
        dpi (Dpi): Resolution of ``binary``; the axes may differ.

    Returns:
        numpy.ndarray of numpy.uint8: New array of the same shape and values with the
            shadows painted as background.
    """
    ink = (binary == _INK).astype(np.uint8)
    reduced = ink
    horizontal, vertical = dpi.horizontal, dpi.vertical
    while (
        horizontal >= _REDUCE_WHILE_AT_LEAST_DPI
        and vertical >= _REDUCE_WHILE_AT_LEAST_DPI
        and min(reduced.shape) >= 2
    ):
        reduced = _halve_keeping_two_of_four(reduced)
        horizontal //= 2
        vertical //= 2

    brick = (
        _from_150_dpi(_BRICK_HEIGHT_AT_BRICK_DPI, vertical),
        _from_150_dpi(_BRICK_WIDTH_AT_BRICK_DPI, horizontal),
    )
    kernel = np.ones(brick, dtype=np.uint8)
    opened = np.asarray(
        cv2.morphologyEx(reduced, cv2.MORPH_OPEN, kernel), dtype=np.uint8
    )
    if not opened.any():
        return binary.copy()

    seed = _upscale_integer_times(opened, ink.shape) & ink
    count, labels = cv2.connectedComponents(ink, connectivity=8)
    touched = np.zeros(count, dtype=bool)
    touched[np.unique(labels[seed == 1])] = True
    touched[0] = False
    cleaned = binary.copy()
    cleaned[touched[labels]] = _WHITE
    return cleaned


def _from_150_dpi(size: int, target_dpi: int) -> int:
    """Scale a length given at 150 DPI to ``target_dpi``, rounding half up."""
    return max(_MIN_BRICK_PX, (size * target_dpi + _BRICK_DPI // 2) // _BRICK_DPI)


def _halve_keeping_two_of_four(ink: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Halve both sides; a pixel is ink when at least two of its four sources are.

    ``ink`` holds 0 and 1, so the uint8 count of four never overflows.
    """
    height, width = ink.shape[0] // 2 * 2, ink.shape[1] // 2 * 2
    count = ink[0:height:2, 0:width:2] + ink[0:height:2, 1:width:2]
    count += ink[1:height:2, 0:width:2]
    count += ink[1:height:2, 1:width:2]
    return (count >= 2).view(np.uint8)


def _upscale_integer_times(
    mask: NDArray[np.uint8], shape: tuple[int, ...]
) -> NDArray[np.uint8]:
    """Repeat each pixel a whole number of times, padding the rest with zeros.

    Args:
        mask (numpy.ndarray of numpy.uint8): Reduced mask.
        shape (tuple[int, ...]): Height and width to reach; at least as large
            as ``mask``.

    Returns:
        numpy.ndarray of numpy.uint8: Mask of ``shape``.
    """
    y_scale = shape[0] // mask.shape[0]
    x_scale = shape[1] // mask.shape[1]
    expanded = np.repeat(np.repeat(mask, y_scale, axis=0), x_scale, axis=1)
    out = np.zeros(shape[:2], dtype=np.uint8)
    out[: expanded.shape[0], : expanded.shape[1]] = expanded
    return out
