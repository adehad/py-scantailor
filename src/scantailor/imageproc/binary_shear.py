"""Shears of black-and-white images by whole-pixel shifts.

Images are boolean arrays with True for black. Every column (vertical shear) or
row (horizontal shear) moves by an integer number of pixels, so runs of equal
shift form blocks that are copied as slices.
"""

import math

import numpy as np
from numpy.typing import NDArray

Block = tuple[int, int, int]


def shift_blocks(count: int, shear: float, origin: float) -> list[Block]:
    """Group columns (or rows) that move by the same whole number of pixels.

    The shift of item ``i`` is ``floor(0.5 + shear * (i + 0.5 - origin))``,
    accumulated one item at a time so that rounding matches a running sum.

    Args:
        count (int): Number of columns (vertical shear) or rows (horizontal).
        shear (float): Shift in pixels per item.
        origin (float): Coordinate of the item line that does not move.

    Returns:
        list[Block]: ``(start, stop, shift)`` triples covering ``range(count)``.
    """
    first = 0.5 + shear * (0.5 - origin)
    last = 0.5 + shear * (count - 0.5 - origin)
    if math.floor(first) == math.floor(last):
        return [(0, count, 0)]

    steps = np.full(count, shear, dtype=np.float64)
    steps[0] = first
    shifts = np.floor(np.cumsum(steps)).astype(np.int64)
    starts = [0, *(np.flatnonzero(np.diff(shifts)) + 1).tolist()]
    stops = [*starts[1:], count]
    return [(a, b, int(shifts[a])) for a, b in zip(starts, stops, strict=True)]


def h_shear(
    image: NDArray[np.bool_], shear: float, y_origin: float, background: bool = False
) -> NDArray[np.bool_]:
    """Shift each row sideways in proportion to its distance from ``y_origin``.

    Args:
        image (NDArray[np.bool_]): (H, W) image, True for black.
        shear (float): Horizontal shift in pixels per row.
        y_origin (float): Row coordinate that does not move.
        background (bool): Value that fills the uncovered pixels.

    Returns:
        NDArray[np.bool_]: New (H, W) image; positive shifts move content right.
    """
    height, width = image.shape
    out = np.full((height, width), background, dtype=np.bool_)
    for start, stop, shift in shift_blocks(height, shear, y_origin):
        if abs(shift) >= width:
            continue
        if shift > 0:
            out[start:stop, shift:] = image[start:stop, : width - shift]
        elif shift < 0:
            out[start:stop, : width + shift] = image[start:stop, -shift:]
        else:
            out[start:stop] = image[start:stop]
    return out


def v_shear(
    image: NDArray[np.bool_], shear: float, x_origin: float, background: bool = False
) -> NDArray[np.bool_]:
    """Shift each column up or down in proportion to its distance from ``x_origin``.

    Args:
        image (NDArray[np.bool_]): (H, W) image, True for black.
        shear (float): Vertical shift in pixels per column.
        x_origin (float): Column coordinate that does not move.
        background (bool): Value that fills the uncovered pixels.

    Returns:
        NDArray[np.bool_]: New (H, W) image; positive shifts move content down.
    """
    height, width = image.shape
    out = np.full((height, width), background, dtype=np.bool_)
    for start, stop, shift in shift_blocks(width, shear, x_origin):
        if abs(shift) >= height:
            continue
        if shift > 0:
            out[shift:, start:stop] = image[: height - shift, start:stop]
        elif shift < 0:
            out[: height + shift, start:stop] = image[-shift:, start:stop]
        else:
            out[:, start:stop] = image[:, start:stop]
    return out
