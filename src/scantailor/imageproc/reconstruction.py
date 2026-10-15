"""Greyscale morphological reconstruction.

The marker is raised by directional sweeps: each line of pixels takes the
maximum of itself and its neighbours in the line before it, capped by the mask,
first down the image and back up, then across it and back. Sweeps repeat until
two in a row change nothing. Every sweep moves values arbitrarily far in its
direction, so natural images settle in a few rounds; only paths that wind back
and forth need more. The result is the unique reconstruction, so it equals
``skimage.morphology.reconstruction`` with a 4 or 8-connected footprint.
"""

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

Connectivity = Literal[4, 8]
Method = Literal["dilation", "erosion"]

# Element types cv2.transpose accepts; it is about three times faster than a
# NumPy transposed copy at page size.
_CV2_TRANSPOSE_DTYPES = frozenset(
    np.dtype(kind)
    for kind in (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)
)


def reconstruction(
    seed: NDArray,
    mask: NDArray,
    method: Method = "dilation",
    connectivity: Connectivity = 8,
) -> NDArray:
    """Reconstruct a mask from a seed by geodesic dilation or erosion.

    Dilation spreads the seed's bright values as far as the mask allows, so the
    result lies between the seed and the mask (seed <= result <= mask).
    Erosion is the dual: dark values spread and mask <= result <= seed.

    Args:
        seed (NDArray): 2-D integer or float array, the starting values. Not
            modified. Must not exceed ``mask`` for dilation, nor fall below it
            for erosion.
        mask (NDArray): 2-D array of the same shape and dtype as ``seed``; the
            limit the seed can grow or shrink to. Not modified.
        method (Method): ``"dilation"`` or ``"erosion"``.
        connectivity (Connectivity): 4 or 8 neighbours per pixel.

    Returns:
        NDArray: New array with the dtype and shape of ``seed``.

    Raises:
        ValueError: If the arguments are not 2-D arrays of equal shape and
            dtype, if ``method`` or ``connectivity`` is unknown, or if the
            seed is on the wrong side of the mask.
    """
    if method not in ("dilation", "erosion"):
        raise ValueError(f"unknown reconstruction method: {method!r}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    if seed.ndim != 2 or seed.shape != mask.shape:
        raise ValueError("seed and mask must be 2-D arrays of the same shape")
    if seed.dtype != mask.dtype:
        raise ValueError("seed and mask must have the same dtype")
    if seed.dtype.kind not in "uif":
        raise ValueError(f"unsupported dtype {seed.dtype}")

    if method == "dilation":
        if (seed > mask).any():
            raise ValueError("seed must not exceed mask for reconstruction by dilation")
        return _dilate(seed, mask, connectivity == 8)
    if (seed < mask).any():
        raise ValueError("seed must not be below mask for reconstruction by erosion")
    # Erosion is dilation of the order-reversed image.
    return _reverse_order(
        _dilate(_reverse_order(seed), _reverse_order(mask), connectivity == 8)
    )


def _reverse_order(image: NDArray) -> NDArray:
    """Map values so that their order is reversed; applying it twice restores them."""
    return -image if image.dtype.kind == "f" else ~image


def _lowest_value(dtype: np.dtype) -> float | int:
    """The value no other value of ``dtype`` is below, so it never wins a maximum."""
    return -np.inf if dtype.kind == "f" else int(np.iinfo(dtype).min)


def _transposed(image: NDArray) -> NDArray:
    """Return a C-contiguous transposed copy."""
    if image.dtype in _CV2_TRANSPOSE_DTYPES:
        return np.asarray(cv2.transpose(image), dtype=image.dtype)
    return np.ascontiguousarray(image.T)


def _with_side_columns(image: NDArray, fill: float | int) -> NDArray:
    """Copy ``image`` with one column of ``fill`` added on the left and right."""
    framed = np.empty((image.shape[0], image.shape[1] + 2), dtype=image.dtype)
    framed[:, 0] = fill
    framed[:, -1] = fill
    framed[:, 1:-1] = image
    return framed


def _sweep_down_and_up(framed: NDArray, mask: NDArray, eight_connected: bool) -> bool:
    """Raise each row to its neighbours in the row before, top down then bottom up.

    Args:
        framed (NDArray): Marker with a side column of the lowest value on each
            side, so the diagonal neighbours of the edge pixels need no special
            case. Raised in place.
        mask (NDArray): Upper limit, without side columns.
        eight_connected (bool): Also take the diagonal neighbours.

    Returns:
        bool: Whether any pixel changed.
    """
    before = framed[:, 1:-1].copy()
    centre = framed[:, 1:-1]
    left = framed[:, :-2]
    right = framed[:, 2:]
    rows = framed.shape[0]
    # A Python loop over rows; every step is a vectorised operation on a whole row.
    for order, step in ((range(1, rows), -1), (range(rows - 2, -1, -1), 1)):
        for row in order:
            line = centre[row]
            previous = row + step
            np.maximum(line, centre[previous], out=line)
            if eight_connected:
                np.maximum(line, left[previous], out=line)
                np.maximum(line, right[previous], out=line)
            np.minimum(line, mask[row], out=line)
    return not np.array_equal(before, centre)


def _dilate(seed: NDArray, mask: NDArray, eight_connected: bool) -> NDArray:
    """Reconstruction by dilation; returns a new array."""
    if seed.size == 0:
        return seed.copy()
    fill = _lowest_value(seed.dtype)
    upright_mask = np.ascontiguousarray(mask)
    masks = (upright_mask, _transposed(upright_mask))
    framed = _with_side_columns(seed, fill)
    transposed = 0
    quiet_sweeps = 0
    # Every neighbour direction is looked at by one of the two orientations, so
    # two quiet sweeps in a row mean no pixel can rise any further.
    while quiet_sweeps < 2:
        changed = _sweep_down_and_up(framed, masks[transposed], eight_connected)
        quiet_sweeps = 0 if changed else quiet_sweeps + 1
        framed = _with_side_columns(_transposed(framed[:, 1:-1]), fill)
        transposed ^= 1
    result = framed[:, 1:-1]
    return np.ascontiguousarray(result.T if transposed else result)
