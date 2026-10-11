"""Greyscale morphological reconstruction.

Hybrid algorithm of Vincent (1993): one raster and one anti-raster scan, then a
FIFO queue that propagates the remaining changes. The result is the unique
reconstruction, so it equals ``skimage.morphology.reconstruction`` with a 4 or
8-connected footprint.
"""

from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray

Connectivity = Literal[4, 8]
Method = Literal["dilation", "erosion"]


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


def _dilate(seed: NDArray, mask: NDArray, eight_connected: bool) -> NDArray:
    """Reconstruction by dilation on contiguous copies; returns the new marker."""
    marker = np.array(seed, order="C")
    _dilate_in_place(marker, np.ascontiguousarray(mask), eight_connected)
    return marker


@njit(cache=True)
def _dilate_in_place(marker, mask, eight_connected):
    """Hybrid reconstruction by dilation; ``marker`` is raised to the result.

    Compiled by numba, so it is written with explicit index loops.
    """
    height, width = marker.shape
    queued = np.zeros((height, width), dtype=np.uint8)
    queue = np.empty(height * width, dtype=np.int32)
    head = 0
    count = 0

    # Raster scan: take the maximum over neighbours already visited.
    for y in range(height):
        for x in range(width):
            value = marker[y, x]
            if x > 0 and marker[y, x - 1] > value:
                value = marker[y, x - 1]
            if y > 0:
                if marker[y - 1, x] > value:
                    value = marker[y - 1, x]
                if eight_connected:
                    if x > 0 and marker[y - 1, x - 1] > value:
                        value = marker[y - 1, x - 1]
                    if x + 1 < width and marker[y - 1, x + 1] > value:
                        value = marker[y - 1, x + 1]
            marker[y, x] = min(value, mask[y, x])

    # Anti-raster scan: same with the other half of the neighbours. A pixel
    # that can still raise a not-yet-visited neighbour starts the queue.
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            value = marker[y, x]
            if x + 1 < width and marker[y, x + 1] > value:
                value = marker[y, x + 1]
            if y + 1 < height:
                if marker[y + 1, x] > value:
                    value = marker[y + 1, x]
                if eight_connected:
                    if x > 0 and marker[y + 1, x - 1] > value:
                        value = marker[y + 1, x - 1]
                    if x + 1 < width and marker[y + 1, x + 1] > value:
                        value = marker[y + 1, x + 1]
            value = min(value, mask[y, x])
            marker[y, x] = value

            pending = False
            if x + 1 < width:
                q = marker[y, x + 1]
                pending = q < value and q < mask[y, x + 1]
            if not pending and y + 1 < height:
                q = marker[y + 1, x]
                pending = q < value and q < mask[y + 1, x]
                if not pending and eight_connected:
                    if x > 0:
                        q = marker[y + 1, x - 1]
                        pending = q < value and q < mask[y + 1, x - 1]
                    if not pending and x + 1 < width:
                        q = marker[y + 1, x + 1]
                        pending = q < value and q < mask[y + 1, x + 1]
            if pending:
                queue[count] = y * width + x
                queued[y, x] = 1
                count += 1

    # Propagate with a circular FIFO. A pixel is queued at most once at a time,
    # so height * width slots always suffice.
    capacity = height * width
    while count > 0:
        flat = queue[head]
        head += 1
        if head == capacity:
            head = 0
        count -= 1
        y = flat // width
        x = flat - y * width
        queued[y, x] = 0
        value = marker[y, x]

        for dy in range(-1, 2):
            ny = y + dy
            if ny < 0 or ny >= height:
                continue
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if not eight_connected and dy != 0 and dx != 0:
                    continue
                nx = x + dx
                if nx < 0 or nx >= width:
                    continue
                if marker[ny, nx] < value and marker[ny, nx] != mask[ny, nx]:
                    marker[ny, nx] = min(value, mask[ny, nx])
                    if queued[ny, nx] == 0:
                        queued[ny, nx] = 1
                        tail = head + count
                        if tail >= capacity:
                            tail -= capacity
                        queue[tail] = ny * width + nx
                        count += 1
