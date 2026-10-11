"""Distance-limited flood fill from the border of an image."""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def _flood(
    mask: NDArray[np.bool_],
    max_steps: int,
) -> NDArray[np.bool_]:
    """Breadth-first flood over 4-connected True pixels, seeded at the border."""
    rows, cols = mask.shape
    reached = np.zeros((rows, cols), dtype=np.bool_)
    queue = np.empty(rows * cols, dtype=np.int64)
    steps = np.zeros((rows, cols), dtype=np.int32)
    head = 0
    tail = 0
    for y in range(rows):
        for x in range(cols):
            on_border = y == 0 or y == rows - 1 or x == 0 or x == cols - 1
            if on_border and mask[y, x]:
                reached[y, x] = True
                queue[tail] = y * cols + x
                tail += 1
    while head < tail:
        cell = queue[head]
        head += 1
        y = cell // cols
        x = cell % cols
        if steps[y, x] >= max_steps:
            continue
        for k in range(4):
            ny = y + (k == 0) - (k == 1)
            nx = x + (k == 2) - (k == 3)
            if 0 <= ny < rows and 0 <= nx < cols:
                if mask[ny, nx] and not reached[ny, nx]:
                    reached[ny, nx] = True
                    steps[ny, nx] = steps[y, x] + 1
                    queue[tail] = ny * cols + nx
                    tail += 1
    return reached


def limited_flood(
    mask: NDArray[np.bool_],
    max_steps: int,
) -> NDArray[np.bool_]:
    """Flood a mask from its border, no further than ``max_steps`` pixels.

    Args:
        mask (NDArray[np.bool_]): 2-D mask of the pixels the flood may enter.
            Not modified.
        max_steps (int): Longest 4-connected path from a border pixel, counted in
            steps, that is still reached.

    Returns:
        NDArray[np.bool_]: New mask of the reached pixels, including the border
            pixels of ``mask`` themselves.
    """
    return _flood(np.ascontiguousarray(mask), max_steps)
