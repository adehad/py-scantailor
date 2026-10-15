"""Distance-limited flood fill from the border of an image."""

import numpy as np
from numpy.typing import NDArray


def limited_flood(
    mask: NDArray[np.bool_],
    max_steps: int,
) -> NDArray[np.bool_]:
    """Flood a mask from its border, no further than ``max_steps`` pixels.

    The flood advances one 4-connected step per round, from the pixels the
    previous round reached, so the work follows the flood front rather than
    the whole image.

    Args:
        mask (NDArray[np.bool_]): 2-D mask of the pixels the flood may enter.
            Not modified.
        max_steps (int): Longest 4-connected path from a border pixel, counted in
            steps, that is still reached.

    Returns:
        NDArray[np.bool_]: New mask of the reached pixels, including the border
            pixels of ``mask`` themselves.
    """
    rows, cols = mask.shape
    # A frame of closed pixels lets every neighbour index stay in range.
    enterable = np.zeros((rows + 2, cols + 2), dtype=bool)
    enterable[1:-1, 1:-1] = mask
    reached = np.zeros_like(enterable)
    border = np.zeros_like(enterable)
    border[1:-1, 1:-1] = True
    border[2:-2, 2:-2] = False
    enterable = enterable.ravel()
    reached = reached.ravel()
    front = np.flatnonzero(border.ravel() & enterable)
    stride = cols + 2
    # Pixels entered from two sides at once would double the next front.
    last_seen = np.empty(enterable.size, dtype=np.int64)
    for _ in range(max(max_steps, 0) + 1):
        if front.size == 0:
            break
        enterable[front] = False
        reached[front] = True
        neighbours = np.concatenate(
            [front - 1, front + 1, front - stride, front + stride]
        )
        neighbours = neighbours[enterable[neighbours]]
        order = np.arange(neighbours.size)
        last_seen[neighbours] = order
        front = neighbours[last_seen[neighbours] == order]
    return reached.reshape(rows + 2, cols + 2)[1:-1, 1:-1].copy()
