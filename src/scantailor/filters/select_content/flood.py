"""Distance-limited flood fill from the border of an image."""

import cv2
import numpy as np
from numpy.typing import NDArray


def limited_flood(
    mask: NDArray[np.bool_],
    max_steps: int,
) -> NDArray[np.bool_]:
    """Flood a mask from its border, no further than ``max_steps`` pixels.

    The flood advances one 4-connected step per round from the pixels the
    previous round reached, so the work follows the flood front rather than
    the whole image. A pixel is closed as soon as it is reached, which keeps
    every front free of repeats, and the reached pixels are then exactly the
    ones of ``mask`` that are no longer open.

    Args:
        mask (NDArray[np.bool_]): 2-D mask of the pixels the flood may enter.
            Not modified.
        max_steps (int): Longest 4-connected path from a border pixel, counted in
            steps, that is still reached.

    Returns:
        NDArray[np.bool_]: New mask of the reached pixels, including the border
            pixels of ``mask`` themselves.
    """
    mask = np.ascontiguousarray(mask, dtype=bool)
    rows, cols = mask.shape
    if mask.size == 0:
        return mask.copy()
    # A frame of closed pixels lets every neighbour index stay in range.
    is_open = cv2.copyMakeBorder(
        mask.view(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    ).view(bool)
    stride = cols + 2
    flat_open = is_open.ravel()
    front = _border_indices(rows, cols)
    front = front[flat_open[front]]
    flat_open[front] = False
    for _ in range(max(max_steps, 0)):
        if front.size == 0:
            break
        reached = []
        for step in (-1, 1, -stride, stride):
            entered = front + step
            entered = entered[flat_open[entered]]
            flat_open[entered] = False
            reached.append(entered)
        front = np.concatenate(reached)
    return mask ^ is_open[1:-1, 1:-1]


def _border_indices(rows: int, cols: int) -> NDArray[np.intp]:
    """Return the flat indices of the border pixels of an image in its frame.

    Args:
        rows (int): Rows of the image, at least 1.
        cols (int): Columns of the image, at least 1.

    Returns:
        NDArray[np.intp]: Sorted indices, each once, into the image with a
            one pixel frame around it.
    """
    stride = cols + 2
    top = stride + np.arange(1, cols + 1)
    bottom = top + (rows - 1) * stride
    sides = stride * np.arange(2, rows)
    return np.unique(np.concatenate([top, bottom, sides + 1, sides + cols]))
