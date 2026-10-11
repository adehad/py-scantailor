"""Largest obstacle-free rectangle of a bilevel image."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray


class WhitespaceRect(NamedTuple):
    """An obstacle-free rectangle.

    Attributes:
        left (int): Column of the left edge.
        top (int): Row of the top edge.
        width (int): Number of columns.
        height (int): Number of rows.
    """

    left: int
    top: int
    width: int
    height: int

    @property
    def area(self) -> int:
        """Return the number of pixels covered."""
        return self.width * self.height


@njit(cache=True)
def _best_rectangle(
    obstacles: NDArray[np.bool_],
    width_power: int,
    height_power: int,
    min_width: int,
    min_height: int,
) -> tuple[int, int, int, int]:
    """Scan every row as a rectangle base with a monotone stack of column heights.

    Every maximal obstacle-free rectangle has its bottom edge on some row and
    is popped from the stack exactly once for that row, so a quality that never
    decreases when a rectangle grows is maximised over all of them.

    Returns:
        tuple[int, int, int, int]: left, top, width and height of the best
            rectangle, or four zeros when none reaches the minimum size.
    """
    rows, cols = obstacles.shape
    heights = np.zeros(cols + 1, dtype=np.int32)
    stack_start = np.empty(cols + 1, dtype=np.int32)
    stack_height = np.empty(cols + 1, dtype=np.int32)
    best_quality = -1
    best_left = 0
    best_top = 0
    best_width = 0
    best_height = 0

    for y in range(rows):
        row = obstacles[y]
        for x in range(cols):
            heights[x] = 0 if row[x] else heights[x] + 1
        depth = 0
        for x in range(cols + 1):
            start = x
            while depth > 0 and stack_height[depth - 1] >= heights[x]:
                depth -= 1
                popped_height = stack_height[depth]
                start = stack_start[depth]
                width = x - start
                if popped_height >= min_height and width >= min_width:
                    quality = 1
                    for _ in range(width_power):
                        quality *= width
                    for _ in range(height_power):
                        quality *= popped_height
                    if quality > best_quality:
                        best_quality = quality
                        best_left = start
                        best_top = y - popped_height + 1
                        best_width = width
                        best_height = popped_height
            stack_start[depth] = start
            stack_height[depth] = heights[x]
            depth += 1
    return best_left, best_top, best_width, best_height


def largest_whitespace(
    obstacles: NDArray[np.bool_],
    width_power: int = 1,
    height_power: int = 1,
    min_width: int = 1,
    min_height: int = 1,
) -> WhitespaceRect | None:
    """Find the obstacle-free rectangle of highest quality.

    The quality of a rectangle is ``width ** width_power * height ** height_power``,
    so ``(1, 1)`` prefers area, ``(2, 1)`` wide rectangles and ``(1, 2)`` tall ones.

    Args:
        obstacles (NDArray[np.bool_]): 2-D mask, True where a rectangle may not
            reach. Not modified.
        width_power (int): Exponent of the width in the quality, at least 1.
        height_power (int): Exponent of the height in the quality, at least 1.
        min_width (int): Narrowest rectangle to report.
        min_height (int): Lowest rectangle to report.

    Returns:
        WhitespaceRect | None: The best rectangle, or None when no obstacle-free
            rectangle reaches the minimum size.
    """
    if obstacles.size == 0:
        return None
    left, top, width, height = _best_rectangle(
        np.ascontiguousarray(obstacles),
        width_power,
        height_power,
        max(1, min_width),
        max(1, min_height),
    )
    if width == 0:
        return None
    return WhitespaceRect(int(left), int(top), int(width), int(height))
