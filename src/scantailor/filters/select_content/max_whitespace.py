"""Largest obstacle-free rectangle of a bilevel image."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

_NONE = -1


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
def _scan_row(
    free_above: NDArray[np.int32],
    width_power: int,
    height_power: int,
    min_width: int,
    min_height: int,
    stack_start: NDArray[np.int32],
    stack_height: NDArray[np.int32],
) -> tuple[int, int, int, int]:
    """Find the best rectangle whose bottom edge lies on one row.

    Every maximal obstacle-free rectangle is popped from a monotone stack of
    column heights exactly once, so a quality that never decreases when a
    rectangle grows is maximised over all of them. Among equal qualities the
    first one popped wins.

    Returns:
        tuple[int, int, int, int]: quality, left, width and height of the best
            rectangle, with a quality of -1 when none reaches the minimum size.
    """
    cols = free_above.shape[0]
    best_quality = _NONE
    best_left = 0
    best_width = 0
    best_height = 0
    depth = 0
    for x in range(cols + 1):
        current = free_above[x] if x < cols else 0
        start = x
        while depth > 0 and stack_height[depth - 1] >= current:
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
                    best_width = width
                    best_height = popped_height
        stack_start[depth] = start
        stack_height[depth] = current
        depth += 1
    return best_quality, best_left, best_width, best_height


@njit(cache=True)
def _scan_all_rows(
    obstacles: NDArray[np.bool_],
    width_power: int,
    height_power: int,
    min_width: int,
    min_height: int,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """Build the free-run heights and the best rectangle of every bottom row."""
    rows, cols = obstacles.shape
    free_above = np.zeros((rows, cols), dtype=np.int32)
    quality = np.full(rows, _NONE, dtype=np.int64)
    left = np.zeros(rows, dtype=np.int32)
    width = np.zeros(rows, dtype=np.int32)
    height = np.zeros(rows, dtype=np.int32)
    stack_start = np.empty(cols + 1, dtype=np.int32)
    stack_height = np.empty(cols + 1, dtype=np.int32)
    for y in range(rows):
        for x in range(cols):
            if not obstacles[y, x]:
                free_above[y, x] = 1 + (free_above[y - 1, x] if y > 0 else 0)
        found = _scan_row(
            free_above[y],
            width_power,
            height_power,
            min_width,
            min_height,
            stack_start,
            stack_height,
        )
        quality[y], left[y], width[y], height[y] = found
    return free_above, quality, left, width, height


@njit(cache=True)
def _block_rectangle(
    free_above: NDArray[np.int32],
    quality: NDArray[np.int64],
    left: NDArray[np.int32],
    width: NDArray[np.int32],
    height: NDArray[np.int32],
    stale: NDArray[np.bool_],
    block_top: int,
    block_bottom: int,
    block_left: int,
    block_right: int,
) -> None:
    """Make a rectangle an obstacle and flag the rows whose best it destroys."""
    rows = free_above.shape[0]
    for y in range(block_top, block_bottom):
        free_above[y, block_left:block_right] = 0
    # Below the block each column changes only down to its next obstacle.
    for x in range(block_left, block_right):
        for y in range(block_bottom, rows):
            if free_above[y, x] == 0:
                break
            free_above[y, x] = free_above[y - 1, x] + 1
    for y in range(rows):
        if quality[y] == _NONE:
            continue
        if (
            y >= block_top
            and y - height[y] + 1 < block_bottom
            and left[y] < block_right
            and left[y] + width[y] > block_left
        ):
            stale[y] = True


@njit(cache=True)
def _best_row(
    free_above: NDArray[np.int32],
    quality: NDArray[np.int64],
    left: NDArray[np.int32],
    width: NDArray[np.int32],
    height: NDArray[np.int32],
    stale: NDArray[np.bool_],
    width_power: int,
    height_power: int,
    min_width: int,
    min_height: int,
) -> int:
    """Return the bottom row of the best rectangle, or -1 when there is none.

    Rows flagged stale hold an upper bound of their best quality, because
    obstacles only remove candidates. The row with the highest bound is
    rescanned until it is up to date; then no other row can beat it.
    """
    rows, cols = free_above.shape
    stack_start = np.empty(cols + 1, dtype=np.int32)
    stack_height = np.empty(cols + 1, dtype=np.int32)
    while True:
        best = _NONE
        best_row = -1
        for y in range(rows):
            if quality[y] > best:
                best = quality[y]
                best_row = y
        if best_row < 0 or not stale[best_row]:
            return best_row
        found = _scan_row(
            free_above[best_row],
            width_power,
            height_power,
            min_width,
            min_height,
            stack_start,
            stack_height,
        )
        quality[best_row], left[best_row], width[best_row], height[best_row] = found
        stale[best_row] = False


class WhitespaceSearch:
    """Repeated largest-whitespace queries on an image that gains obstacles.

    The first query scans the whole image. Later queries rescan only the rows
    whose best rectangle was hit by an obstacle added since, so a sequence of
    "find the best rectangle, block it, find the next" costs little more than
    one scan. Results are identical to a fresh scan of the updated image.

    Args:
        obstacles (NDArray[np.bool_]): 2-D mask, True where a rectangle may not
            reach. Not modified.
        width_power (int): Exponent of the width in the quality, at least 1.
        height_power (int): Exponent of the height in the quality, at least 1.
        min_width (int): Narrowest rectangle to report.
        min_height (int): Lowest rectangle to report.
    """

    def __init__(
        self,
        obstacles: NDArray[np.bool_],
        width_power: int = 1,
        height_power: int = 1,
        min_width: int = 1,
        min_height: int = 1,
    ) -> None:
        """Scan the whole image once for the best rectangle of every bottom row."""
        self._powers = (width_power, height_power)
        self._minimum = (max(1, min_width), max(1, min_height))
        self._is_empty = obstacles.size == 0
        self._shape = obstacles.shape
        if self._is_empty:
            return
        (
            self._free_above,
            self._quality,
            self._left,
            self._width,
            self._height,
        ) = _scan_all_rows(
            np.ascontiguousarray(obstacles), *self._powers, *self._minimum
        )
        self._stale = np.zeros(obstacles.shape[0], dtype=np.bool_)

    def largest(self) -> WhitespaceRect | None:
        """Return the best rectangle now, or None when none reaches the minimum."""
        if self._is_empty:
            return None
        row = _best_row(
            self._free_above,
            self._quality,
            self._left,
            self._width,
            self._height,
            self._stale,
            *self._powers,
            *self._minimum,
        )
        if row < 0:
            return None
        height = int(self._height[row])
        return WhitespaceRect(
            int(self._left[row]), row - height + 1, int(self._width[row]), height
        )

    def block(self, top: int, left: int, bottom: int, right: int) -> None:
        """Make rows ``top:bottom`` and columns ``left:right`` an obstacle.

        Args:
            top (int): First row.
            left (int): First column.
            bottom (int): One past the last row.
            right (int): One past the last column.
        """
        if self._is_empty:
            return
        rows, cols = self._shape
        top, left = max(top, 0), max(left, 0)
        bottom, right = min(bottom, rows), min(right, cols)
        if top >= bottom or left >= right:
            return
        _block_rectangle(
            self._free_above,
            self._quality,
            self._left,
            self._width,
            self._height,
            self._stale,
            top,
            bottom,
            left,
            right,
        )


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
    return WhitespaceSearch(
        obstacles, width_power, height_power, min_width, min_height
    ).largest()
