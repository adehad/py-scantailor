"""Largest obstacle-free rectangle of a bilevel image."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# Rows of a rectangle table: one column per rectangle, far edges exclusive.
_TOP, _LEFT, _BOTTOM, _RIGHT = range(4)
# Quality of a removed rectangle; real qualities are never negative.
_GONE = -1


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


def _pixel_rectangles(
    obstacles: NDArray[np.bool_],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Find the largest rectangle each free pixel bounds from below.

    That rectangle is as high as the free run above the pixel and as wide as
    every row of the run allows, so its left edge is the largest "first free
    column" over the rows of the run, and likewise for the right edge. The
    rows are walked one at a time: whole-row ufuncs are faster than a
    cumulative maximum down the columns, which would also need a trick to
    restart at every obstacle.

    Args:
        obstacles (NDArray[np.bool_]): 2-D mask, True where a rectangle may not reach.

    Returns:
        tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]: Height,
            first column and last column of the rectangle of every free pixel;
            0 height on obstacles, where the columns are meaningless.
    """
    rows, cols = obstacles.shape
    col_index = np.arange(cols, dtype=np.int32)
    free = (~obstacles).astype(np.int32)
    first_free = np.maximum.accumulate(
        np.where(obstacles, col_index + 1, np.int32(0)), axis=1
    )
    # Counting columns from the right turns the right edge into a maximum too.
    mirrored_last_free = np.maximum.accumulate(
        np.where(obstacles, cols - col_index, np.int32(0))[:, ::-1], axis=1
    )[:, ::-1]
    height = np.empty((rows, cols), dtype=np.int32)
    left = np.empty((rows, cols), dtype=np.int32)
    mirrored_right = np.empty((rows, cols), dtype=np.int32)
    height[0] = free[0]
    left[0] = first_free[0] * free[0]
    mirrored_right[0] = mirrored_last_free[0] * free[0]
    for y in range(1, rows):
        np.add(height[y - 1], free[y], out=height[y])
        height[y] *= free[y]
        np.maximum(left[y - 1], first_free[y], out=left[y])
        left[y] *= free[y]
        np.maximum(mirrored_right[y - 1], mirrored_last_free[y], out=mirrored_right[y])
        mirrored_right[y] *= free[y]
    return height, left, cols - 1 - mirrored_right


def _maximal_rectangles(obstacles: NDArray[np.bool_]) -> NDArray[np.int32]:
    """List every maximal obstacle-free rectangle.

    The rectangle a free pixel bounds from below cannot grow up, left or
    right. It is maximal when it cannot grow down either: when the pixel below
    is an obstacle or bounds a rectangle of other columns. A run of pixels of
    equal height names one rectangle, so only the first pixel of each run
    counts; the few repeats left are harmless.

    Args:
        obstacles (NDArray[np.bool_]): 2-D mask, True where a rectangle may not reach.

    Returns:
        NDArray[np.int32]: Rectangle table, one column of top, left, bottom
            and right per rectangle.
    """
    height, left, right = _pixel_rectangles(obstacles)
    named_here = height > 0
    named_here[:, 1:] &= height[:, 1:] != height[:, :-1]
    named_here[:-1] &= (
        obstacles[1:] | (left[1:] != left[:-1]) | (right[1:] != right[:-1])
    )
    rows, cols = np.nonzero(named_here)
    table = np.empty((4, rows.size), dtype=np.int32)
    table[_TOP] = rows - height[rows, cols] + 1
    table[_LEFT] = left[rows, cols]
    table[_BOTTOM] = rows + 1
    table[_RIGHT] = right[rows, cols] + 1
    return table


def _parts_beside(
    rectangles: NDArray[np.int32], top: int, left: int, bottom: int, right: int
) -> NDArray[np.int32]:
    """Cut the parts above, below, left and right of a block out of rectangles.

    A rectangle inside one of them that misses the block lies wholly on one
    side of the block, so the four parts hold every maximal rectangle the
    block leaves behind.

    Args:
        rectangles (NDArray[np.int32]): Rectangle table; every one overlaps the block.
        top (int): First row of the block.
        left (int): First column of the block.
        bottom (int): One past the last row of the block.
        right (int): One past the last column of the block.

    Returns:
        NDArray[np.int32]: Rectangle table of the parts that are not empty.
    """
    above, below, before, after = (rectangles.copy() for _ in range(4))
    np.minimum(above[_BOTTOM], top, out=above[_BOTTOM])
    np.maximum(below[_TOP], bottom, out=below[_TOP])
    np.minimum(before[_RIGHT], left, out=before[_RIGHT])
    np.maximum(after[_LEFT], right, out=after[_LEFT])
    parts = np.concatenate([above, below, before, after], axis=1)
    return parts[:, (parts[_TOP] < parts[_BOTTOM]) & (parts[_LEFT] < parts[_RIGHT])]


class WhitespaceSearch:
    """Repeated largest-whitespace queries on an image that gains obstacles.

    The image is scanned once for every maximal obstacle-free rectangle. A
    block replaces the rectangles it overlaps by their parts beside it, so a
    sequence of "find the best rectangle, block it, find the next" costs
    little more than the scan. Results are identical to a fresh scan of the
    updated image.

    Args:
        obstacles (numpy.ndarray of numpy.bool):
            2-D mask, True where a rectangle may not
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
        """List the maximal rectangles of the image."""
        self._powers = (width_power, height_power)
        self._minimum = (max(1, min_width), max(1, min_height))
        self._shape = obstacles.shape
        if obstacles.size == 0:
            found = np.zeros((4, 0), dtype=np.int32)
        else:
            found = _maximal_rectangles(np.asarray(obstacles, dtype=bool))
        self._store(*self._rate(found))

    def _rate(
        self, rectangles: NDArray[np.int32]
    ) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
        """Keep the rectangles that reach the minimum size, with their qualities.

        Args:
            rectangles (NDArray[np.int32]): Rectangle table.

        Returns:
            tuple[NDArray[np.int32], NDArray[np.int64]]: The rectangles kept and
                their qualities.
        """
        width = rectangles[_RIGHT] - rectangles[_LEFT]
        height = rectangles[_BOTTOM] - rectangles[_TOP]
        quality = np.power(width, self._powers[0], dtype=np.int64) * np.power(
            height, self._powers[1], dtype=np.int64
        )
        # A quality past the int64 range wraps around; negative ones never win.
        kept = (
            (width >= self._minimum[0]) & (height >= self._minimum[1]) & (quality >= 0)
        )
        return rectangles[:, kept], quality[kept]

    def largest(self) -> WhitespaceRect | None:
        """Return the best rectangle now, or None when none reaches the minimum.

        Among equal qualities the rectangle whose bottom edge is highest wins,
        then the one ending furthest left, then the taller one: the order in
        which a monotone stack of column heights, run along each row from the
        top, meets them.
        """
        best = self._quality.max(initial=_GONE)
        if best == _GONE:
            return None
        tied = self._found[:, self._quality == best]
        top, left, bottom, right = (
            int(edge)
            for edge in tied[
                :, np.lexsort((tied[_TOP], tied[_RIGHT], tied[_BOTTOM]))[0]
            ]
        )
        return WhitespaceRect(left, top, right - left, bottom - top)

    def block(self, top: int, left: int, bottom: int, right: int) -> None:
        """Make rows ``top:bottom`` and columns ``left:right`` an obstacle.

        Args:
            top (int): First row.
            left (int): First column.
            bottom (int): One past the last row.
            right (int): One past the last column.
        """
        rows, cols = self._shape
        top, left = max(top, 0), max(left, 0)
        bottom, right = min(bottom, rows), min(right, cols)
        found = self._found
        hit = np.flatnonzero((found[_TOP] < bottom) & (found[_BOTTOM] > top))
        hit = hit[
            (found[_LEFT, hit] < right)
            & (found[_RIGHT, hit] > left)
            & (self._quality[hit] != _GONE)
        ]
        if hit.size == 0:
            return
        parts, quality = self._rate(
            _parts_beside(found[:, hit], top, left, bottom, right)
        )
        self._quality[hit] = _GONE
        self._append(parts, quality)

    def _store(self, rectangles: NDArray[np.int32], quality: NDArray[np.int64]) -> None:
        """Hold rectangles in tables with as much room again to add more.

        Args:
            rectangles (NDArray[np.int32]): Rectangle table.
            quality (NDArray[np.int64]): Their qualities.
        """
        capacity = 2 * quality.size + 64
        self._found = np.zeros((4, capacity), dtype=np.int32)
        self._found[:, : quality.size] = rectangles
        self._quality = np.full(capacity, _GONE, dtype=np.int64)
        self._quality[: quality.size] = quality
        self._used = quality.size

    def _append(
        self, rectangles: NDArray[np.int32], quality: NDArray[np.int64]
    ) -> None:
        """Add rectangles after the others, dropping removed ones when room runs out.

        Removed rectangles stay in place with a quality that never wins, so a
        block does not copy the whole table each time.

        Args:
            rectangles (NDArray[np.int32]): Rectangle table to add.
            quality (NDArray[np.int64]): Their qualities.
        """
        end = self._used + quality.size
        if end > self._quality.size:
            used = slice(0, self._used)
            kept = self._quality[used] != _GONE
            self._store(
                np.concatenate([self._found[:, used][:, kept], rectangles], axis=1),
                np.concatenate([self._quality[used][kept], quality]),
            )
            return
        self._found[:, self._used : end] = rectangles
        self._quality[self._used : end] = quality
        self._used = end


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
        obstacles (numpy.ndarray of numpy.bool):
            2-D mask, True where a rectangle may not
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
