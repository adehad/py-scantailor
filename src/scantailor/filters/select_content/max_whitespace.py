"""Largest obstacle-free rectangle of a bilevel image."""

import copy
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


def _rectangle_edges(obstacles: NDArray[np.bool_]) -> NDArray[np.signedinteger]:
    """Find the largest rectangle each free pixel bounds from below.

    That rectangle is as high as the free run above the pixel and as wide as
    every row of the run allows, so its left edge is the largest "first free
    column" over the rows of the run, and likewise for the right edge. Each
    row follows from the one above with one maximum and one minimum over all
    three edges at once: the maximum carries the run down and narrows it to
    the free span of the row, and the minimum, which only bites on obstacles,
    starts a new run below them. Edges are stored so that both steps act the
    same way on every edge.

    Args:
        obstacles (NDArray[np.bool_]): 2-D mask, True where a rectangle may not reach.

    Returns:
        NDArray[np.signedinteger]: Array of shape (rows, 3, cols) holding, per
            pixel, minus the top row, the left column, and the number of
            columns between the right edge and the right of the image.
            Obstacles hold a negative value in every channel.
    """
    rows, cols = obstacles.shape
    dtype = np.int16 if max(rows, cols) < np.iinfo(np.int16).max else np.int32
    col_number = np.arange(1, cols + 1, dtype=dtype)
    # (rows, 1): minus the top row of a run that starts on the next row.
    below = -np.arange(1, rows + 1, dtype=dtype)[:, None]
    edges = np.empty((rows, 3, cols), dtype=dtype)
    # A run carries its top row down unchanged.
    edges[:, 0] = np.iinfo(dtype).min
    # First free column of the row segment of every pixel.
    np.multiply(obstacles, col_number, out=edges[:, 1])
    np.maximum.accumulate(edges[:, 1], axis=1, out=edges[:, 1])
    # Columns right of the row segment, counted on the mirrored image.
    mirrored = edges[:, 2, ::-1]
    np.multiply(obstacles[:, ::-1], col_number, out=mirrored)
    np.maximum.accumulate(mirrored, axis=1, out=mirrored)
    # (rows, 1, cols): forces all three edges of an obstacle down to below.
    ceiling = np.where(obstacles, below, np.iinfo(dtype).max)[:, None, :]
    # Above the image is a run of height 0 that starts at row 0.
    np.maximum(edges[0], 0, out=edges[0])
    np.minimum(edges[0], ceiling[0], out=edges[0])
    for y in range(1, rows):
        np.maximum(edges[y - 1], edges[y], out=edges[y])
        np.minimum(edges[y], ceiling[y], out=edges[y])
    return edges


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
    cols = obstacles.shape[1]
    edges = _rectangle_edges(obstacles)
    minus_top, left, after_right = edges[:, 0], edges[:, 1], edges[:, 2]
    named_here = ~obstacles
    named_here[:, 1:] &= minus_top[:, 1:] != minus_top[:, :-1]
    named_here[:-1] &= (
        obstacles[1:] | (left[1:] != left[:-1]) | (after_right[1:] != after_right[:-1])
    )
    row, col = np.divmod(np.flatnonzero(named_here), cols)
    table = np.empty((4, row.size), dtype=np.int32)
    table[_TOP] = -minus_top[row, col]
    table[_LEFT] = left[row, col]
    table[_BOTTOM] = row + 1
    table[_RIGHT] = cols - after_right[row, col]
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
        self._maximal = found
        self._store(*self._rate(found))

    def with_powers(self, width_power: int, height_power: int) -> WhitespaceSearch:
        """Start a second search of the same image with other quality exponents.

        The new search sees the image as it was given to the constructor,
        without the blocks made on this one since, and keeps the minimum size.
        It reuses the scan, so it costs much less than a new instance.

        Args:
            width_power (int): Exponent of the width in the quality, at least 1.
            height_power (int): Exponent of the height in the quality, at least 1.

        Returns:
            WhitespaceSearch: An independent search; blocks on either one do not
                reach the other.
        """
        search = copy.copy(self)
        search._powers = (width_power, height_power)
        search._store(*search._rate(self._maximal))
        return search

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
        quality = self._quality[: self._used]
        best = quality.max(initial=_GONE)
        if best == _GONE:
            return None
        tied = self._found[:, np.flatnonzero(quality == best)]
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
        found = self._found[:, : self._used]
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
