"""Tests for the largest obstacle-free rectangle search."""

import time

import numpy as np
import pytest
from numba import njit
from numpy.typing import NDArray

from scantailor.filters.select_content.max_whitespace import (
    WhitespaceRect,
    WhitespaceSearch,
    largest_whitespace,
)


@njit(cache=True)
def _reference_scan(
    obstacles: NDArray[np.bool_],
    width_power: int,
    height_power: int,
    min_width: int,
    min_height: int,
) -> tuple[int, int, int, int]:
    """Full monotone stack scan of every row; the behaviour to be matched."""
    rows, cols = obstacles.shape
    heights = np.zeros(cols + 1, dtype=np.int32)
    stack_start = np.empty(cols + 1, dtype=np.int32)
    stack_height = np.empty(cols + 1, dtype=np.int32)
    best_quality = -1
    best = (0, 0, 0, 0)
    for y in range(rows):
        for x in range(cols):
            heights[x] = 0 if obstacles[y, x] else heights[x] + 1
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
                        best = (start, y - popped_height + 1, width, popped_height)
            stack_start[depth] = start
            stack_height[depth] = heights[x]
            depth += 1
    return best


def _reference(obstacles: np.ndarray, *args: int) -> WhitespaceRect | None:
    left, top, width, height = _reference_scan(obstacles, *args)
    return WhitespaceRect(left, top, width, height) if width else None


def _block_and_search(
    obstacles: np.ndarray, rng: np.random.Generator, powers: tuple[int, int]
) -> list[tuple[WhitespaceRect | None, WhitespaceRect | None]]:
    """Alternate best-rectangle queries and random blocks on both implementations."""
    reference_map = obstacles.copy()
    search = WhitespaceSearch(obstacles, *powers, 2, 2)
    pairs = []
    rows, cols = obstacles.shape
    for _ in range(25):
        pairs.append((search.largest(), _reference(reference_map, *powers, 2, 2)))
        found = pairs[-1][1]
        if found is None:
            break
        if rng.random() < 0.5:
            top, left = found.top, found.left
            bottom, right = top + found.height, left + found.width
            top += found.height // 5
            bottom -= found.height // 5
        else:
            top, left = int(rng.integers(0, rows)), int(rng.integers(0, cols))
            bottom = top + int(rng.integers(1, rows // 3 + 2))
            right = left + int(rng.integers(1, cols // 3 + 2))
        search.block(top, left, bottom, right)
        reference_map[max(top, 0) : bottom, max(left, 0) : right] = True
    return pairs


def _brute_force(
    obstacles: np.ndarray,
    width_power: int,
    height_power: int,
    min_width: int = 1,
    min_height: int = 1,
) -> int:
    """Return the best quality over every obstacle-free rectangle."""
    rows, cols = obstacles.shape
    best = 0
    for top in range(rows):
        for left in range(cols):
            for bottom in range(top, rows):
                for right in range(left, cols):
                    if obstacles[top : bottom + 1, left : right + 1].any():
                        break
                    width = right - left + 1
                    height = bottom - top + 1
                    if width >= min_width and height >= min_height:
                        best = max(best, width**width_power * height**height_power)
    return best


def _blocked_mask(
    shape: tuple[int, int], top: int, left: int, bottom: int, right: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


class TestLargestWhitespace:
    def test_empty_image_is_one_rectangle(self) -> None:
        found = largest_whitespace(np.zeros((6, 9), dtype=bool))

        assert found == WhitespaceRect(left=0, top=0, width=9, height=6)

    def test_rectangle_avoids_the_obstacle(self) -> None:
        obstacles = np.zeros((10, 10), dtype=bool)
        obstacles[:, 7] = True

        found = largest_whitespace(obstacles)

        assert found == WhitespaceRect(left=0, top=0, width=7, height=10)

    def test_every_pixel_blocked_gives_none(self) -> None:
        assert largest_whitespace(np.ones((5, 5), dtype=bool)) is None

    def test_zero_size_image_gives_none(self) -> None:
        assert largest_whitespace(np.zeros((0, 4), dtype=bool)) is None

    def test_wide_preference_picks_the_wide_rectangle(self) -> None:
        obstacles = np.ones((12, 12), dtype=bool)
        obstacles[0:2, 0:12] = False  # 12 x 2 wide strip, area 24
        obstacles[2:12, 0:3] = False  # 3 x 10 tall strip, area 30, joined at the corner

        wide = largest_whitespace(obstacles, width_power=2, height_power=1)
        tall = largest_whitespace(obstacles, width_power=1, height_power=2)

        assert wide is not None and tall is not None
        assert wide.width > wide.height
        assert tall.height > tall.width

    def test_minimum_size_excludes_thin_rectangles(self) -> None:
        obstacles = np.ones((10, 10), dtype=bool)
        obstacles[4, :] = False  # Only a one row strip is free.

        assert largest_whitespace(obstacles, min_height=2) is None
        assert largest_whitespace(obstacles, min_height=1) is not None

    def test_input_is_not_modified(self) -> None:
        obstacles = np.zeros((8, 8), dtype=bool)
        obstacles[3, 3] = True
        before = obstacles.copy()

        largest_whitespace(obstacles)

        assert np.array_equal(obstacles, before)

    def test_area_property(self) -> None:
        assert WhitespaceRect(0, 0, 4, 5).area == 20

    @pytest.mark.parametrize("powers", [(1, 1), (2, 1), (1, 2)])
    def test_quality_matches_brute_force_on_random_images(
        self, powers: tuple[int, int]
    ) -> None:
        rng = np.random.default_rng(7)
        for _ in range(8):
            obstacles = rng.random((9, 11)) < 0.12

            found = largest_whitespace(obstacles, *powers)
            expected = _brute_force(obstacles, *powers)

            if found is None:
                assert expected == 0
                continue
            assert not obstacles[
                found.top : found.top + found.height,
                found.left : found.left + found.width,
            ].any()
            assert found.width ** powers[0] * found.height ** powers[1] == expected

    def test_minimum_size_matches_brute_force(self) -> None:
        rng = np.random.default_rng(11)
        obstacles = rng.random((10, 12)) < 0.15

        found = largest_whitespace(obstacles, 1, 1, 3, 2)
        expected = _brute_force(obstacles, 1, 1, 3, 2)

        assert (found.area if found else 0) == expected


class TestWhitespaceSearch:
    @pytest.mark.parametrize("powers", [(1, 1), (2, 1), (1, 2)])
    def test_sequence_of_blocks_matches_a_fresh_scan_each_time(
        self, powers: tuple[int, int]
    ) -> None:
        rng = np.random.default_rng(3)
        for density in (0.0, 0.02, 0.1):
            obstacles = rng.random((40, 55)) < density

            for incremental, fresh in _block_and_search(obstacles, rng, powers):
                assert incremental == fresh

    def test_blocking_outside_the_image_is_clipped(self) -> None:
        search = WhitespaceSearch(np.zeros((4, 4), dtype=bool))

        search.block(-3, -3, 2, 9)

        assert search.largest() == WhitespaceRect(left=0, top=2, width=4, height=2)

    def test_empty_image_has_no_rectangle_and_ignores_blocks(self) -> None:
        search = WhitespaceSearch(np.zeros((0, 3), dtype=bool))

        search.block(0, 0, 1, 1)

        assert search.largest() is None

    def test_strided_input_is_searched_and_left_unmodified(self) -> None:
        base = np.zeros((12, 20), dtype=bool)
        base[:, 11] = True
        strided = base[:, ::2]
        before = base.copy()
        search = WhitespaceSearch(strided)

        search.block(0, 0, 12, 3)
        found = search.largest()

        assert found == _reference(
            np.ascontiguousarray(strided) | _blocked_mask(strided.shape, 0, 0, 12, 3),
            1,
            1,
            1,
            1,
        )
        assert np.array_equal(base, before)

    def test_blocking_everything_leaves_no_rectangle(self) -> None:
        search = WhitespaceSearch(np.zeros((5, 6), dtype=bool))

        search.block(0, 0, 5, 6)

        assert search.largest() is None


def _page_like_map(rng: np.random.Generator) -> np.ndarray:
    """Text-like obstacle map: many short ink runs in lines on a 900 x 640 page."""
    page = np.zeros((900, 640), dtype=bool)
    for top in range(40, 860, 14):
        for left in range(40, 600, 9):
            if rng.random() < 0.8:
                page[top : top + 8, left : left + int(rng.integers(3, 8))] = True
    return page


def _cut_sequence(finder_factory, obstacles: np.ndarray) -> list:
    """Run the block-cutting loop used by content detection."""
    found_all = []
    finder = finder_factory(obstacles)
    for _ in range(60):
        found = finder.largest()
        if found is None:
            break
        found_all.append(found)
        margin = found.height // 5
        finder.block(
            found.top + margin,
            found.left,
            found.top + found.height - margin,
            found.left + found.width,
        )
    return found_all


class _ReferenceFinder:
    def __init__(self, obstacles: np.ndarray) -> None:
        self._map = obstacles.copy()

    def largest(self) -> WhitespaceRect | None:
        return _reference(self._map, 2, 1, 1, 1)

    def block(self, top: int, left: int, bottom: int, right: int) -> None:
        self._map[top:bottom, left:right] = True


class TestWhitespaceSearchSpeed:
    def test_block_cutting_loop_is_much_faster_than_rescanning(self) -> None:
        obstacles = _page_like_map(np.random.default_rng(5))
        _cut_sequence(lambda m: WhitespaceSearch(m, 2, 1), obstacles[:20, :20])
        _reference(obstacles[:20, :20], 2, 1, 1, 1)

        start = time.perf_counter()
        expected = _cut_sequence(_ReferenceFinder, obstacles)
        reference_time = time.perf_counter() - start
        start = time.perf_counter()
        actual = _cut_sequence(lambda m: WhitespaceSearch(m, 2, 1), obstacles)
        incremental_time = time.perf_counter() - start

        assert actual == expected
        assert len(actual) > 20
        assert reference_time / incremental_time >= 3
