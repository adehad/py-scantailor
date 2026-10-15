"""Tests for the largest obstacle-free rectangle search."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from scantailor.filters.select_content.content_box_finder import _block_found
from scantailor.filters.select_content.max_whitespace import (
    WhitespaceRect,
    WhitespaceSearch,
    _maximal_rectangles,
    largest_whitespace,
)
from tests.filters.select_content.test_content_rect_pages import _page

_CUTS_FIXTURE = (
    Path(__file__).parents[2] / "fixtures" / "select_content" / ("whitespace_cuts.json")
)

# Name -> (powers, minimum size, shrink of each block, most cuts), as the
# content box finder runs its three searches.
_SEARCHES = {
    "wide": ((2, 1), 1, "height", 80),
    "tall": ((1, 2), 1, "width", 40),
    "area": ((1, 1), 4, None, 10),
}


def _reference_scan(
    obstacles: np.ndarray,
    width_power: int,
    height_power: int,
    min_width: int,
    min_height: int,
) -> tuple[int, int, int, int]:
    """Monotone stack scan of every row; the behaviour, ties included, to match."""
    rows, cols = obstacles.shape
    heights = [0] * (cols + 1)
    best_quality = -1
    best = (0, 0, 0, 0)
    for y in range(rows):
        for x in range(cols):
            heights[x] = 0 if obstacles[y, x] else heights[x] + 1
        stack: list[tuple[int, int]] = []
        for x in range(cols + 1):
            start = x
            while stack and stack[-1][1] >= heights[x]:
                start, popped_height = stack.pop()
                width = x - start
                if popped_height >= min_height and width >= min_width:
                    quality = width**width_power * popped_height**height_power
                    if quality > best_quality:
                        best_quality = quality
                        best = (start, y - popped_height + 1, width, popped_height)
            stack.append((start, heights[x]))
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


def _maximal_by_brute_force(obstacles: np.ndarray) -> set[tuple[int, int, int, int]]:
    """Every obstacle-free rectangle that no free row or column can extend."""
    rows, cols = obstacles.shape
    found = set()
    for top in range(rows):
        for bottom in range(top + 1, rows + 1):
            for left in range(cols):
                for right in range(left + 1, cols + 1):
                    if obstacles[top:bottom, left:right].any():
                        break
                    grows = (
                        (top > 0 and not obstacles[top - 1, left:right].any())
                        or (bottom < rows and not obstacles[bottom, left:right].any())
                        or (left > 0 and not obstacles[top:bottom, left - 1].any())
                        or (right < cols and not obstacles[top:bottom, right].any())
                    )
                    if not grows:
                        found.add((top, left, bottom, right))
    return found


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

    @pytest.mark.parametrize("period", [3, 4, 6])
    @pytest.mark.parametrize("powers", [(1, 1), (2, 1), (1, 2)])
    def test_ties_on_a_regular_grid_resolve_like_the_stack_scan(
        self, period: int, powers: tuple[int, int]
    ) -> None:
        """Equal cells everywhere: the top row, then the left end, then the taller."""
        obstacles = np.zeros((30, 40), dtype=bool)
        obstacles[::period, :] = True
        obstacles[:, ::period] = True

        assert largest_whitespace(obstacles, *powers) == _reference(
            obstacles, *powers, 1, 1
        )

    def test_equal_quality_rectangles_ending_in_one_column_pick_the_taller(
        self,
    ) -> None:
        """A 2 x 3 and a 3 x 2 rectangle with the same right edge and area 6."""
        obstacles = np.ones((4, 5), dtype=bool)
        obstacles[1:4, 3:5] = False
        obstacles[2:4, 2:5] = False

        found = largest_whitespace(obstacles)

        assert found == WhitespaceRect(left=3, top=1, width=2, height=3)
        assert found == _reference(obstacles, 1, 1, 1, 1)

    @pytest.mark.parametrize("density", [0.0, 0.1, 0.3, 1.0])
    def test_scan_lists_exactly_the_maximal_rectangles(self, density: float) -> None:
        rng = np.random.default_rng(17)
        for _ in range(6):
            obstacles = rng.random((7, 9)) < density

            listed = {
                tuple(edges) for edges in _maximal_rectangles(obstacles).T.tolist()
            }

            assert listed == _maximal_by_brute_force(obstacles)

    @pytest.mark.parametrize("shape", [(33_000, 4), (4, 33_000)])
    def test_image_too_long_for_16_bit_edges_matches_the_stack_scan(
        self, shape: tuple[int, int]
    ) -> None:
        """One free line runs the whole long side, past the 16-bit range."""
        obstacles = np.random.default_rng(13).random(shape) < 0.3
        if shape[0] > shape[1]:
            obstacles[:, 1] = False
        else:
            obstacles[1, :] = False

        for powers in ((1, 1), (2, 1), (1, 2)):
            assert largest_whitespace(obstacles, *powers) == _reference(
                obstacles, *powers, 1, 1
            )


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

    @pytest.mark.parametrize("min_size", [1, 3])
    def test_search_with_other_powers_matches_a_new_search_of_the_unblocked_image(
        self, min_size: int
    ) -> None:
        """Blocks before or after the split stay in the search they were made on."""
        obstacles = np.random.default_rng(11).random((40, 55)) < 0.05
        wide = WhitespaceSearch(obstacles, 2, 1, min_size, min_size)
        wide.block(5, 5, 30, 40)

        tall = wide.with_powers(1, 2)
        fresh = WhitespaceSearch(obstacles, 1, 2, min_size, min_size)
        for _ in range(15):
            found = fresh.largest()
            assert tall.largest() == found
            if found is None:
                break
            bottom, right = found.top + found.height, found.left + found.width
            tall.block(found.top, found.left, bottom, right)
            fresh.block(found.top, found.left, bottom, right)

        assert wide.largest() == _reference(
            obstacles | _blocked_mask(obstacles.shape, 5, 5, 30, 40),
            2,
            1,
            min_size,
            min_size,
        )

    def test_many_blocks_on_one_search_match_fresh_scans(self) -> None:
        """Enough blocks to fill the spare room of the rectangle table several times."""
        rng = np.random.default_rng(9)
        obstacles = rng.random((60, 70)) < 0.05
        search = WhitespaceSearch(obstacles, 1, 1)
        current = obstacles.copy()
        for _ in range(60):
            top, left = int(rng.integers(0, 60)), int(rng.integers(0, 70))
            bottom, right = (
                top + int(rng.integers(1, 6)),
                left + int(rng.integers(1, 6)),
            )
            search.block(top, left, bottom, right)
            current[top:bottom, left:right] = True

            assert search.largest() == largest_whitespace(current)


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


class _RescanningFinder:
    """Search the whole updated map afresh on every query."""

    def __init__(self, obstacles: np.ndarray) -> None:
        self._map = obstacles.copy()

    def largest(self) -> WhitespaceRect | None:
        return largest_whitespace(self._map, 2, 1, 1, 1)

    def block(self, top: int, left: int, bottom: int, right: int) -> None:
        self._map[top:bottom, left:right] = True


class TestWhitespaceSearchSpeed:
    def test_block_cutting_loop_is_much_faster_than_rescanning(self) -> None:
        obstacles = _page_like_map(np.random.default_rng(5))

        start = time.perf_counter()
        expected = _cut_sequence(_RescanningFinder, obstacles)
        rescan_time = time.perf_counter() - start
        start = time.perf_counter()
        actual = _cut_sequence(lambda m: WhitespaceSearch(m, 2, 1), obstacles)
        incremental_time = time.perf_counter() - start

        assert actual == expected
        assert len(actual) > 20
        assert rescan_time / incremental_time >= 3


def _cuts(ink: np.ndarray, search: str) -> list[list[int]]:
    """Cut whitespace from a page the way the content box finder does."""
    powers, min_size, shrink, max_count = _SEARCHES[search]
    area_threshold = min(ink.shape)
    finder = WhitespaceSearch(ink, *powers, min_size, min_size)
    found_all = []
    for _ in range(max_count):
        found = finder.largest()
        if found is None or found.area < area_threshold:
            break
        found_all.append(list(found))
        _block_found(finder, found, shrink)
    return found_all


@pytest.mark.parametrize("search", sorted(_SEARCHES))
@pytest.mark.parametrize("seed", [0, 1, 4, 6])
def test_cuts_on_a_seeded_page_match_the_recorded_ones(seed: int, search: str) -> None:
    """Every rectangle, tie-breaks included, as recorded from the stack scan."""
    expected = json.loads(_CUTS_FIXTURE.read_text())[str(seed)][search]

    assert _cuts(_page(seed) < 128, search) == expected
