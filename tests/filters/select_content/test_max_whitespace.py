"""Tests for the largest obstacle-free rectangle search."""

import numpy as np
import pytest

from scantailor.filters.select_content.max_whitespace import (
    WhitespaceRect,
    largest_whitespace,
)


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
