"""Tests for the distance-limited border flood."""

import numpy as np

from scantailor.filters.select_content.flood import limited_flood


class TestLimitedFlood:
    def test_floods_a_strip_from_the_border_up_to_the_limit(self) -> None:
        mask = np.zeros((5, 20), dtype=bool)
        mask[2, :] = True

        reached = limited_flood(mask, 4)

        assert reached[2, :5].tolist() == [True] * 5
        assert not reached[2, 5:15].any()
        # The right end is a border too and floods inwards.
        assert reached[2, 15:].tolist() == [True] * 5

    def test_pixels_not_connected_to_the_border_are_not_reached(self) -> None:
        mask = np.zeros((9, 9), dtype=bool)
        mask[3:6, 3:6] = True

        assert not limited_flood(mask, 100).any()

    def test_diagonal_neighbours_do_not_connect(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True

        reached = limited_flood(mask, 10)

        assert reached[0, 0]
        assert not reached[1, 1]

    def test_zero_steps_reaches_only_the_border_pixels(self) -> None:
        mask = np.ones((5, 5), dtype=bool)

        reached = limited_flood(mask, 0)

        assert reached.sum() == 16
        assert not reached[1:4, 1:4].any()

    def test_empty_mask_reaches_nothing(self) -> None:
        assert not limited_flood(np.zeros((4, 4), dtype=bool), 5).any()

    def test_input_is_not_modified(self) -> None:
        mask = np.ones((4, 4), dtype=bool)
        before = mask.copy()

        limited_flood(mask, 2)

        assert np.array_equal(mask, before)
