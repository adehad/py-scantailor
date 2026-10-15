"""Tests for the distance-limited border flood."""

import cv2
import numpy as np
import pytest

from scantailor.filters.select_content.flood import limited_flood
from tests.filters.select_content.test_content_rect_pages import _page


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

    def test_negative_steps_reach_only_the_border_pixels(self) -> None:
        mask = np.ones((5, 5), dtype=bool)

        assert np.array_equal(limited_flood(mask, -1), limited_flood(mask, 0))


def _dilation_flood(mask: np.ndarray, max_steps: int) -> np.ndarray:
    """Reference: grow the border pixels by one cross-shaped step at a time."""
    reached = np.zeros_like(mask)
    reached[[0, -1], :] = mask[[0, -1], :]
    reached[:, [0, -1]] = mask[:, [0, -1]]
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_steps):
        grown = cv2.dilate(reached.view(np.uint8), cross).view(bool) & mask
        if np.array_equal(grown, reached):
            break
        reached = grown
    return reached


@pytest.mark.parametrize(
    ("shape", "density", "max_steps"),
    [((300, 200), 0.55, 40), ((300, 200), 0.7, 25), ((1, 9), 0.7, 3), ((9, 1), 0.7, 3)],
)
def test_matches_step_by_step_dilation_on_random_masks(
    shape: tuple[int, int], density: float, max_steps: int
) -> None:
    mask = np.random.default_rng(1).random(shape) < density

    assert np.array_equal(
        limited_flood(mask, max_steps), _dilation_flood(mask, max_steps)
    )


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_matches_step_by_step_dilation_on_a_seeded_page(seed: int) -> None:
    """Paper of a full page, which reaches far in from the border, at the real limit."""
    paper = _page(seed) >= 128
    max_steps = min(paper.shape) // 4 - 1

    assert np.array_equal(
        limited_flood(paper, max_steps), _dilation_flood(paper, max_steps)
    )
