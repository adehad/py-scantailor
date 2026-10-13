"""Whole-pixel shears of black-and-white images."""

from itertools import pairwise

import numpy as np
import pytest

from scantailor.imageproc.binary_shear import h_shear, shift_blocks, v_shear
from tests.filters.page_split import cpp_reference as cpp


def _noise(height: int, width: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).random((height, width)) < 0.3


@pytest.mark.parametrize("shear", [0.0, 0.01, -0.01, 0.1227, -0.0875, 0.5, 3.0, -9.0])
@pytest.mark.parametrize("shape", [(60, 90), (61, 33), (1, 20)])
def test_vertical_shear_matches_upstream(shear: float, shape: tuple[int, int]) -> None:
    image = _noise(*shape)
    origin = 0.5 * shape[1]

    got = v_shear(image, shear, origin)

    np.testing.assert_array_equal(got, cpp.v_shear(image, shear, origin))


@pytest.mark.parametrize("shear", [0.0, 0.02, -0.02, 0.1227, -3.0])
@pytest.mark.parametrize("shape", [(60, 90), (33, 61)])
def test_horizontal_shear_matches_upstream(
    shear: float, shape: tuple[int, int]
) -> None:
    image = _noise(*shape)
    origin = 0.5 * shape[0]

    got = h_shear(image, shear, origin)

    np.testing.assert_array_equal(got, cpp.h_shear(image, shear, origin))


def test_a_positive_horizontal_shear_moves_lower_rows_right() -> None:
    image = np.zeros((10, 12), dtype=bool)
    image[:, 6] = True

    got = h_shear(image, 0.5, 5.0)

    assert np.flatnonzero(got[0]).tolist() == [4]
    assert np.flatnonzero(got[9]).tolist() == [8]


def test_uncovered_pixels_take_the_background() -> None:
    image = np.ones((4, 8), dtype=bool)

    white = h_shear(image, 1.0, 2.0)
    black = h_shear(image, 1.0, 2.0, background=True)

    assert not white[3, 0]
    assert black[3, 0]


def test_a_shear_too_small_to_move_anything_has_one_block() -> None:
    assert shift_blocks(100, 0.001, 50.0) == [(0, 100, 0)]


def test_blocks_cover_every_item_once_with_growing_shifts() -> None:
    blocks = shift_blocks(100, 0.07, 50.0)

    assert blocks[0][0] == 0
    assert blocks[-1][1] == 100
    assert all(a[1] == b[0] for a, b in pairwise(blocks))
    shifts = [shift for _start, _stop, shift in blocks]
    assert shifts == sorted(set(shifts))
