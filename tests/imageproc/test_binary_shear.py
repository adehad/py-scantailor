"""Whole-pixel shears of black-and-white images."""

import math
from itertools import pairwise

import numpy as np
import pytest

from scantailor.imageproc.binary_shear import h_shear, shift_blocks, v_shear


def _noise(height: int, width: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).random((height, width)) < 0.3


def _expected_shift(index: int, shear: float, origin: float) -> int:
    """Whole-pixel shift of item ``index``, rounded from its centre's offset.

    Args:
        index (int): Column of a vertical shear or row of a horizontal one.
        shear (float): Pixels moved per pixel away from ``origin``.
        origin (float): Position that does not move.

    Returns:
        int: The shift, with halves rounded up.
    """
    return math.floor(0.5 + shear * (index + 0.5 - origin))


@pytest.mark.parametrize("shape", [(60, 90), (61, 33), (1, 20)])
def test_zero_vertical_shear_is_the_identity(shape: tuple[int, int]) -> None:
    image = _noise(*shape)

    np.testing.assert_array_equal(v_shear(image, 0.0, 0.5 * shape[1]), image)


@pytest.mark.parametrize("shape", [(60, 90), (33, 61)])
def test_zero_horizontal_shear_is_the_identity(shape: tuple[int, int]) -> None:
    image = _noise(*shape)

    np.testing.assert_array_equal(h_shear(image, 0.0, 0.5 * shape[0]), image)


# Shears of a hundredth of a pixel per column take many columns to round to the
# next pixel; whole and large shears move columns past the image edge.
_SHEARS = [0.01, -0.01, 0.1227, -0.0875, 0.5, 3.0, -9.0]


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("shear", _SHEARS)
def test_vertical_shear_moves_each_column_by_its_rounded_offset(
    shear: float, background: bool
) -> None:
    height, width = 61, 33
    image = _noise(height, width)
    origin = 0.5 * width

    got = v_shear(image, shear, origin, background=background)

    for column in range(width):
        shift = _expected_shift(column, shear, origin)
        expected = np.full(height, background)
        if shift >= 0:
            expected[shift:] = image[: max(0, height - shift), column]
        else:
            expected[: max(0, height + shift)] = image[-shift:, column]
        np.testing.assert_array_equal(got[:, column], expected, err_msg=f"{column=}")


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("shear", [0.02, -0.02, 0.1227, -3.0])
def test_horizontal_shear_moves_each_row_by_its_rounded_offset(
    shear: float, background: bool
) -> None:
    height, width = 33, 61
    image = _noise(height, width)
    origin = 0.5 * height

    got = h_shear(image, shear, origin, background=background)

    for row in range(height):
        shift = _expected_shift(row, shear, origin)
        expected = np.full(width, background)
        if shift >= 0:
            expected[shift:] = image[row, : max(0, width - shift)]
        else:
            expected[: max(0, width + shift)] = image[row, -shift:]
        np.testing.assert_array_equal(got[row], expected, err_msg=f"{row=}")


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
