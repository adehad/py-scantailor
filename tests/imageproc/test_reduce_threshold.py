"""Block reduction of black-and-white images."""

import numpy as np
import pytest

from scantailor.imageproc.reduce_threshold import reduce_threshold
from tests.filters.page_split import cpp_reference as cpp


@pytest.mark.parametrize("threshold", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(8, 8), (9, 13), (2, 2), (3, 3), (40, 7)])
def test_a_block_is_black_when_enough_of_its_pixels_are(
    threshold: int, shape: tuple[int, int]
) -> None:
    image = np.random.default_rng(sum(shape)).random(shape) < 0.45

    got = reduce_threshold(image, threshold)

    np.testing.assert_array_equal(got, cpp.reduce_threshold(image, threshold))
    assert got.shape == (shape[0] // 2, shape[1] // 2)


def test_a_single_row_reduces_along_its_length() -> None:
    row = np.array([[True, False, False, False, True, True, True]])

    assert reduce_threshold(row, 1).tolist() == [[True, False, True]]
    assert reduce_threshold(row, 4).tolist() == [[False, False, True]]


def test_a_single_column_reduces_along_its_height() -> None:
    column = np.array([[True], [False], [True], [True]])

    assert reduce_threshold(column, 2).tolist() == [[True], [True]]
    assert reduce_threshold(column, 3).tolist() == [[False], [True]]


def test_a_single_pixel_is_returned_unchanged() -> None:
    pixel = np.array([[True]])

    assert reduce_threshold(pixel, 4).tolist() == [[True]]


@pytest.mark.parametrize("threshold", [0, 5])
def test_a_threshold_outside_one_to_four_is_rejected(threshold: int) -> None:
    with pytest.raises(ValueError, match="threshold"):
        reduce_threshold(np.zeros((4, 4), dtype=bool), threshold)
