"""Block reduction of black-and-white images."""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc.reduce_threshold import reduce_threshold


def _blocks_with_black_counts() -> NDArray[np.bool_]:
    """A 2x10 image of five 2x2 blocks holding 0, 1, 2, 3 and 4 black pixels.

    Returns:
        NDArray[np.bool_]: The image, True for black.
    """
    image = np.zeros((2, 10), dtype=bool)
    for block, cells in enumerate(
        [
            [],
            [(0, 0)],
            [(0, 1), (1, 0)],
            [(0, 0), (0, 1), (1, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        ]
    ):
        for y, x in cells:
            image[y, 2 * block + x] = True
    return image


@pytest.mark.parametrize("threshold", [1, 2, 3, 4])
def test_a_block_is_black_exactly_when_at_least_threshold_pixels_are(
    threshold: int,
) -> None:
    got = reduce_threshold(_blocks_with_black_counts(), threshold)

    assert got.tolist() == [[count >= threshold for count in range(5)]]


@pytest.mark.parametrize("threshold", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(8, 8), (9, 13), (2, 2), (3, 3), (40, 7)])
def test_a_trailing_odd_row_and_column_are_dropped(
    threshold: int, shape: tuple[int, int]
) -> None:
    image = np.random.default_rng(sum(shape)).random(shape) < 0.45
    even = image[: shape[0] // 2 * 2, : shape[1] // 2 * 2]
    flipped = image.copy()
    flipped[shape[0] // 2 * 2 :, :] ^= True
    flipped[:, shape[1] // 2 * 2 :] ^= True

    got = reduce_threshold(image, threshold)

    assert got.shape == (shape[0] // 2, shape[1] // 2)
    np.testing.assert_array_equal(got, reduce_threshold(even, threshold))
    np.testing.assert_array_equal(got, reduce_threshold(flipped, threshold))


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
