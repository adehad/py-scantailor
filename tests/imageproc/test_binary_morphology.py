"""Opening by rectangles and seed fill on black-and-white images."""

import cv2
import numpy as np
import pytest

from scantailor.imageproc.binary_morphology import open_brick, seed_fill


def _blobs(height: int, width: int, seed: int) -> np.ndarray:
    """Random rectangles, some reaching the border."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=bool)
    for _ in range(25):
        y0 = int(rng.integers(-5, height))
        x0 = int(rng.integers(-5, width))
        image[
            max(0, y0) : y0 + int(rng.integers(1, 25)),
            max(0, x0) : x0 + int(rng.integers(1, 30)),
        ] = True
    return image


class TestOpenBrick:
    @pytest.mark.parametrize("black_surroundings", [False, True])
    @pytest.mark.parametrize(
        ("width", "height"), [(4, 1), (1, 4), (7, 3), (20, 6), (6, 20), (3, 3), (1, 1)]
    )
    def test_opening_never_adds_pixels_and_a_second_opening_changes_nothing(
        self, width: int, height: int, black_surroundings: bool
    ) -> None:
        image = _blobs(70, 90, seed=width * 31 + height)

        once = open_brick(image, width, height, black_surroundings=black_surroundings)
        twice = open_brick(once, width, height, black_surroundings=black_surroundings)

        assert not (once & ~image).any()
        np.testing.assert_array_equal(twice, once)

    @pytest.mark.parametrize("black_surroundings", [False, True])
    def test_every_pixel_kept_lies_in_a_brick_that_fits_the_black(
        self, black_surroundings: bool
    ) -> None:
        width, height = 7, 3
        image = _blobs(70, 90, seed=5)
        padded = np.pad(
            image,
            ((height, height), (width, width)),
            constant_values=black_surroundings,
        )
        fits = np.zeros_like(padded)
        for y in range(padded.shape[0] - height + 1):
            for x in range(padded.shape[1] - width + 1):
                if padded[y : y + height, x : x + width].all():
                    fits[y : y + height, x : x + width] = True
        expected = fits[height:-height, width:-width] & image

        got = open_brick(image, width, height, black_surroundings=black_surroundings)

        np.testing.assert_array_equal(got, expected)

    def test_a_brick_of_one_pixel_keeps_the_image(self) -> None:
        image = _blobs(30, 40, seed=2)

        np.testing.assert_array_equal(open_brick(image, 1, 1), image)

    def test_a_bar_shorter_than_the_brick_disappears(self) -> None:
        image = np.zeros((5, 20), dtype=bool)
        image[2, 3:6] = True
        image[2, 10:15] = True

        got = open_brick(image, 4, 1)

        assert np.flatnonzero(got[2]).tolist() == [10, 11, 12, 13, 14]

    def test_a_bar_at_the_border_survives_only_with_black_surroundings(self) -> None:
        image = np.zeros((5, 20), dtype=bool)
        image[2, 0:3] = True

        assert not open_brick(image, 4, 1).any()
        assert open_brick(image, 4, 1, black_surroundings=True)[2, 0:3].all()

    def test_does_not_modify_its_input(self) -> None:
        image = _blobs(30, 30, seed=1)
        before = image.copy()

        open_brick(image, 5, 5)

        np.testing.assert_array_equal(image, before)


class TestSeedFill:
    @pytest.mark.parametrize("seed", range(4))
    def test_fills_whole_components_that_hold_a_seed_and_nothing_else(
        self, seed: int
    ) -> None:
        mask = _blobs(60, 70, seed=seed)
        rng = np.random.default_rng(seed)
        seeds = rng.random(mask.shape) < 0.01

        got = seed_fill(seeds, mask)

        assert not (got & ~mask).any()
        assert (got[seeds & mask]).all()
        _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for label in np.unique(labels[mask]):
            component = labels == label
            assert got[component].all() == (seeds & component).any()
            assert got[component].any() == got[component].all()

    def test_a_diagonal_only_link_joins_two_blocks(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:3, 0:3] = True
        mask[3:6, 3:6] = True
        seeds = np.zeros_like(mask)
        seeds[0, 0] = True

        got = seed_fill(seeds, mask)

        np.testing.assert_array_equal(got, mask)

    def test_keeps_whole_components_and_joins_diagonal_neighbours(self) -> None:
        mask = np.zeros((4, 6), dtype=bool)
        mask[0, 0] = mask[1, 1] = mask[2, 2] = True
        mask[3, 5] = True
        seeds = np.zeros_like(mask)
        seeds[2, 2] = True

        got = seed_fill(seeds, mask)

        assert got.sum() == 3
        assert not got[3, 5]

    def test_a_seed_outside_the_mask_grows_nothing(self) -> None:
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        seeds = np.zeros_like(mask)
        seeds[2, 2] = True

        assert not seed_fill(seeds, mask).any()


def test_seed_fill_of_an_empty_image_is_empty():
    empty = np.zeros((0, 4), dtype=bool)

    assert seed_fill(empty, empty).shape == (0, 4)
