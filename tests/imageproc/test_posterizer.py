"""Tests for posterization."""

import numpy as np
import pytest

from scantailor.imageproc.posterizer import posterize


def _bands(values_and_counts: dict[int, int]) -> np.ndarray:
    """A gray image holding each value that many times."""
    flat = np.concatenate(
        [
            np.full(count, value, dtype=np.uint8)
            for value, count in values_and_counts.items()
        ]
    )
    return flat.reshape(1, -1)


def _colour_image(colours_and_counts: dict[tuple[int, int, int], int]) -> np.ndarray:
    """A BGR image holding each colour that many times."""
    flat = np.concatenate(
        [
            np.tile(np.array(c, dtype=np.uint8), (n, 1))
            for c, n in colours_and_counts.items()
        ]
    )
    return flat.reshape(1, -1, 3)


class TestPosterize:
    """Reducing an image to the most frequent colour of each RGB cell."""

    def test_most_frequent_colour_of_each_cell_replaces_the_cell(self):
        # With a range of 10..255 these stretch to 0, 31, 62, 198 and 255, so the
        # first three share a cell (cells are 63.75 wide at level 4).
        image = _bands({10: 100, 40: 300, 70: 50, 200: 200, 255: 10})

        result = posterize(image, 4, force_black_and_white=False)

        assert set(np.unique(result)) == {40, 200, 255}
        assert (result[image == 10] == 40).all()
        assert (result[image == 70] == 40).all()
        assert (result[image == 200] == 200).all()

    def test_output_never_has_more_colours_than_cells(self):
        image = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)

        result = posterize(image, 3, force_black_and_white=False)

        colours = {tuple(c) for c in result.reshape(-1, 3)}
        assert len(colours) <= (3 + 1) ** 3

    def test_every_output_colour_is_an_input_colour_without_normalisation(self):
        image = np.random.default_rng(1).integers(0, 256, (40, 40, 3), dtype=np.uint8)

        result = posterize(image, 4, force_black_and_white=False)

        input_colours = {tuple(c) for c in image.reshape(-1, 3)}
        assert {tuple(c) for c in result.reshape(-1, 3)} <= input_colours

    def test_level_255_without_options_returns_an_equal_copy(self):
        image = np.random.default_rng(2).integers(0, 256, (10, 10, 3), dtype=np.uint8)

        result = posterize(image, 255)

        assert np.array_equal(result, image)
        assert result is not image

    @pytest.mark.parametrize("level", [1, 256, 0, -3])
    def test_level_out_of_range_is_rejected(self, level):
        with pytest.raises(ValueError, match="level"):
            posterize(np.zeros((2, 2), dtype=np.uint8), level)

    def test_normalisation_stretches_a_pale_page(self):
        image = _bands({200: 500, 215: 300, 240: 200})

        result = posterize(image, 8, normalize=True, force_black_and_white=False)

        assert result.min() == 0
        assert result.max() == 255

    def test_normalisation_keeps_pure_black_and_white(self):
        image = _bands({0: 100, 255: 100, 120: 400, 180: 400})

        result = posterize(image, 8, normalize=True, force_black_and_white=False)

        assert (result[image == 0] == 0).all()
        assert (result[image == 255] == 255).all()

    def test_force_black_and_white_snaps_dark_and_light_grays(self):
        image = _bands({30: 300, 235: 500, 150: 200})

        result = posterize(image, 4, force_black_and_white=True)

        assert (result[image == 30] == 0).all()
        assert (result[image == 235] == 255).all()

    def test_force_black_and_white_keeps_saturated_colours(self):
        red = (30, 30, 200)
        image = _colour_image({red: 300, (235, 235, 235): 500, (30, 30, 30): 200})

        result = posterize(image, 4, force_black_and_white=True)

        assert tuple(result[0, 0]) == red
        assert tuple(result[0, 400]) == (255, 255, 255)
        assert tuple(result[0, -1]) == (0, 0, 0)

    def test_colour_channel_order_is_preserved(self):
        blue_dominant = (200, 40, 40)
        image = _colour_image({blue_dominant: 300, (235, 235, 235): 300})

        result = posterize(image, 255 - 1, force_black_and_white=False)

        assert tuple(result[0, 0]) == blue_dominant

    def test_gray_input_gives_gray_output(self):
        image = _bands({20: 10, 120: 10, 220: 10})

        result = posterize(image, 3)

        assert result.shape == image.shape
        assert result.ndim == 2

    def test_alpha_channel_passes_through(self):
        image = np.random.default_rng(3).integers(0, 256, (8, 8, 4), dtype=np.uint8)

        result = posterize(image, 3)

        assert np.array_equal(result[..., 3], image[..., 3])

    def test_input_is_not_modified(self):
        image = np.random.default_rng(4).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        before = image.copy()

        posterize(image, 3, normalize=True)

        assert np.array_equal(image, before)

    def test_thinly_spread_black_ink_is_not_merged_into_a_dense_grey(self):
        # Like a scanned page with a grey frame: the grey sits on one level well
        # above the noise share, while the black ink is spread over 60 levels with
        # a tenth of the noise share on each.
        black_ink = {level: 40 for level in range(20, 80)}
        image = _bands({**black_ink, 150: 5_000, 255: 1_000_000})

        result = posterize(image, 3, force_black_and_white=False)

        assert result[image < 80].max() < 80
        assert (result[image == 150] == 150).all()
        assert (result[image == 255] == 255).all()

    def test_a_few_outlier_pixels_do_not_widen_the_range(self):
        # Two stray pixels at 0 and 254 are below the noise share at either end,
        # so the 100..200 content still stretches to the full range.
        image = _bands({0: 1, 100: 5_000, 200: 5_000, 254: 1})

        result = posterize(image, 8, normalize=True, force_black_and_white=False)

        assert (result[image == 100] == 0).all()
        assert (result[image == 200] == 255).all()

    def test_result_is_deterministic_when_cells_tie(self):
        image = _bands({0: 100, 150: 5, 160: 5, 255: 100})

        result = posterize(image, 2, force_black_and_white=False)

        assert (result[(image == 150) | (image == 160)] == 150).all()
