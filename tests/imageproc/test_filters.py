"""Tests for the Wiener, Savitzky-Golay and gray-range filters."""

import numpy as np
import pytest

from scantailor.imageproc.filters import (
    savgol_filter_2d,
    stretch_gray_range,
    wiener_color_filter,
    wiener_filter,
)


def _noisy(shape=(60, 80), low=100, high=160, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(low, high, size=shape, dtype=np.uint8)


class TestWienerFilter:
    """Adaptive noise reduction with a window clipped at the border."""

    def test_zero_noise_sigma_is_the_identity(self):
        image = _noisy()

        assert np.array_equal(wiener_filter(image, 5, 0.0), image)

    def test_flat_image_is_unchanged(self):
        image = np.full((20, 30), 137, dtype=np.uint8)

        assert np.array_equal(wiener_filter(image, 5, 60.0), image)

    def test_noise_in_a_flat_area_is_reduced(self):
        image = _noisy()

        result = wiener_filter(image, 5, 40.0)

        assert result.std() < image.std() * 0.5

    def test_strong_edge_keeps_most_of_its_contrast(self):
        image = np.zeros((40, 40), dtype=np.uint8)
        image[:, 20:] = 200

        result = wiener_filter(image, 5, 10.0)

        assert int(result[:, 25].mean()) - int(result[:, 14].mean()) > 150

    def test_window_may_exceed_the_image(self):
        image = _noisy((3, 4))

        result = wiener_filter(image, (9, 9), 30.0)

        assert result.shape == image.shape

    def test_input_is_not_modified(self):
        image = _noisy()
        before = image.copy()

        wiener_filter(image, 5, 40.0)

        assert np.array_equal(image, before)

    def test_negative_sigma_is_rejected(self):
        with pytest.raises(ValueError, match="noise_sigma"):
            wiener_filter(_noisy(), 5, -1.0)

    def test_empty_window_is_rejected(self):
        with pytest.raises(ValueError, match="window"):
            wiener_filter(_noisy(), 0, 1.0)

    def test_non_square_window_is_accepted(self):
        image = _noisy()

        result = wiener_filter(image, (7, 3), 40.0)

        assert result.std() < image.std()


class TestWienerColorFilter:
    """Brightness denoising that keeps hues."""

    def test_coefficient_zero_is_the_identity_for_gray(self):
        image = _noisy()

        result = wiener_color_filter(image, 5, 0.0)

        assert np.array_equal(result, image)
        assert result is not image

    def test_coefficient_zero_is_the_identity_for_colour(self):
        image = np.random.default_rng(1).integers(0, 256, (30, 40, 3), dtype=np.uint8)

        assert np.array_equal(wiener_color_filter(image, 5, 0.0), image)

    def test_positive_coefficient_reduces_noise_of_a_gray_image(self):
        image = _noisy()

        result = wiener_color_filter(image, 5, 0.5)

        assert result.std() < image.std()

    def test_colour_image_keeps_its_hue_direction(self):
        base = _noisy((50, 50), 100, 160, seed=4).astype(np.float64)
        image = np.stack([base * 0.6, base * 0.8, base], axis=2).astype(np.uint8)

        result = wiener_color_filter(image, 5, 0.5)

        assert result.std() < image.std()
        red_over_blue = result[..., 2].mean() / result[..., 0].mean()
        assert red_over_blue == pytest.approx(1 / 0.6, rel=0.05)

    def test_alpha_channel_passes_through(self):
        image = np.random.default_rng(2).integers(0, 256, (20, 20, 4), dtype=np.uint8)

        result = wiener_color_filter(image, 5, 0.6)

        assert np.array_equal(result[..., 3], image[..., 3])

    def test_input_is_not_modified(self):
        image = np.random.default_rng(3).integers(0, 256, (20, 20, 3), dtype=np.uint8)
        before = image.copy()

        wiener_color_filter(image, 5, 0.6)

        assert np.array_equal(image, before)


def _staircase_edge_positions(image: np.ndarray) -> np.ndarray:
    """Sub-pixel column where each row of a dark-left image crosses mid-gray."""
    positions = []
    for row in image.astype(float):
        crossing = int(np.argmax(row >= 127.5))
        before, after = row[crossing - 1], row[crossing]
        positions.append(crossing - 1 + (127.5 - before) / (after - before))
    return np.array(positions)


class TestSavgolFilter:
    """Local polynomial smoothing."""

    def _staircase(self) -> np.ndarray:
        rows = np.arange(80)
        edge = 30 + rows // 4
        columns = np.arange(100)
        return np.where(columns[None, :] < edge[:, None], 0, 255).astype(np.uint8)

    def test_staircase_edge_gets_smoother_without_moving(self):
        image = self._staircase()
        ideal = 30 + (np.arange(80) + 0.5) / 4 - 0.5

        smooth = savgol_filter_2d(image, 7, 2, 2)

        before = _staircase_edge_positions(image)[10:70]
        after = _staircase_edge_positions(smooth)[10:70]
        assert np.abs(after - ideal[10:70]).std() < np.abs(before - ideal[10:70]).std()
        assert abs(after.mean() - before.mean()) < 0.5

    def test_polynomial_surface_within_the_degree_is_reproduced_everywhere(self):
        y, x = np.mgrid[0:50, 0:60].astype(float)
        surface = 60 + 0.8 * x + 0.5 * y + 0.01 * x * y + 0.005 * x * x + 0.004 * y * y
        image = np.round(surface).astype(np.uint8)

        result = savgol_filter_2d(image, 7, 2, 2)

        assert np.abs(result.astype(int) - image).max() <= 1
        assert np.abs(result[:3].astype(int) - image[:3]).max() <= 1
        assert np.abs(result[:, -3:].astype(int) - image[:, -3:]).max() <= 1

    def test_constant_image_stays_constant(self):
        image = np.full((30, 30), 99, dtype=np.uint8)

        assert np.array_equal(savgol_filter_2d(image, 5, 3, 3), image)

    def test_isolated_noise_pixel_is_flattened(self):
        image = np.full((30, 30), 120, dtype=np.uint8)
        image[15, 15] = 255

        result = savgol_filter_2d(image, 7, 2, 2)

        assert result[15, 15] < 200

    def test_image_smaller_than_the_window_is_returned_unchanged(self):
        image = _noisy((4, 40))

        result = savgol_filter_2d(image, 5, 2, 2)

        assert np.array_equal(result, image)
        assert result is not image

    def test_input_is_not_modified(self):
        image = _noisy()
        before = image.copy()

        savgol_filter_2d(image, 5, 2, 2)

        assert np.array_equal(image, before)

    @pytest.mark.parametrize(("hor", "vert"), [(-1, 2), (2, -1), (5, 2), (2, 5)])
    def test_invalid_degree_is_rejected(self, hor, vert):
        with pytest.raises(ValueError, match="degree"):
            savgol_filter_2d(_noisy(), 5, hor, vert)

    def test_horizontal_and_vertical_degrees_apply_to_their_own_axes(self):
        # Linear along x, constant along y: only a horizontal degree of 1 keeps x.
        columns = np.linspace(40, 200, 60)
        image = np.tile(columns, (50, 1)).round().astype(np.uint8)

        result = savgol_filter_2d(image, (7, 5), 1, 0)

        assert np.abs(result.astype(int) - image).max() <= 1

    def test_non_square_window_matches_square_on_a_symmetric_image(self):
        image = _noisy((40, 40))

        square = savgol_filter_2d(image, 7, 2, 2)
        pair = savgol_filter_2d(image, (7, 7), 2, 2)

        assert np.array_equal(square, pair)


class TestStretchGrayRange:
    """Stretching the used gray range to the full range."""

    def test_pale_image_spans_the_full_range(self):
        image = np.tile(np.linspace(100, 150, 200).astype(np.uint8), (20, 1))

        result = stretch_gray_range(image, 0.0, 0.0)

        assert result.min() == 0
        assert result.max() == 255

    def test_clipping_fraction_saturates_the_tail(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)

        result = stretch_gray_range(image, 0.1, 0.1)

        assert (result[image < 25] == 0).all()
        assert (result[image > 230] == 255).all()

    def test_flat_image_becomes_black_or_white(self):
        dark = np.full((5, 5), 30, dtype=np.uint8)
        light = np.full((5, 5), 220, dtype=np.uint8)

        assert set(np.unique(stretch_gray_range(dark, 0.0, 0.0))) <= {0, 255}
        assert set(np.unique(stretch_gray_range(light, 0.0, 0.0))) <= {0, 255}

    def test_absolute_levels_are_used_at_one_and_above(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)

        result = stretch_gray_range(image, 50.0, 200.0)

        assert (result[image <= 50] == 0).all()
        assert (result[image >= 200] == 255).all()

    def test_input_is_not_modified(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        before = image.copy()

        stretch_gray_range(image, 0.01, 0.01)

        assert np.array_equal(image, before)
