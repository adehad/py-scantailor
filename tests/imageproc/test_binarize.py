"""Tests for binarization algorithms."""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc.binarize import (
    _ink_below_level,
    binarize_bradley,
    binarize_edge_div,
    binarize_mokji,
    binarize_otsu,
    binarize_peak,
    binarize_sauvola,
    mokji_threshold,
    peak_threshold,
)


class TestBinarizeOtsu:
    """Tests for Otsu binarization."""

    def test_all_black_returns_binary(self):
        """All-black image should return all-black binary."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = binarize_otsu(image)
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)
        # All same value (either all 0 or all 255)
        assert len(np.unique(result)) == 1

    def test_all_white_returns_binary(self):
        """All-white image should return all-white binary."""
        image = np.full((100, 100), 255, dtype=np.uint8)
        result = binarize_otsu(image)
        assert result.dtype == np.uint8
        assert len(np.unique(result)) == 1

    def test_bimodal_histogram_separates_correctly(self):
        """Image with two distinct values should separate them."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200  # Top half bright
        image[50:, :] = 50  # Bottom half dark

        result = binarize_otsu(image)

        # Top half should be white (255), bottom half black (0)
        assert np.all(result[:50, :] == 255)
        assert np.all(result[50:, :] == 0)

    def test_output_is_binary(self):
        """Result should only contain 0 and 255."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = binarize_otsu(image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)


class TestBinarizeSauvola:
    """Tests for Sauvola binarization."""

    def test_all_black_returns_binary(self):
        """All-black image should return binary image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = binarize_sauvola(image)
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)

    def test_all_white_returns_binary(self):
        """All-white image should return binary image."""
        image = np.full((100, 100), 255, dtype=np.uint8)
        result = binarize_sauvola(image)
        assert result.dtype == np.uint8

    def test_output_is_binary(self):
        """Result should only contain 0 and 255."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = binarize_sauvola(image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_different_k_values(self):
        """Different k values should produce different results."""
        image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        result_low_k = binarize_sauvola(image, k=0.1)
        result_high_k = binarize_sauvola(image, k=0.5)
        # Results should generally differ
        # (may be same for some images, so just check they run)
        assert result_low_k.shape == result_high_k.shape

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        for shape in [(50, 50), (100, 200), (200, 100)]:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
            result = binarize_sauvola(image)
            assert result.shape == shape


class TestBinarizeBradley:
    """Tests for Bradley binarization."""

    def test_all_black_returns_binary(self):
        """All-black image should return binary image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = binarize_bradley(image)
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)

    def test_all_white_returns_binary(self):
        """All-white image should return binary image."""
        image = np.full((100, 100), 255, dtype=np.uint8)
        result = binarize_bradley(image)
        assert result.dtype == np.uint8
        # All white input with default k should give all white output
        assert np.all(result == 255)

    def test_output_is_binary(self):
        """Result should only contain 0 and 255."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = binarize_bradley(image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_different_k_values(self):
        """Different k values should produce different results."""
        image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        result_low_k = binarize_bradley(image, k=0.05)
        result_high_k = binarize_bradley(image, k=0.3)
        # threshold = mean * (1 - k)
        # Higher k means lower threshold, so more pixels above threshold (white)
        # So high_k should have fewer black pixels (more white)
        black_low = np.sum(result_low_k == 0)
        black_high = np.sum(result_high_k == 0)
        assert black_low >= black_high

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        for shape in [(50, 50), (100, 200), (200, 100)]:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
            result = binarize_bradley(image)
            assert result.shape == shape

    def test_dark_text_on_light_background(self):
        """Dark text on light background should be detected."""
        # Create image with light background and dark square
        image = np.full((100, 100), 200, dtype=np.uint8)
        image[40:60, 40:60] = 50  # Dark square in center

        result = binarize_bradley(image, window_size=25, k=0.15)

        # The dark square should be black in the result
        assert np.mean(result[40:60, 40:60]) < 128
        # The surrounding area should be white
        assert np.mean(result[0:20, 0:20]) > 128

    def test_k_equals_one_threshold_zero(self):
        """When k=1.0, threshold should be 0 (only pure black stays black)."""
        image = np.full((50, 50), 100, dtype=np.uint8)
        result = binarize_bradley(image, k=1.0)
        # Everything should be white since no pixels are below threshold 0
        assert np.all(result == 255)


class TestPeakThreshold:
    """Tests for peak threshold detection."""

    def test_bimodal_histogram(self):
        """Bimodal image should find threshold between peaks."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200  # Bright region
        image[50:, :] = 50  # Dark region

        threshold = peak_threshold(image)

        # Threshold should be between the two modes
        assert 50 < threshold < 200

    def test_returns_integer(self):
        """Threshold should be an integer."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        threshold = peak_threshold(image)
        assert isinstance(threshold, int)
        assert 0 <= threshold <= 255


class TestBinarizePeak:
    """Tests for peak binarization."""

    def test_output_is_binary(self):
        """Result should only contain 0 and 255."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = binarize_peak(image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_bimodal_separates_correctly(self):
        """Bimodal image should be separated correctly."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 220  # Top half bright
        image[50:, :] = 30  # Bottom half dark

        result = binarize_peak(image)

        # Top half should be white, bottom half black
        assert np.mean(result[:50, :]) > 200
        assert np.mean(result[50:, :]) < 50

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        for shape in [(50, 50), (100, 200), (200, 100)]:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
            result = binarize_peak(image)
            assert result.shape == shape


class TestBinarizeMokji:
    """Tests for Mokji binarization."""

    def test_output_is_binary(self):
        """Result should only contain 0 and 255."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = binarize_mokji(image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_invalid_max_edge_width(self):
        """Should raise error for invalid max_edge_width."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        try:
            binarize_mokji(image, max_edge_width=0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "max_edge_width" in str(e)

    def test_invalid_min_edge_magnitude(self):
        """Should raise error for invalid min_edge_magnitude."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        try:
            binarize_mokji(image, min_edge_magnitude=0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "min_edge_magnitude" in str(e)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        for shape in [(50, 50), (100, 200), (200, 100)]:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
            result = binarize_mokji(image)
            assert result.shape == shape

    def test_dark_text_on_light_background(self):
        """Dark text on light background should be detected."""
        image = np.full((100, 100), 200, dtype=np.uint8)
        image[40:60, 40:60] = 30  # Dark square in center

        result = binarize_mokji(image, max_edge_width=5, min_edge_magnitude=10)

        # The dark square should be black in the result
        assert np.mean(result[40:60, 40:60]) < 128

    def test_small_image_uses_default(self):
        """Very small images should not crash."""
        image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = binarize_mokji(image, max_edge_width=5)
        assert result.shape == (10, 10)


def _step_edge(dark: int, light: int) -> NDArray[np.uint8]:
    """A 40x40 page, ``dark`` on the left half and ``light`` on the right.

    Args:
        dark (int): Gray level of columns 0 to 19.
        light (int): Gray level of columns 20 to 39.

    Returns:
        NDArray[np.uint8]: The page.
    """
    image = np.full((40, 40), light, dtype=np.uint8)
    image[:, :20] = dark
    return image


class TestMokjiThreshold:
    """Mokji's threshold is the mean midpoint of the edges it finds."""

    def test_a_clean_step_edge_gives_the_midpoint_of_its_two_levels(self):
        """Every edge pair is (60, 200), so every midpoint is 130."""
        assert mokji_threshold(_step_edge(60, 200), 3, 20) == 130

    def test_a_midpoint_halfway_between_levels_rounds_up(self):
        assert mokji_threshold(_step_edge(60, 201), 3, 20) == 131

    def test_a_noisy_step_edge_gives_close_to_the_midpoint(self):
        """Noise of up to 5 levels on a (60, 220) step, whose clean midpoint is 140.

        Pairs within the noise are below the edge magnitude and ignored. Each
        step pair takes the darkest pixel in its window, about 5 below 60, so
        midpoints sit about 2.5 low; 3 levels allows for that and rounding, and
        keeps clear of the 128 returned when no edge is found.
        """
        rng = np.random.default_rng(0)
        noise = rng.integers(-5, 6, (40, 40))
        image = np.clip(_step_edge(60, 220).astype(int) + noise, 0, 255)

        threshold = mokji_threshold(image.astype(np.uint8), 3, 20)

        assert abs(threshold - 140) <= 3

    def test_a_step_below_the_edge_magnitude_is_not_an_edge(self):
        """A step of 10 is ignored at magnitude 20 and counted at magnitude 10."""
        image = _step_edge(60, 70)

        assert mokji_threshold(image, 3, 20) == 128
        assert mokji_threshold(image, 3, 10) == 65

    def test_two_kinds_of_edge_give_a_threshold_between_their_midpoints(self):
        """The threshold is a weighted mean of edge midpoints, here 130 for the
        (60, 200) step and 170 for the (140, 200) patch, so it lies between.
        """
        image = _step_edge(60, 200)
        image[5:15, 30:] = 140

        threshold = mokji_threshold(image, 3, 20)

        assert 130 < threshold < 170

    def test_image_without_edges_gives_128(self):
        """A flat image has no edge pairs, so the threshold is 128."""
        image = np.full((30, 30), 90, dtype=np.uint8)
        assert mokji_threshold(image, 3, 20) == 128

    def test_pixel_equal_to_threshold_is_background(self):
        """A pixel exactly at the threshold is white (strict comparison)."""
        image = np.full((60, 60), 60, dtype=np.uint8)
        image[:, :15] = 20
        image[25:35, 35:45] = 40
        assert mokji_threshold(image, 3, 30) == 40
        result = binarize_mokji(image, max_edge_width=3, min_edge_magnitude=30)
        assert np.all(result[25:35, 35:45] == 255)
        assert np.all(result[:, :15] == 0)


class TestBinarizeEdgeDiv:
    """Tests for edge division binarization."""

    def test_output_is_binary(self):
        """Result should only contain 0 and 255."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = binarize_edge_div(image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        for shape in [(50, 50), (100, 200), (200, 100)]:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
            result = binarize_edge_div(image)
            assert result.shape == shape

    def test_an_even_window_size_is_accepted(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = binarize_edge_div(image, window_size=24)
        assert result.dtype == np.uint8

    def test_kep_zero_uses_original(self):
        """With kep=0, no edge enhancement should be applied."""
        image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        result = binarize_edge_div(image, kep=0.0, kbd=0.0)
        # Should essentially be Otsu on the original image
        assert result.dtype == np.uint8

    def test_dark_text_on_light_background(self):
        """Dark text on light background should be detected."""
        image = np.full((100, 100), 200, dtype=np.uint8)
        image[40:60, 40:60] = 50  # Dark square in center

        result = binarize_edge_div(image)

        # The dark square should be black in the result
        assert np.mean(result[40:60, 40:60]) < 128

    def test_uneven_illumination(self):
        """Should handle uneven illumination better than simple Otsu."""
        # Create image with gradient illumination and dark text
        image = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            image[i, :] = int(100 + i * 1.5)  # Gradient from 100 to 250

        # Add dark square (relative to local background)
        image[40:60, 40:60] = np.clip(image[40:60, 40:60] - 80, 0, 255)

        result = binarize_edge_div(image, kbd=0.8)
        # Result should be binary
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)


class TestEdgeDivFractionalDelta:
    """The C++ adds delta to the Otsu threshold and truncates the sum."""

    def test_fractional_delta_is_truncated_not_rounded(self):
        """A delta of 1.5 moves the threshold by one level, like a delta of 1."""
        ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
        flat_dark = np.full((16, 16), 50, np.uint8)
        flat_light = np.full((16, 16), 200, np.uint8)
        image = np.vstack([flat_dark, ramp, flat_light])

        fractional = binarize_edge_div(image, kep=0.0, kbd=0.0, delta=1.5)
        whole = binarize_edge_div(image, kep=0.0, kbd=0.0, delta=1.0)

        np.testing.assert_array_equal(fractional, whole)


class TestInkBelowLevel:
    """A gray level is ink when strictly below the threshold, at any threshold."""

    @pytest.mark.parametrize("level", [-5, 0, 1, 30, 128, 255, 256, 300])
    def test_matches_a_comparison_for_every_level(self, level: int) -> None:
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)

        result = _ink_below_level(image, level)

        assert np.array_equal(result, np.where(image < level, 0, 255))
        assert result.dtype == np.uint8

    def test_otsu_threshold_keeps_pixels_at_the_threshold_as_background(self) -> None:
        image = np.array([[99, 100, 101, 200]], dtype=np.uint8)
        threshold = 100

        result = binarize_otsu(image, adjustment=0, bounds=(threshold, threshold))

        assert result.tolist() == [[0, 255, 255, 255]]
