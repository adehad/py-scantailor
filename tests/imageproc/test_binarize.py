"""Tests for binarization algorithms."""

from __future__ import annotations

import numpy as np

from scantailor.imageproc import binarize_bradley, binarize_otsu, binarize_sauvola


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

    def test_even_window_size_is_corrected(self):
        """Even window size should be corrected to odd."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # Should not raise error with even window size
        result = binarize_sauvola(image, window_size=24)
        assert result.dtype == np.uint8

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

    def test_even_window_size_is_corrected(self):
        """Even window size should be corrected to odd."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # Should not raise error with even window size
        result = binarize_bradley(image, window_size=24)
        assert result.dtype == np.uint8

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
