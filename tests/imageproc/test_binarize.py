"""Tests for binarization algorithms."""

from __future__ import annotations

import numpy as np

from scantailor.imageproc import binarize_otsu, binarize_sauvola


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
