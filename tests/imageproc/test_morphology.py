"""Tests for morphological operations."""

from __future__ import annotations

import numpy as np
import pytest

from scantailor.imageproc import close_morph, dilate, erode, open_morph


class TestDilate:
    """Tests for dilation."""

    def test_dilates_white_region(self):
        """White region should expand after dilation."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[45:55, 45:55] = 255  # 10x10 white square in center

        result = dilate(image, kernel_size=3)

        # White region should be larger
        white_before = np.sum(image == 255)
        white_after = np.sum(result == 255)
        assert white_after > white_before

    def test_all_black_stays_black(self):
        """All-black image should remain all-black."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = dilate(image)
        assert np.all(result == 0)

    def test_all_white_stays_white(self):
        """All-white image should remain all-white."""
        image = np.full((100, 100), 255, dtype=np.uint8)
        result = dilate(image)
        assert np.all(result == 255)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        image = np.zeros((100, 150), dtype=np.uint8)
        result = dilate(image)
        assert result.shape == image.shape

    def test_iterations(self):
        """More iterations should dilate more."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[49, 49] = 255  # Single white pixel

        result_1 = dilate(image, kernel_size=3, iterations=1)
        result_2 = dilate(image, kernel_size=3, iterations=2)

        white_1 = np.sum(result_1 == 255)
        white_2 = np.sum(result_2 == 255)
        assert white_2 > white_1


class TestErode:
    """Tests for erosion."""

    def test_erodes_white_region(self):
        """White region should shrink after erosion."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255  # 20x20 white square

        result = erode(image, kernel_size=3)

        white_before = np.sum(image == 255)
        white_after = np.sum(result == 255)
        assert white_after < white_before

    def test_small_region_disappears(self):
        """Small white region should disappear with large kernel."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[48:52, 48:52] = 255  # 4x4 white square

        result = erode(image, kernel_size=5)

        # Square should be mostly or entirely gone
        white_after = np.sum(result == 255)
        assert white_after < np.sum(image == 255)

    def test_all_black_stays_black(self):
        """All-black image should remain all-black."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = erode(image)
        assert np.all(result == 0)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        image = np.zeros((100, 150), dtype=np.uint8)
        result = erode(image)
        assert result.shape == image.shape


class TestOpenMorph:
    """Tests for morphological opening."""

    def test_removes_small_white_spots(self):
        """Small isolated white spots should be removed."""
        image = np.zeros((100, 100), dtype=np.uint8)
        # Large white region
        image[20:80, 20:80] = 255
        # Small white spot
        image[5:7, 5:7] = 255

        result = open_morph(image, kernel_size=5)

        # Large region should mostly remain
        assert np.sum(result[30:70, 30:70] == 255) > 0
        # Small spot should be gone or reduced
        assert np.sum(result[5:7, 5:7] == 255) < np.sum(image[5:7, 5:7] == 255)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        image = np.zeros((100, 150), dtype=np.uint8)
        result = open_morph(image)
        assert result.shape == image.shape


class TestCloseMorph:
    """Tests for morphological closing."""

    def test_fills_small_black_holes(self):
        """Small black holes in white regions should be filled."""
        image = np.full((100, 100), 255, dtype=np.uint8)
        # Small black hole
        image[48:52, 48:52] = 0

        result = close_morph(image, kernel_size=5)

        # Hole should be filled or reduced
        black_before = np.sum(image == 0)
        black_after = np.sum(result == 0)
        assert black_after < black_before

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        image = np.zeros((100, 150), dtype=np.uint8)
        result = close_morph(image)
        assert result.shape == image.shape

    def test_different_shapes(self):
        """Different structuring element shapes should work."""
        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        for shape in ["rect", "ellipse", "cross"]:
            result = close_morph(image, kernel_size=3, shape=shape)
            assert result.shape == image.shape
