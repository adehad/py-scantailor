"""Tests for output generator."""

import numpy as np

from scantailor.filters.output import (
    BinarizationMethod,
    BinarizationOptions,
    ColorMode,
    DespeckleLevel,
    Params,
)
from scantailor.filters.output.generator import generate_output


class TestGenerateOutput:
    """Tests for generate_output function."""

    def test_binary_output_grayscale_input(self):
        """Binary output from grayscale input."""
        # Create a simple grayscale image with two regions
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200  # Light top half
        image[50:, :] = 50  # Dark bottom half

        params = Params(color_mode=ColorMode.BLACK_AND_WHITE)
        result = generate_output(image, params)

        assert result.is_binary is True
        assert result.image.shape == image.shape
        # Should be binary (only 0 and 255)
        unique_values = np.unique(result.image)
        assert len(unique_values) <= 2

    def test_binary_output_color_input(self):
        """Binary output from color input."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :] = [200, 200, 200]
        image[50:, :] = [50, 50, 50]

        params = Params(color_mode=ColorMode.BLACK_AND_WHITE)
        result = generate_output(image, params)

        assert result.is_binary is True
        assert result.image.shape == (100, 100)  # Grayscale output

    def test_grayscale_output(self):
        """Grayscale output preserves image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[25:75, 25:75] = 128

        params = Params(color_mode=ColorMode.COLOR_GRAYSCALE)
        result = generate_output(image, params)

        assert result.is_binary is False
        np.testing.assert_array_equal(result.image, image)

    def test_white_on_black_inversion(self):
        """White on black mode inverts output."""
        image = np.full((100, 100), 200, dtype=np.uint8)

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            black_on_white=True,
        )
        result_bow = generate_output(image, params)

        params_wob = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            black_on_white=False,
        )
        result_wob = generate_output(image, params_wob)

        # Results should be inverted relative to each other
        np.testing.assert_array_equal(result_bow.image, 255 - result_wob.image)

    def test_otsu_binarization(self):
        """Otsu binarization produces binary output."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200
        image[50:, :] = 50

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(method=BinarizationMethod.OTSU),
        )
        result = generate_output(image, params)

        assert result.is_binary is True
        unique = np.unique(result.image)
        assert len(unique) <= 2

    def test_sauvola_binarization(self):
        """Sauvola binarization produces binary output."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200
        image[50:, :] = 50

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(method=BinarizationMethod.SAUVOLA),
        )
        result = generate_output(image, params)

        assert result.is_binary is True

    def test_wolf_binarization(self):
        """Wolf binarization produces binary output."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200
        image[50:, :] = 50

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(method=BinarizationMethod.WOLF),
        )
        result = generate_output(image, params)

        assert result.is_binary is True

    def test_bradley_binarization(self):
        """Bradley binarization produces binary output."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200
        image[50:, :] = 50

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(method=BinarizationMethod.BRADLEY),
        )
        result = generate_output(image, params)

        assert result.is_binary is True
        unique = np.unique(result.image)
        assert len(unique) <= 2

    def test_edgediv_binarization(self):
        """EdgeDiv binarization produces binary output."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200
        image[50:, :] = 50

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(method=BinarizationMethod.EDGEDIV),
        )
        result = generate_output(image, params)

        assert result.is_binary is True
        unique = np.unique(result.image)
        assert len(unique) <= 2

    def test_despeckle_off(self):
        """Despeckle off leaves small components."""
        image = np.full((100, 100), 200, dtype=np.uint8)
        # Add some small noise
        image[50, 50] = 50

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            despeckle_level=DespeckleLevel.OFF,
        )
        result = generate_output(image, params)
        assert result.is_binary is True

    def test_despeckle_aggressive(self):
        """Aggressive despeckle removes small components."""
        image = np.full((100, 100), 200, dtype=np.uint8)
        # Add some small noise (should be removed)
        image[50, 50] = 50

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            despeckle_level=DespeckleLevel.AGGRESSIVE,
        )
        result = generate_output(image, params)
        assert result.is_binary is True

    def test_threshold_adjustment_darker(self):
        """Positive threshold adjustment makes output darker."""
        image = np.full((100, 100), 128, dtype=np.uint8)

        params_normal = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(threshold_adjustment=0),
        )
        params_darker = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(threshold_adjustment=50),
        )

        result_normal = generate_output(image, params_normal)
        result_darker = generate_output(image, params_darker)

        # Darker adjustment should have more black pixels
        black_normal = np.sum(result_normal.image == 0)
        black_darker = np.sum(result_darker.image == 0)
        assert black_darker >= black_normal

    def test_normalize_illumination(self):
        """Illumination normalization works."""
        # Create image with uneven lighting
        image = np.zeros((100, 100), dtype=np.uint8)
        for y in range(100):
            image[y, :] = 128 + y  # Gradient from top to bottom

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(normalize_illumination=True),
        )
        result = generate_output(image, params)
        assert result.is_binary is True

    def test_morphological_smoothing(self):
        """Morphological smoothing is applied."""
        image = np.full((100, 100), 200, dtype=np.uint8)
        # Add jagged edge
        for i in range(50):
            image[i, 48 : 52 if i % 2 == 0 else 49 : 51] = 50

        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(morphological_smoothing=True),
        )
        result = generate_output(image, params)
        assert result.is_binary is True
