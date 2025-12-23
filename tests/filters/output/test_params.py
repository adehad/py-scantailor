"""Tests for output Params class."""

import pytest

from scantailor.core import Dpi
from scantailor.filters.output import (
    BinarizationMethod,
    BinarizationOptions,
    ColorMode,
    DespeckleLevel,
    Params,
)


class TestParams:
    """Tests for Params class."""

    def test_default_params(self):
        """Default params have expected values."""
        params = Params()
        assert params.output_dpi == Dpi.uniform(600)
        assert params.color_mode == ColorMode.BLACK_AND_WHITE
        assert params.despeckle_level == DespeckleLevel.NORMAL
        assert params.black_on_white is True

    def test_with_output_dpi(self):
        """Create params with different DPI."""
        params = Params()
        new_params = params.with_output_dpi(Dpi.uniform(300))
        assert new_params.output_dpi == Dpi.uniform(300)
        # Original unchanged
        assert params.output_dpi == Dpi.uniform(600)

    def test_with_color_mode(self):
        """Create params with different color mode."""
        params = Params()
        new_params = params.with_color_mode(ColorMode.COLOR_GRAYSCALE)
        assert new_params.color_mode == ColorMode.COLOR_GRAYSCALE

    def test_with_binarization(self):
        """Create params with different binarization options."""
        params = Params()
        options = BinarizationOptions(method=BinarizationMethod.SAUVOLA)
        new_params = params.with_binarization(options)
        assert new_params.binarization.method == BinarizationMethod.SAUVOLA

    def test_with_binarization_method(self):
        """Create params with different binarization method."""
        params = Params()
        new_params = params.with_binarization_method(BinarizationMethod.WOLF)
        assert new_params.binarization.method == BinarizationMethod.WOLF

    def test_with_despeckle_level(self):
        """Create params with different despeckle level."""
        params = Params()
        new_params = params.with_despeckle_level(DespeckleLevel.AGGRESSIVE)
        assert new_params.despeckle_level == DespeckleLevel.AGGRESSIVE

    def test_with_black_on_white(self):
        """Create params with different black_on_white."""
        params = Params()
        new_params = params.with_black_on_white(False)
        assert new_params.black_on_white is False

    def test_needs_binarization_bw(self):
        """Black and white mode needs binarization."""
        params = Params(color_mode=ColorMode.BLACK_AND_WHITE)
        assert params.needs_binarization() is True

    def test_needs_binarization_grayscale(self):
        """Grayscale mode doesn't need binarization."""
        params = Params(color_mode=ColorMode.COLOR_GRAYSCALE)
        assert params.needs_binarization() is False

    def test_needs_binarization_mixed(self):
        """Mixed mode needs binarization."""
        params = Params(color_mode=ColorMode.MIXED)
        assert params.needs_binarization() is True

    def test_frozen_model(self):
        """Params is immutable."""
        params = Params()
        with pytest.raises(Exception):
            params.color_mode = ColorMode.MIXED  # type: ignore[misc]
