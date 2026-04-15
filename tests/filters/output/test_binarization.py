"""Tests for binarization options."""

import pytest

from scantailor.filters.output.binarization import (
    BinarizationMethod,
    BinarizationOptions,
)


class TestBinarizationMethod:
    """Tests for BinarizationMethod enum."""

    def test_otsu_value(self):
        """Otsu has correct string value."""
        assert BinarizationMethod.OTSU.value == "otsu"

    def test_sauvola_value(self):
        """Sauvola has correct string value."""
        assert BinarizationMethod.SAUVOLA.value == "sauvola"

    def test_wolf_value(self):
        """Wolf has correct string value."""
        assert BinarizationMethod.WOLF.value == "wolf"

    def test_bradley_value(self):
        """Bradley has correct string value."""
        assert BinarizationMethod.BRADLEY.value == "bradley"

    def test_edgediv_value(self):
        """EdgeDiv has correct string value."""
        assert BinarizationMethod.EDGEDIV.value == "edgediv"

    def test_from_string_otsu(self):
        """Parse otsu string."""
        assert BinarizationMethod.from_string("otsu") == BinarizationMethod.OTSU

    def test_from_string_sauvola(self):
        """Parse sauvola string."""
        assert BinarizationMethod.from_string("sauvola") == BinarizationMethod.SAUVOLA

    def test_from_string_wolf(self):
        """Parse wolf string."""
        assert BinarizationMethod.from_string("wolf") == BinarizationMethod.WOLF

    def test_from_string_bradley(self):
        """Parse bradley string."""
        assert BinarizationMethod.from_string("bradley") == BinarizationMethod.BRADLEY

    def test_from_string_edgediv(self):
        """Parse edgediv string."""
        assert BinarizationMethod.from_string("edgediv") == BinarizationMethod.EDGEDIV

    def test_from_string_unknown_defaults_to_otsu(self):
        """Unknown string defaults to Otsu."""
        assert BinarizationMethod.from_string("unknown") == BinarizationMethod.OTSU


class TestBinarizationOptions:
    """Tests for BinarizationOptions class."""

    def test_default_values(self):
        """Default values are correct."""
        options = BinarizationOptions()
        assert options.method == BinarizationMethod.OTSU
        assert options.threshold_adjustment == 0
        assert options.window_size == 200
        assert options.normalize_illumination is True
        assert options.morphological_smoothing is True

    def test_with_method(self):
        """Create options with different method."""
        options = BinarizationOptions()
        new_options = options.with_method(BinarizationMethod.SAUVOLA)
        assert new_options.method == BinarizationMethod.SAUVOLA
        # Original unchanged
        assert options.method == BinarizationMethod.OTSU

    def test_with_threshold_adjustment(self):
        """Create options with different threshold."""
        options = BinarizationOptions()
        new_options = options.with_threshold_adjustment(10)
        assert new_options.threshold_adjustment == 10

    def test_with_window_size(self):
        """Create options with different window size."""
        options = BinarizationOptions()
        new_options = options.with_window_size(300)
        assert new_options.window_size == 300

    def test_threshold_adjustment_bounds(self):
        """Threshold adjustment has valid bounds."""
        # Valid values
        BinarizationOptions(threshold_adjustment=-100)
        BinarizationOptions(threshold_adjustment=100)

        # Invalid values
        with pytest.raises(ValueError):
            BinarizationOptions(threshold_adjustment=-101)
        with pytest.raises(ValueError):
            BinarizationOptions(threshold_adjustment=101)

    def test_window_size_bounds(self):
        """Window size has valid bounds."""
        BinarizationOptions(window_size=10)
        BinarizationOptions(window_size=1000)

        with pytest.raises(ValueError):
            BinarizationOptions(window_size=9)
        with pytest.raises(ValueError):
            BinarizationOptions(window_size=1001)

    def test_sauvola_coef_bounds(self):
        """Sauvola coefficient has valid bounds."""
        BinarizationOptions(sauvola_coef=0.0)
        BinarizationOptions(sauvola_coef=1.0)

        with pytest.raises(ValueError):
            BinarizationOptions(sauvola_coef=-0.1)
        with pytest.raises(ValueError):
            BinarizationOptions(sauvola_coef=1.1)

    def test_bradley_coef_bounds(self):
        """Bradley coefficient has valid bounds."""
        BinarizationOptions(bradley_coef=0.0)
        BinarizationOptions(bradley_coef=1.0)

        with pytest.raises(ValueError):
            BinarizationOptions(bradley_coef=-0.1)
        with pytest.raises(ValueError):
            BinarizationOptions(bradley_coef=1.1)

    def test_edge_div_kep_bounds(self):
        """EdgeDiv kep coefficient has valid bounds."""
        BinarizationOptions(edge_div_kep=0.0)
        BinarizationOptions(edge_div_kep=1.0)

        with pytest.raises(ValueError):
            BinarizationOptions(edge_div_kep=-0.1)
        with pytest.raises(ValueError):
            BinarizationOptions(edge_div_kep=1.1)

    def test_edge_div_kbd_bounds(self):
        """EdgeDiv kbd coefficient has valid bounds."""
        BinarizationOptions(edge_div_kbd=0.0)
        BinarizationOptions(edge_div_kbd=1.0)

        with pytest.raises(ValueError):
            BinarizationOptions(edge_div_kbd=-0.1)
        with pytest.raises(ValueError):
            BinarizationOptions(edge_div_kbd=1.1)

    def test_bradley_default_values(self):
        """Bradley options have correct defaults."""
        options = BinarizationOptions(method=BinarizationMethod.BRADLEY)
        assert options.bradley_coef == 0.34

    def test_edge_div_default_values(self):
        """EdgeDiv options have correct defaults."""
        options = BinarizationOptions(method=BinarizationMethod.EDGEDIV)
        assert options.edge_div_kep == 0.5
        assert options.edge_div_kbd == 0.5

    def test_frozen_model(self):
        """Options is immutable."""
        options = BinarizationOptions()
        with pytest.raises(Exception):
            options.method = BinarizationMethod.WOLF  # type: ignore[misc]
