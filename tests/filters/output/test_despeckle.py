"""Tests for DespeckleLevel enum."""

import pytest

from scantailor.filters.output.despeckle import DespeckleLevel, DespeckleStrength


class TestDespeckleLevel:
    """Tests for DespeckleLevel enum."""

    def test_off_value(self):
        """Off has correct string value."""
        assert DespeckleLevel.OFF.value == "off"

    def test_cautious_value(self):
        """Cautious has correct string value."""
        assert DespeckleLevel.CAUTIOUS.value == "cautious"

    def test_normal_value(self):
        """Normal has correct string value."""
        assert DespeckleLevel.NORMAL.value == "normal"

    def test_aggressive_value(self):
        """Aggressive has correct string value."""
        assert DespeckleLevel.AGGRESSIVE.value == "aggressive"

    def test_from_string_off(self):
        """Parse off string."""
        assert DespeckleLevel.from_string("off") == DespeckleLevel.OFF

    def test_from_string_cautious(self):
        """Parse cautious string."""
        assert DespeckleLevel.from_string("cautious") == DespeckleLevel.CAUTIOUS

    def test_from_string_normal(self):
        """Parse normal string."""
        assert DespeckleLevel.from_string("normal") == DespeckleLevel.NORMAL

    def test_from_string_aggressive(self):
        """Parse aggressive string."""
        assert DespeckleLevel.from_string("aggressive") == DespeckleLevel.AGGRESSIVE

    def test_from_string_unknown_raises_instead_of_turning_despeckle_off(self):
        """An unknown name is an error, not a silent Off."""
        with pytest.raises(ValueError, match="unknown despeckle level 'unknown'"):
            DespeckleLevel.from_string("unknown")

    @pytest.mark.parametrize(
        ("factor", "level"),
        [
            (0.0, DespeckleLevel.OFF),
            (0.49, DespeckleLevel.OFF),
            (0.5, DespeckleLevel.CAUTIOUS),
            (1.49, DespeckleLevel.CAUTIOUS),
            (1.5, DespeckleLevel.NORMAL),
            (2.49, DespeckleLevel.NORMAL),
            (2.5, DespeckleLevel.AGGRESSIVE),
            (3.5, DespeckleLevel.AGGRESSIVE),
        ],
    )
    def test_nearest_picks_the_level_by_the_midpoints_between_presets(
        self, factor, level
    ):
        assert DespeckleLevel.nearest(factor) == level


class TestDespeckleStrength:
    def test_it_answers_the_generator_as_a_level_does(self):
        strength = DespeckleStrength(1.7)

        assert strength.to_factor() == 1.7
        assert strength.is_enabled()

    def test_zero_is_not_enabled(self):
        assert not DespeckleStrength(0.0).is_enabled()

    def test_str_representation(self):
        """String representation returns value."""
        assert str(DespeckleLevel.NORMAL) == "normal"

    def test_to_factor_maps_levels_to_cpp_factors(self):
        """Levels map to the factors 0, 1, 2 and 3."""
        assert DespeckleLevel.OFF.to_factor() == 0.0
        assert DespeckleLevel.CAUTIOUS.to_factor() == 1.0
        assert DespeckleLevel.NORMAL.to_factor() == 2.0
        assert DespeckleLevel.AGGRESSIVE.to_factor() == 3.0

    def test_is_enabled(self):
        """is_enabled returns correct values."""
        assert DespeckleLevel.OFF.is_enabled() is False
        assert DespeckleLevel.CAUTIOUS.is_enabled() is True
        assert DespeckleLevel.NORMAL.is_enabled() is True
        assert DespeckleLevel.AGGRESSIVE.is_enabled() is True
