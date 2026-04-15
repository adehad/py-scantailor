"""Tests for DespeckleLevel enum."""

from scantailor.filters.output.despeckle import DespeckleLevel


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

    def test_from_string_unknown_defaults_to_off(self):
        """Unknown string defaults to off."""
        assert DespeckleLevel.from_string("unknown") == DespeckleLevel.OFF

    def test_str_representation(self):
        """String representation returns value."""
        assert str(DespeckleLevel.NORMAL) == "normal"

    def test_to_component_size_off(self):
        """Off returns 0 component size."""
        assert DespeckleLevel.OFF.to_component_size() == 0

    def test_to_component_size_cautious(self):
        """Cautious returns small component size."""
        assert DespeckleLevel.CAUTIOUS.to_component_size() == 10

    def test_to_component_size_normal(self):
        """Normal returns medium component size."""
        assert DespeckleLevel.NORMAL.to_component_size() == 20

    def test_to_component_size_aggressive(self):
        """Aggressive returns large component size."""
        assert DespeckleLevel.AGGRESSIVE.to_component_size() == 40

    def test_is_enabled(self):
        """is_enabled returns correct values."""
        assert DespeckleLevel.OFF.is_enabled() is False
        assert DespeckleLevel.CAUTIOUS.is_enabled() is True
        assert DespeckleLevel.NORMAL.is_enabled() is True
        assert DespeckleLevel.AGGRESSIVE.is_enabled() is True
