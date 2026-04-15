"""Tests for ColorMode enum."""

from scantailor.filters.output.color_mode import ColorMode


class TestColorMode:
    """Tests for ColorMode enum."""

    def test_black_and_white_value(self):
        """Black and white has correct string value."""
        assert ColorMode.BLACK_AND_WHITE.value == "bw"

    def test_color_grayscale_value(self):
        """Color/grayscale has correct string value."""
        assert ColorMode.COLOR_GRAYSCALE.value == "colorOrGray"

    def test_mixed_value(self):
        """Mixed has correct string value."""
        assert ColorMode.MIXED.value == "mixed"

    def test_from_string_bw(self):
        """Parse bw string."""
        assert ColorMode.from_string("bw") == ColorMode.BLACK_AND_WHITE

    def test_from_string_color(self):
        """Parse colorOrGray string."""
        assert ColorMode.from_string("colorOrGray") == ColorMode.COLOR_GRAYSCALE

    def test_from_string_mixed(self):
        """Parse mixed string."""
        assert ColorMode.from_string("mixed") == ColorMode.MIXED

    def test_from_string_unknown_defaults_to_bw(self):
        """Unknown string defaults to black and white."""
        assert ColorMode.from_string("unknown") == ColorMode.BLACK_AND_WHITE

    def test_str_representation(self):
        """String representation returns value."""
        assert str(ColorMode.BLACK_AND_WHITE) == "bw"

    def test_is_binary(self):
        """is_binary returns correct values."""
        assert ColorMode.BLACK_AND_WHITE.is_binary() is True
        assert ColorMode.COLOR_GRAYSCALE.is_binary() is False
        assert ColorMode.MIXED.is_binary() is False

    def test_is_grayscale_or_color(self):
        """is_grayscale_or_color returns correct values."""
        assert ColorMode.BLACK_AND_WHITE.is_grayscale_or_color() is False
        assert ColorMode.COLOR_GRAYSCALE.is_grayscale_or_color() is True
        assert ColorMode.MIXED.is_grayscale_or_color() is False

    def test_is_mixed(self):
        """is_mixed returns correct values."""
        assert ColorMode.BLACK_AND_WHITE.is_mixed() is False
        assert ColorMode.COLOR_GRAYSCALE.is_mixed() is False
        assert ColorMode.MIXED.is_mixed() is True
