"""Tests for DewarpingOptions."""

import json

import pytest

from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions


class TestDewarpingMode:
    """Tests for DewarpingMode enum."""

    def test_values(self) -> None:
        """DewarpingMode has expected values."""
        assert DewarpingMode.OFF.value == "off"
        assert DewarpingMode.AUTO.value == "auto"
        assert DewarpingMode.MANUAL.value == "manual"
        assert DewarpingMode.MARGINAL.value == "marginal"

    def test_string_enum(self) -> None:
        """DewarpingMode is a string enum."""
        assert DewarpingMode.AUTO == "auto"
        assert DewarpingMode.OFF == "off"


class TestDewarpingOptions:
    """Tests for DewarpingOptions class."""

    def test_defaults(self) -> None:
        """Default values match C++."""
        options = DewarpingOptions()

        assert options.mode == DewarpingMode.OFF
        assert options.post_deskew is True
        assert options.post_deskew_angle == 0.0

    def test_custom_values(self) -> None:
        """Can create with custom values."""
        options = DewarpingOptions(
            mode=DewarpingMode.AUTO,
            post_deskew=False,
            post_deskew_angle=2.5,
        )

        assert options.mode == DewarpingMode.AUTO
        assert options.post_deskew is False
        assert options.post_deskew_angle == 2.5

    def test_is_enabled(self) -> None:
        """is_enabled returns True when mode is not OFF."""
        assert DewarpingOptions(mode=DewarpingMode.OFF).is_enabled() is False
        assert DewarpingOptions(mode=DewarpingMode.AUTO).is_enabled() is True
        assert DewarpingOptions(mode=DewarpingMode.MANUAL).is_enabled() is True
        assert DewarpingOptions(mode=DewarpingMode.MARGINAL).is_enabled() is True

    def test_is_auto(self) -> None:
        """is_auto returns True only for AUTO mode."""
        assert DewarpingOptions(mode=DewarpingMode.AUTO).is_auto() is True
        assert DewarpingOptions(mode=DewarpingMode.OFF).is_auto() is False
        assert DewarpingOptions(mode=DewarpingMode.MANUAL).is_auto() is False

    def test_is_manual(self) -> None:
        """is_manual returns True only for MANUAL mode."""
        assert DewarpingOptions(mode=DewarpingMode.MANUAL).is_manual() is True
        assert DewarpingOptions(mode=DewarpingMode.OFF).is_manual() is False
        assert DewarpingOptions(mode=DewarpingMode.AUTO).is_manual() is False

    def test_is_marginal(self) -> None:
        """is_marginal returns True only for MARGINAL mode."""
        assert DewarpingOptions(mode=DewarpingMode.MARGINAL).is_marginal() is True
        assert DewarpingOptions(mode=DewarpingMode.OFF).is_marginal() is False
        assert DewarpingOptions(mode=DewarpingMode.AUTO).is_marginal() is False

    def test_with_mode(self) -> None:
        """with_mode creates copy with new mode."""
        options = DewarpingOptions(mode=DewarpingMode.OFF, post_deskew=False)
        new_options = options.with_mode(DewarpingMode.AUTO)

        # Original unchanged
        assert options.mode == DewarpingMode.OFF

        # New has updated mode but preserves other settings
        assert new_options.mode == DewarpingMode.AUTO
        assert new_options.post_deskew is False

    def test_with_post_deskew(self) -> None:
        """with_post_deskew creates copy with new setting."""
        options = DewarpingOptions(mode=DewarpingMode.AUTO, post_deskew=True)
        new_options = options.with_post_deskew(False)

        assert options.post_deskew is True
        assert new_options.post_deskew is False
        assert new_options.mode == DewarpingMode.AUTO

    def test_with_post_deskew_angle(self) -> None:
        """with_post_deskew_angle creates copy with new angle."""
        options = DewarpingOptions()
        new_options = options.with_post_deskew_angle(3.5)

        assert options.post_deskew_angle == 0.0
        assert new_options.post_deskew_angle == 3.5

    def test_post_deskew_angle_validation(self) -> None:
        """Post-deskew angle is validated within bounds."""
        # Valid values
        DewarpingOptions(post_deskew_angle=0.0)
        DewarpingOptions(post_deskew_angle=45.0)
        DewarpingOptions(post_deskew_angle=-45.0)

        # Invalid values
        with pytest.raises(ValueError):
            DewarpingOptions(post_deskew_angle=46.0)
        with pytest.raises(ValueError):
            DewarpingOptions(post_deskew_angle=-46.0)

    def test_json_serialization(self) -> None:
        """DewarpingOptions serializes to JSON."""
        options = DewarpingOptions(
            mode=DewarpingMode.AUTO,
            post_deskew=False,
            post_deskew_angle=1.5,
        )

        json_str = options.model_dump_json()
        data = json.loads(json_str)

        assert data["mode"] == "auto"
        assert data["post_deskew"] is False
        assert data["post_deskew_angle"] == 1.5

    def test_json_deserialization(self) -> None:
        """DewarpingOptions deserializes from JSON."""
        json_data = {
            "mode": "manual",
            "post_deskew": True,
            "post_deskew_angle": -2.0,
        }

        options = DewarpingOptions.model_validate(json_data)

        assert options.mode == DewarpingMode.MANUAL
        assert options.post_deskew is True
        assert options.post_deskew_angle == -2.0

    def test_partial_json_uses_defaults(self) -> None:
        """Partial JSON uses defaults for missing fields."""
        json_data = {"mode": "auto"}

        options = DewarpingOptions.model_validate(json_data)

        assert options.mode == DewarpingMode.AUTO
        assert options.post_deskew is True  # Default
        assert options.post_deskew_angle == 0.0  # Default
