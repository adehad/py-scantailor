"""Tests for Deskew filter parameters."""

import pytest

from scantailor.filters.deskew.params import AutoManualMode, Params


class TestAutoManualMode:
    """Tests for AutoManualMode enum."""

    def test_auto_value(self):
        """AUTO should have value 'auto'."""
        assert AutoManualMode.AUTO.value == "auto"

    def test_manual_value(self):
        """MANUAL should have value 'manual'."""
        assert AutoManualMode.MANUAL.value == "manual"

    def test_from_string(self):
        """Should be creatable from string values."""
        assert AutoManualMode("auto") == AutoManualMode.AUTO
        assert AutoManualMode("manual") == AutoManualMode.MANUAL


class TestParams:
    """Tests for Params class."""

    def test_default_values(self):
        """Default values should be 0 angle and AUTO mode."""
        params = Params()
        assert params.deskew_angle_deg == 0.0
        assert params.mode == AutoManualMode.AUTO

    def test_construction_with_values(self):
        """Should accept angle and mode."""
        params = Params(deskew_angle_deg=5.0, mode=AutoManualMode.MANUAL)
        assert params.deskew_angle_deg == 5.0
        assert params.mode == AutoManualMode.MANUAL

    def test_is_manual(self):
        """is_manual should return True only for MANUAL mode."""
        auto_params = Params(mode=AutoManualMode.AUTO)
        manual_params = Params(mode=AutoManualMode.MANUAL)

        assert auto_params.is_manual() is False
        assert manual_params.is_manual() is True

    def test_is_auto(self):
        """is_auto should return True only for AUTO mode."""
        auto_params = Params(mode=AutoManualMode.AUTO)
        manual_params = Params(mode=AutoManualMode.MANUAL)

        assert auto_params.is_auto() is True
        assert manual_params.is_auto() is False

    def test_with_angle_preserves_mode(self):
        """with_angle without mode should preserve current mode."""
        params = Params(deskew_angle_deg=1.0, mode=AutoManualMode.MANUAL)

        new_params = params.with_angle(5.0)

        assert new_params.deskew_angle_deg == 5.0
        assert new_params.mode == AutoManualMode.MANUAL

    def test_with_angle_changes_mode(self):
        """with_angle with mode should change the mode."""
        params = Params(deskew_angle_deg=1.0, mode=AutoManualMode.MANUAL)

        new_params = params.with_angle(5.0, mode=AutoManualMode.AUTO)

        assert new_params.deskew_angle_deg == 5.0
        assert new_params.mode == AutoManualMode.AUTO

    def test_with_auto_angle(self):
        """with_auto_angle should set AUTO mode."""
        params = Params(mode=AutoManualMode.MANUAL)

        new_params = params.with_auto_angle(3.5)

        assert new_params.deskew_angle_deg == 3.5
        assert new_params.mode == AutoManualMode.AUTO

    def test_with_manual_angle(self):
        """with_manual_angle should set MANUAL mode."""
        params = Params(mode=AutoManualMode.AUTO)

        new_params = params.with_manual_angle(-2.5)

        assert new_params.deskew_angle_deg == -2.5
        assert new_params.mode == AutoManualMode.MANUAL

    def test_angle_validation_max(self):
        """Angle should be limited to 45 degrees."""
        with pytest.raises(ValueError):
            Params(deskew_angle_deg=50.0)

    def test_angle_validation_min(self):
        """Angle should be limited to -45 degrees."""
        with pytest.raises(ValueError):
            Params(deskew_angle_deg=-50.0)

    def test_angle_at_bounds(self):
        """Angles at exactly ±45 should be valid."""
        params_max = Params(deskew_angle_deg=45.0)
        params_min = Params(deskew_angle_deg=-45.0)

        assert params_max.deskew_angle_deg == 45.0
        assert params_min.deskew_angle_deg == -45.0

    def test_json_round_trip(self):
        """Should serialize and deserialize correctly."""
        original = Params(deskew_angle_deg=7.5, mode=AutoManualMode.MANUAL)

        json_str = original.model_dump_json()
        restored = Params.model_validate_json(json_str)

        assert restored.deskew_angle_deg == original.deskew_angle_deg
        assert restored.mode == original.mode
