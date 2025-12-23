"""Tests for detection mode enums."""

from __future__ import annotations

from scantailor.filters.select_content import ContentDetectionMode, PageDetectionMode


class TestContentDetectionMode:
    """Tests for ContentDetectionMode enum."""

    def test_auto_value(self):
        """AUTO should have value 'auto'."""
        assert ContentDetectionMode.AUTO.value == "auto"

    def test_manual_value(self):
        """MANUAL should have value 'manual'."""
        assert ContentDetectionMode.MANUAL.value == "manual"

    def test_disabled_value(self):
        """DISABLED should have value 'disabled'."""
        assert ContentDetectionMode.DISABLED.value == "disabled"

    def test_from_string(self):
        """Should be creatable from string values."""
        assert ContentDetectionMode("auto") == ContentDetectionMode.AUTO
        assert ContentDetectionMode("manual") == ContentDetectionMode.MANUAL
        assert ContentDetectionMode("disabled") == ContentDetectionMode.DISABLED


class TestPageDetectionMode:
    """Tests for PageDetectionMode enum."""

    def test_auto_value(self):
        """AUTO should have value 'auto'."""
        assert PageDetectionMode.AUTO.value == "auto"

    def test_manual_value(self):
        """MANUAL should have value 'manual'."""
        assert PageDetectionMode.MANUAL.value == "manual"

    def test_disabled_value(self):
        """DISABLED should have value 'disabled'."""
        assert PageDetectionMode.DISABLED.value == "disabled"

    def test_from_string(self):
        """Should be creatable from string values."""
        assert PageDetectionMode("auto") == PageDetectionMode.AUTO
        assert PageDetectionMode("manual") == PageDetectionMode.MANUAL
        assert PageDetectionMode("disabled") == PageDetectionMode.DISABLED
