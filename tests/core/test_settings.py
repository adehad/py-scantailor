"""Tests for ApplicationSettings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scantailor.core import (
    ApplicationSettings,
    ColorScheme,
    DeviationSettings,
    ThumbnailSize,
    TiffCompression,
    Units,
)


@pytest.fixture
def temp_config_path(tmp_path: Path) -> Path:
    """Create a temporary config file path."""
    return tmp_path / "test_settings.json"


@pytest.fixture
def reset_singleton() -> None:
    """Reset the singleton before and after each test."""
    ApplicationSettings.reset_instance()
    yield
    ApplicationSettings.reset_instance()


class TestColorScheme:
    """Tests for ColorScheme enum."""

    def test_values(self) -> None:
        """ColorScheme has expected values."""
        assert ColorScheme.DARK.value == "dark"
        assert ColorScheme.LIGHT.value == "light"

    def test_string_enum(self) -> None:
        """ColorScheme is a string enum."""
        assert str(ColorScheme.DARK) == "ColorScheme.DARK"
        assert ColorScheme.DARK == "dark"


class TestUnits:
    """Tests for Units enum."""

    def test_values(self) -> None:
        """Units has expected values."""
        assert Units.MILLIMETERS.value == "mm"
        assert Units.INCHES.value == "in"
        assert Units.CENTIMETERS.value == "cm"


class TestTiffCompression:
    """Tests for TiffCompression enum."""

    def test_values(self) -> None:
        """TiffCompression has expected values matching libtiff."""
        assert TiffCompression.NONE.value == 1
        assert TiffCompression.LZW.value == 5
        assert TiffCompression.JPEG.value == 7
        assert TiffCompression.DEFLATE.value == 8
        assert TiffCompression.CCITT_FAX4.value == 4


class TestDeviationSettings:
    """Tests for DeviationSettings model."""

    def test_defaults(self) -> None:
        """Default values are 1.0."""
        settings = DeviationSettings()
        assert settings.coefficient == 1.0
        assert settings.threshold == 1.0

    def test_custom_values(self) -> None:
        """Can create with custom values."""
        settings = DeviationSettings(coefficient=1.5, threshold=0.5)
        assert settings.coefficient == 1.5
        assert settings.threshold == 0.5

    def test_validation_non_negative(self) -> None:
        """Values must be non-negative."""
        with pytest.raises(ValueError):
            DeviationSettings(coefficient=-1.0)
        with pytest.raises(ValueError):
            DeviationSettings(threshold=-0.1)


class TestThumbnailSize:
    """Tests for ThumbnailSize dataclass."""

    def test_creation(self) -> None:
        """Can create ThumbnailSize."""
        size = ThumbnailSize(width=200, height=150)
        assert size.width == 200
        assert size.height == 150

    def test_immutable(self) -> None:
        """ThumbnailSize is immutable."""
        size = ThumbnailSize(width=200, height=150)
        with pytest.raises(AttributeError):
            size.width = 300  # type: ignore[misc]


class TestApplicationSettings:
    """Tests for ApplicationSettings."""

    def test_defaults(self, reset_singleton: None) -> None:
        """Default values match C++ defaults."""
        settings = ApplicationSettings()

        # Display settings
        assert settings.opengl_enabled is False
        assert settings.color_scheme == ColorScheme.DARK
        assert settings.units == Units.MILLIMETERS

        # Project settings
        assert settings.auto_save_project is False
        assert settings.show_canceling_selection_question is True

        # Output settings
        assert settings.tiff_bw_compression == TiffCompression.CCITT_FAX4
        assert settings.tiff_color_compression == TiffCompression.LZW

        # Detection settings
        assert settings.black_on_white_detection is True
        assert settings.black_on_white_detection_output is True

        # Deviation highlighting
        assert settings.highlight_deviation is True
        assert settings.deskew_deviation.coefficient == 1.5
        assert settings.deskew_deviation.threshold == 1.0
        assert settings.select_content_deviation.coefficient == 0.35
        assert settings.margins_deviation.coefficient == 0.35

        # Thumbnail settings
        assert settings.thumbnail_quality_width == 200
        assert settings.thumbnail_quality_height == 200
        assert settings.max_thumbnail_width == 250.0
        assert settings.max_thumbnail_height == 160.0
        assert settings.single_column_thumbnail_display is False

        # Profile
        assert settings.current_profile == "Default"

    def test_custom_values(self, reset_singleton: None) -> None:
        """Can create with custom values."""
        settings = ApplicationSettings(
            color_scheme=ColorScheme.LIGHT,
            units=Units.INCHES,
            auto_save_project=True,
            tiff_bw_compression=TiffCompression.LZW,
        )

        assert settings.color_scheme == ColorScheme.LIGHT
        assert settings.units == Units.INCHES
        assert settings.auto_save_project is True
        assert settings.tiff_bw_compression == TiffCompression.LZW

    def test_thumbnail_quality_property(self, reset_singleton: None) -> None:
        """thumbnail_quality property returns ThumbnailSize."""
        settings = ApplicationSettings()
        quality = settings.thumbnail_quality

        assert isinstance(quality, ThumbnailSize)
        assert quality.width == 200
        assert quality.height == 200

    def test_max_thumbnail_size_property(self, reset_singleton: None) -> None:
        """max_thumbnail_size property returns tuple."""
        settings = ApplicationSettings()
        size = settings.max_thumbnail_size

        assert size == (250.0, 160.0)

    def test_save_and_load(
        self, temp_config_path: Path, reset_singleton: None
    ) -> None:
        """Can save and load settings."""
        ApplicationSettings.set_config_path(temp_config_path)

        # Create and save settings
        settings = ApplicationSettings(
            color_scheme=ColorScheme.LIGHT,
            units=Units.INCHES,
            auto_save_project=True,
            current_profile="Custom",
        )
        settings.save()

        # Verify file exists
        assert temp_config_path.exists()

        # Load settings
        loaded = ApplicationSettings.load()

        assert loaded.color_scheme == ColorScheme.LIGHT
        assert loaded.units == Units.INCHES
        assert loaded.auto_save_project is True
        assert loaded.current_profile == "Custom"

    def test_load_missing_file_returns_defaults(
        self, temp_config_path: Path, reset_singleton: None
    ) -> None:
        """Loading from missing file returns defaults."""
        ApplicationSettings.set_config_path(temp_config_path)

        # File doesn't exist
        assert not temp_config_path.exists()

        # Load returns defaults
        settings = ApplicationSettings.load()
        assert settings.color_scheme == ColorScheme.DARK
        assert settings.auto_save_project is False

    def test_load_corrupted_file_returns_defaults(
        self, temp_config_path: Path, reset_singleton: None
    ) -> None:
        """Loading from corrupted file returns defaults."""
        ApplicationSettings.set_config_path(temp_config_path)

        # Write invalid JSON
        temp_config_path.write_text("not valid json {{{")

        # Load returns defaults
        settings = ApplicationSettings.load()
        assert settings.color_scheme == ColorScheme.DARK

    def test_singleton_pattern(
        self, temp_config_path: Path, reset_singleton: None
    ) -> None:
        """get_instance returns same instance."""
        ApplicationSettings.set_config_path(temp_config_path)

        instance1 = ApplicationSettings.get_instance()
        instance2 = ApplicationSettings.get_instance()

        assert instance1 is instance2

    def test_singleton_loads_from_file(
        self, temp_config_path: Path, reset_singleton: None
    ) -> None:
        """Singleton loads settings from file."""
        ApplicationSettings.set_config_path(temp_config_path)

        # Save custom settings
        settings = ApplicationSettings(current_profile="TestProfile")
        settings.save()

        # Reset instance only (not config path) and get instance
        ApplicationSettings._instance = None  # Reset instance without clearing path
        instance = ApplicationSettings.get_instance()

        assert instance.current_profile == "TestProfile"

    def test_validation_on_assignment(self, reset_singleton: None) -> None:
        """Settings validate on assignment."""
        settings = ApplicationSettings()

        # Valid assignment works
        settings.color_scheme = ColorScheme.LIGHT
        assert settings.color_scheme == ColorScheme.LIGHT

        # Invalid assignment raises
        with pytest.raises(ValueError):
            settings.thumbnail_quality_width = 0  # Must be >= 1

    def test_json_serialization(self, reset_singleton: None) -> None:
        """Settings serialize to JSON correctly."""
        settings = ApplicationSettings(
            color_scheme=ColorScheme.LIGHT,
            deskew_deviation=DeviationSettings(coefficient=2.0, threshold=0.5),
        )

        json_str = settings.model_dump_json()
        data = json.loads(json_str)

        assert data["color_scheme"] == "light"
        assert data["deskew_deviation"]["coefficient"] == 2.0
        assert data["deskew_deviation"]["threshold"] == 0.5

    def test_json_deserialization(self, reset_singleton: None) -> None:
        """Settings deserialize from JSON correctly."""
        json_data = {
            "color_scheme": "light",
            "units": "in",
            "auto_save_project": True,
            "tiff_bw_compression": 5,  # LZW
        }

        settings = ApplicationSettings.model_validate(json_data)

        assert settings.color_scheme == ColorScheme.LIGHT
        assert settings.units == Units.INCHES
        assert settings.auto_save_project is True
        assert settings.tiff_bw_compression == TiffCompression.LZW

    def test_partial_json_uses_defaults(self, reset_singleton: None) -> None:
        """Partial JSON uses defaults for missing fields."""
        json_data = {"color_scheme": "light"}

        settings = ApplicationSettings.model_validate(json_data)

        assert settings.color_scheme == ColorScheme.LIGHT
        # Other values are defaults
        assert settings.units == Units.MILLIMETERS
        assert settings.auto_save_project is False

    def test_creates_config_directory(
        self, tmp_path: Path, reset_singleton: None
    ) -> None:
        """save() creates config directory if needed."""
        config_path = tmp_path / "nested" / "dir" / "settings.json"
        ApplicationSettings.set_config_path(config_path)

        settings = ApplicationSettings()
        settings.save()

        assert config_path.exists()
        assert config_path.parent.exists()
