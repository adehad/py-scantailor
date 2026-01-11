"""Application-wide settings with JSON persistence.

This module provides the ApplicationSettings class for storing and retrieving
user preferences. Settings are persisted to a JSON file in the user's config
directory.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar, Self

from loguru import logger
from pydantic import BaseModel, Field


class ColorScheme(str, Enum):
    """Available color schemes."""

    DARK = "dark"
    LIGHT = "light"


class Units(str, Enum):
    """Measurement units."""

    MILLIMETERS = "mm"
    INCHES = "in"
    CENTIMETERS = "cm"


class TiffCompression(int, Enum):
    """TIFF compression methods.

    Values match libtiff constants.
    """

    NONE = 1
    LZW = 5
    JPEG = 7
    DEFLATE = 8
    CCITT_FAX4 = 4  # Group 4 fax compression (good for B&W)


@dataclass(frozen=True)
class ThumbnailSize:
    """Thumbnail dimensions."""

    width: int
    height: int


class DeviationSettings(BaseModel):
    """Settings for deviation highlighting in a filter stage."""

    coefficient: float = Field(default=1.0, ge=0.0)
    threshold: float = Field(default=1.0, ge=0.0)


class ApplicationSettings(BaseModel):
    """Application-wide settings.

    This class stores user preferences that persist across sessions.
    Settings are saved to a JSON file in the user's config directory.

    Use `ApplicationSettings.get_instance()` for singleton access, or create
    instances directly for testing.

    Example:
        >>> settings = ApplicationSettings.get_instance()
        >>> settings.color_scheme
        <ColorScheme.DARK: 'dark'>
        >>> settings.color_scheme = ColorScheme.LIGHT
        >>> settings.save()
    """

    model_config = {"validate_assignment": True}

    # Display settings
    opengl_enabled: bool = Field(default=False)
    color_scheme: ColorScheme = Field(default=ColorScheme.DARK)
    language: str = Field(default="en")
    units: Units = Field(default=Units.MILLIMETERS)

    # Project settings
    auto_save_project: bool = Field(default=False)
    show_canceling_selection_question: bool = Field(default=True)

    # Output settings
    tiff_bw_compression: TiffCompression = Field(default=TiffCompression.CCITT_FAX4)
    tiff_color_compression: TiffCompression = Field(default=TiffCompression.LZW)

    # Detection settings
    black_on_white_detection: bool = Field(default=True)
    black_on_white_detection_output: bool = Field(default=True)

    # Deviation highlighting
    highlight_deviation: bool = Field(default=True)
    deskew_deviation: DeviationSettings = Field(
        default_factory=lambda: DeviationSettings(coefficient=1.5, threshold=1.0)
    )
    select_content_deviation: DeviationSettings = Field(
        default_factory=lambda: DeviationSettings(coefficient=0.35, threshold=1.0)
    )
    margins_deviation: DeviationSettings = Field(
        default_factory=lambda: DeviationSettings(coefficient=0.35, threshold=1.0)
    )

    # Thumbnail settings
    thumbnail_quality_width: int = Field(default=200, ge=1)
    thumbnail_quality_height: int = Field(default=200, ge=1)
    max_thumbnail_width: float = Field(default=250.0, ge=1.0)
    max_thumbnail_height: float = Field(default=160.0, ge=1.0)
    single_column_thumbnail_display: bool = Field(default=False)

    # Profile settings
    current_profile: str = Field(default="Default")

    # Class-level singleton storage
    _instance: ClassVar[ApplicationSettings | None] = None
    _config_path: ClassVar[Path | None] = None

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the path to the settings file.

        Returns:
            Path to the settings JSON file.
        """
        if cls._config_path is not None:
            return cls._config_path

        # Use platform-appropriate config directory
        import sys

        if sys.platform == "win32":
            # Windows: %APPDATA%/ScanTailor/settings.json
            config_dir = Path.home() / "AppData" / "Roaming" / "ScanTailor"
        elif sys.platform == "darwin":
            # macOS: ~/Library/Application Support/ScanTailor/settings.json
            config_dir = Path.home() / "Library" / "Application Support" / "ScanTailor"
        else:
            # Linux/Unix: ~/.config/scantailor/settings.json
            config_dir = Path.home() / ".config" / "scantailor"

        return config_dir / "settings.json"

    @classmethod
    def set_config_path(cls, path: Path) -> None:
        """Set the config file path (mainly for testing).

        Args:
            path: Path to use for the settings file.
        """
        cls._config_path = path
        cls._instance = None  # Reset singleton

    @classmethod
    def get_instance(cls) -> ApplicationSettings:
        """Get the singleton settings instance.

        Loads settings from the config file if it exists, otherwise
        creates default settings.

        Returns:
            The singleton ApplicationSettings instance.
        """
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
        cls._config_path = None

    @classmethod
    def load(cls) -> ApplicationSettings:
        """Load settings from the config file.

        Returns:
            ApplicationSettings loaded from file, or defaults if file doesn't exist.
        """
        config_path = cls.get_config_path()
        if config_path.exists():
            try:
                return cls.model_validate_json(config_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load settings from {config_path}: {e}")
                logger.info("Using default settings")
        return cls()

    def save(self) -> None:
        """Save settings to the config file."""
        config_path = self.get_config_path()
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
            logger.debug(f"Settings saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save settings to {config_path}: {e}")
            raise

    @property
    def thumbnail_quality(self) -> ThumbnailSize:
        """Get thumbnail quality as a ThumbnailSize.

        Returns:
            ThumbnailSize with width and height.
        """
        return ThumbnailSize(
            width=self.thumbnail_quality_width,
            height=self.thumbnail_quality_height,
        )

    @property
    def max_thumbnail_size(self) -> tuple[float, float]:
        """Get maximum logical thumbnail size.

        Returns:
            Tuple of (width, height) in logical units.
        """
        return (self.max_thumbnail_width, self.max_thumbnail_height)
