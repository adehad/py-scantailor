"""Application-wide settings with JSON persistence.

This module provides the ApplicationSettings class for storing and retrieving
user preferences. Settings are persisted to a JSON file in the user's config
directory.
"""

from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import ClassVar, Self

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator


class Units(StrEnum):
    """Measurement units."""

    PIXELS = "px"
    MILLIMETERS = "mm"
    INCHES = "in"
    CENTIMETERS = "cm"


class ZoneCreationMode(int, Enum):
    """How a new zone is drawn in the Output view.

    Values match the ones upstream stores, and the Z, X and C shortcuts that
    choose them.
    """

    POLYGONAL = 0
    LASSO = 1
    RECTANGULAR = 2


class TiffCompression(int, Enum):
    """TIFF compression methods.

    Values match libtiff constants.
    """

    NONE = 1
    LZW = 5
    JPEG = 7
    DEFLATE = 8
    CCITT_FAX4 = 4  # Group 4 fax compression (good for B&W)


# What the Settings dialog offers: a black-and-white image cannot use JPEG and a
# grey or colour image cannot use CCITT G4.
BW_COMPRESSIONS = (
    TiffCompression.NONE,
    TiffCompression.LZW,
    TiffCompression.DEFLATE,
    TiffCompression.CCITT_FAX4,
)
COLOR_COMPRESSIONS = (
    TiffCompression.NONE,
    TiffCompression.LZW,
    TiffCompression.DEFLATE,
    TiffCompression.JPEG,
)


@dataclass(frozen=True)
class TiffCompressions:
    """TIFF compression per kind of image, as plain data for worker processes.

    Args:
        bw (TiffCompression): Used for bilevel images.
        color (TiffCompression): Used for grey and colour images.
    """

    bw: TiffCompression = TiffCompression.CCITT_FAX4
    color: TiffCompression = TiffCompression.LZW

    def __post_init__(self) -> None:
        """Reject a method that cannot compress the kind of image it is meant for."""
        if self.bw not in BW_COMPRESSIONS:
            raise ValueError(f"{self.bw.name} cannot compress a bilevel image")
        if self.color not in COLOR_COMPRESSIONS:
            raise ValueError(
                f"{self.color.name} cannot compress a grey or colour image"
            )


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
        >>> settings.units
        <Units.MILLIMETERS: 'mm'>
        >>> settings.units = Units.INCHES
        >>> settings.save()
    """

    model_config = {"validate_assignment": True}

    # Display settings; the colour scheme lives in QSettings under the key the
    # theme reads, so a change applies at once and at the next start.
    opengl_enabled: bool = Field(default=False)
    # Empty follows the operating system's locale, as upstream's default does.
    language: str = Field(default="")
    units: Units = Field(default=Units.MILLIMETERS)

    # Project settings
    auto_save_project: bool = Field(default=False)
    show_canceling_selection_question: bool = Field(default=True)

    # Output view
    default_zone_creation_mode: ZoneCreationMode = Field(
        default=ZoneCreationMode.POLYGONAL
    )

    # Millimetre grids over the page in the Margins and Output views
    margins_grid_shown: bool = Field(default=False)
    output_grid_shown: bool = Field(default=False)

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

    @field_validator("default_zone_creation_mode", mode="before")
    @classmethod
    def _unknown_zone_mode_means_polygonal(cls, value: object) -> object:
        """Fall back to polygonal for a stored value no mode has, as upstream does."""
        try:
            return ZoneCreationMode(value)
        except ValueError:
            return ZoneCreationMode.POLYGONAL

    @field_validator("tiff_bw_compression")
    @classmethod
    def _bw_compression_applies_to_bilevel(
        cls, value: TiffCompression
    ) -> TiffCompression:
        if value not in BW_COMPRESSIONS:
            raise ValueError(f"{value.name} cannot compress a bilevel image")
        return value

    @field_validator("tiff_color_compression")
    @classmethod
    def _color_compression_applies_to_color(
        cls, value: TiffCompression
    ) -> TiffCompression:
        if value not in COLOR_COMPRESSIONS:
            raise ValueError(f"{value.name} cannot compress a grey or colour image")
        return value

    # Class-level singleton storage
    _instance: ClassVar[Self | None] = None
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
    def get_instance(cls) -> Self:
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
    def load(cls) -> Self:
        """Load settings from the config file.

        Returns:
            ApplicationSettings loaded from file, or defaults if file doesn't exist.
        """
        config_path = cls.get_config_path()
        if config_path.exists():
            try:
                return cls.model_validate_json(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValidationError) as e:
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
    def tiff_compressions(self) -> TiffCompressions:
        """Get the TIFF compression choices as one picklable value.

        Returns:
            TiffCompressions: The black-and-white and the colour choice.
        """
        return TiffCompressions(
            bw=self.tiff_bw_compression, color=self.tiff_color_compression
        )

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
