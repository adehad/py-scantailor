"""Dewarping options for the output filter.

This module provides configuration for dewarping (page curvature correction)
during output generation.
"""

from enum import Enum

from pydantic import BaseModel, Field


class DewarpingMode(str, Enum):
    """Dewarping mode selection.

    OFF: No dewarping applied.
    AUTO: Automatically detect and correct page curvature.
    MANUAL: Use manually specified distortion model.
    MARGINAL: Apply dewarping only to page margins (book binding area).
    """

    OFF = "off"
    AUTO = "auto"
    MANUAL = "manual"
    MARGINAL = "marginal"


class DewarpingOptions(BaseModel):
    """Options for dewarping during output generation.

    Dewarping corrects perspective distortion from curved page surfaces,
    like the curvature near book spines.

    Attributes:
        mode: The dewarping mode (off, auto, manual, marginal).
        post_deskew: Whether to apply additional deskewing after dewarping.
        post_deskew_angle: The angle for post-dewarping deskew (degrees).

    Example:
        >>> options = DewarpingOptions(mode=DewarpingMode.AUTO)
        >>> options.is_enabled()
        True
        >>> options = DewarpingOptions(mode=DewarpingMode.OFF)
        >>> options.is_enabled()
        False
    """

    mode: DewarpingMode = DewarpingMode.OFF
    post_deskew: bool = Field(default=True)
    post_deskew_angle: float = Field(default=0.0, ge=-45.0, le=45.0)

    def is_enabled(self) -> bool:
        """Return True if dewarping is enabled (not OFF)."""
        return self.mode != DewarpingMode.OFF

    def is_auto(self) -> bool:
        """Return True if automatic dewarping is enabled."""
        return self.mode == DewarpingMode.AUTO

    def is_manual(self) -> bool:
        """Return True if manual dewarping is enabled."""
        return self.mode == DewarpingMode.MANUAL

    def is_marginal(self) -> bool:
        """Return True if marginal dewarping is enabled."""
        return self.mode == DewarpingMode.MARGINAL

    def with_mode(self, mode: DewarpingMode) -> DewarpingOptions:
        """Return a copy with a different mode."""
        return self.model_copy(update={"mode": mode})

    def with_post_deskew(self, enabled: bool) -> DewarpingOptions:
        """Return a copy with post-deskew setting changed."""
        return self.model_copy(update={"post_deskew": enabled})

    def with_post_deskew_angle(self, angle: float) -> DewarpingOptions:
        """Return a copy with a different post-deskew angle."""
        return self.model_copy(update={"post_deskew_angle": angle})
