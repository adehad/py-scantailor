"""Parameters for the Deskew filter.

This module defines the parameter structures for storing deskew settings
on a per-page basis.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AutoManualMode(str, Enum):
    """Mode for deskew angle determination.

    AUTO: Automatically detect skew angle using the find_skew algorithm.
    MANUAL: Use a user-specified angle.
    """

    AUTO = "auto"
    MANUAL = "manual"


class Params(BaseModel):
    """Parameters for a single page's deskew settings.

    Attributes:
        deskew_angle_deg: The deskew angle in degrees. Positive values rotate
            clockwise, negative values rotate counter-clockwise. This is the
            correction angle (opposite of detected skew).
        mode: Whether the angle was auto-detected or manually specified.
    """

    deskew_angle_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    mode: AutoManualMode = AutoManualMode.AUTO

    def is_manual(self) -> bool:
        """Return True if the angle was manually specified."""
        return self.mode == AutoManualMode.MANUAL

    def is_auto(self) -> bool:
        """Return True if the angle was auto-detected."""
        return self.mode == AutoManualMode.AUTO

    def with_angle(self, angle: float, mode: AutoManualMode | None = None) -> Params:
        """Return a new Params with the specified angle.

        Args:
            angle: The new deskew angle in degrees.
            mode: The mode (auto/manual). If None, keeps current mode.

        Returns:
            New Params instance with the updated angle.
        """
        return Params(
            deskew_angle_deg=angle,
            mode=mode if mode is not None else self.mode,
        )

    def with_auto_angle(self, angle: float) -> Params:
        """Return a new Params with an auto-detected angle.

        Args:
            angle: The detected deskew angle in degrees.

        Returns:
            New Params instance with AUTO mode.
        """
        return Params(deskew_angle_deg=angle, mode=AutoManualMode.AUTO)

    def with_manual_angle(self, angle: float) -> Params:
        """Return a new Params with a manually specified angle.

        Args:
            angle: The manual deskew angle in degrees.

        Returns:
            New Params instance with MANUAL mode.
        """
        return Params(deskew_angle_deg=angle, mode=AutoManualMode.MANUAL)
