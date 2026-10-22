"""Parameters for the Deskew filter.

This module defines the parameter structures for storing deskew settings
on a per-page basis.
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field

from .dependencies import Dependencies

MAX_OBLIQUE_DEG = 5.0


class AutoManualMode(StrEnum):
    """Mode for deskew angle determination.

    AUTO: Automatically detect skew angle using the find_skew algorithm.
    MANUAL: Use a user-specified angle.
    """

    AUTO = "auto"
    MANUAL = "manual"


class Params(BaseModel):
    """Parameters for a single page's deskew settings.

    The rotation and the oblique (shear) correction each have their own angle
    and mode, so redetecting one never touches the other.

    Attributes:
        deskew_angle_deg: The deskew angle in degrees. Positive values rotate
            clockwise, negative values rotate counter-clockwise. This is the
            correction angle (opposite of detected skew).
        mode: Whether the angle was auto-detected or manually specified.
        oblique_angle_deg: The oblique (shear) correction in degrees, applied
            after the rotation. Zero leaves the page unsheared.
        oblique_mode: Whether the oblique angle is detected or manual. A page
            only gets an oblique correction when this is AUTO and detection
            found one, or when the user entered one.
        redetect_rotation: An automatic rotation that the next run has to
            detect again whatever its dependencies say.
        redetect_oblique: Same for the oblique angle.
        dependencies: Upstream geometry the angles were detected under; null if
            unknown, in which case the angles are trusted.
    """

    deskew_angle_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    mode: AutoManualMode = AutoManualMode.AUTO
    oblique_angle_deg: float = Field(
        default=0.0, ge=-MAX_OBLIQUE_DEG, le=MAX_OBLIQUE_DEG
    )
    oblique_mode: AutoManualMode = AutoManualMode.MANUAL
    redetect_rotation: bool = False
    redetect_oblique: bool = False
    dependencies: Dependencies = Field(default_factory=Dependencies)

    def is_manual(self) -> bool:
        """Return True if the angle was manually specified."""
        return self.mode == AutoManualMode.MANUAL

    def is_auto(self) -> bool:
        """Return True if the angle was auto-detected."""
        return self.mode == AutoManualMode.AUTO

    def is_oblique_manual(self) -> bool:
        """Return True if the oblique angle was manually specified."""
        return self.oblique_mode == AutoManualMode.MANUAL

    def is_oblique_auto(self) -> bool:
        """Return True if the oblique angle is detected."""
        return self.oblique_mode == AutoManualMode.AUTO

    def with_angle(self, angle: float, mode: AutoManualMode | None = None) -> Self:
        """Return a new Params with the specified angle.

        Args:
            angle: The new deskew angle in degrees.
            mode: The mode (auto/manual). If None, keeps current mode.

        Returns:
            New Params instance with the updated angle.
        """
        new_mode = mode if mode is not None else self.mode
        return self.model_copy(
            update={
                "deskew_angle_deg": angle,
                "mode": new_mode,
                "redetect_rotation": self.redetect_rotation
                and new_mode == AutoManualMode.AUTO,
            }
        )

    def with_auto_angle(self, angle: float) -> Self:
        """Return a new Params with an auto-detected angle.

        Args:
            angle: The detected deskew angle in degrees.

        Returns:
            New Params instance with AUTO mode.
        """
        return self.with_angle(angle, AutoManualMode.AUTO)

    def with_manual_angle(self, angle: float) -> Self:
        """Return a new Params with a manually specified angle.

        Args:
            angle: The manual deskew angle in degrees.

        Returns:
            New Params instance with MANUAL mode.
        """
        return self.with_angle(angle, AutoManualMode.MANUAL)

    def with_oblique(self, angle: float, mode: AutoManualMode | None = None) -> Self:
        """Return a new Params with the specified oblique angle.

        Args:
            angle: The new oblique angle in degrees, within the oblique limit.
            mode: The oblique mode. If None, keeps the current one.

        Returns:
            New Params instance with the updated oblique angle.
        """
        new_mode = mode if mode is not None else self.oblique_mode
        return self.model_copy(
            update={
                "oblique_angle_deg": angle,
                "oblique_mode": new_mode,
                "redetect_oblique": self.redetect_oblique
                and new_mode == AutoManualMode.AUTO,
            }
        )

    def needs_redetection(self) -> bool:
        """Return True if a part was asked to be detected again."""
        return self.redetect_rotation or self.redetect_oblique

    def with_manual_oblique(self, angle: float) -> Self:
        """Return a new Params with a manually specified oblique angle.

        Args:
            angle: The manual oblique angle in degrees.

        Returns:
            New Params instance with MANUAL oblique mode.
        """
        return self.with_oblique(angle, AutoManualMode.MANUAL)


def shown_deskew_angle(deskew_angle_deg: float) -> float:
    """Convert a stored deskew angle to the one the user sees and types.

    The stored angle turns clockwise for positive values because the image Y
    axis points down. The user sees the usual geometric convention instead,
    counter-clockwise positive, as ScanTailor Advanced shows it.

    Args:
        deskew_angle_deg (float): The angle as ``Params.deskew_angle_deg``
            stores it.

    Returns:
        float: The same rotation, counter-clockwise positive.
    """
    # Subtracting from 0.0 keeps a zero angle unsigned, so it never reads "-0.00".
    return 0.0 - deskew_angle_deg


def stored_deskew_angle(shown_deg: float) -> float:
    """Convert an angle the user typed to the one ``Params`` stores.

    Args:
        shown_deg (float): The angle as the user sees it, counter-clockwise
            positive.

    Returns:
        float: The same rotation, clockwise positive.
    """
    return 0.0 - shown_deg
