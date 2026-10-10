"""Despeckle options for the output filter.

This module defines despeckle levels for removing noise from binary images.
"""

from enum import Enum


class DespeckleLevel(Enum):
    """Level of despeckling (noise removal) to apply.

    OFF: No despeckling.
    CAUTIOUS: Light despeckling - only remove small noise.
    NORMAL: Standard despeckling.
    AGGRESSIVE: Heavy despeckling - may remove small details.
    """

    OFF = "off"
    CAUTIOUS = "cautious"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"

    @classmethod
    def from_string(cls, value: str) -> "DespeckleLevel":
        """Parse a DespeckleLevel from its string value."""
        for level in cls:
            if level.value == value:
                return level
        return cls.OFF

    def __str__(self) -> str:
        """Return the string value."""
        return self.value

    def to_factor(self) -> float:
        """Return the despeckle strength passed to the despeckling algorithm.

        Returns:
            float: 0.0 for OFF, otherwise 1.0, 2.0 or 3.0 for cautious, normal
                or aggressive.
        """
        return _FACTORS[self]

    def is_enabled(self) -> bool:
        """Return True if despeckling is enabled."""
        return self != DespeckleLevel.OFF


_FACTORS: dict[DespeckleLevel, float] = {
    DespeckleLevel.OFF: 0.0,
    DespeckleLevel.CAUTIOUS: 1.0,
    DespeckleLevel.NORMAL: 2.0,
    DespeckleLevel.AGGRESSIVE: 3.0,
}
