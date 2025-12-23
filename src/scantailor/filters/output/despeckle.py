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

    def to_component_size(self) -> int:
        """Return the minimum connected component size to keep.

        Components smaller than this size will be removed.
        Returns 0 for OFF (no filtering).
        """
        if self == DespeckleLevel.OFF:
            return 0
        if self == DespeckleLevel.CAUTIOUS:
            return 10
        if self == DespeckleLevel.NORMAL:
            return 20
        return 40  # AGGRESSIVE

    def is_enabled(self) -> bool:
        """Return True if despeckling is enabled."""
        return self != DespeckleLevel.OFF
