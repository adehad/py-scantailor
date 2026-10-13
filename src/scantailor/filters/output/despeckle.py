"""Despeckle options for the output filter.

This module defines despeckle levels for removing noise from binary images, and
the continuous strength a C++ project stores between them.
"""

from dataclasses import dataclass
from enum import Enum

# C++ stores a number between 0.5 and 3.5 (0 is off); the named levels are the
# presets 1, 2 and 3 of it.
MAX_DESPECKLE_FACTOR = 3.5
# Midpoints between the preset strengths 0, 1, 2 and 3.
_NEAREST_LEVEL_UPPER_BOUNDS = (0.5, 1.5, 2.5)


class DespeckleLevel(Enum):
    """Named steps of the despeckle strength.

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
        """Parse a DespeckleLevel from its string value.

        Raises:
            ValueError: If ``value`` names no level.
        """
        for level in cls:
            if level.value == value:
                return level
        names = ", ".join(level.value for level in cls)
        msg = f"unknown despeckle level {value!r}; expected one of {names}"
        raise ValueError(msg)

    @classmethod
    def nearest(cls, factor: float) -> "DespeckleLevel":
        """Return the named level closest to a continuous strength.

        Args:
            factor (float): Despeckle strength; 0 is off and 1, 2 and 3 are
                cautious, normal and aggressive.

        Returns:
            DespeckleLevel: The nearest level.
        """
        levels = tuple(cls)
        for level, upper_bound in zip(levels, _NEAREST_LEVEL_UPPER_BOUNDS):
            if factor < upper_bound:
                return level
        return levels[-1]

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


@dataclass(frozen=True)
class DespeckleStrength:
    """A despeckle strength that need not be one of the named levels.

    It answers what the output generator asks of a ``DespeckleLevel``, so the
    generator can despeckle at a stored in-between strength.

    Attributes:
        factor (float): Strength passed to the despeckling algorithm; 0 is off.
    """

    factor: float

    def to_factor(self) -> float:
        """Return the strength passed to the despeckling algorithm."""
        return self.factor

    def is_enabled(self) -> bool:
        """Return True if despeckling is enabled."""
        return self.factor > 0.0
