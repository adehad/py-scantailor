"""Color mode options for the output filter.

This module defines the different color modes available for output images.
"""

from enum import Enum


class ColorMode(Enum):
    """Output color mode.

    BLACK_AND_WHITE: Binary (1-bit) black and white output.
    COLOR_GRAYSCALE: Grayscale output (8-bit).
    MIXED: Mixed mode - background is black/white, foreground keeps colors.
    """

    BLACK_AND_WHITE = "bw"
    COLOR_GRAYSCALE = "colorOrGray"
    MIXED = "mixed"

    @classmethod
    def from_string(cls, value: str) -> "ColorMode":
        """Parse a ColorMode from its string value."""
        for mode in cls:
            if mode.value == value:
                return mode
        return cls.BLACK_AND_WHITE

    def __str__(self) -> str:
        """Return the string value."""
        return self.value

    def is_binary(self) -> bool:
        """Return True if this mode produces binary output."""
        return self == ColorMode.BLACK_AND_WHITE

    def is_grayscale_or_color(self) -> bool:
        """Return True if this mode produces grayscale or color output."""
        return self == ColorMode.COLOR_GRAYSCALE

    def is_mixed(self) -> bool:
        """Return True if this is mixed mode."""
        return self == ColorMode.MIXED
