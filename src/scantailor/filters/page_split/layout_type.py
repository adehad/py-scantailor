"""Layout type enumeration for page split filter.

This module defines the different layout types that can be detected or
specified for splitting scanned images into pages.
"""

from enum import Enum


class LayoutType(Enum):
    """Type of page layout for splitting.

    AUTO_LAYOUT_TYPE: Automatically detect whether it's single or two pages.
    SINGLE_PAGE_UNCUT: Single page with no cutting needed.
    PAGE_PLUS_OFFCUT: Single page with garbage on one or both sides to cut.
    TWO_PAGES: Two pages (like an open book) to split in the middle.
    """

    AUTO_LAYOUT_TYPE = "auto-detect"
    SINGLE_PAGE_UNCUT = "single-uncut"
    PAGE_PLUS_OFFCUT = "single-cut"
    TWO_PAGES = "two-pages"

    @classmethod
    def from_string(cls, value: str) -> "LayoutType":
        """Parse a LayoutType from its string value."""
        for layout_type in cls:
            if layout_type.value == value:
                return layout_type
        return cls.AUTO_LAYOUT_TYPE

    def __str__(self) -> str:
        """Return the string value."""
        return self.value
