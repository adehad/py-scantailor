"""Layout types for splitting a scan into pages.

The type is what the user picks in the Split Pages panel, or what the detector
chooses when the type is automatic.
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

    def sub_page_count(self) -> int | None:
        """Return how many logical pages the type yields, None when undecided."""
        if self == LayoutType.AUTO_LAYOUT_TYPE:
            return None
        return 2 if self == LayoutType.TWO_PAGES else 1

    def __str__(self) -> str:
        """Return the string value."""
        return self.value
