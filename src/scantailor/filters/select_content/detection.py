"""Detection mode enums for content and page detection."""

from enum import Enum


class ContentDetectionMode(str, Enum):
    """Mode for content detection.

    AUTO: Automatically detect content boundaries.
    MANUAL: Use manually specified content box.
    DISABLED: Treat entire page as content.
    """

    AUTO = "auto"
    MANUAL = "manual"
    DISABLED = "disabled"


class PageDetectionMode(str, Enum):
    """Mode for page boundary detection.

    AUTO: Automatically detect page edges.
    MANUAL: Use manually specified page box.
    DISABLED: Use full image as page (default).
    """

    AUTO = "auto"
    MANUAL = "manual"
    DISABLED = "disabled"
