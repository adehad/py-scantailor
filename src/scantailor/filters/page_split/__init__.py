"""Page Split filter module.

This module provides functionality for splitting scanned images into
individual pages. It handles:
- Single page images (no split needed)
- Two-page spreads (like open books)
- Pages with offcut garbage to remove

The filter can automatically detect the appropriate split point or
accept manual user input.
"""

from .filter import Filter
from .layout_type import LayoutType
from .page_layout import PageLayout, PageLayoutType
from .params import AutoManualMode, Params
from .settings import Settings
from .split_finder import (
    ContentSpan,
    SplitResult,
    detect_split,
    find_content_spans,
    find_vertical_lines,
)

__all__ = [
    "AutoManualMode",
    "ContentSpan",
    "Filter",
    "LayoutType",
    "PageLayout",
    "PageLayoutType",
    "Params",
    "Settings",
    "SplitResult",
    "detect_split",
    "find_content_spans",
    "find_vertical_lines",
]
