"""Parameters for the page split filter.

This module defines the Params class which stores the page split settings
for a specific page/image.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from .layout_type import LayoutType
from .page_layout import PageLayout


class AutoManualMode(Enum):
    """Mode for split line detection."""

    AUTO = "auto"
    MANUAL = "manual"


class Params(BaseModel):
    """Parameters for page split filter.

    Attributes:
        layout_type: The requested layout type for splitting.
        page_layout: The detected or manually set page layout.
        split_line_mode: Whether the split line was auto-detected or manually set.
    """

    model_config = ConfigDict(frozen=True)

    layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE
    page_layout: PageLayout | None = None
    split_line_mode: AutoManualMode = AutoManualMode.AUTO

    def with_layout_type(self, layout_type: LayoutType) -> "Params":
        """Create a copy with a different layout type."""
        return self.model_copy(update={"layout_type": layout_type})

    def with_page_layout(
        self, page_layout: PageLayout, mode: AutoManualMode = AutoManualMode.AUTO
    ) -> "Params":
        """Create a copy with a different page layout."""
        return self.model_copy(
            update={"page_layout": page_layout, "split_line_mode": mode}
        )

    def with_manual_layout(self, page_layout: PageLayout) -> "Params":
        """Create a copy with a manually set page layout."""
        return self.with_page_layout(page_layout, AutoManualMode.MANUAL)

    def with_auto_layout(self, page_layout: PageLayout) -> "Params":
        """Create a copy with an auto-detected page layout."""
        return self.with_page_layout(page_layout, AutoManualMode.AUTO)

    def is_auto(self) -> bool:
        """Return True if the split line is in auto mode."""
        return self.split_line_mode == AutoManualMode.AUTO

    def is_manual(self) -> bool:
        """Return True if the split line was manually set."""
        return self.split_line_mode == AutoManualMode.MANUAL
