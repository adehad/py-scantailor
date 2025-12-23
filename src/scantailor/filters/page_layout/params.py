"""Parameters for the Page Layout filter.

This module defines the parameter structures for storing page layout
settings on a per-page basis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from scantailor.core.models import Margins
from scantailor.filters.page_layout.alignment import Alignment


class ContentRect(BaseModel):
    """Rectangle defining content boundaries in pixels.

    Attributes:
        x: Left edge x-coordinate.
        y: Top edge y-coordinate.
        width: Width of the content area.
        height: Height of the content area.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def is_empty(self) -> bool:
        """Return True if the rectangle has no area."""
        return self.width <= 0 or self.height <= 0


class ContentSize(BaseModel):
    """Size of content in millimeters.

    Attributes:
        width_mm: Width in millimeters.
        height_mm: Height in millimeters.
    """

    width_mm: float = 0.0
    height_mm: float = 0.0

    def is_empty(self) -> bool:
        """Return True if the size is effectively zero."""
        return self.width_mm <= 0 or self.height_mm <= 0


class Params(BaseModel):
    """Parameters for a single page's layout settings.

    Attributes:
        hard_margins_mm: User-specified margins in millimeters.
        alignment: How the page is aligned relative to others.
        content_rect: Content bounding box in pixels (from select_content).
        page_rect: Page bounding box in pixels.
        content_size_mm: Content size in millimeters.
        auto_margins: If True, margins were auto-calculated.
    """

    hard_margins_mm: Margins = Field(default_factory=Margins)
    alignment: Alignment = Field(default_factory=Alignment.centered)
    content_rect: ContentRect = Field(default_factory=ContentRect)
    page_rect: ContentRect = Field(default_factory=ContentRect)
    content_size_mm: ContentSize = Field(default_factory=ContentSize)
    auto_margins: bool = True

    def hard_width_mm(self) -> float:
        """Calculate total hard width including margins."""
        return (
            self.content_size_mm.width_mm
            + self.hard_margins_mm.left
            + self.hard_margins_mm.right
        )

    def hard_height_mm(self) -> float:
        """Calculate total hard height including margins."""
        return (
            self.content_size_mm.height_mm
            + self.hard_margins_mm.top
            + self.hard_margins_mm.bottom
        )

    def with_margins(self, margins: Margins) -> Params:
        """Return a new Params with updated margins.

        Args:
            margins: New margins in millimeters.

        Returns:
            New Params instance with updated margins.
        """
        return self.model_copy(
            update={"hard_margins_mm": margins, "auto_margins": False}
        )

    def with_alignment(self, alignment: Alignment) -> Params:
        """Return a new Params with updated alignment.

        Args:
            alignment: New alignment settings.

        Returns:
            New Params instance with updated alignment.
        """
        return self.model_copy(update={"alignment": alignment})

    def with_auto_margins(self, auto: bool = True) -> Params:
        """Return a new Params with auto_margins flag set.

        Args:
            auto: Whether margins should be auto-calculated.

        Returns:
            New Params instance with updated auto_margins.
        """
        return self.model_copy(update={"auto_margins": auto})
