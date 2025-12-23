"""Alignment types for the Page Layout filter.

This module defines how pages are aligned both vertically and horizontally
when their sizes differ.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class VerticalAlignment(str, Enum):
    """Vertical alignment mode for page content.

    TOP: Align content to top of page, soft margin at bottom.
    VCENTER: Center content vertically, equal soft margins.
    BOTTOM: Align content to bottom of page, soft margin at top.
    VAUTO: Automatically determine alignment based on content position.
    VORIGINAL: Preserve original content position proportion.
    """

    TOP = "top"
    VCENTER = "vcenter"
    BOTTOM = "bottom"
    VAUTO = "vauto"
    VORIGINAL = "voriginal"


class HorizontalAlignment(str, Enum):
    """Horizontal alignment mode for page content.

    LEFT: Align content to left of page, soft margin at right.
    HCENTER: Center content horizontally, equal soft margins.
    RIGHT: Align content to right of page, soft margin at left.
    HAUTO: Automatically determine alignment based on content position.
    HORIGINAL: Preserve original content position proportion.
    """

    LEFT = "left"
    HCENTER = "hcenter"
    RIGHT = "right"
    HAUTO = "hauto"
    HORIGINAL = "horiginal"


class Alignment(BaseModel):
    """Page alignment configuration.

    Defines how page content is aligned both vertically and horizontally.
    When pages have different sizes, soft margins are added to create
    uniform output sizes. The alignment determines where these soft
    margins are placed.

    Attributes:
        vertical: Vertical alignment mode.
        horizontal: Horizontal alignment mode.
        is_null: If True, this page doesn't participate in aggregate
            size calculation (independent sizing).
    """

    vertical: VerticalAlignment = VerticalAlignment.VCENTER
    horizontal: HorizontalAlignment = HorizontalAlignment.HCENTER
    is_null: bool = False

    def is_auto_vertical(self) -> bool:
        """Return True if vertical alignment is automatic."""
        return self.vertical == VerticalAlignment.VAUTO

    def is_auto_horizontal(self) -> bool:
        """Return True if horizontal alignment is automatic."""
        return self.horizontal == HorizontalAlignment.HAUTO

    def is_original_vertical(self) -> bool:
        """Return True if vertical alignment preserves original position."""
        return self.vertical == VerticalAlignment.VORIGINAL

    def is_original_horizontal(self) -> bool:
        """Return True if horizontal alignment preserves original position."""
        return self.horizontal == HorizontalAlignment.HORIGINAL

    def aligned_with_others(self) -> bool:
        """Return True if this page participates in aggregate sizing."""
        return not self.is_null

    @classmethod
    def centered(cls) -> Alignment:
        """Create a centered alignment (default)."""
        return cls(
            vertical=VerticalAlignment.VCENTER,
            horizontal=HorizontalAlignment.HCENTER,
        )

    @classmethod
    def top_left(cls) -> Alignment:
        """Create a top-left alignment."""
        return cls(
            vertical=VerticalAlignment.TOP,
            horizontal=HorizontalAlignment.LEFT,
        )

    @classmethod
    def independent(cls) -> Alignment:
        """Create an independent alignment (not aligned with others)."""
        return cls(is_null=True)
