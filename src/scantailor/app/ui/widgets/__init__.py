"""Reusable UI widgets for the ScanTailor application."""

from __future__ import annotations

from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.app.ui.widgets.color_pickup import (
    ColorPickerButton,
    ColorPickupInteraction,
)
from scantailor.app.ui.widgets.non_owning_widget import NonOwningWidget

__all__ = [
    "CollapsibleGroupBox",
    "ColorPickerButton",
    "ColorPickupInteraction",
    "NonOwningWidget",
]
