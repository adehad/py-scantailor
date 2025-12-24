"""Options widgets for filter UI panels.

Each filter has its own OptionsWidget class that provides the UI controls
specific to that filter's parameters.
"""

from __future__ import annotations

from scantailor.app.ui.options_widgets.deskew import (
    OptionsWidget as DeskewOptionsWidget,
)
from scantailor.app.ui.options_widgets.fix_orientation import (
    OptionsWidget as FixOrientationOptionsWidget,
)
from scantailor.app.ui.options_widgets.page_split import (
    OptionsWidget as PageSplitOptionsWidget,
)

__all__ = [
    "DeskewOptionsWidget",
    "FixOrientationOptionsWidget",
    "PageSplitOptionsWidget",
]
