"""Options widgets for filter UI panels.

Each filter has its own OptionsWidget class that provides the UI controls
specific to that filter's parameters.
"""

from scantailor.app.ui.options_widgets.deskew import (
    OptionsWidget as DeskewOptionsWidget,
)
from scantailor.app.ui.options_widgets.fix_orientation import (
    OptionsWidget as FixOrientationOptionsWidget,
)
from scantailor.app.ui.options_widgets.output import (
    OptionsWidget as OutputOptionsWidget,
)
from scantailor.app.ui.options_widgets.page_layout import (
    OptionsWidget as PageLayoutOptionsWidget,
)
from scantailor.app.ui.options_widgets.page_split import (
    OptionsWidget as PageSplitOptionsWidget,
)
from scantailor.app.ui.options_widgets.select_content import (
    OptionsWidget as SelectContentOptionsWidget,
)

__all__ = [
    "DeskewOptionsWidget",
    "FixOrientationOptionsWidget",
    "OutputOptionsWidget",
    "PageLayoutOptionsWidget",
    "PageSplitOptionsWidget",
    "SelectContentOptionsWidget",
]
