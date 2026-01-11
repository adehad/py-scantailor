"""Page Layout filter.

This filter manages page margins and alignment. It ensures consistent page
sizes across a project by calculating hard margins (user-specified) and
soft margins (to align pages with each other).
"""

from scantailor.filters.page_layout.alignment import (
    Alignment,
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_layout.filter import Filter
from scantailor.filters.page_layout.params import Params
from scantailor.filters.page_layout.settings import Settings

__all__ = [
    Alignment.__name__,
    Filter.__name__,
    HorizontalAlignment.__name__,
    Params.__name__,
    Settings.__name__,
    VerticalAlignment.__name__,
]
