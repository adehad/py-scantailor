"""Deskew filter.

This filter detects and corrects page skew (rotation). It can automatically
detect skew angle using the find_skew algorithm, or use a manually specified
angle.

The deskew angle is stored as metadata and applied during rendering/output.
"""

from __future__ import annotations

from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.params import AutoManualMode, Params
from scantailor.filters.deskew.settings import Settings

__all__ = [
    AutoManualMode.__name__,
    Filter.__name__,
    Params.__name__,
    Settings.__name__,
]
