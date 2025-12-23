"""Fix Orientation filter.

This is the first filter in the processing pipeline. It allows applying
orthogonal rotations (0, 90, 180, 270 degrees) to images.

The rotation is stored as metadata and applied during rendering/output,
not directly to image pixels during this stage.
"""

from __future__ import annotations

from scantailor.filters.fix_orientation.filter import Filter
from scantailor.filters.fix_orientation.settings import Settings

__all__ = [
    Filter.__name__,
    Settings.__name__,
]
