"""Core data models for ScanTailor."""

from __future__ import annotations

from scantailor.core.models import (
    Dpi,
    ImageId,
    Margins,
    OrthogonalDegrees,
    OrthogonalRotation,
    PageId,
    SubPage,
)
from scantailor.core.project import (
    DpiStatus,
    ImageInfo,
    ImageMetadata,
    LayoutDirection,
    PageInfo,
    Project,
)
from scantailor.core.transformation import (
    ImageTransformation,
    Rect,
)

__all__ = [
    Dpi.__name__,
    DpiStatus.__name__,
    ImageId.__name__,
    ImageInfo.__name__,
    ImageMetadata.__name__,
    ImageTransformation.__name__,
    "LayoutDirection",
    Margins.__name__,
    "OrthogonalDegrees",
    OrthogonalRotation.__name__,
    PageId.__name__,
    PageInfo.__name__,
    Project.__name__,
    Rect.__name__,
    SubPage.__name__,
]
