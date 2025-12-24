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
from scantailor.core.default_params import (
    DefaultParams,
    DeskewDefaults,
    FixOrientationDefaults,
    OutputDefaults,
    PageLayoutDefaults,
    PageSplitDefaults,
    SelectContentDefaults,
)
from scantailor.core.settings import (
    ApplicationSettings,
    ColorScheme,
    DeviationSettings,
    ThumbnailSize,
    TiffCompression,
    Units,
)
from scantailor.core.stage_sequence import (
    FilterStage,
    StageSequence,
)
from scantailor.core.pipeline import (
    PipelineOptions,
    PipelineResult,
    process_batch,
    process_page,
)
from scantailor.core.transformation import (
    ImageTransformation,
    Rect,
)
from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
    Zone,
    ZoneCategory,
    ZoneSet,
    ZoneSpline,
)

__all__ = [
    ApplicationSettings.__name__,
    ColorScheme.__name__,
    DefaultParams.__name__,
    DeskewDefaults.__name__,
    DeviationSettings.__name__,
    Dpi.__name__,
    DpiStatus.__name__,
    FillZoneProperties.__name__,
    FilterStage.__name__,
    FixOrientationDefaults.__name__,
    ImageId.__name__,
    ImageInfo.__name__,
    ImageMetadata.__name__,
    ImageTransformation.__name__,
    "LayoutDirection",
    Margins.__name__,
    "OrthogonalDegrees",
    OrthogonalRotation.__name__,
    OutputDefaults.__name__,
    PageId.__name__,
    PageInfo.__name__,
    PageLayoutDefaults.__name__,
    PageSplitDefaults.__name__,
    PictureLayer.__name__,
    PictureZoneProperties.__name__,
    PipelineOptions.__name__,
    PipelineResult.__name__,
    "process_batch",
    "process_page",
    Project.__name__,
    Rect.__name__,
    SelectContentDefaults.__name__,
    StageSequence.__name__,
    SubPage.__name__,
    ThumbnailSize.__name__,
    TiffCompression.__name__,
    Units.__name__,
    Zone.__name__,
    ZoneCategory.__name__,
    ZoneSet.__name__,
    ZoneSpline.__name__,
]
