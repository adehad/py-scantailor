"""Default parameters for all filter stages.

This module provides the DefaultParams class which stores default parameter
values for each stage of the ScanTailor processing pipeline. These defaults
are applied when processing new pages.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from scantailor.core.models import Dpi, Margins, OrthogonalRotation
from scantailor.core.settings import Units
from scantailor.filters.deskew.params import AutoManualMode as DeskewMode
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.params import Params as PageLayoutParams
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.params import Params as PageSplitParams
from scantailor.filters.select_content.content_box import PhysicalSize
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)
from scantailor.filters.select_content.params import Params as SelectContentParams


class FixOrientationDefaults(BaseModel):
    """Default parameters for the Fix Orientation filter.

    Attributes:
        image_rotation: Default rotation to apply (0, 90, 180, 270 degrees).
    """

    image_rotation: OrthogonalRotation = Field(default_factory=OrthogonalRotation)


class DeskewDefaults(BaseModel):
    """Default parameters for the Deskew filter.

    Attributes:
        deskew_angle_deg: Default deskew angle in degrees.
        mode: Default mode (AUTO or MANUAL).
    """

    deskew_angle_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    mode: DeskewMode = DeskewMode.AUTO

    def to_params(self) -> DeskewParams:
        """Convert to filter Params instance."""
        return DeskewParams(
            deskew_angle_deg=self.deskew_angle_deg,
            mode=self.mode,
        )


class PageSplitDefaults(BaseModel):
    """Default parameters for the Page Split filter.

    Attributes:
        layout_type: Default layout type for page splitting.
    """

    layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE

    def to_params(self) -> PageSplitParams:
        """Convert to filter Params instance."""
        return PageSplitParams(layout_type=self.layout_type)


class SelectContentDefaults(BaseModel):
    """Default parameters for the Select Content filter.

    Attributes:
        page_rect_size_mm: Default page size in millimeters (A4 by default).
        content_detection_mode: Default content detection mode.
        page_detection_mode: Default page detection mode.
        fine_tune_corners: Whether to enable corner fine-tuning.
    """

    page_rect_size_mm: PhysicalSize = Field(
        default_factory=lambda: PhysicalSize(width_mm=210.0, height_mm=297.0)  # A4
    )
    content_detection_mode: ContentDetectionMode = ContentDetectionMode.AUTO
    page_detection_mode: PageDetectionMode = PageDetectionMode.DISABLED
    fine_tune_corners: bool = False

    def to_params(self) -> SelectContentParams:
        """Convert to filter Params instance."""
        return SelectContentParams(
            content_detection_mode=self.content_detection_mode,
            page_detection_mode=self.page_detection_mode,
            fine_tune_corners=self.fine_tune_corners,
        )


class PageLayoutDefaults(BaseModel):
    """Default parameters for the Page Layout filter.

    Attributes:
        hard_margins_mm: Default margins in millimeters.
        alignment: Default content alignment.
        auto_margins: Whether to auto-calculate margins.
    """

    hard_margins_mm: Margins = Field(
        default_factory=lambda: Margins(left=10.0, right=5.0, top=10.0, bottom=5.0)
    )
    alignment: Alignment = Field(default_factory=Alignment.centered)
    auto_margins: bool = False

    def to_params(self) -> PageLayoutParams:
        """Convert to filter Params instance."""
        return PageLayoutParams(
            hard_margins_mm=self.hard_margins_mm,
            alignment=self.alignment,
            auto_margins=self.auto_margins,
        )


class OutputDefaults(BaseModel):
    """Default parameters for the Output filter.

    Attributes:
        dpi: Default output resolution.
        color_mode: Default color mode.
        binarization: Default binarization options.
        despeckle_level: Default despeckling level.
        black_on_white: Default text polarity.
    """

    dpi: Dpi = Field(default_factory=lambda: Dpi.uniform(600))
    color_mode: ColorMode = ColorMode.BLACK_AND_WHITE
    binarization: BinarizationOptions = Field(default_factory=BinarizationOptions)
    despeckle_level: DespeckleLevel = DespeckleLevel.NORMAL
    black_on_white: bool = True

    def to_params(self) -> OutputParams:
        """Convert to filter Params instance."""
        return OutputParams(
            output_dpi=self.dpi,
            color_mode=self.color_mode,
            binarization=self.binarization,
            despeckle_level=self.despeckle_level,
            black_on_white=self.black_on_white,
        )


class DefaultParams(BaseModel):
    """Default parameters for all filter stages.

    This class aggregates default parameters for each stage of the
    ScanTailor processing pipeline. When processing a new page, these
    defaults are used as the initial parameter values.

    Example:
        >>> defaults = DefaultParams()
        >>> defaults.output.dpi
        Dpi(horizontal=600, vertical=600)
        >>> defaults.deskew.mode
        <AutoManualMode.AUTO: 'auto'>

    Attributes:
        fix_orientation: Default Fix Orientation parameters.
        deskew: Default Deskew parameters.
        page_split: Default Page Split parameters.
        select_content: Default Select Content parameters.
        page_layout: Default Page Layout parameters.
        output: Default Output parameters.
        units: Measurement units for display.
    """

    fix_orientation: FixOrientationDefaults = Field(
        default_factory=FixOrientationDefaults
    )
    deskew: DeskewDefaults = Field(default_factory=DeskewDefaults)
    page_split: PageSplitDefaults = Field(default_factory=PageSplitDefaults)
    select_content: SelectContentDefaults = Field(default_factory=SelectContentDefaults)
    page_layout: PageLayoutDefaults = Field(default_factory=PageLayoutDefaults)
    output: OutputDefaults = Field(default_factory=OutputDefaults)
    units: Units = Units.MILLIMETERS

    def get_deskew_params(self) -> DeskewParams:
        """Get default Deskew filter parameters."""
        return self.deskew.to_params()

    def get_page_split_params(self) -> PageSplitParams:
        """Get default Page Split filter parameters."""
        return self.page_split.to_params()

    def get_select_content_params(self) -> SelectContentParams:
        """Get default Select Content filter parameters."""
        return self.select_content.to_params()

    def get_page_layout_params(self) -> PageLayoutParams:
        """Get default Page Layout filter parameters."""
        return self.page_layout.to_params()

    def get_output_params(self) -> OutputParams:
        """Get default Output filter parameters."""
        return self.output.to_params()
