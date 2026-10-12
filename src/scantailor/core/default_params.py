"""Default parameters for all filter stages.

This module provides the DefaultParams class which stores default parameter
values for each stage of the ScanTailor processing pipeline. These defaults
are applied when processing new pages.
"""

from collections.abc import Iterable
from typing import Self

from pydantic import BaseModel, Field

from scantailor.core.models import (
    Dpi,
    ImageId,
    Margins,
    OrthogonalRotation,
    PageId,
    SubPage,
)
from scantailor.core.settings import Units
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.deskew.params import AutoManualMode as DeskewMode
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import ColorCommonOptions
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingOptions
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.output.picture_shape import PictureShapeOptions
from scantailor.filters.output.splitting_options import SplittingOptions
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
        default_factory=lambda: Margins(left=10.0, right=10.0, top=5.0, bottom=5.0)
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
        color_common: Default options of the grayscale and colour paths.
        picture_shape: Default automatic picture detection.
        splitting: Default layer splitting of mixed output.
        dewarping: Default dewarping options.
        depth_perception: Default dewarping depth perception, 1 to 3.
    """

    dpi: Dpi = Field(default_factory=lambda: Dpi.uniform(600))
    color_mode: ColorMode = ColorMode.BLACK_AND_WHITE
    binarization: BinarizationOptions = Field(default_factory=BinarizationOptions)
    despeckle_level: DespeckleLevel = DespeckleLevel.CAUTIOUS
    black_on_white: bool = True
    color_common: ColorCommonOptions = Field(default_factory=ColorCommonOptions)
    picture_shape: PictureShapeOptions = Field(default_factory=PictureShapeOptions)
    splitting: SplittingOptions = Field(default_factory=SplittingOptions)
    dewarping: DewarpingOptions = Field(default_factory=DewarpingOptions)
    depth_perception: float = Field(default=2.0, ge=1.0, le=3.0)

    def to_params(self) -> OutputParams:
        """Convert to filter Params instance."""
        return OutputParams(
            output_dpi=self.dpi,
            color_mode=self.color_mode,
            binarization=self.binarization,
            despeckle_level=self.despeckle_level,
            black_on_white=self.black_on_white,
            color_common=self.color_common,
            picture_shape=self.picture_shape,
            splitting=self.splitting,
            dewarping=self.dewarping,
            depth_perception=self.depth_perception,
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

    @classmethod
    def source(cls) -> Self:
        """Return the Source profile: leave the scan as it is.

        Pages are not split, deskewed, cropped or given margins, and the output
        stays grayscale.
        """
        return cls(
            deskew=DeskewDefaults(mode=DeskewMode.MANUAL),
            page_split=PageSplitDefaults(layout_type=LayoutType.SINGLE_PAGE_UNCUT),
            select_content=SelectContentDefaults(
                content_detection_mode=ContentDetectionMode.DISABLED
            ),
            page_layout=PageLayoutDefaults(
                hard_margins_mm=Margins(left=0.0, right=0.0, top=0.0, bottom=0.0),
                alignment=Alignment.independent(),
            ),
            output=OutputDefaults(
                color_mode=ColorMode.COLOR_GRAYSCALE,
                color_common=ColorCommonOptions(fill_margins=False),
            ),
        )

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


def apply_default_params(
    stages: StageSequence, images: Iterable[ImageId], defaults: DefaultParams
) -> None:
    """Make the stages of a new project start from ``defaults``.

    Stages whose defaults equal the factory ones are left alone so that the
    pipeline keeps detecting by itself. Deskew is only seeded when manual, as an
    explicit automatic entry would be trusted instead of detected.

    Pages that a split will create later (left and right) are seeded together
    with the single page, because the filters keep no default for them.

    Args:
        stages (StageSequence): Stages of the new project.
        images (Iterable[ImageId]): Images of the project.
        defaults (DefaultParams): The selected profile.
    """
    factory = DefaultParams()
    image_list = list(images)
    pages = [
        PageId(image_id=image_id, sub_page=sub_page)
        for image_id in image_list
        for sub_page in SubPage
    ]

    if defaults.fix_orientation != factory.fix_orientation:
        for image_id in image_list:
            stages.fix_orientation_filter.settings.apply_rotation(
                image_id, defaults.fix_orientation.image_rotation
            )

    if defaults.page_split != factory.page_split:
        stages.page_split_filter.settings.set_default_layout_type(
            defaults.page_split.layout_type
        )

    if defaults.deskew.mode is DeskewMode.MANUAL:
        deskew_params = defaults.get_deskew_params()
        for page in pages:
            stages.deskew_filter.settings.set_params(page, deskew_params)

    if defaults.select_content != factory.select_content:
        size = defaults.select_content.page_rect_size_mm
        stages.select_content_filter.settings.set_page_detection_box(
            size.width_mm, size.height_mm
        )
        select_params = defaults.get_select_content_params()
        for page in pages:
            stages.select_content_filter.settings.set_params(page, select_params)

    if defaults.page_layout != factory.page_layout:
        layout_params = defaults.get_page_layout_params()
        for page in pages:
            stages.page_layout_filter.settings.set_params(page, layout_params)

    if defaults.output != factory.output:
        output_params = defaults.get_output_params()
        for page in pages:
            stages.output_filter.settings.set_params(page, output_params)
