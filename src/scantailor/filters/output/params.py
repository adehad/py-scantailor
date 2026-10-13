"""Parameters for the output filter.

This module defines the Params class which stores all output settings
for a specific page.
"""

from enum import Enum
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scantailor.core.models import Dpi
from scantailor.core.zones import (
    FillZoneProperties,
    PictureZoneProperties,
    Zone,
    ZoneSpline,
)

from .binarization import BinarizationMethod, BinarizationOptions
from .color_mode import ColorMode
from .color_options import ColorCommonOptions
from .despeckle import MAX_DESPECKLE_FACTOR, DespeckleLevel, DespeckleStrength
from .dewarping_options import DewarpingMode, DewarpingOptions
from .picture_shape import PictureShapeOptions
from .splitting_options import SplittingOptions

_MIN_CURVE_POINTS = 2


class ApplyGroup(Enum):
    """A group of output options that "Apply to..." copies on its own."""

    DPI = "dpi"
    COLOR_MODE = "color_mode"
    SPLITTING = "splitting"
    BINARIZATION = "binarization"
    DESPECKLE = "despeckle"
    DEWARPING = "dewarping"
    DEPTH_PERCEPTION = "depth_perception"


# Zones, the manual distortion model and every other per-page geometry are
# deliberately in no group: they only make sense for the page they were drawn on.
_GROUP_FIELDS: dict[ApplyGroup, tuple[str, ...]] = {
    ApplyGroup.DPI: ("output_dpi",),
    ApplyGroup.COLOR_MODE: (
        "color_mode",
        "color_common",
        "picture_shape",
        "black_on_white",
    ),
    ApplyGroup.SPLITTING: ("splitting",),
    ApplyGroup.BINARIZATION: ("binarization",),
    ApplyGroup.DESPECKLE: ("despeckle_level", "despeckle_factor"),
    ApplyGroup.DEWARPING: ("dewarping",),
    ApplyGroup.DEPTH_PERCEPTION: ("depth_perception",),
}


class ZoneOutline(BaseModel):
    """Vertices of a zone boundary.

    Attributes:
        points: Control points as (x, y) pairs.
        is_smooth: True when the boundary is interpolated between the points.
    """

    model_config = ConfigDict(frozen=True)

    points: tuple[tuple[float, float], ...] = ()
    is_smooth: bool = False

    def to_spline(self) -> ZoneSpline:
        """Return the boundary as a ZoneSpline."""
        return ZoneSpline(
            points=np.asarray(self.points, dtype=np.float64), is_smooth=self.is_smooth
        )


class PictureZoneSpec(ZoneOutline):
    """A manual picture zone for mixed output.

    Attributes:
        properties: Which layer of the mixed output the zone edits.
    """

    properties: PictureZoneProperties = Field(default_factory=PictureZoneProperties)

    def to_zone(self) -> Zone:
        """Return the zone with the properties it was defined with."""
        return Zone(spline=self.to_spline(), properties=self.properties)


class FillZoneSpec(ZoneOutline):
    """A zone painted with a fixed colour.

    Attributes:
        properties: The colour to paint.
    """

    properties: FillZoneProperties = Field(default_factory=FillZoneProperties)

    def to_zone(self) -> Zone:
        """Return the zone with the properties it was defined with."""
        return Zone(spline=self.to_spline(), properties=self.properties)


class DistortionModelSpec(BaseModel):
    """Page curvature given by the page's top and bottom edges.

    Attributes:
        top: Polyline along the top edge, left to right, as (x, y) pairs.
        bottom: Polyline along the bottom edge, left to right.
    """

    model_config = ConfigDict(frozen=True)

    top: tuple[tuple[float, float], ...] = ()
    bottom: tuple[tuple[float, float], ...] = ()

    def is_defined(self) -> bool:
        """Return True when both edges have at least two points."""
        return (
            len(self.top) >= _MIN_CURVE_POINTS and len(self.bottom) >= _MIN_CURVE_POINTS
        )


class Params(BaseModel):
    """Parameters for output filter.

    Attributes:
        output_dpi: Resolution for the output image.
        color_mode: Color mode (B&W, grayscale, or mixed).
        binarization: Binarization options for B&W conversion.
        despeckle_level: Noise removal level as the options widget's named
            step; the nearest step when ``despeckle_factor`` is set.
        despeckle_factor: Noise removal strength as C++ stores it, 0 for off
            and 0.5 to 3.5 otherwise; None when the named level says it.
        black_on_white: True for black text on white background.
        dewarping: Dewarping options for curved page correction.
        color_common: Options shared by the grayscale and colour output paths.
        picture_shape: Automatic picture detection for mixed output.
        depth_perception: Distance from the camera to the plane through the
            page's outer edges, from 1 to 3; used by dewarping.
        distortion_model: Page edges for manual dewarping.
        picture_zones: Manual picture zones for mixed output.
        fill_zones: Zones painted with a fixed colour.
        splitting: Saving mixed output as separate layers.

    Zones and the distortion model are in the coordinates of the page image
    after orientation fixing, before any crop, rotation or scaling, so they stay
    valid when margins or the output DPI change.
    """

    model_config = ConfigDict(frozen=True)

    output_dpi: Dpi = Field(default_factory=lambda: Dpi.uniform(600))
    color_mode: ColorMode = ColorMode.BLACK_AND_WHITE
    binarization: BinarizationOptions = Field(default_factory=BinarizationOptions)
    despeckle_level: DespeckleLevel = DespeckleLevel.CAUTIOUS
    despeckle_factor: float | None = Field(
        default=None, ge=0.0, le=MAX_DESPECKLE_FACTOR
    )
    black_on_white: bool = True
    dewarping: DewarpingOptions = Field(default_factory=DewarpingOptions)
    color_common: ColorCommonOptions = Field(default_factory=ColorCommonOptions)
    picture_shape: PictureShapeOptions = Field(default_factory=PictureShapeOptions)
    depth_perception: float = Field(default=2.0, ge=1.0, le=3.0)
    distortion_model: DistortionModelSpec = Field(default_factory=DistortionModelSpec)
    picture_zones: tuple[PictureZoneSpec, ...] = ()
    fill_zones: tuple[FillZoneSpec, ...] = ()
    splitting: SplittingOptions = Field(default_factory=SplittingOptions)

    def with_output_dpi(self, dpi: Dpi) -> Self:
        """Create a copy with different output DPI.

        Automatic dewarping becomes manual when a model is stored, so the page
        keeps its shape at the new resolution instead of being measured again.
        """
        return self.model_copy(update=self._dpi_update(dpi))

    def _dpi_update(self, dpi: Dpi) -> dict[str, object]:
        update: dict[str, object] = {"output_dpi": dpi}
        if (
            dpi != self.output_dpi
            and self.dewarping.is_auto()
            and self.distortion_model.is_defined()
        ):
            update["dewarping"] = self.dewarping.with_mode(DewarpingMode.MANUAL)
        return update

    def with_color_mode(self, mode: ColorMode) -> Self:
        """Create a copy with different color mode."""
        return self.model_copy(update={"color_mode": mode})

    def with_binarization(self, options: BinarizationOptions) -> Self:
        """Create a copy with different binarization options."""
        return self.model_copy(update={"binarization": options})

    def with_binarization_method(self, method: BinarizationMethod) -> Self:
        """Create a copy with different binarization method."""
        new_binarization = self.binarization.with_method(method)
        return self.model_copy(update={"binarization": new_binarization})

    @model_validator(mode="before")
    @classmethod
    def _level_follows_factor(cls, data: object) -> object:
        """Derive the named level from a stored factor that comes without one."""
        if (
            isinstance(data, dict)
            and data.get("despeckle_factor") is not None
            and "despeckle_level" not in data
        ):
            level = DespeckleLevel.nearest(float(data["despeckle_factor"]))
            return {**data, "despeckle_level": level}
        return data

    @property
    def despeckle_strength(self) -> float:
        """Strength despeckling runs at, whether set by a step or a factor."""
        if self.despeckle_factor is not None:
            return self.despeckle_factor
        return self.despeckle_level.to_factor()

    def with_despeckle_level(self, level: DespeckleLevel) -> Self:
        """Create a copy with a named despeckle level and no in-between factor."""
        return self.model_copy(
            update={"despeckle_level": level, "despeckle_factor": None}
        )

    def with_despeckle_factor(self, factor: float) -> Self:
        """Create a copy with a continuous despeckle strength.

        Args:
            factor (float): 0 for off, otherwise 0.5 to 3.5 as C++ allows.

        Returns:
            Self: Copy whose named level is the step nearest to ``factor``.

        Raises:
            ValueError: If ``factor`` is outside 0 to 3.5.
        """
        if not 0.0 <= factor <= MAX_DESPECKLE_FACTOR:
            msg = f"despeckle factor {factor} is outside 0 to {MAX_DESPECKLE_FACTOR}"
            raise ValueError(msg)
        return self.model_copy(
            update={
                "despeckle_level": DespeckleLevel.nearest(factor),
                "despeckle_factor": factor,
            }
        )

    def for_generation(self) -> Self:
        """Return a copy the output generator despeckles at the stored strength.

        The generator asks ``despeckle_level`` for the strength; an in-between
        factor takes its place there so that the generator needs no knowledge
        of it.

        Returns:
            Self: ``self`` when the strength is that of a named level.
        """
        if self.despeckle_factor is None:
            return self
        if self.despeckle_factor == self.despeckle_level.to_factor():
            return self
        strength = DespeckleStrength(self.despeckle_factor)
        return self.model_copy(update={"despeckle_level": strength})

    def with_black_on_white(self, black_on_white: bool) -> Self:
        """Create a copy with different black_on_white setting."""
        return self.model_copy(update={"black_on_white": black_on_white})

    def with_dewarping(self, options: DewarpingOptions) -> Self:
        """Create a copy with different dewarping options."""
        return self.model_copy(update={"dewarping": options})

    def with_color_common(self, options: ColorCommonOptions) -> Self:
        """Create a copy with different grayscale and colour options."""
        return self.model_copy(update={"color_common": options})

    def with_picture_shape(self, options: PictureShapeOptions) -> Self:
        """Create a copy with different picture detection options."""
        return self.model_copy(update={"picture_shape": options})

    def with_depth_perception(self, value: float) -> Self:
        """Create a copy with different dewarping depth perception."""
        return self.model_copy(update={"depth_perception": value})

    def with_distortion_model(self, model: DistortionModelSpec) -> Self:
        """Create a copy with a different manual distortion model."""
        return self.model_copy(update={"distortion_model": model})

    def with_picture_zones(self, zones: tuple[PictureZoneSpec, ...]) -> Self:
        """Create a copy with different picture zones."""
        return self.model_copy(update={"picture_zones": zones})

    def with_fill_zones(self, zones: tuple[FillZoneSpec, ...]) -> Self:
        """Create a copy with different fill zones."""
        return self.model_copy(update={"fill_zones": zones})

    def with_splitting(self, options: SplittingOptions) -> Self:
        """Create a copy with different layer splitting options."""
        return self.model_copy(update={"splitting": options})

    def with_group_from(self, source: "Params", group: ApplyGroup) -> Self:
        """Create a copy that takes one group of options from another page.

        Args:
            source (Params): Parameters of the page being copied from.
            group (ApplyGroup): The options to copy; everything else, zones and
                the distortion model included, stays as it is.

        Returns:
            Self: The merged parameters.
        """
        update = {name: getattr(source, name) for name in _GROUP_FIELDS[group]}
        if group is ApplyGroup.DPI:
            update = self._dpi_update(source.output_dpi)
        return self.model_copy(update=update)

    def needs_binarization(self) -> bool:
        """Return True if the color mode requires binarization."""
        return self.color_mode in (ColorMode.BLACK_AND_WHITE, ColorMode.MIXED)

    def needs_dewarping(self) -> bool:
        """Return True if dewarping is enabled."""
        return self.dewarping.is_enabled()
