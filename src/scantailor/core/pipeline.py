"""Pipeline execution for the ScanTailor filter chain.

This module provides functions to execute the 6-stage filter pipeline
on images, processing them through each stage in sequence.

The pipeline stages are:
1. Fix Orientation - Apply orthogonal rotation
2. Page Split - Crop to the selected sub-page
3. Deskew - Rotate by the detected or manual skew angle
4. Select Content - Detect content and page boundaries
5. Page Layout - Apply margins and alignment, fixing the final page size
6. Output - Generate final output with binarization

Stages 2, 3 and 5 only edit an ImageTransformation. Pixels are resampled from
the oriented source image through the accumulated transformation, either to
give a later stage something to detect on or to produce the result.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.black_on_white_estimator import is_black_on_white
from scantailor.core.debug_images import DebugImages
from scantailor.core.models import Dpi, PageId, SubPage, page_name
from scantailor.core.page_geometry import (
    page_crop_area,
    page_polygon,
    selection_rects,
    transformation_dpi,
)
from scantailor.core.settings import BlackOnWhiteDetection
from scantailor.core.stage_summary import summarize_stage
from scantailor.core.transformation import (
    ImageTransformation,
    Rect,
    _whiten_outside,
    render_transformed,
    warp_area_averaged,
)
from scantailor.core.zones import PictureZoneProperties, Zone, ZoneSpline
from scantailor.dewarping.auto_model import DewarpMetrics, detect_distortion_model
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.dewarping.distortion_model import Curve, DistortionModel
from scantailor.dewarping.marginal_model import build_marginal_model
from scantailor.dewarping.page_dewarper import (
    dewarper_from_model,
    dewarper_from_polylines,
)
from scantailor.filters.deskew.dependencies import Dependencies as DeskewDependencies
from scantailor.filters.deskew.filter import Filter as DeskewFilter
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.generator import DewarperSource, OutputResult
from scantailor.filters.output.params import (
    AutoPictureZones,
    DistortionModelSpec,
    FillZoneSpec,
    Params,
    PictureZoneSpec,
)
from scantailor.filters.output.post_deskew import measure_post_deskew
from scantailor.filters.output.split_output import SplitImages
from scantailor.filters.page_layout.page_rect import (
    adapt_content_rect,
    calc_margins_mm,
    rect_size_mm,
)
from scantailor.filters.page_layout.params import ContentRect, ContentSize
from scantailor.filters.page_split.page_layout import PageLayout
from scantailor.imageproc.filters import bgr_to_gray

from .stage_sequence import FilterStage, StageSequence


class PipelineCancelled(Exception):
    """Raised between stages when ``PipelineOptions.should_cancel`` says to stop."""


@dataclass(frozen=True)
class OutputExtras:
    """What the Output view's tabs show besides the output page.

    Attributes:
        page: The page at output resolution as it is before dewarping and
            binarisation, in the pixels the output image has.
        despeckle_source: The text as a bilevel page just before despeckling,
            0 being ink; None when the output is not binarised.
        despeckle_dpi: Resolution of ``despeckle_source``, the smaller axis.
        picture_mask: True where the output keeps the original; only for mixed
            output whose pixels are those of ``page`` (not dewarped).
        dewarp_metrics: What automatic dewarping found on the page; None unless
            the page is dewarped automatically.
    """

    page: NDArray[np.uint8]
    despeckle_source: NDArray[np.uint8] | None
    despeckle_dpi: int
    picture_mask: NDArray[np.bool_] | None = None
    dewarp_metrics: DewarpMetrics | None = None


@dataclass
class PipelineResult:
    """Result of processing an image through the pipeline.

    Attributes:
        image: The processed output image.
        page_id: The page ID that was processed.
        stages_completed: List of stages that were executed.
        metadata: Additional metadata from processing.
        dpi: Resolution of ``image`` once the output stage has run; None before.
        split: Foreground and background layers when the output splits them.
        transformation: Source pixels to output pixels as accumulated by the
            geometry stages; None only when no stage ran.
        distortion_model: Page edges that automatic or marginal dewarping
            found, in the coordinates ``Params.distortion_model`` uses; None
            when the output stage did not build one.
        output_extras: What the Output view's tabs show; only when the options
            ask for it and the output stage ran.
        dewarp_metrics: What automatic dewarping found on the page, whether or
            not it built a model; None unless the output stage dewarped the
            page automatically.
        auto_picture_zones: The auto picture zones the output stage settled
            for the page, already stored in its parameters; None when they
            stayed as they were.
    """

    image: NDArray[np.uint8]
    page_id: PageId
    stages_completed: list[FilterStage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    dpi: Dpi | None = None
    split: SplitImages | None = None
    transformation: ImageTransformation | None = None
    distortion_model: DistortionModelSpec | None = None
    output_extras: OutputExtras | None = None
    dewarp_metrics: DewarpMetrics | None = None
    auto_picture_zones: AutoPictureZones | None = None


@dataclass
class PipelineOptions:
    """Options for pipeline execution.

    Attributes:
        start_stage: First stage to execute (default: FIX_ORIENTATION).
        end_stage: Last stage to execute (default: OUTPUT).
        dpi: Resolution of the image as decoded, before Fix Orientation; a
            number means the same on both axes. The views after Fix
            Orientation are resampled to square pixels at the smaller axis's.
        dewarper: Dewarper to use for every page instead of the one the output
            parameters describe.
        detect_only: If True, only run detection without applying transforms.
        keep_output_extras: Also return the pictures the Output view's tabs
            show, in ``PipelineResult.output_extras``.
        should_cancel: Asked before each stage; when it returns True the run
            stops with ``PipelineCancelled``. Called on the thread that runs
            the pipeline.
        render_image: False when a run that ends before Output is only wanted
            for its parameters and transformation; ``PipelineResult.image`` is
            then the source instead of a resampled view.
        black_on_white_detection: Whether Deskew and Select Content invert a
            page of light content on a dark background before detecting on it,
            and whether Output takes that polarity.
        debug_images: Receives the intermediate images of ``end_stage`` only,
            as upstream's debug mode records just the stage on screen. A
            detecting stage whose parameters are already stored detects once
            more without storing the result, so that its images are recorded.
    """

    start_stage: FilterStage = FilterStage.FIX_ORIENTATION
    end_stage: FilterStage = FilterStage.OUTPUT
    dpi: Dpi | float = 300.0
    dewarper: CylindricalSurfaceDewarper | None = None
    detect_only: bool = False
    keep_output_extras: bool = False
    should_cancel: Callable[[], bool] | None = None
    render_image: bool = True
    black_on_white_detection: BlackOnWhiteDetection = field(
        default_factory=BlackOnWhiteDetection
    )
    debug_images: DebugImages | None = None

    def debug_images_for(self, stage: FilterStage) -> DebugImages | None:
        """Return where ``stage`` records its intermediate images, or None."""
        return self.debug_images if stage == self.end_stage else None


class _PageState:
    """Oriented source pixels plus the transformation accumulated so far.

    The transformed view is rendered on demand and cached until the
    transformation changes.
    """

    def __init__(
        self, source: NDArray[np.uint8], dpi: Dpi | float, rotation_degrees: int = 0
    ) -> None:
        """Start from pixels that Fix Orientation turned by ``rotation_degrees``.

        Args:
            source (NDArray[np.uint8]): The pixels.
            dpi (Dpi | float): Resolution of the image as decoded, unturned.
            rotation_degrees (int): Orthogonal rotation already applied.
        """
        self._decoded_dpi = dpi
        self._source_dpi = transformation_dpi(dpi, rotation_degrees)
        self._view: NDArray[np.uint8] | None = None
        self._black_on_white: bool | None = None
        self._polarity_recorded = False
        self.source = source
        self.xform = self._new_transformation(source)
        # Content rectangle in source coordinates, set by the page layout stage;
        # kept there because the later transformation changes the view's frame.
        # Without content it is Page Layout's stand-in, a speck at the page's
        # centre, which tells Output to leave the page blank.
        self.content_polygon_source: NDArray[np.float64] | None = None
        # Region Fix Orientation's manual trim keeps, in source coordinates.
        self.trim_area: NDArray[np.float64] | None = None

    def _new_transformation(self, source: NDArray[np.uint8]) -> ImageTransformation:
        height, width = source.shape[:2]
        return ImageTransformation.from_size(width, height, self._source_dpi)

    @property
    def source_dpi(self) -> Dpi:
        """Resolution of ``source``; its axes may differ."""
        return self._source_dpi

    @property
    def dpi(self) -> Dpi:
        """Resolution of the view, whose pixels are square."""
        return self.xform.pre_scaled_dpi

    def replace_source(self, source: NDArray[np.uint8], rotation_degrees: int) -> None:
        """Swap in turned source pixels and discard the accumulated transformation.

        Args:
            source (NDArray[np.uint8]): The decoded image turned by
                ``rotation_degrees``.
            rotation_degrees (int): Orthogonal rotation applied to the pixels.
        """
        self._source_dpi = transformation_dpi(self._decoded_dpi, rotation_degrees)
        self.source = source
        self.xform = self._new_transformation(source)
        self.content_polygon_source = None
        self.trim_area = None
        self._view = None
        self._black_on_white = None
        self._polarity_recorded = False

    def view(self) -> NDArray[np.uint8]:
        """Return the source resampled through the current transformation."""
        if self._view is None:
            self._view = render_transformed(self.source, self.xform)
        return self._view

    def is_black_on_white(self, dbg: DebugImages | None = None) -> bool:
        """Tell whether the page inside the pre-crop area has dark content on light.

        Args:
            dbg (DebugImages | None): Receives the estimator's images. The
                answer is cached, so the first call given a collector measures
                again to record them; later ones reuse the answer.

        Returns:
            bool: True for dark content on a light background.
        """
        record = dbg is not None and not self._polarity_recorded
        if self._black_on_white is None or record:
            self._black_on_white = is_black_on_white(self.source, self.xform, dbg)
            self._polarity_recorded = self._polarity_recorded or record
        return self._black_on_white

    def detection_view(
        self, detection: BlackOnWhiteDetection, dbg: DebugImages | None = None
    ) -> NDArray[np.uint8]:
        """Return ``view``, inverted when it shows light content on a dark page.

        The source is inverted before it is resampled, so the area with no
        source pixel stays white, as it is in ``view``. ``dbg`` receives the
        polarity estimator's images.
        """
        if not detection.enabled or self.is_black_on_white(dbg):
            return self.view()
        return render_transformed(255 - self.source, self.xform)

    def view_size(self) -> tuple[int, int]:
        """Return the (width, height) of ``view`` without resampling."""
        if self._view is not None:
            height, width = self._view.shape[:2]
            return width, height
        rect = self.xform.resulting_rect
        return max(1, round(rect.width)), max(1, round(rect.height))

    def crop_to(self, area: NDArray[np.float64]) -> None:
        """Crop to a polygon given in source coordinates."""
        self.xform.set_pre_crop_area(self.xform.orig_polygon_to_pre_crop_space(area))
        self._view = None
        self._black_on_white = None
        self._polarity_recorded = False

    def apply_deskew(self, filter_: DeskewFilter, params: DeskewParams) -> None:
        """Rotate by the deskew angle, then shear by the oblique angle."""
        filter_.apply_geometry(self.xform, params)
        self._view = None

    def crop_to_page(self, area: NDArray[np.float64]) -> None:
        """Crop to a polygon given in the current transformed coordinates."""
        self.xform.set_post_crop_area(area)
        self._view = None

    def scale_to_dpi(self, dpi: Dpi) -> None:
        """Resample the view to a resolution; pixels are still taken from the source."""
        self.xform.post_scale_to_dpi(dpi)
        self._view = None

    def to_view(self, polygon: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map a polygon from source coordinates to the view's pixel grid."""
        rect = self.xform.resulting_rect
        return self.xform.map_polygon(polygon) - [rect.x, rect.y]


def process_page(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    """Process an image through the filter pipeline.

    This function executes the specified stages of the pipeline in order.
    Geometry stages accumulate an ImageTransformation; the returned image is
    the source resampled once through it (and binarized etc. by OUTPUT).

    Args:
        image: Input image to process.
        page_id: The page ID for parameter lookup.
        sequence: The stage sequence with configured filters.
        options: Pipeline execution options.

    Returns:
        PipelineResult with the processed image and metadata.

    Example:
        >>> import numpy as np
        >>> from scantailor.core import PageId, ImageId, SubPage, StageSequence
        >>> from scantailor.core.pipeline import (
        ...     process_page, PipelineOptions, FilterStage,
        ... )
        >>> from pathlib import Path
        >>>
        >>> # Create a test image and page ID
        >>> image = np.zeros((300, 200), dtype=np.uint8)
        >>> page_id = PageId(
        ...     image_id=ImageId(file_path=Path("test.tiff")),
        ...     sub_page=SubPage.SINGLE_PAGE,
        ... )
        >>>
        >>> # Process through all stages
        >>> sequence = StageSequence()
        >>> result = process_page(image, page_id, sequence)
    """
    if options is None:
        options = PipelineOptions()
    return _run_stages(
        image,
        page_id,
        sequence,
        options,
        detect_from=options.start_stage,
        render=options.render_image,
    )


def render_from_reduced_source(
    source: NDArray[np.uint8],
    full_size: tuple[int, int],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
    fit: tuple[int, int],
) -> tuple[NDArray[np.uint8], tuple[int, int]] | None:
    """Render a stage before Output from a downscaled copy of the source.

    The geometry is built as ``process_page`` builds it for the full-size
    source, from stored parameters only: no stage detects, so the caller must
    know that every stage up to ``options.end_stage`` already holds what a full
    run would settle on. Only the final resampling reads pixels, from ``source``.

    Args:
        source (NDArray[np.uint8]): The decoded image shrunk evenly, before Fix
            Orientation; ``(H, W)`` gray or ``(H, W, 3)`` BGR. Only read.
        full_size (tuple[int, int]): Width and height of the full-size decode
            that ``source`` was shrunk from.
        page_id (PageId): The page.
        sequence (StageSequence): Stages holding the page's parameters. Page
            Layout records the page's content size, as a full run does.
        options (PipelineOptions): ``end_stage`` must come before Output;
            ``dpi`` is the full-size source's resolution.
        fit (tuple[int, int]): Width and height of a box; the view, scaled
            to fit it, needs at least this much detail.

    Returns:
        tuple[NDArray[np.uint8], tuple[int, int]] | None: The stage's view
            shrunk as ``source`` is, in the channel layout of ``source``, and
            the width and height the view has at full size; None when ``source``
            is coarser than ``fit`` needs.

    Raises:
        ValueError: If ``options.end_stage`` is Output.
    """
    if options.end_stage >= FilterStage.OUTPUT:
        msg = "a reduced source cannot render the Output stage"
        raise ValueError(msg)
    orientation = sequence.fix_orientation_filter
    oriented = orientation.process_image(source, page_id.image_id)
    full_width, full_height = full_size
    if orientation.get_rotation(page_id.image_id).degrees % 180:
        full_width, full_height = full_height, full_width
    # Zero strides: the geometry only reads the full-size source's shape.
    stand_in = np.broadcast_to(np.uint8(0), (full_height, full_width))
    rotation = orientation.get_rotation(page_id.image_id).degrees
    state = _PageState(stand_in, options.dpi, rotation)
    _apply_trim(state, page_id, sequence)
    for stage in FilterStage:
        if FilterStage.FIX_ORIENTATION < stage <= options.end_stage:
            _process_stage(state, page_id, sequence, stage, options, detect=False)
    source_scale = (oriented.shape[1] / full_width, oriented.shape[0] / full_height)
    return _resample_reduced(oriented, source_scale, state.xform, fit)


def _resample_reduced(
    source: NDArray[np.uint8],
    source_scale: tuple[float, float],
    xform: ImageTransformation,
    fit: tuple[int, int],
) -> tuple[NDArray[np.uint8], tuple[int, int]] | None:
    """Resample a shrunk source through a full-size transformation, shrunk too.

    Like ``render_transformed``, areas with no source pixel or outside the
    pre-crop area are white. The view is shrunk as the source is, so that the
    resampling neither blurs nor drops detail the source still has.

    Returns:
        The view and its full-size width and height, or None when the view
        scaled to fit ``fit`` needs more detail than the source has.
    """
    rect = xform.resulting_rect
    full_size = (max(1, round(rect.width)), max(1, round(rect.height)))
    view_width, view_height = max(1.0, rect.width), max(1.0, rect.height)
    scale = min(source_scale)
    if min(fit[0] / view_width, fit[1] / view_height, 1.0) > scale * (1 + 1e-9):
        return None
    out_width = max(1, round(view_width * scale))
    out_height = max(1, round(view_height * scale))
    # OpenCV addresses pixel centres at integer coordinates while the
    # transformation addresses pixel corners, hence the half pixel shifts.
    result_to_view = np.array(
        [
            [1 / scale, 0.0, rect.x + 0.5 / scale],
            [0.0, 1 / scale, rect.y + 0.5 / scale],
            [0.0, 0.0, 1.0],
        ]
    )
    full_to_reduced = np.array(
        [[source_scale[0], 0.0, -0.5], [0.0, source_scale[1], -0.5], [0.0, 0.0, 1.0]]
    )
    dst_to_src = full_to_reduced @ xform.transform_back @ result_to_view
    result = warp_area_averaged(
        np.ascontiguousarray(source), dst_to_src, (out_width, out_height)
    )
    if result.ndim < source.ndim:
        result = result.reshape(out_height, out_width, 1)
    _whiten_outside(result, (xform.resulting_pre_crop_area - [rect.x, rect.y]) * scale)
    return result, full_size


def _run_stages(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
    detect_from: FilterStage,
    render: bool = True,
) -> PipelineResult:
    """Run every stage up to options.end_stage, detecting only from detect_from on.

    Stages before detect_from still shape the transformation, but only from
    parameters already stored; they never run detection. Only stages from
    options.start_stage on are reported.

    Args:
        image: Input image.
        page_id: The page ID.
        sequence: The stage sequence.
        options: Pipeline options.
        detect_from: First stage allowed to run detection.
        render: If False, skip resampling; the result image is the source.

    Returns:
        PipelineResult for the stages from options.start_stage on.
    """
    state = _PageState(image.copy(), options.dpi)
    stages_completed: list[FilterStage] = []
    metadata: dict = {}
    output_image: NDArray[np.uint8] | None = None
    output_dpi: Dpi | None = None
    split: SplitImages | None = None
    distortion_model: DistortionModelSpec | None = None
    extras: OutputExtras | None = None
    dewarp_metrics: DewarpMetrics | None = None
    auto_zones: AutoPictureZones | None = None

    with logger.contextualize(page=page_name(page_id)):
        for stage in FilterStage:
            if stage > options.end_stage:
                break
            detect = stage >= detect_from
            if options.should_cancel is not None and options.should_cancel():
                raise PipelineCancelled

            with logger.contextualize(stage=stage.name.lower()):
                started = time.perf_counter()
                if stage == FilterStage.OUTPUT:
                    (
                        stage_meta,
                        generated,
                        distortion_model,
                        extras,
                        dewarp_metrics,
                        auto_zones,
                    ) = _process_output(state, page_id, sequence, options)
                    if generated is not None:
                        output_image = generated.image
                        split = generated.split
                        output_dpi = sequence.output_filter.get_params(
                            page_id
                        ).output_dpi
                else:
                    stage_meta = _process_stage(
                        state, page_id, sequence, stage, options, detect
                    )
                if stage < options.start_stage:
                    continue
                if not render or stage == options.end_stage:
                    _log_stage(
                        stage, stage_meta, dewarp_metrics, time.perf_counter() - started
                    )
            stages_completed.append(stage)
            if stage_meta:
                metadata[stage.name] = stage_meta

    if output_image is None:
        skip_pixels = options.detect_only or not render
        output_image = state.source if skip_pixels else state.view()

    return PipelineResult(
        image=output_image,
        page_id=page_id,
        stages_completed=stages_completed,
        metadata=metadata,
        dpi=output_dpi,
        split=split,
        transformation=state.xform,
        distortion_model=distortion_model,
        output_extras=extras,
        dewarp_metrics=dewarp_metrics,
        auto_picture_zones=auto_zones,
    )


def _log_stage(
    stage: FilterStage,
    stage_meta: dict,
    dewarp_metrics: DewarpMetrics | None,
    seconds: float,
) -> None:
    """Log what a stage settled on for the page in the current log context.

    A run that renders only reports its last stage: the stages before it were
    decided, and reported, by the geometry pass or the earlier view that
    detected them, and repeating them would multiply the lines per page.
    """
    logger.info(
        "{}; {:.2f} s",
        summarize_stage(stage, stage_meta, dewarp_metrics),
        seconds,
    )


def _process_stage(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    stage: FilterStage,
    options: PipelineOptions,
    detect: bool,
) -> dict:
    """Process a single stage of the pipeline.

    Args:
        state: Source pixels and accumulated transformation.
        page_id: The page ID.
        sequence: The stage sequence.
        stage: The stage to execute.
        options: Pipeline options.
        detect: Whether the stage may detect parameters that are not stored.

    Returns:
        Stage metadata.
    """
    if stage == FilterStage.FIX_ORIENTATION:
        return _process_fix_orientation(state, page_id, sequence, options)
    if stage == FilterStage.PAGE_SPLIT:
        return _process_page_split(state, page_id, sequence, options, detect)
    if stage == FilterStage.DESKEW:
        return _process_deskew(state, page_id, sequence, options, detect)
    if stage == FilterStage.SELECT_CONTENT:
        return _process_select_content(state, page_id, sequence, options, detect)
    return _process_page_layout(state, page_id, sequence, options)


def _process_fix_orientation(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> dict:
    """Process the Fix Orientation stage.

    The rotation is applied to the pixels directly: an orthogonal rotation is
    lossless, and every later stage then works in the upright image's coordinates.
    The manual trim becomes the transformation's first crop, which Page Split
    narrows further.
    """
    filter_ = sequence.fix_orientation_filter
    rotation = filter_.get_rotation(page_id.image_id)
    metadata: dict[str, object] = {"rotation_degrees": rotation.degrees}

    if not options.detect_only:
        state.replace_source(
            filter_.process_image(state.source, page_id.image_id), rotation.degrees
        )
        trim_area = _apply_trim(state, page_id, sequence)
        if trim_area is not None:
            metadata["trim_area"] = trim_area.tolist()
    return metadata


def _apply_trim(
    state: _PageState, page_id: PageId, sequence: StageSequence
) -> NDArray[np.float64] | None:
    """Crop the oriented source to Fix Orientation's manual trim, if there is one.

    Returns:
        The trimmed region in oriented source pixels, or None without a trim.
    """
    height, width = state.source.shape[:2]
    trim_area = sequence.fix_orientation_filter.trim_crop_area(
        page_id.image_id, (width, height)
    )
    if trim_area is not None:
        state.trim_area = trim_area
        state.crop_to(trim_area)
    return trim_area


def _process_page_split(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
    detect: bool,
) -> dict:
    """Process the Page Split stage, cropping to this page's region."""
    filter_ = sequence.page_split_filter
    dbg = options.debug_images_for(FilterStage.PAGE_SPLIT)
    rotation_degrees = sequence.fix_orientation_filter.get_rotation(
        page_id.image_id
    ).degrees

    detected = False
    if detect:
        detected = filter_.settle(
            state.source,
            page_id,
            dpi=state.source_dpi,
            rotation_degrees=rotation_degrees,
            trim_area=state.trim_area,
            dbg=dbg,
        ).detected
    if dbg is not None and not detected:
        filter_.detect_split(
            state.source,
            page_id,
            dpi=state.source_dpi,
            store_result=False,
            rotation_degrees=rotation_degrees,
            trim_area=state.trim_area,
            dbg=dbg,
        )

    params = filter_.get_params(page_id)
    metadata: dict[str, object] = {
        "layout_type": params.layout_type.value,
        "is_manual": params.is_manual(),
    }

    if params.page_layout is None:
        return metadata

    metadata["num_sub_pages"] = params.page_layout.num_sub_pages
    height, width = state.source.shape[:2]
    metadata["page_layout"] = params.page_layout.layout_type.value
    metadata["cutters_x"] = _cutters_x(params.page_layout)
    metadata["image_width"] = width
    area = page_crop_area(
        params.page_layout, page_id.sub_page, (width, height), state.trim_area
    )
    if area is None:
        logger.debug(
            "No usable {} region in layout {}; keeping the whole image",
            page_id.sub_page,
            params.page_layout.layout_type,
        )
        return metadata

    state.crop_to(area)
    metadata["crop_area"] = area.tolist()
    return metadata


def _cutters_x(layout: PageLayout) -> list[tuple[float, float]]:
    """Return the x of each cutter of a layout at its top and its bottom end."""
    spans = []
    for cutter in (layout.cutter1, layout.cutter2)[: layout.num_cutters]:
        if cutter is not None:
            top, bottom = sorted(cutter, key=lambda point: point[1])
            spans.append((top[0], bottom[0]))
    return spans


def _process_deskew(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
    detect: bool,
) -> dict:
    """Process the Deskew stage, rotating by the correction angle."""
    filter_ = sequence.deskew_filter
    dependencies = DeskewDependencies.of(
        state.xform.pre_crop_area,
        sequence.fix_orientation_filter.get_rotation(page_id.image_id).degrees,
    )

    dbg = options.debug_images_for(FilterStage.DESKEW)
    skew = None
    if detect and filter_.needs_detection(page_id, dependencies):
        skew = filter_.detect_skew(
            state.detection_view(options.black_on_white_detection, dbg),
            page_id,
            dependencies=dependencies,
            dpi=state.dpi,
            dbg=dbg,
        )
    elif dbg is not None:
        filter_.detect_skew(
            state.detection_view(options.black_on_white_detection, dbg),
            page_id,
            store_result=False,
            dpi=state.dpi,
            dbg=dbg,
        )

    params = filter_.get_params(page_id)
    if detect and params.is_manual() and params.dependencies != dependencies:
        recorded = params.model_copy(update={"dependencies": dependencies})
        if filter_.settings.replace_params(page_id, params, recorded):
            params = recorded
        else:
            params = filter_.get_params(page_id)
    state.apply_deskew(filter_, params)
    return {
        "skew_angle": params.deskew_angle_deg,
        "oblique_angle": params.oblique_angle_deg,
        "is_manual": params.is_manual(),
        "detected_by": None if skew is None else skew.method,
        "confident": skew is None or skew.is_confident(),
    }


def _outside_source_mask(state: _PageState) -> NDArray[np.bool_]:
    """Mark view pixels that have no source pixel (deskew corners, crops).

    Returns:
        NDArray[np.bool_]: (H, W) mask of the transformed view, True where the
        resampling had nothing to sample.
    """
    # Every channel resamples alike, so one channel gives the same mask.
    blank = np.zeros(state.source.shape[:2], dtype=np.uint8)
    return render_transformed(blank, state.xform) > 127


def _outside_source_key(state: _PageState) -> tuple:
    """Identify the mask ``_outside_source_mask`` would make for this state."""
    xform = state.xform
    rect = xform.resulting_rect
    return (
        state.source.shape,
        xform.transform.tobytes(),
        np.asarray(xform.resulting_pre_crop_area).tobytes(),
        (rect.x, rect.y, rect.width, rect.height),
    )


def _process_select_content(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
    detect: bool,
) -> dict:
    """Process the Select Content stage.

    Boxes are expressed in the transformed view's coordinates and the
    transformation itself is left unchanged. Without detection the stored
    boxes are used as they are.
    """
    filter_ = sequence.select_content_filter
    dbg = options.debug_images_for(FilterStage.SELECT_CONTENT)
    page_detected = content_detected = False
    if detect:
        params, page_detected, content_detected = filter_.settle(
            lambda: state.detection_view(options.black_on_white_detection, dbg),
            page_id,
            dpi=float(state.dpi.horizontal),
            outside=lambda: _outside_source_mask(state),
            outside_key=_outside_source_key(state),
            page_outline=_polygon_points(state.xform.resulting_pre_crop_area),
            view_transform=state.xform.transform,
            image_size=state.view_size(),
            dbg=dbg,
        )
    else:
        params = filter_.get_params(page_id)
    if dbg is not None and not (page_detected and content_detected):
        _debug_select_content(
            state,
            page_id,
            sequence,
            options,
            dbg,
            page=not page_detected,
            content=not content_detected,
        )

    box = params.content_box
    page_box = params.page_box
    return {
        "content_detection_mode": params.content_detection_mode.value,
        "page_detection_mode": params.page_detection_mode.value,
        "content_box": {
            "x": box.x,
            "y": box.y,
            "width": box.width,
            "height": box.height,
        },
        "page_box": {
            "x": page_box.x,
            "y": page_box.y,
            "width": page_box.width,
            "height": page_box.height,
        },
    }


def _debug_select_content(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
    dbg: DebugImages,
    *,
    page: bool,
    content: bool,
) -> None:
    """Run the finders that did not run for their images, storing nothing."""
    filter_ = sequence.select_content_filter
    view = state.detection_view(options.black_on_white_detection, dbg)
    dpi = float(state.dpi.horizontal)
    outside = _outside_source_mask(state)
    if page:
        filter_.detect_page(
            view, page_id, dpi=dpi, store_result=False, outside=outside, dbg=dbg
        )
    if content:
        filter_.detect_content(
            view,
            page_id,
            dpi=dpi,
            store_result=False,
            outside=outside,
            page_box=filter_.get_params(page_id).page_box.to_content_box(),
            dbg=dbg,
        )


def _process_page_layout(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> dict:
    """Process the Page Layout stage, fixing the final page rectangle.

    Margins and alignment decide the page polygon around the content box; that
    polygon becomes the transformation's post-crop area.
    """
    filter_ = sequence.page_layout_filter
    selection = sequence.select_content_filter.get_params(page_id)
    xform = state.xform

    page_rect, content_rect = selection_rects(selection, xform.resulting_rect)
    adapted_content = adapt_content_rect(xform, content_rect)

    content_width_mm, content_height_mm = rect_size_mm(xform, content_rect)
    auto_margins_mm = (
        calc_margins_mm(xform, page_rect, content_rect)
        if filter_.get_params(page_id).auto_margins
        else None
    )
    update = filter_.settings.update_content_size_and_get_aggregate(
        page_id,
        _as_content_rect(page_rect),
        _as_content_rect(content_rect),
        ContentSize(width_mm=content_width_mm, height_mm=content_height_mm),
        auto_margins_mm,
    )
    params = update.params
    aggregate = update.aggregate_after
    content_polygon_source = xform.map_polygon_back(adapted_content.to_polygon())
    state.crop_to_page(
        page_polygon(
            xform,
            content_rect,
            params,
            (aggregate.width_mm, aggregate.height_mm),
        )
    )
    state.content_polygon_source = content_polygon_source

    margins = params.hard_margins_mm
    final_rect = state.xform.resulting_rect
    return {
        "aggregate_size_changed": update.aggregate_changed,
        "margins": {
            "left": margins.left,
            "right": margins.right,
            "top": margins.top,
            "bottom": margins.bottom,
        },
        "alignment": {
            "horizontal": params.alignment.horizontal.value,
            "vertical": params.alignment.vertical.value,
        },
        "page_size_px": (round(final_rect.width), round(final_rect.height)),
    }


def _as_content_rect(rect: Rect) -> ContentRect:
    return ContentRect(x=rect.x, y=rect.y, width=rect.width, height=rect.height)


def _process_output(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> tuple[
    dict,
    OutputResult | None,
    DistortionModelSpec | None,
    OutputExtras | None,
    DewarpMetrics | None,
    AutoPictureZones | None,
]:
    """Process the Output stage.

    The view is resampled once, at the output resolution, through the
    accumulated transformation. Content, page outline and zones are mapped into
    that view's pixels. Auto picture zones the render settles are stored in the
    page's parameters.

    Returns:
        Tuple of (stage metadata, generated output or None in detect-only mode,
        the distortion model automatic or marginal dewarping built or None,
        the pictures for the view's tabs or None unless the options ask, what
        automatic dewarping found or None, the auto picture zones the render
        settled or None when they stayed).
    """
    from scantailor.filters.output.generator import generate_output

    params = _settled_polarity(state, page_id, sequence, options)

    metadata: dict = {
        "black_on_white": params.black_on_white,
        "color_mode": params.color_mode.value,
        "despeckle_level": params.despeckle_level.value,
        "despeckle_strength": params.despeckle_strength,
        "binarization_method": params.binarization.method.value,
        "dewarping_mode": params.dewarping.mode.value,
    }

    if options.detect_only:
        return metadata, None, None, None, None, None

    state.scale_to_dpi(params.output_dpi)
    view = state.view()
    polygon = state.content_polygon_source
    content_polygon = (
        None if polygon is None else _polygon_points(state.to_view(polygon))
    )
    rect = state.xform.resulting_rect
    page_outline = _polygon_points(
        state.xform.resulting_pre_crop_area - [rect.x, rect.y]
    )
    height, width = state.source.shape[:2]
    source_outline = _polygon_points(
        state.to_view(
            np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=float)
        )
    )
    found = _FoundModel()
    source = _dewarper_source(state, view, page_id, params, options, found)
    post_deskew_angle, source = measure_post_deskew(view, params, source, page_outline)
    if (
        params.needs_dewarping()
        and post_deskew_angle != params.dewarping.post_deskew_angle
    ):
        params = params.with_dewarping(
            params.dewarping.model_copy(update={"post_deskew_angle": post_deskew_angle})
        )
        sequence.output_filter.settings.set_post_deskew_angle(
            page_id, post_deskew_angle
        )
    result = generate_output(
        view,
        params,
        dewarper=source,
        color_options=params.color_common,
        content_polygon=content_polygon,
        page_outline=page_outline,
        picture_zones=_view_zones(state, params.picture_zones),
        fill_zones=_view_zones(state, params.fill_zones),
        source_outline=source_outline,
        keep_despeckle_source=options.keep_output_extras,
        dbg=options.debug_images_for(FilterStage.OUTPUT),
    )

    metadata["is_binary"] = result.is_binary
    metadata["was_dewarped"] = result.was_dewarped
    metadata["output_dpi"] = (params.output_dpi.horizontal, params.output_dpi.vertical)
    metadata["post_deskew_angle"] = post_deskew_angle

    extras = (
        OutputExtras(
            page=view,
            despeckle_source=result.despeckle_source,
            despeckle_dpi=round(
                min(params.output_dpi.horizontal, params.output_dpi.vertical)
            ),
            picture_mask=_mask_on_page(result.picture_mask, view),
            dewarp_metrics=found.metrics,
        )
        if options.keep_output_extras
        else None
    )
    auto_zones = None
    if result.auto_picture_zones is not None:
        auto_zones = AutoPictureZones(
            zones=tuple(
                _source_picture_zone(state, zone) for zone in result.auto_picture_zones
            ),
            picture_shape=params.picture_shape,
        )
        sequence.output_filter.settings.replace_auto_picture_zones(page_id, auto_zones)
    return (
        metadata,
        result,
        found.as_source_spec(state),
        extras,
        found.metrics,
        auto_zones,
    )


def _settled_polarity(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> Params:
    """Return the page's output parameters with the polarity Output works in.

    When Output takes the measured polarity and the user has not chosen one,
    a measured polarity that differs from the stored one is stored.
    """
    settings = sequence.output_filter.settings
    if options.black_on_white_detection.output and not options.detect_only:
        dbg = options.debug_images_for(FilterStage.OUTPUT)
        settings.set_detected_black_on_white(page_id, state.is_black_on_white(dbg))
    return settings.get_params(page_id)


def _mask_on_page(
    mask: NDArray[np.bool_] | None, page: NDArray[np.uint8]
) -> NDArray[np.bool_] | None:
    """Return the picture mask if it lies on the pixels of ``page``."""
    if mask is None or mask.shape != page.shape[:2]:
        return None
    return mask


@dataclass
class _FoundModel:
    """The distortion model a dewarper factory built, in view pixels.

    Attributes:
        model: The model, once one worked.
        metrics: What automatic detection found, whether or not it gave a
            model; None for the other modes.
    """

    model: DistortionModel | None = None
    metrics: DewarpMetrics | None = None

    def dewarper_for(
        self, model: DistortionModel | None, depth: float
    ) -> CylindricalSurfaceDewarper | None:
        """Build the dewarper for ``model``, remembering the model if it works."""
        dewarper = dewarper_from_model(model, depth)
        if dewarper is not None:
            self.model = model
        return dewarper

    def dewarper_for_detected(
        self, detected: tuple[DistortionModel | None, DewarpMetrics], depth: float
    ) -> CylindricalSurfaceDewarper | None:
        """Like ``dewarper_for``, also keeping the metrics of the detection."""
        model, self.metrics = detected
        return self.dewarper_for(model, depth)

    def as_source_spec(self, state: _PageState) -> DistortionModelSpec | None:
        """Return the model as page edges in source coordinates, or None."""
        model = self.model
        if model is None or model.top_curve is None or model.bottom_curve is None:
            return None
        rect = state.xform.resulting_rect
        origin = np.array([rect.x, rect.y])

        def edge(polyline: NDArray[np.float64]) -> tuple[tuple[float, float], ...]:
            return tuple(
                (float(x), float(y))
                for x, y in state.xform.map_polygon_back(polyline + origin)
            )

        return DistortionModelSpec(
            top=edge(model.top_curve.polyline),
            bottom=edge(model.bottom_curve.polyline),
        )


def _detected_model(
    gray: NDArray[np.uint8],
    dpi: tuple[float, float],
    content: tuple[float, float, float, float],
    spine_on_right: bool,
    params: Params,
    state: _PageState,
    dbg: DebugImages | None,
) -> tuple[DistortionModel | None, DewarpMetrics]:
    """Fit the page curvature to its text lines and page edges.

    Text lines are traced on ``gray``, the page as laid out for output. The
    page edges are traced on the source image, where the paper still meets its
    surroundings; the view shows margins only. The model comes back in the
    view's pixels, with what the detection found. The model is None when the
    curves give no valid model or the model is refused; the page is then left
    undewarped.
    """
    rect = state.xform.resulting_rect
    view_to_resulting = np.array(
        [[1.0, 0.0, rect.x], [0.0, 1.0, rect.y], [0.0, 0.0, 1.0]]
    )
    to_source = state.xform.transform_back @ view_to_resulting
    source_gray = bgr_to_gray(state.source)
    detected = detect_distortion_model(
        gray,
        dpi,
        content,
        spine_on_right,
        edge_gray=source_gray if params.black_on_white else 255 - source_gray,
        to_edge_image=to_source[:2],
        min_span_percent=params.dewarp_min_span_percent,
        dbg=dbg,
    )
    model = detected.model
    if model is None or model.top_curve is None or model.bottom_curve is None:
        return None, detected.metrics
    return (
        DistortionModel(
            top_curve=Curve.from_polyline(state.to_view(model.top_curve.polyline)),
            bottom_curve=Curve.from_polyline(
                state.to_view(model.bottom_curve.polyline)
            ),
        ),
        detected.metrics,
    )


def _polygon_points(polygon: NDArray[np.float64]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in polygon]


def _view_zones(
    state: _PageState, specs: Sequence[PictureZoneSpec | FillZoneSpec]
) -> list[Zone]:
    """Zones from the output parameters, moved into the view's pixels."""
    return [
        Zone(
            spline=ZoneSpline(
                points=state.to_view(np.asarray(spec.points, dtype=np.float64)),
                is_smooth=spec.is_smooth,
            ),
            properties=spec.properties,
        )
        for spec in specs
        if spec.points
    ]


def _source_picture_zone(state: _PageState, zone: Zone) -> PictureZoneSpec:
    """A picture zone in the view's pixels, moved back into source coordinates."""
    rect = state.xform.resulting_rect
    points = state.xform.map_polygon_back(
        zone.spline.points + np.array([rect.x, rect.y])
    )
    assert isinstance(zone.properties, PictureZoneProperties)
    return PictureZoneSpec(
        points=tuple((float(x), float(y)) for x, y in points),
        properties=zone.properties,
    )


def _dewarper_source(
    state: _PageState,
    view: NDArray[np.uint8],
    page_id: PageId,
    params: Params,
    options: PipelineOptions,
    found: _FoundModel,
) -> DewarperSource | None:
    """Pick how the page is dewarped, as the output parameters ask.

    Automatic and marginal models are detected on the illumination-normalised
    gray page that the output generator hands to the returned function; the
    automatic one also reads the page edges off the source. The model that
    works is left in ``found``.
    """
    if options.dewarper is not None:
        return options.dewarper
    mode = params.dewarping.mode
    depth = params.depth_perception

    if mode == DewarpingMode.MANUAL:
        spec = params.distortion_model
        if not spec.is_defined():
            return None
        return dewarper_from_polylines(
            state.to_view(np.asarray(spec.top, dtype=np.float64)),
            state.to_view(np.asarray(spec.bottom, dtype=np.float64)),
            depth,
        )

    polygon = state.content_polygon_source
    if polygon is None:
        content = (0.0, 0.0, float(view.shape[1]), float(view.shape[0]))
    else:
        corners = state.to_view(polygon)
        low, high = corners.min(axis=0), corners.max(axis=0)
        content = (
            float(low[0]),
            float(low[1]),
            float(high[0] - low[0]),
            float(high[1] - low[1]),
        )

    dbg = options.debug_images_for(FilterStage.OUTPUT)
    if mode == DewarpingMode.AUTO:
        dpi = (params.output_dpi.horizontal, params.output_dpi.vertical)
        spine_on_right = page_id.sub_page != SubPage.RIGHT_PAGE
        return lambda gray: found.dewarper_for_detected(
            _detected_model(gray, dpi, content, spine_on_right, params, state, dbg),
            depth,
        )
    if mode == DewarpingMode.MARGINAL:
        spine_on_right = page_id.sub_page != SubPage.RIGHT_PAGE
        return lambda gray: found.dewarper_for(
            build_marginal_model(gray, content, spine_on_right, dbg=dbg), depth
        )
    return None


def process_batch(
    images: list[tuple[NDArray[np.uint8], PageId]],
    sequence: StageSequence,
    options: PipelineOptions | None = None,
) -> list[PipelineResult]:
    """Process multiple images through the pipeline.

    When the run reaches Page Layout, every page's geometry is computed first so
    that all pages are sized against the same aggregate, whatever their order.

    Args:
        images: List of (image, page_id) tuples.
        sequence: The stage sequence with configured filters.
        options: Pipeline execution options.

    Returns:
        List of PipelineResult for each input image.

    Example:
        >>> results = process_batch(
        ...     [(image1, page_id1), (image2, page_id2)],
        ...     sequence,
        ... )
    """
    if options is None:
        options = PipelineOptions()

    if options.end_stage >= FilterStage.PAGE_LAYOUT:
        fix_aggregate_page_size(images, sequence, options)
        detect_from = FilterStage.OUTPUT
    else:
        detect_from = options.start_stage

    return [
        _run_stages(image, page_id, sequence, options, detect_from)
        for image, page_id in images
    ]


def fix_aggregate_page_size(
    images: list[tuple[NDArray[np.uint8], PageId]],
    sequence: StageSequence,
    options: PipelineOptions,
) -> None:
    """Run every page's geometry so the page layout settings know all content sizes.

    The final page size depends on the largest aligned page, which is only
    known once every page has been measured. Pixels from this pass are discarded.
    Detection stops at ``options.end_stage`` when it comes before Margins.
    """
    end_stage = min(options.end_stage, FilterStage.PAGE_LAYOUT)
    geometry_options = replace(options, end_stage=end_stage)
    for image, page_id in images:
        _run_stages(
            image,
            page_id,
            sequence,
            geometry_options,
            detect_from=options.start_stage,
            render=False,
        )
