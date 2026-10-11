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

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.models import Dpi, PageId, SubPage
from scantailor.core.transformation import (
    ImageTransformation,
    Rect,
    render_transformed,
)
from scantailor.core.zones import Zone, ZoneSpline
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.dewarping.page_dewarper import (
    auto_dewarper,
    dewarper_from_polylines,
    marginal_dewarper,
)
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.generator import DewarperSource, OutputResult
from scantailor.filters.output.params import FillZoneSpec, Params, PictureZoneSpec
from scantailor.filters.output.split_output import SplitImages
from scantailor.filters.page_layout.page_rect import (
    adapt_content_rect,
    calc_margins_mm,
    calc_page_polygon,
    rect_size_mm,
)
from scantailor.filters.page_layout.params import ContentRect, ContentSize

from .stage_sequence import FilterStage, StageSequence

# Smallest crop edge, in pixels, that is still worth applying.
_MIN_CROP_EDGE_PX = 1.0


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
    """

    image: NDArray[np.uint8]
    page_id: PageId
    stages_completed: list[FilterStage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    dpi: Dpi | None = None
    split: SplitImages | None = None


@dataclass
class PipelineOptions:
    """Options for pipeline execution.

    Attributes:
        start_stage: First stage to execute (default: FIX_ORIENTATION).
        end_stage: Last stage to execute (default: OUTPUT).
        dpi: Image resolution in dots per inch.
        dewarper: Dewarper to use for every page instead of the one the output
            parameters describe.
        detect_only: If True, only run detection without applying transforms.
    """

    start_stage: FilterStage = FilterStage.FIX_ORIENTATION
    end_stage: FilterStage = FilterStage.OUTPUT
    dpi: float = 300.0
    dewarper: CylindricalSurfaceDewarper | None = None
    detect_only: bool = False


class _PageState:
    """Oriented source pixels plus the transformation accumulated so far.

    The transformed view is rendered on demand and cached until the
    transformation changes.
    """

    def __init__(self, source: NDArray[np.uint8], dpi: float) -> None:
        self._dpi = Dpi.uniform(max(2, round(dpi)))
        self._view: NDArray[np.uint8] | None = None
        self.source = source
        self.xform = self._new_transformation(source)
        # Content rectangle in source coordinates, set by the page layout stage;
        # kept there because the later transformation changes the view's frame.
        self.content_polygon_source: NDArray[np.float64] | None = None

    def _new_transformation(self, source: NDArray[np.uint8]) -> ImageTransformation:
        height, width = source.shape[:2]
        return ImageTransformation.from_size(width, height, self._dpi)

    def replace_source(self, source: NDArray[np.uint8]) -> None:
        """Swap in new source pixels and discard the accumulated transformation."""
        self.source = source
        self.xform = self._new_transformation(source)
        self.content_polygon_source = None
        self._view = None

    def view(self) -> NDArray[np.uint8]:
        """Return the source resampled through the current transformation."""
        if self._view is None:
            self._view = render_transformed(self.source, self.xform)
        return self._view

    def crop_to(self, area: NDArray[np.float64]) -> None:
        """Crop to a polygon given in source coordinates."""
        self.xform.set_pre_crop_area(area)
        self._view = None

    def rotate(self, degrees: float) -> None:
        """Rotate clockwise by the deskew angle, expanding the canvas."""
        self.xform.set_post_rotation(degrees)
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
        image, page_id, sequence, options, detect_from=options.start_stage
    )


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

    for stage in FilterStage:
        if stage > options.end_stage:
            break
        detect = stage >= detect_from

        if stage == FilterStage.OUTPUT:
            stage_meta, generated = _process_output(state, page_id, sequence, options)
            if generated is not None:
                output_image = generated.image
                split = generated.split
                output_dpi = sequence.output_filter.get_params(page_id).output_dpi
        else:
            stage_meta = _process_stage(
                state, page_id, sequence, stage, options, detect
            )
        if stage < options.start_stage:
            continue
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
        return _process_deskew(state, page_id, sequence, detect)
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
    """
    filter_ = sequence.fix_orientation_filter
    rotation = filter_.get_rotation(page_id.image_id)

    if not options.detect_only:
        state.replace_source(filter_.process_image(state.source, page_id.image_id))
    return {"rotation_degrees": rotation.degrees}


def _process_page_split(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
    detect: bool,
) -> dict:
    """Process the Page Split stage, cropping to this page's region."""
    filter_ = sequence.page_split_filter

    if detect and not filter_.is_params_set(page_id):
        filter_.process(state.source, page_id, dpi=options.dpi)

    params = filter_.get_params(page_id)
    metadata: dict[str, object] = {
        "layout_type": params.layout_type.value,
        "is_manual": params.is_manual(),
    }

    if params.page_layout is None:
        return metadata

    metadata["num_sub_pages"] = params.page_layout.num_sub_pages
    outline = params.page_layout.get_page_outline(page_id.sub_page)
    if outline is None:
        logger.debug(
            "No {} region in layout {}; keeping the whole image",
            page_id.sub_page,
            params.page_layout.layout_type,
        )
        return metadata

    area = _clip_to_image(outline, state.source.shape)
    if area is not None:
        state.crop_to(area)
        metadata["crop_area"] = area.tolist()
    return metadata


def _clip_to_image(
    polygon: NDArray[np.floating], image_shape: tuple[int, ...]
) -> NDArray[np.float64] | None:
    """Clamp a polygon to the image bounds; None if it would be degenerate."""
    height, width = image_shape[:2]
    clipped = np.column_stack(
        [
            np.clip(polygon[:, 0], 0, width),
            np.clip(polygon[:, 1], 0, height),
        ]
    ).astype(np.float64)
    extent = clipped.max(axis=0) - clipped.min(axis=0)
    if np.any(extent < _MIN_CROP_EDGE_PX):
        return None
    return clipped


def _process_deskew(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    detect: bool,
) -> dict:
    """Process the Deskew stage, rotating by the correction angle."""
    filter_ = sequence.deskew_filter

    if detect and not filter_.settings.is_params_set(page_id):
        filter_.detect_skew(state.view(), page_id)

    params = filter_.get_params(page_id)
    state.rotate(params.deskew_angle_deg)
    return {
        "skew_angle": params.deskew_angle_deg,
        "is_manual": params.is_manual(),
    }


def _outside_source_mask(state: _PageState) -> NDArray[np.bool_]:
    """Mark view pixels that have no source pixel (deskew corners, crops).

    Returns:
        NDArray[np.bool_]: (H, W) mask of the transformed view, True where the
        resampling had nothing to sample.
    """
    rendered = render_transformed(np.zeros_like(state.source), state.xform)
    mask = rendered > 127
    if mask.ndim == 3:
        return np.any(mask, axis=2)
    return mask


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
    if detect:
        params = filter_.process(
            state.view(),
            page_id,
            dpi=options.dpi,
            outside=_outside_source_mask(state),
        )
    else:
        params = filter_.get_params(page_id)

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

    view_rect = xform.resulting_rect
    page_rect = (
        Rect(
            x=selection.page_box.x,
            y=selection.page_box.y,
            width=selection.page_box.width,
            height=selection.page_box.height,
        )
        if selection.page_box.is_valid()
        else view_rect
    )
    content_rect = Rect(
        x=selection.content_box.x,
        y=selection.content_box.y,
        width=selection.content_box.width,
        height=selection.content_box.height,
    )
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
    page_polygon = calc_page_polygon(
        xform,
        adapted_content,
        params,
        (aggregate.width_mm, aggregate.height_mm),
    )
    state.crop_to_page(page_polygon)
    state.content_polygon_source = (
        None if content_rect.is_empty() else content_polygon_source
    )

    margins = params.hard_margins_mm
    final_rect = state.xform.resulting_rect
    return {
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
) -> tuple[dict, OutputResult | None]:
    """Process the Output stage.

    The view is resampled once, at the output resolution, through the
    accumulated transformation. Content, page outline and zones are mapped into
    that view's pixels.

    Returns:
        Tuple of (stage metadata, generated output or None in detect-only mode).
    """
    from scantailor.filters.output.generator import generate_output

    params = sequence.output_filter.get_params(page_id)

    metadata: dict = {
        "color_mode": params.color_mode.value,
        "despeckle_level": params.despeckle_level.value,
        "binarization_method": params.binarization.method.value,
    }

    if options.detect_only:
        return metadata, None

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
    result = generate_output(
        view,
        params,
        dewarper=_dewarper_source(state, view, page_id, params, options),
        color_options=params.color_common,
        content_polygon=content_polygon,
        page_outline=page_outline,
        picture_zones=_view_zones(state, params.picture_zones),
        fill_zones=_view_zones(state, params.fill_zones),
    )

    metadata["is_binary"] = result.is_binary
    metadata["was_dewarped"] = result.was_dewarped
    metadata["output_dpi"] = (params.output_dpi.horizontal, params.output_dpi.vertical)

    return metadata, result


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


def _dewarper_source(
    state: _PageState,
    view: NDArray[np.uint8],
    page_id: PageId,
    params: Params,
    options: PipelineOptions,
) -> DewarperSource | None:
    """Pick how the page is dewarped, as the output parameters ask.

    Automatic and marginal models are detected on the illumination-normalised
    gray page that the output generator hands to the returned function.
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

    if mode == DewarpingMode.AUTO:
        dpi = (params.output_dpi.horizontal, params.output_dpi.vertical)
        return lambda gray: auto_dewarper(gray, dpi, content, depth)
    if mode == DewarpingMode.MARGINAL:
        spine_on_right = page_id.sub_page != SubPage.RIGHT_PAGE
        return lambda gray: marginal_dewarper(gray, content, spine_on_right, depth)
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
    """
    geometry_options = replace(options, end_stage=FilterStage.PAGE_LAYOUT)
    for image, page_id in images:
        _run_stages(
            image,
            page_id,
            sequence,
            geometry_options,
            detect_from=options.start_stage,
            render=False,
        )
