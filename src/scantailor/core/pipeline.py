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

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.models import Dpi, PageId
from scantailor.core.transformation import (
    ImageTransformation,
    Rect,
    render_transformed,
)
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
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
    """

    image: NDArray[np.uint8]
    page_id: PageId
    stages_completed: list[FilterStage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineOptions:
    """Options for pipeline execution.

    Attributes:
        start_stage: First stage to execute (default: FIX_ORIENTATION).
        end_stage: Last stage to execute (default: OUTPUT).
        dpi: Image resolution in dots per inch.
        dewarper: Optional dewarper for curved page correction.
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

    def _new_transformation(self, source: NDArray[np.uint8]) -> ImageTransformation:
        height, width = source.shape[:2]
        return ImageTransformation.from_size(width, height, self._dpi)

    def replace_source(self, source: NDArray[np.uint8]) -> None:
        """Swap in new source pixels and discard the accumulated transformation."""
        self.source = source
        self.xform = self._new_transformation(source)
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

    state = _PageState(image.copy(), options.dpi)
    stages_completed: list[FilterStage] = []
    metadata: dict = {}
    output_image: NDArray[np.uint8] | None = None

    for stage in FilterStage:
        if stage < options.start_stage:
            continue
        if stage > options.end_stage:
            break

        if stage == FilterStage.OUTPUT:
            stage_meta, output_image = _process_output(
                state, page_id, sequence, options
            )
        else:
            stage_meta = _process_stage(state, page_id, sequence, stage, options)
        stages_completed.append(stage)
        if stage_meta:
            metadata[stage.name] = stage_meta

    if output_image is None:
        output_image = state.source if options.detect_only else state.view()

    return PipelineResult(
        image=output_image,
        page_id=page_id,
        stages_completed=stages_completed,
        metadata=metadata,
    )


def _process_stage(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    stage: FilterStage,
    options: PipelineOptions,
) -> dict:
    """Process a single stage of the pipeline.

    Args:
        state: Source pixels and accumulated transformation.
        page_id: The page ID.
        sequence: The stage sequence.
        stage: The stage to execute.
        options: Pipeline options.

    Returns:
        Stage metadata.
    """
    if stage == FilterStage.FIX_ORIENTATION:
        return _process_fix_orientation(state, page_id, sequence, options)
    if stage == FilterStage.PAGE_SPLIT:
        return _process_page_split(state, page_id, sequence, options)
    if stage == FilterStage.DESKEW:
        return _process_deskew(state, page_id, sequence, options)
    if stage == FilterStage.SELECT_CONTENT:
        return _process_select_content(state, page_id, sequence, options)
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
) -> dict:
    """Process the Page Split stage, cropping to this page's region."""
    filter_ = sequence.page_split_filter

    if not filter_.is_params_set(page_id):
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
    options: PipelineOptions,
) -> dict:
    """Process the Deskew stage, rotating by the correction angle."""
    filter_ = sequence.deskew_filter

    if not filter_.settings.is_params_set(page_id):
        filter_.detect_skew(state.view(), page_id)

    params = filter_.get_params(page_id)
    state.rotate(params.deskew_angle_deg)
    return {
        "skew_angle": params.deskew_angle_deg,
        "is_manual": params.is_manual(),
    }


def _process_select_content(
    state: _PageState,
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> dict:
    """Process the Select Content stage.

    Boxes are expressed in the transformed view's coordinates and the
    transformation itself is left unchanged.
    """
    filter_ = sequence.select_content_filter
    params = filter_.process(state.view(), page_id, dpi=options.dpi)

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

    params = filter_.get_params(page_id)
    content_width_mm, content_height_mm = rect_size_mm(xform, content_rect)
    updates: dict[str, object] = {
        "content_rect": _as_content_rect(content_rect),
        "page_rect": _as_content_rect(page_rect),
        "content_size_mm": ContentSize(
            width_mm=content_width_mm, height_mm=content_height_mm
        ),
    }
    if params.auto_margins:
        updates["hard_margins_mm"] = calc_margins_mm(xform, page_rect, adapted_content)
    params = params.model_copy(update=updates)
    filter_.set_params(page_id, params)

    aggregate = filter_.get_aggregate_hard_size()
    page_polygon = calc_page_polygon(
        xform,
        adapted_content,
        params,
        (aggregate.width_mm, aggregate.height_mm),
    )
    state.crop_to_page(page_polygon)

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
) -> tuple[dict, NDArray[np.uint8] | None]:
    """Process the Output stage on the fully transformed view.

    Returns:
        Tuple of (stage metadata, output image or None in detect-only mode).
    """
    from scantailor.filters.output.generator import generate_output

    filter_ = sequence.output_filter
    params = filter_.get_params(page_id)

    metadata: dict = {
        "color_mode": params.color_mode.value,
        "despeckle_level": params.despeckle_level.value,
        "binarization_method": params.binarization.method.value,
    }

    if options.detect_only:
        return metadata, None

    result = generate_output(state.view(), params, dewarper=options.dewarper)

    metadata["is_binary"] = result.is_binary
    metadata["was_dewarped"] = result.was_dewarped

    return metadata, result.image


def process_batch(
    images: list[tuple[NDArray[np.uint8], PageId]],
    sequence: StageSequence,
    options: PipelineOptions | None = None,
) -> list[PipelineResult]:
    """Process multiple images through the pipeline.

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
    return [
        process_page(image, page_id, sequence, options) for image, page_id in images
    ]
