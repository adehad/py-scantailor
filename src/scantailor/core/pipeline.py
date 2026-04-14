"""Pipeline execution for the ScanTailor filter chain.

This module provides functions to execute the 6-stage filter pipeline
on images, processing them through each stage in sequence.

The pipeline stages are:
1. Fix Orientation - Apply orthogonal rotation
2. Page Split - Detect and apply page splits
3. Deskew - Detect and correct skew angle
4. Select Content - Detect content and page boundaries
5. Page Layout - Apply margins and alignment
6. Output - Generate final output with binarization
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import PageId
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper

from .stage_sequence import FilterStage, StageSequence


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


def process_page(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    """Process an image through the filter pipeline.

    This function executes the specified stages of the pipeline in order,
    transforming the image at each stage.

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

    current_image = image.copy()
    stages_completed: list[FilterStage] = []
    metadata: dict = {}

    # Process each stage in order
    for stage in FilterStage:
        if stage < options.start_stage:
            continue
        if stage > options.end_stage:
            break

        current_image, stage_meta = _process_stage(
            current_image, page_id, sequence, stage, options
        )
        stages_completed.append(stage)
        if stage_meta:
            metadata[stage.name] = stage_meta

    return PipelineResult(
        image=current_image,
        page_id=page_id,
        stages_completed=stages_completed,
        metadata=metadata,
    )


def _process_stage(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    stage: FilterStage,
    options: PipelineOptions,
) -> tuple[NDArray[np.uint8], dict]:
    """Process a single stage of the pipeline.

    Args:
        image: Current image state.
        page_id: The page ID.
        sequence: The stage sequence.
        stage: The stage to execute.
        options: Pipeline options.

    Returns:
        Tuple of (processed image, stage metadata).
    """
    if stage == FilterStage.FIX_ORIENTATION:
        return _process_fix_orientation(image, page_id, sequence, options)
    if stage == FilterStage.PAGE_SPLIT:
        return _process_page_split(image, page_id, sequence, options)
    if stage == FilterStage.DESKEW:
        return _process_deskew(image, page_id, sequence, options)
    if stage == FilterStage.SELECT_CONTENT:
        return _process_select_content(image, page_id, sequence, options)
    if stage == FilterStage.PAGE_LAYOUT:
        return _process_page_layout(image, page_id, sequence, options)
    # OUTPUT stage
    return _process_output(image, page_id, sequence, options)


def _process_fix_orientation(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> tuple[NDArray[np.uint8], dict]:
    """Process the Fix Orientation stage."""
    filter_ = sequence.fix_orientation_filter
    rotation = filter_.get_rotation(page_id.image_id)

    if options.detect_only:
        return image, {"rotation_degrees": rotation.degrees}

    result = filter_.process_image(image, page_id.image_id)
    return result, {"rotation_degrees": rotation.degrees}


def _process_page_split(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> tuple[NDArray[np.uint8], dict]:
    """Process the Page Split stage."""
    filter_ = sequence.page_split_filter

    # Run detection if not already set
    if not filter_.is_params_set(page_id):
        filter_.process(image, page_id, dpi=options.dpi)

    params = filter_.get_params(page_id)
    metadata: dict[str, object] = {
        "layout_type": params.layout_type.value,
        "is_manual": params.is_manual(),
    }

    if params.page_layout is not None:
        metadata["num_sub_pages"] = params.page_layout.num_sub_pages

    # For page split, we don't transform the image here
    # The split is applied by selecting the appropriate sub-page region
    return image, metadata


def _process_deskew(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> tuple[NDArray[np.uint8], dict]:
    """Process the Deskew stage."""
    filter_ = sequence.deskew_filter

    # Run detection if not already set
    if not filter_.settings.is_params_set(page_id):
        filter_.detect_skew(image, page_id)

    params = filter_.get_params(page_id)
    metadata = {
        "skew_angle": params.deskew_angle_deg,
        "is_manual": params.is_manual(),
    }

    if options.detect_only:
        return image, metadata

    result = filter_.process_image(image, page_id)
    return result, metadata


def _process_select_content(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> tuple[NDArray[np.uint8], dict]:
    """Process the Select Content stage."""
    filter_ = sequence.select_content_filter

    # Run detection if not already set
    if not filter_.is_params_set(page_id):
        filter_.process(image, page_id, dpi=options.dpi)

    params = filter_.get_params(page_id)
    metadata: dict[str, object] = {
        "content_detection_mode": params.content_detection_mode.value,
        "page_detection_mode": params.page_detection_mode.value,
    }

    if params.content_box is not None:
        box = params.content_box
        metadata["content_box"] = {
            "x": box.x,
            "y": box.y,
            "width": box.width,
            "height": box.height,
        }

    # For select content, we store the box but don't crop yet
    # The crop is typically applied during output
    return image, metadata


def _process_page_layout(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> tuple[NDArray[np.uint8], dict]:
    """Process the Page Layout stage."""
    filter_ = sequence.page_layout_filter
    params = filter_.get_params(page_id)

    margins = params.hard_margins_mm
    metadata = {
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
    }

    # Page layout doesn't transform the image directly
    # Margins and alignment are applied during output
    return image, metadata


def _process_output(
    image: NDArray[np.uint8],
    page_id: PageId,
    sequence: StageSequence,
    options: PipelineOptions,
) -> tuple[NDArray[np.uint8], dict]:
    """Process the Output stage."""
    from scantailor.filters.output.generator import generate_output

    filter_ = sequence.output_filter
    params = filter_.get_params(page_id)

    metadata: dict = {
        "color_mode": params.color_mode.value,
        "despeckle_level": params.despeckle_level.value,
        "binarization_method": params.binarization.method.value,
    }

    if options.detect_only:
        return image, metadata

    # Process the output - use generate_output directly to pass dewarper
    result = generate_output(image, params, dewarper=options.dewarper)

    metadata["is_binary"] = result.is_binary
    metadata["was_dewarped"] = result.was_dewarped

    return result.image, metadata


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
