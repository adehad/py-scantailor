"""Page transformations rebuilt from stored stage parameters.

The image views show the oriented source image through the transformation
that the stages before them define. This module builds those transformations
from the parameters alone, without detection and without touching pixels, so
it is cheap enough for the GUI thread. It applies the stages the same way
``core.pipeline`` does.
"""

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi, PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation, Rect
from scantailor.filters.page_layout.page_rect import (
    adapt_content_rect,
    calc_page_polygon,
)
from scantailor.filters.page_split.page_layout_adapter import (
    correct_page_layout_type,
)

# Smallest crop edge, in pixels, that is still worth applying.
_MIN_CROP_EDGE_PX = 1.0
_MIN_DPI = 2


def transformation_dpi(dpi: float) -> Dpi:
    """Return the integer resolution a transformation is built with.

    Args:
        dpi (float): Source resolution in dots per inch.

    Returns:
        Dpi: Square-pixel resolution, at least 2.
    """
    return Dpi.uniform(max(_MIN_DPI, round(dpi)))


def build_transformation(
    sequence: StageSequence,
    page_id: PageId,
    oriented_size: tuple[int, int],
    dpi: float,
    through: FilterStage,
) -> ImageTransformation:
    """Build the transformation defined by the stages up to ``through``.

    Args:
        sequence (StageSequence): Stages holding the page's parameters; only read.
        page_id (PageId): The page.
        oriented_size (tuple[int, int]): Width and height in pixels of the source
            image after Fix Orientation.
        dpi (float): Resolution of the source image.
        through (FilterStage): Last stage whose geometry is included.
            ``FIX_ORIENTATION`` gives the identity, ``PAGE_SPLIT`` adds the
            crop to the page, ``DESKEW`` adds the rotation and ``PAGE_LAYOUT``
            adds the crop to the final page.

    Returns:
        ImageTransformation: Maps oriented source pixels to the coordinates the
        stages after ``through`` work in.
    """
    width, height = oriented_size
    xform = ImageTransformation.from_size(width, height, transformation_dpi(dpi))
    if through < FilterStage.PAGE_SPLIT:
        return xform

    crop = _page_crop_area(sequence, page_id, oriented_size)
    if crop is not None:
        xform.set_pre_crop_area(crop)
    if through < FilterStage.DESKEW:
        return xform

    deskew = sequence.deskew_filter
    deskew.apply_geometry(xform, deskew.get_params(page_id))
    if through >= FilterStage.PAGE_LAYOUT:
        xform.set_post_crop_area(_page_polygon(sequence, page_id, xform))
    return xform


def _page_crop_area(
    sequence: StageSequence, page_id: PageId, size: tuple[int, int]
) -> NDArray[np.float64] | None:
    layout = sequence.page_split_filter.get_params(page_id).page_layout
    if layout is None:
        return None
    layout = correct_page_layout_type(layout)
    outline = layout.get_page_outline(page_id.sub_page)
    if outline is None:
        return None

    width, height = size
    clipped = np.column_stack(
        [np.clip(outline[:, 0], 0, width), np.clip(outline[:, 1], 0, height)]
    ).astype(np.float64)
    extent = clipped.max(axis=0) - clipped.min(axis=0)
    return None if np.any(extent < _MIN_CROP_EDGE_PX) else clipped


def content_rect_of(sequence: StageSequence, page_id: PageId) -> Rect:
    """Return the stored content box of a page.

    Args:
        sequence (StageSequence): Stages holding the page's parameters.
        page_id (PageId): The page.

    Returns:
        Rect: The box in the coordinates of the Select Content stage; empty
        when no content was found or selected.
    """
    box = sequence.select_content_filter.get_params(page_id).content_box
    return Rect(x=box.x, y=box.y, width=box.width, height=box.height)


def _page_polygon(
    sequence: StageSequence, page_id: PageId, xform: ImageTransformation
) -> NDArray[np.float64]:
    layout_filter = sequence.page_layout_filter
    aggregate = layout_filter.get_aggregate_hard_size()
    return calc_page_polygon(
        xform,
        adapt_content_rect(xform, content_rect_of(sequence, page_id)),
        layout_filter.get_params(page_id),
        (aggregate.width_mm, aggregate.height_mm),
    )
