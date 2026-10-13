"""Page transformations rebuilt from stored stage parameters.

The image views show the oriented source image through the transformation
that the stages before them define. This module builds those transformations
from the parameters alone, without detection and without touching pixels, so
it is cheap enough for the GUI thread. It applies the stages the same way
``core.pipeline`` does.
"""

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi, PageId, SubPage
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation, Rect
from scantailor.filters.fix_orientation.image_trim import merge_with_trim
from scantailor.filters.page_layout.page_rect import (
    adapt_content_rect,
    calc_page_polygon,
)
from scantailor.filters.page_layout.params import Params as PageLayoutParams
from scantailor.filters.page_split.page_layout import PageLayout
from scantailor.filters.page_split.page_layout_adapter import (
    correct_page_layout_type,
)
from scantailor.filters.select_content.params import Params as SelectContentParams

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
            ``FIX_ORIENTATION`` gives the identity, or the crop to the manual
            trim when the image has one, ``PAGE_SPLIT`` adds the crop to the
            page, ``DESKEW`` adds the rotation and ``PAGE_LAYOUT`` adds the
            crop to the final page.

    Returns:
        ImageTransformation: Maps oriented source pixels to the coordinates the
        stages after ``through`` work in.
    """
    width, height = oriented_size
    xform = ImageTransformation.from_size(width, height, transformation_dpi(dpi))
    trim_area = sequence.fix_orientation_filter.trim_crop_area(
        page_id.image_id, oriented_size
    )
    if trim_area is not None:
        xform.set_pre_crop_area(trim_area)
    if through < FilterStage.PAGE_SPLIT:
        return xform

    layout = sequence.page_split_filter.get_params(page_id).page_layout
    crop = (
        None
        if layout is None
        else page_crop_area(layout, page_id.sub_page, oriented_size, trim_area)
    )
    if crop is not None:
        xform.set_pre_crop_area(crop)
    if through < FilterStage.DESKEW:
        return xform

    deskew = sequence.deskew_filter
    deskew.apply_geometry(xform, deskew.get_params(page_id))
    if through >= FilterStage.PAGE_LAYOUT:
        xform.set_post_crop_area(_page_polygon(sequence, page_id, xform))
    return xform


def page_crop_area(
    layout: PageLayout,
    sub_page: SubPage,
    size: tuple[int, int],
    trim_area: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    """Return the region Page Split keeps of the oriented image for one page.

    Args:
        layout (PageLayout): The page's split layout; its type is corrected to
            the cutters it has.
        sub_page (SubPage): Which page of the layout.
        size (tuple[int, int]): Width and height in pixels of the oriented image.
        trim_area (NDArray[np.float64] | None): Region Fix Orientation's manual
            trim keeps, in the same coordinates; None when there is no trim.

    Returns:
        NDArray[np.float64] | None: Polygon in oriented source pixels, clipped to
        the image and to the trim; None when the layout has no such page or the
        clipped region is degenerate, in which case the whole image is kept.
    """
    layout = correct_page_layout_type(layout)
    outline = layout.get_page_outline(sub_page)
    if outline is None:
        return None

    width, height = size
    clipped = np.column_stack(
        [np.clip(outline[:, 0], 0, width), np.clip(outline[:, 1], 0, height)]
    ).astype(np.float64)
    extent = clipped.max(axis=0) - clipped.min(axis=0)
    if np.any(extent < _MIN_CROP_EDGE_PX):
        return None
    return merge_with_trim(clipped, trim_area)


def content_rect_of(sequence: StageSequence, page_id: PageId) -> Rect:
    """Return the stored content box of a page.

    Args:
        sequence (StageSequence): Stages holding the page's parameters.
        page_id (PageId): The page.

    Returns:
        Rect: The box in the coordinates of the Select Content stage; empty
        when no content was found or selected.
    """
    return selection_rects(
        sequence.select_content_filter.get_params(page_id),
        Rect(x=0, y=0, width=0, height=0),
    )[1]


def selection_rects(
    selection: SelectContentParams, view_rect: Rect
) -> tuple[Rect, Rect]:
    """Return the page and content rectangles Select Content stored.

    Args:
        selection (SelectContentParams): The page's Select Content parameters.
        view_rect (Rect): Rectangle of the transformed view; stands in for the
            page rectangle while no valid page box is stored.

    Returns:
        tuple[Rect, Rect]: The page rectangle and the content rectangle, in the
        coordinates of the Select Content stage.
    """
    page_box, content_box = selection.page_box, selection.content_box
    page_rect = (
        Rect(x=page_box.x, y=page_box.y, width=page_box.width, height=page_box.height)
        if page_box.is_valid()
        else view_rect
    )
    content_rect = Rect(
        x=content_box.x,
        y=content_box.y,
        width=content_box.width,
        height=content_box.height,
    )
    return page_rect, content_rect


def page_polygon(
    xform: ImageTransformation,
    content_rect: Rect,
    layout_params: PageLayoutParams,
    aggregate_hard_size_mm: tuple[float, float],
) -> NDArray[np.float64]:
    """Return the final page polygon Page Layout crops to.

    Args:
        xform (ImageTransformation): Transformation the content rectangle is in.
        content_rect (Rect): Content rectangle as Select Content stored it.
        layout_params (PageLayoutParams): The page's margins and alignment.
        aggregate_hard_size_mm (tuple[float, float]): Largest hard size among
            aligned pages.

    Returns:
        NDArray[np.float64]: (4, 2) polygon in the coordinates of ``xform``.
    """
    return calc_page_polygon(
        xform,
        adapt_content_rect(xform, content_rect),
        layout_params,
        aggregate_hard_size_mm,
    )


def _page_polygon(
    sequence: StageSequence, page_id: PageId, xform: ImageTransformation
) -> NDArray[np.float64]:
    layout_filter = sequence.page_layout_filter
    aggregate = layout_filter.get_aggregate_hard_size()
    return page_polygon(
        xform,
        content_rect_of(sequence, page_id),
        layout_filter.get_params(page_id),
        (aggregate.width_mm, aggregate.height_mm),
    )
