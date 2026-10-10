"""Page rectangle geometry for the page layout stage.

Margins are specified in millimetres, content and page rectangles live in the
coordinates of an ImageTransformation. These helpers convert between the two and
work out the final page polygon: content plus hard margins plus the soft margins
that bring every aligned page to the same physical size.
"""

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi, Margins
from scantailor.core.transformation import ImageTransformation, Rect
from scantailor.filters.page_layout.alignment import (
    Alignment,
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_layout.params import ContentRect, Params

MM_PER_INCH = 25.4

# Tolerance below which a margin distance is treated as no space at all.
_MIN_SPACE_PX = 1.0

# Half extent of the stand-in rectangle used when no content was found.
_EMPTY_CONTENT_HALF_EXTENT_PX = 0.01

# Auto alignment snaps to a side when the other side has this many times more room.
_AUTO_ALIGN_RATIO = 3


def adapt_content_rect(xform: ImageTransformation, content_rect: Rect) -> Rect:
    """Return the content rectangle, or a tiny one at the page centre if empty.

    Args:
        xform (ImageTransformation): Transformation whose resulting rect
            provides the fallback centre.
        content_rect (Rect): Content rectangle in resulting coordinates.

    Returns:
        Rect: content_rect when it has area, otherwise a near-point rectangle.
    """
    if not content_rect.is_empty():
        return content_rect
    cx, cy = xform.resulting_rect.center()
    extent = 2 * _EMPTY_CONTENT_HALF_EXTENT_PX
    return Rect(
        x=cx - _EMPTY_CONTENT_HALF_EXTENT_PX,
        y=cy - _EMPTY_CONTENT_HALF_EXTENT_PX,
        width=extent,
        height=extent,
    )


def rect_size_mm(xform: ImageTransformation, rect: Rect) -> tuple[float, float]:
    """Measure a rectangle in millimetres on the original image.

    Args:
        xform (ImageTransformation): Transformation the rectangle is expressed in.
        rect (Rect): Rectangle in resulting coordinates.

    Returns:
        tuple[float, float]: (width_mm, height_mm) measured along the rectangle's
        top and left edges after undoing rotation.
    """
    corners_mm = _pixels_to_mm(
        xform.map_polygon_back(rect.to_polygon()), xform.orig_dpi
    )
    return (
        float(np.linalg.norm(corners_mm[1] - corners_mm[0])),
        float(np.linalg.norm(corners_mm[3] - corners_mm[0])),
    )


def calc_margins_mm(
    xform: ImageTransformation, page_rect: Rect, content_rect: Rect
) -> Margins:
    """Derive hard margins from where the content sits inside the page.

    Args:
        xform (ImageTransformation): Transformation both rectangles are in.
        page_rect (Rect): Page rectangle in resulting coordinates.
        content_rect (Rect): Content rectangle in resulting coordinates.

    Returns:
        Margins: Millimetre margins that, added to the content size, reproduce
        the page size, split in proportion to the pixel distances.
    """
    content_w, content_h = rect_size_mm(xform, content_rect)
    page_w, page_h = rect_size_mm(xform, page_rect)
    spare_w_mm = page_w - content_w
    spare_h_mm = page_h - content_h

    left = content_rect.left - page_rect.left
    right = page_rect.right - content_rect.right
    top = content_rect.top - page_rect.top
    bottom = page_rect.bottom - content_rect.bottom

    h_space = left + right
    v_space = top + bottom
    return Margins(
        left=0.0 if h_space < _MIN_SPACE_PX else left * spare_w_mm / h_space,
        right=0.0 if h_space < _MIN_SPACE_PX else right * spare_w_mm / h_space,
        top=0.0 if v_space < _MIN_SPACE_PX else top * spare_h_mm / v_space,
        bottom=0.0 if v_space < _MIN_SPACE_PX else bottom * spare_h_mm / v_space,
    )


def calc_soft_margins_mm(
    hard_size_mm: tuple[float, float],
    aggregate_size_mm: tuple[float, float],
    alignment: Alignment,
    content_rect: Rect,
    content_size_mm: tuple[float, float],
    page_rect: Rect,
) -> Margins:
    """Distribute the size shortfall against the aggregate over the sides.

    Args:
        hard_size_mm (tuple[float, float]): (width, height) of content plus hard
            margins.
        aggregate_size_mm (tuple[float, float]): Largest hard size among aligned
            pages.
        alignment (Alignment): Where the shortfall goes.
        content_rect (Rect): Content rectangle in pixels.
        content_size_mm (tuple[float, float]): Content size in millimetres.
        page_rect (Rect): Page rectangle in pixels.

    Returns:
        Margins: Soft margins in millimetres; all zero for a page that is not
        aligned with others.
    """
    if alignment.is_null:
        return Margins()

    delta_w = aggregate_size_mm[0] - hard_size_mm[0]
    delta_h = aggregate_size_mm[1] - hard_size_mm[1]

    left_border, top_border = 0.0, 0.0
    right_border, bottom_border = delta_w, delta_h

    has_geometry = (
        content_size_mm[0] > 0
        and content_size_mm[1] > 0
        and not content_rect.is_empty()
        and not page_rect.is_empty()
    )
    if has_geometry:
        px_per_mm_x = content_rect.width / content_size_mm[0]
        px_per_mm_y = content_rect.height / content_size_mm[1]
        page_w_mm = page_rect.width / px_per_mm_x
        page_h_mm = page_rect.height / px_per_mm_y
        center_x, center_y = content_rect.center()
        center_x_mm = (center_x - page_rect.left) / page_rect.width * page_w_mm
        center_y_mm = (center_y - page_rect.top) / page_rect.height * page_h_mm

        if delta_w > 0:
            left_border = float(
                np.clip(
                    center_x_mm / page_w_mm * aggregate_size_mm[0]
                    - hard_size_mm[0] / 2,
                    0.0,
                    delta_w,
                )
            )
            right_border = delta_w - left_border
        if delta_h > 0:
            top_border = float(
                np.clip(
                    center_y_mm / page_h_mm * aggregate_size_mm[1]
                    - hard_size_mm[1] / 2,
                    0.0,
                    delta_h,
                )
            )
            bottom_border = delta_h - top_border

    left = right = top = bottom = 0.0
    if delta_w > 0:
        left, right = _split_shortfall(
            delta_w,
            _resolve_horizontal(alignment.horizontal, left_border, right_border),
            left_border,
            right_border,
        )
    if delta_h > 0:
        top, bottom = _split_shortfall(
            delta_h,
            _resolve_vertical(alignment.vertical, top_border, bottom_border),
            top_border,
            bottom_border,
        )
    return Margins(left=left, right=right, top=top, bottom=bottom)


def calc_page_polygon(
    xform: ImageTransformation,
    content_rect: Rect,
    params: Params,
    aggregate_hard_size_mm: tuple[float, float],
) -> NDArray[np.float64]:
    """Compute the page outline in the transformation's resulting coordinates.

    Args:
        xform (ImageTransformation): Transformation the content rectangle is in.
        content_rect (Rect): Non-empty content rectangle in resulting
            coordinates.
        params (Params): Page layout parameters with margins, alignment and the
            stored content and page rectangles.
        aggregate_hard_size_mm (tuple[float, float]): Largest hard size among
            aligned pages.

    Returns:
        NDArray[np.float64]: (4, 2) polygon, top-left first, whose bounding
        rectangle origin is shifted to whole pixels so the later crop lands on
        the pixel grid.
    """
    dpi = xform.orig_dpi
    content_mm = _pixels_to_mm(xform.map_polygon_back(content_rect.to_polygon()), dpi)
    with_hard = _extend_with_margins(content_mm, params.hard_margins_mm)

    hard_size_mm = (
        float(np.linalg.norm(with_hard[1] - with_hard[0])),
        float(np.linalg.norm(with_hard[3] - with_hard[0])),
    )
    soft = calc_soft_margins_mm(
        hard_size_mm,
        aggregate_hard_size_mm,
        params.alignment,
        _rect_from_params(params.content_rect),
        (params.content_size_mm.width_mm, params.content_size_mm.height_mm),
        _rect_from_params(params.page_rect),
    )
    page_mm = _extend_with_margins(with_hard, soft)

    page_polygon = xform.map_polygon(_mm_to_pixels(page_mm, dpi))
    return _shift_to_rounded_origin(page_polygon)


def _rect_from_params(rect: ContentRect) -> Rect:
    """Copy a stored rectangle into a Rect."""
    return Rect(x=rect.x, y=rect.y, width=rect.width, height=rect.height)


def _mm_scale(dpi: Dpi) -> NDArray[np.float64]:
    """Per-axis millimetres per pixel."""
    return np.array(
        [MM_PER_INCH / dpi.horizontal, MM_PER_INCH / dpi.vertical], dtype=np.float64
    )


def _pixels_to_mm(points: NDArray[np.float64], dpi: Dpi) -> NDArray[np.float64]:
    return points * _mm_scale(dpi)


def _mm_to_pixels(points: NDArray[np.float64], dpi: Dpi) -> NDArray[np.float64]:
    return points / _mm_scale(dpi)


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 0 else vector


def _extend_with_margins(
    polygon: NDArray[np.float64], margins: Margins
) -> NDArray[np.float64]:
    """Grow a (possibly rotated) rectangle outwards along its own edges.

    Args:
        polygon (NDArray[np.float64]): (4, 2) rectangle, top-left first, clockwise.
        margins (Margins): Amount to add on each side, in the polygon's units.

    Returns:
        NDArray[np.float64]: Extended copy of the polygon.
    """
    down = _unit(polygon[3] - polygon[0])
    right = _unit(polygon[1] - polygon[0])
    extended = polygon.copy()
    extended[0] += -down * margins.top - right * margins.left
    extended[1] += -down * margins.top + right * margins.right
    extended[2] += down * margins.bottom + right * margins.right
    extended[3] += down * margins.bottom - right * margins.left
    return extended


def _shift_to_rounded_origin(polygon: NDArray[np.float64]) -> NDArray[np.float64]:
    origin = polygon.min(axis=0)
    return polygon - (origin - np.round(origin))


def _resolve_horizontal(
    mode: HorizontalAlignment, left_border: float, right_border: float
) -> HorizontalAlignment:
    if mode != HorizontalAlignment.HAUTO:
        return mode
    if _AUTO_ALIGN_RATIO * left_border < right_border:
        return HorizontalAlignment.LEFT
    if _AUTO_ALIGN_RATIO * right_border < left_border:
        return HorizontalAlignment.RIGHT
    return HorizontalAlignment.HCENTER


def _resolve_vertical(
    mode: VerticalAlignment, top_border: float, bottom_border: float
) -> VerticalAlignment:
    if mode != VerticalAlignment.VAUTO:
        return mode
    if _AUTO_ALIGN_RATIO * top_border < bottom_border:
        return VerticalAlignment.TOP
    if _AUTO_ALIGN_RATIO * bottom_border < top_border:
        return VerticalAlignment.BOTTOM
    return VerticalAlignment.VCENTER


def _split_shortfall(
    delta: float,
    mode: HorizontalAlignment | VerticalAlignment,
    before_border: float,
    after_border: float,
) -> tuple[float, float]:
    """Return the (before, after) margins for one axis.

    Args:
        delta: Total shortfall to distribute.
        mode: Resolved alignment for this axis.
        before_border: Share placed before the content under original-position
            alignment.
        after_border: Share placed after the content under original-position
            alignment.

    Returns:
        (before, after) margins, where before is left/top.
    """
    if mode in (HorizontalAlignment.LEFT, VerticalAlignment.TOP):
        return 0.0, delta
    if mode in (HorizontalAlignment.RIGHT, VerticalAlignment.BOTTOM):
        return delta, 0.0
    if mode in (HorizontalAlignment.HCENTER, VerticalAlignment.VCENTER):
        return delta / 2, delta / 2
    return before_border, after_border
