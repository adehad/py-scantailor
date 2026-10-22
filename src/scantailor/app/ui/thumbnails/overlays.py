"""Each stage's result drawn over its thumbnails.

Page Split tints the pages of the layout and draws its cutters, Deskew lays a
grid over the page, Select Content outlines the page and content boxes, and
Margins tints the margins around the content.

The overlay is painted into the thumbnail on the worker thread that rendered it,
so it is cached with the picture under the same parameter digest. The colours
are the light scheme's view colours whatever the scheme shown: a cached
thumbnail outlives a scheme change, and the overlay sits on the page's own
paper either way.
"""

from collections.abc import Callable

from PySide6.QtCore import QLineF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QTransform

from scantailor.app.ui.image_views.base import matrix_to_qtransform
from scantailor.app.ui.image_views.page_split import layout_cutters, region_polygons
from scantailor.app.ui.theme.tokens import LIGHT, qcolor
from scantailor.core.models import Dpi, PageId, SubPage
from scantailor.core.page_geometry import build_transformation, content_rect_of
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation

__all__ = ["paint_stage_overlay", "view_stage"]

# The deskew grid has this many cells along the thumbnail's longer side.
_GRID_CELLS = 20
# Fill opacity of boxes and margins; the views' own fills are too faint at
# thumbnail size.
_FILL_ALPHA = 60
_LINE_WIDTH = 1.5
_GRID_ALPHA = 70

# The geometry each stage's thumbnail shows: Page Split shows the whole image,
# as its view does, so that both pages and the cutters are visible.
_SHOWN_THROUGH = {
    FilterStage.PAGE_SPLIT: FilterStage.FIX_ORIENTATION,
    FilterStage.DESKEW: FilterStage.DESKEW,
    FilterStage.SELECT_CONTENT: FilterStage.DESKEW,
    FilterStage.PAGE_LAYOUT: FilterStage.PAGE_LAYOUT,
}


def view_stage(stage: FilterStage) -> FilterStage:
    """Return the stage whose geometry a thumbnail of ``stage`` is rendered with.

    Args:
        stage (FilterStage): The stage the thumbnail is shown at.

    Returns:
        FilterStage: ``FIX_ORIENTATION`` for Page Split, ``stage`` otherwise.
    """
    return FilterStage.FIX_ORIENTATION if stage == FilterStage.PAGE_SPLIT else stage


def paint_stage_overlay(
    thumbnail: QImage,
    stage: FilterStage,
    page_id: PageId,
    sequence: StageSequence,
    source: tuple[tuple[int, int], Dpi],
    removed_halves: tuple[bool, bool] = (False, False),
) -> QImage:
    """Return a copy of a thumbnail with the stage's result drawn over it.

    Safe to call from worker threads. The stage's parameters are only read.

    Args:
        thumbnail (QImage): The page rendered at the stage, through the geometry
            ``view_stage`` names; not modified.
        stage (FilterStage): The stage shown.
        page_id (PageId): The page.
        sequence (StageSequence): Stages holding the page's parameters.
        source (tuple[tuple[int, int], Dpi]): Width and height of the source
            after Fix Orientation, and the resolution it was decoded with, as
            the render used them.
        removed_halves (tuple[bool, bool]): Whether the left and the right half
            of the page's image were removed from the project.

    Returns:
        QImage: The overlaid copy in an RGB format, or ``thumbnail`` itself for
        a stage without an overlay or a page without a result there.
    """
    painters: dict[FilterStage, Callable[[QPainter, _Frame], bool]] = {
        FilterStage.PAGE_SPLIT: lambda painter, frame: _paint_split(
            painter, frame, sequence, page_id, removed_halves
        ),
        FilterStage.DESKEW: _paint_grid,
        FilterStage.SELECT_CONTENT: lambda painter, frame: _paint_boxes(
            painter, frame, sequence, page_id
        ),
        FilterStage.PAGE_LAYOUT: lambda painter, frame: _paint_margins(
            painter, frame, sequence, page_id
        ),
    }
    paint = painters.get(stage)
    if paint is None or thumbnail.isNull():
        return thumbnail
    oriented_size, dpi = source
    xform = build_transformation(
        sequence, page_id, oriented_size, dpi, _SHOWN_THROUGH[stage]
    )
    image = thumbnail.convertToFormat(QImage.Format.Format_RGB32)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        drawn = paint(painter, _Frame(xform, image.width(), image.height()))
    finally:
        painter.end()
    return image if drawn else thumbnail


class _Frame:
    """Maps the stage's virtual coordinates onto the thumbnail's pixels."""

    def __init__(self, xform: ImageTransformation, width: int, height: int) -> None:
        rect = xform.resulting_rect
        self.width = width
        self.height = height
        self.bounds = QRectF(0, 0, width, height)
        self.from_virtual = QTransform.fromTranslate(
            -rect.x, -rect.y
        ) * QTransform.fromScale(
            width / max(rect.width, 1e-9), height / max(rect.height, 1e-9)
        )
        self.from_source = matrix_to_qtransform(xform.transform) * self.from_virtual


def _color(name: str, alpha: int | None = None) -> QColor:
    color = qcolor(getattr(LIGHT.view, name))
    if alpha is not None:
        color.setAlpha(alpha)
    return color


def _pen(name: str, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(_color(name), _LINE_WIDTH)
    pen.setStyle(style)
    return pen


def _paint_split(
    painter: QPainter,
    frame: _Frame,
    sequence: StageSequence,
    page_id: PageId,
    removed_halves: tuple[bool, bool],
) -> bool:
    layout = sequence.page_split_filter.get_params(page_id).page_layout
    if layout is None:
        return False
    cutters = layout_cutters(layout)
    removed = {
        SubPage.LEFT_PAGE: removed_halves[0],
        SubPage.RIGHT_PAGE: removed_halves[1],
    }
    painter.setPen(Qt.PenStyle.NoPen)
    for polygon, role in region_polygons(layout, cutters, removed):
        painter.setBrush(_color(role))
        painter.drawPolygon(frame.from_source.map(polygon))
    painter.setPen(_pen("split_line"))
    for cutter in cutters:
        painter.drawLine(frame.from_source.map(cutter))
    return True


def _paint_grid(painter: QPainter, frame: _Frame) -> bool:
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setPen(QPen(_color("deskew_grid", _GRID_ALPHA), 1))
    step = max(frame.width, frame.height) / _GRID_CELLS
    center = frame.bounds.center()

    def offsets(origin: float, extent: float) -> list[float]:
        before = int(origin // step)
        after = int((extent - origin) // step)
        return [origin + index * step for index in range(-before, after + 1)]

    for y in offsets(center.y(), frame.height):
        painter.drawLine(QLineF(0, y, frame.width, y))
    for x in offsets(center.x(), frame.width):
        painter.drawLine(QLineF(x, 0, x, frame.height))
    return True


def _paint_boxes(
    painter: QPainter, frame: _Frame, sequence: StageSequence, page_id: PageId
) -> bool:
    params = sequence.select_content_filter.get_params(page_id)
    page_box = params.page_box
    content_box = params.content_box
    page_rect = (
        frame.from_virtual.mapRect(
            QRectF(page_box.x, page_box.y, page_box.width, page_box.height)
        )
        if page_box.is_valid()
        else frame.bounds
    )
    drawn = False
    if not params.is_page_disabled() and page_box.is_valid():
        painter.setPen(_pen("page_border"))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(page_rect)
        drawn = True
    if content_box.is_valid():
        content = frame.from_virtual.mapRect(
            QRectF(content_box.x, content_box.y, content_box.width, content_box.height)
        )
        painter.setPen(_pen("content_border"))
        painter.setBrush(_color("content_fill", _FILL_ALPHA))
        painter.drawRect(content.intersected(page_rect))
        drawn = True
    return drawn


def _paint_margins(
    painter: QPainter, frame: _Frame, sequence: StageSequence, page_id: PageId
) -> bool:
    aligned = sequence.page_layout_filter.get_params(
        page_id
    ).alignment.aligned_with_others()
    # Pages matched to the others' size get the outer rectangle's colour, the
    # ones left at their own size the content's, as in the Margins view.
    role = "margin_outer" if aligned else "margin_inner"
    content = content_rect_of(sequence, page_id)
    outline = QRectF(frame.bounds).adjusted(0.5, 0.5, -0.5, -0.5)
    fill = _color(f"{role}_fill", _FILL_ALPHA)
    if content.is_empty():
        painter.fillRect(frame.bounds, fill)
    else:
        inner = frame.from_virtual.mapRect(
            QRectF(content.x, content.y, content.width, content.height)
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        for band in _bands_around(frame.bounds, inner):
            painter.drawRect(band)
        painter.setPen(_pen(f"{role}_border"))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(inner)
    painter.setPen(
        _pen(
            f"{role}_border",
            Qt.PenStyle.DashLine if aligned else Qt.PenStyle.SolidLine,
        )
    )
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(outline)
    return True


def _bands_around(outer: QRectF, inner: QRectF) -> list[QRectF]:
    """Return the rectangles that cover ``outer`` except ``inner``."""
    inner = inner.intersected(outer)
    if inner.isEmpty():
        return [outer]
    return [
        QRectF(outer.left(), outer.top(), outer.width(), inner.top() - outer.top()),
        QRectF(
            outer.left(), inner.bottom(), outer.width(), outer.bottom() - inner.bottom()
        ),
        QRectF(outer.left(), inner.top(), inner.left() - outer.left(), inner.height()),
        QRectF(
            inner.right(), inner.top(), outer.right() - inner.right(), inner.height()
        ),
    ]
