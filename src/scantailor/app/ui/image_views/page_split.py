"""Page Split filter image view.

Shows the oriented source image with the cutting lines of the page layout: one
line splits a spread into two pages, two lines cut the offcut away from a single
page. A line is dragged by its body or by either end handle; dragging one handle
tilts it. Pages that were removed after a split are darkened and can be restored
from a button on the page.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import (
    QT_TRANSLATE_NOOP,
    QCoreApplication,
    QLineF,
    QPointF,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QPainter, QPen, QPolygonF

from scantailor.app.ui.icons import icon
from scantailor.app.ui.image_views.base import ImageViewBase, point_to_segment_distance
from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.models import SubPage
from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType

_DRAG_TIP = cast(
    str, QT_TRANSLATE_NOOP("page_split::ImageView", "Drag the line or the handles.")
)
_UNREMOVE_TIP = cast(
    str, QT_TRANSLATE_NOOP("page_split::UnremoveButton", "Restore removed page.")
)

Part = Literal["top", "bottom", "line"]


@dataclass
class _Drag:
    """A cutter being dragged, with the pointer and line as they were at the press.

    Attributes:
        index: Cutter being dragged.
        part: The end handle or the whole line.
        start_pos: Pointer position at the press, widget coordinates.
        start_line: The line as displayed at the press, widget coordinates.
        grab_offset: Horizontal distance from the pointer to the grabbed handle at
            the press, so the handle does not jump to the pointer.
        moved: Whether the line changed since the press.
    """

    index: int
    part: Part
    start_pos: QPointF
    start_line: QLineF
    grab_offset: float = 0.0
    moved: bool = False


def _x_at(line: QLineF, y: float) -> float:
    """Return where a line crosses a horizontal; a horizontal line gives its middle."""
    if line.y1() == line.y2():
        return (line.x1() + line.x2()) / 2
    slope = (line.x2() - line.x1()) / (line.y2() - line.y1())
    return line.x1() + (y - line.y1()) * slope


def _inscribed(line: QLineF, rect: QRectF) -> QLineF:
    """Return the line with its ends on the top and bottom of a rectangle.

    The ends keep the direction of the line but may not leave the rectangle
    sideways, which can change the angle.

    Args:
        line (QLineF): The line to extend or shorten.
        rect (QRectF): The rectangle.

    Returns:
        QLineF: Line from the top edge to the bottom edge.
    """
    left, right = rect.left(), rect.right()
    top_x = max(left, min(right, _x_at(line, rect.top())))
    bottom_x = max(left, min(right, _x_at(line, rect.bottom())))
    return QLineF(top_x, rect.top(), bottom_x, rect.bottom())


def outline_rect(layout: PageLayout) -> QRectF:
    """Return the outline a page layout covers, in oriented image pixels."""
    x, y, width, height = layout.outline
    return QRectF(x, y, width, height)


def layout_cutters(layout: PageLayout | None) -> list[QLineF]:
    """Return the cutters of a layout from the top to the bottom of its outline.

    Args:
        layout (PageLayout | None): The layout; None has no cutters.

    Returns:
        list[QLineF]: The cutters in oriented image pixels, left one first; empty
        for an uncut page.
    """
    if layout is None or layout.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT:
        return []
    outline = outline_rect(layout)
    stored = [layout.cutter1]
    if layout.layout_type == PageLayoutType.SINGLE_PAGE_CUT:
        stored.append(layout.cutter2)
    lines = [
        _inscribed(QLineF(QPointF(*line[0]), QPointF(*line[1])), outline)
        for line in stored
        if line is not None
    ]
    return sorted(lines, key=lambda line: line.x1() + line.x2())


def region_polygons(
    layout: PageLayout,
    cutters: list[QLineF],
    removed: Mapping[SubPage, bool],
) -> list[tuple[QPolygonF, str]]:
    """Return the regions a layout tints, with the view colour role of each.

    Args:
        layout (PageLayout): The layout, for its type and outline.
        cutters (list[QLineF]): Its cutters as ``layout_cutters`` returns them,
            or as the user is dragging them.
        removed (Mapping[SubPage, bool]): Which halves of a two-page layout were
            removed from the project; those get the offcut tint.

    Returns:
        list[tuple[QPolygonF, str]]: Polygons in oriented image pixels.
    """
    outline = outline_rect(layout)
    top, bottom = outline.top(), outline.bottom()

    def side(line: QLineF | None, x: float) -> tuple[QPointF, QPointF]:
        """Return the top and bottom of a bounding line, or of an outline side."""
        if line is None:
            return QPointF(x, top), QPointF(x, bottom)
        return line.p1(), line.p2()

    def between(first: QLineF | None, second: QLineF | None) -> QPolygonF:
        left = side(first, outline.left())
        right = side(second, outline.right())
        return QPolygonF([left[0], right[0], right[1], left[1]])

    kind = layout.layout_type
    if kind == PageLayoutType.SINGLE_PAGE_UNCUT or not cutters:
        return [(QPolygonF(outline), "split_page_left")]
    if kind == PageLayoutType.SINGLE_PAGE_CUT:
        return [(between(*cutters[:2]), "split_page_left")]
    return [
        (
            between(None, cutters[0]),
            "split_offcut" if removed.get(SubPage.LEFT_PAGE) else "split_page_left",
        ),
        (
            between(cutters[0], None),
            "split_offcut" if removed.get(SubPage.RIGHT_PAGE) else "split_page_right",
        ),
    ]


class PageSplitImageView(ImageViewBase):
    """Image view of the Page Split stage with draggable cutting lines.

    Layout coordinates are image coordinates of the displayed image.

    Signals:
        page_layout_set_locally: The user moved a line. Args: layout (PageLayout).
        unremove_requested: The user asked for a removed page back. Args:
            sub_page (SubPage), LEFT_PAGE or RIGHT_PAGE.
    """

    page_layout_set_locally = Signal(object)
    unremove_requested = Signal(object)

    HANDLE_RADIUS = 8
    HANDLE_HIT_RADIUS = 15
    LINE_HIT_DISTANCE = 10
    UNREMOVE_BUTTON_SIZE = 64

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._layout: PageLayout | None = None
        # Image-space cutters from the top edge of the outline to the bottom
        # edge, left to right.
        self._cutters: list[QLineF] = []
        self._drag: _Drag | None = None
        self._hover: tuple[int, Part] | None = None
        self._removed = {SubPage.LEFT_PAGE: False, SubPage.RIGHT_PAGE: False}
        self._hovered_unremove: SubPage | None = None
        self._unremove_icons = {
            False: icon("trash-2", tint="danger"),
            True: icon("archive-restore"),
        }
        self.setMouseTracking(True)

    @property
    def page_layout(self) -> PageLayout | None:
        """The layout being shown, including any unreleased drag."""
        return self._current_layout()

    def set_page_layout(self, layout: PageLayout | None) -> None:
        """Show a layout.

        Args:
            layout (PageLayout | None): Layout in image coordinates; None shows
                no lines.
        """
        self._layout = layout
        self._cutters = self._cutters_of(layout)
        self._drag = None
        self.update()

    def set_removed_pages(self, left: bool, right: bool) -> None:
        """Mark halves of a split page as removed from the project.

        Args:
            left (bool): The left half was removed.
            right (bool): The right half was removed.
        """
        self._removed = {SubPage.LEFT_PAGE: left, SubPage.RIGHT_PAGE: right}
        self._hovered_unremove = None
        self.update()

    @staticmethod
    def _outline_rect(layout: PageLayout) -> QRectF:
        return outline_rect(layout)

    @staticmethod
    def _cutters_of(layout: PageLayout | None) -> list[QLineF]:
        return layout_cutters(layout)

    def _current_layout(self) -> PageLayout | None:
        layout = self._layout
        if layout is None or not self._cutters:
            return layout
        lines = [
            ((line.x1(), line.y1()), (line.x2(), line.y2())) for line in self._cutters
        ]
        if layout.layout_type == PageLayoutType.TWO_PAGES:
            return layout.model_copy(update={"cutter1": lines[0], "cutter2": lines[0]})
        if len(lines) == 2:
            return layout.model_copy(update={"cutter1": lines[0], "cutter2": lines[1]})
        return layout

    def _page_rect_in_widget(self) -> QRectF:
        assert self._layout is not None
        return self.image_to_widget().mapRect(self._outline_rect(self._layout))

    def _valid_rect(self) -> QRectF:
        """Return the part of the page that is on screen."""
        viewport = self.viewport()
        assert viewport is not None
        return self._page_rect_in_widget().intersected(QRectF(viewport.rect()))

    def _widget_cutter(self, index: int) -> QLineF:
        """Return a cutter as displayed: ends on the visible top and bottom.

        The ends sit a handle radius inside the visible area so that a handle is
        reachable when the page is larger than the window. While one handle is
        dragged the other stays where the page really ends.
        """
        line = self.image_to_widget().map(self._cutters[index])
        valid = self._valid_rect()
        page = self._page_rect_in_widget()
        reduced = QRectF(
            page.left(),
            valid.top() + self.HANDLE_RADIUS,
            page.width(),
            max(0.0, valid.height() - 2 * self.HANDLE_RADIUS),
        )
        shown = _inscribed(line, reduced)
        drag = self._drag
        if drag is not None and drag.index == index and drag.part == "bottom":
            shown.setP1(line.p1())
        elif drag is not None and drag.index == index and drag.part == "top":
            shown.setP2(line.p2())
        return shown

    def line_handle_positions(self) -> list[tuple[QPointF, QPointF]]:
        """Return the end handles of every line.

        Returns:
            list[tuple[QPointF, QPointF]]: Top and bottom handle of each line in
            widget coordinates, left line first.
        """
        if self._layout is None:
            return []
        lines = [self._widget_cutter(i) for i in range(len(self._cutters))]
        return [(line.p1(), line.p2()) for line in lines]

    def _part_at(self, pos: QPointF) -> tuple[int, Part] | None:
        """Return the cutter handle or body under a widget position."""
        lines = self.line_handle_positions()
        for index, (top, bottom) in enumerate(lines):
            for part, handle in (("top", top), ("bottom", bottom)):
                if math.hypot(pos.x() - handle.x(), pos.y() - handle.y()) <= (
                    self.HANDLE_HIT_RADIUS
                ):
                    return index, part  # type: ignore[return-value]
        for index, (top, bottom) in enumerate(lines):
            if point_to_segment_distance(pos, top, bottom) <= self.LINE_HIT_DISTANCE:
                return index, "line"
        return None

    def _bounds_from_neighbours(
        self, index: int, valid: QRectF
    ) -> tuple[float, float, float, float]:
        """Return how far left and right the ends of a cutter may go.

        A cutter may not reach the other one, so the limits are where the other
        cutter crosses the top and bottom of the visible page.

        Returns:
            tuple[float, float, float, float]: Smallest and largest x at the top,
            then smallest and largest x at the bottom.
        """
        min_top = min_bottom = valid.left()
        max_top = max_bottom = valid.right()
        to_widget = self.image_to_widget()
        for other, cutter in enumerate(self._cutters):
            if other == index:
                continue
            line = to_widget.map(cutter)
            top_x = _x_at(line, valid.top())
            bottom_x = _x_at(line, valid.bottom())
            if any(not valid.left() <= x <= valid.right() for x in (top_x, bottom_x)):
                continue
            if index > other:
                min_top, min_bottom = top_x, bottom_x
            else:
                max_top, max_bottom = top_x, bottom_x
        return min_top, max_top, min_bottom, max_bottom

    def _move_handle(self, drag: _Drag, pos: QPointF) -> None:
        valid = self._valid_rect()
        min_top, max_top, min_bottom, max_bottom = self._bounds_from_neighbours(
            drag.index, valid
        )
        to_image = self.widget_to_image()
        line = QLineF(self._cutters[drag.index])
        target = pos.x() + drag.grab_offset
        if drag.part == "top":
            x = max(min_top, min(max_top, target))
            line.setP1(to_image.map(QPointF(x, valid.top())))
        else:
            x = max(min_bottom, min(max_bottom, target))
            line.setP2(to_image.map(QPointF(x, valid.bottom())))
        self._set_cutter(drag, line)

    def _move_line(self, drag: _Drag, pos: QPointF) -> None:
        valid = self._valid_rect()
        min_top, max_top, min_bottom, max_bottom = self._bounds_from_neighbours(
            drag.index, valid
        )
        line = drag.start_line.translated(pos - drag.start_pos)
        top_x, bottom_x = _x_at(line, valid.top()), _x_at(line, valid.bottom())
        left = max(min_top - top_x, min_bottom - bottom_x)
        right = max(top_x - max_top, bottom_x - max_bottom)
        if left > right and left > 0:
            line.translate(left, 0)
        elif right > 0:
            line.translate(-right, 0)
        self._set_cutter(drag, self.widget_to_image().map(line))

    def _set_cutter(self, drag: _Drag, line: QLineF) -> None:
        assert self._layout is not None
        inscribed = _inscribed(line, self._outline_rect(self._layout))
        if inscribed != self._cutters[drag.index]:
            self._cutters[drag.index] = inscribed
            drag.moved = True
            self.update()

    def _region_polygons(self) -> list[tuple[QPolygonF, str]]:
        """Return the tinted page regions, in image pixels, with their colour role."""
        if self._layout is None:
            return []
        return region_polygons(self._layout, self._cutters, self._removed)

    def _unremove_button_rect(self, page: SubPage) -> QRectF | None:
        """Return the restore button of a removed half in widget coordinates."""
        layout = self._layout
        if (
            layout is None
            or layout.layout_type != PageLayoutType.TWO_PAGES
            or not self._removed[page]
            or not self._cutters
        ):
            return None
        regions = self._region_polygons()
        left_box = regions[0][0].boundingRect()
        right_box = regions[1][0].boundingRect()
        middle = (left_box.right() + right_box.left()) / 2
        if page == SubPage.LEFT_PAGE:
            left_box.setRight(middle)
            center = left_box.center()
        else:
            right_box.setLeft(middle)
            center = right_box.center()
        size = self.UNREMOVE_BUTTON_SIZE
        rect = QRectF(0, 0, size, size)
        rect.moveCenter(self.image_to_widget().map(center))
        return rect

    def _unremove_button_at(self, pos: QPointF) -> SubPage | None:
        for page in (SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE):
            rect = self._unremove_button_rect(page)
            if rect is not None and rect.contains(pos):
                return page
        return None

    def _status_tip_at(
        self, widget_pos: QPointF, modifiers: Qt.KeyboardModifier
    ) -> str:
        if self._unremove_button_at(widget_pos) is not None:
            return QCoreApplication.translate(
                "page_split::UnremoveButton", _UNREMOVE_TIP
            )
        if self._part_at(widget_pos) is None:
            return ""
        return QCoreApplication.translate("page_split::ImageView", _DRAG_TIP)

    def _paint_overlay(self, painter: QPainter) -> None:
        if self._layout is None:
            return
        to_widget = self.image_to_widget()
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for polygon, role in self._region_polygons():
            painter.setBrush(view_color(role))
            painter.drawPolygon(to_widget.map(polygon))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        active = self._drag.index if self._drag is not None else None
        if active is None and self._hover is not None:
            active = self._hover[0]
        for index in range(len(self._cutters)):
            line = self._widget_cutter(index)
            highlighted = index == active
            painter.setPen(
                QPen(
                    view_color("split_line_active" if highlighted else "split_line"),
                    3 if highlighted else 2,
                )
            )
            painter.drawLine(line)
            painter.setPen(QPen(view_color("handle_ring"), 2))
            painter.setBrush(
                view_color("split_line_active" if highlighted else "split_handle")
            )
            for handle in (line.p1(), line.p2()):
                painter.drawEllipse(handle, self.HANDLE_RADIUS, self.HANDLE_RADIUS)
        painter.restore()
        self._paint_unremove_buttons(painter)

    def _paint_unremove_buttons(self, painter: QPainter) -> None:
        for page in (SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE):
            rect = self._unremove_button_rect(page)
            if rect is None:
                continue
            hovered = self._hovered_unremove == page
            self._unremove_icons[hovered].paint(painter, rect.toRect())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Restore a removed page, or pick up a line or handle.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            page = self._unremove_button_at(event.position())
            if page is not None:
                self._removed[page] = False
                self._hovered_unremove = None
                self.update()
                self.unremove_requested.emit(page)
                event.accept()
                return
            hit = self._part_at(event.position())
            if hit is not None:
                index, part = hit
                line = self._widget_cutter(index)
                handle = {"top": line.p1(), "bottom": line.p2()}.get(part)
                self._drag = _Drag(
                    index,
                    part,
                    event.position(),
                    line,
                    0.0 if handle is None else handle.x() - event.position().x(),
                )
                self.setCursor(Qt.CursorShape.SplitHCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Move the picked line or handle, or update the hover state.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        drag = self._drag
        if drag is not None:
            if drag.part == "line":
                self._move_line(drag, event.position())
            else:
                self._move_handle(drag, event.position())
            event.accept()
            return

        if not self._panning:
            pos = event.position()
            hovered = self._part_at(pos)
            button = self._unremove_button_at(pos)
            if hovered != self._hover or button != self._hovered_unremove:
                self._hover = hovered
                self._hovered_unremove = button
                if button is not None:
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                elif hovered is not None:
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop the line and report the new layout if it moved.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._drag is not None:
            moved = self._drag.moved
            self._drag = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.set_status_tip(
                self._status_tip_at(event.position(), event.modifiers())
            )
            event.accept()
            self.update()
            layout = self._current_layout()
            if moved and layout is not None:
                self._layout = layout
                self.page_layout_set_locally.emit(layout)
            return
        super().mouseReleaseEvent(event)
