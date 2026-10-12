"""Page Layout filter image view.

Shows the deskewed page with three nested rectangles: the content box, the
content plus hard margins, and that plus the soft margins that bring every
aligned page to one size. Dragging a side of the middle rectangle sets the hard
margins. Alignment guides can be added from the context menu.
"""

from dataclasses import dataclass

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QPainter, QPen

from scantailor.app.ui.image_views.base import (
    Edge,
    ImageViewBase,
    point_to_segment_distance,
)
from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.models import Margins


@dataclass(eq=False)
class Guide:
    """A horizontal or vertical alignment guide.

    Attributes:
        position: Coordinate in virtual units, y for a horizontal guide.
        is_horizontal: True for a horizontal guide.
    """

    position: float
    is_horizontal: bool


class PageLayoutImageView(ImageViewBase):
    """Image view of the Page Layout stage with draggable hard margins.

    Signals:
        margins_set_locally: The user dragged a margin. Args: margins (Margins)
            in millimetres.
    """

    margins_set_locally = Signal(Margins)

    EDGE_HIT_DISTANCE = 10
    GUIDE_HIT_DISTANCE = 6

    _EDGES = (Edge.LEFT, Edge.RIGHT, Edge.TOP, Edge.BOTTOM)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._inner_rect = QRectF()
        self._middle_rect = QRectF()
        self._outer_rect = QRectF()
        self._mm_per_pixel = (1.0, 1.0)
        self._left_right_linked = False
        self._top_bottom_linked = False

        self._dragged_edge = Edge.NONE
        self._drag_offset = QPointF()
        self._drag_moved = False
        self._hovered_edge = Edge.NONE

        self._guides: list[Guide] = []
        self._dragged_guide: Guide | None = None
        self._hovered_guide: Guide | None = None
        self._context_pos = QPointF()
        self._show_middle_rect = True
        self._build_context_menu()
        self.setMouseTracking(True)

    def _build_context_menu(self) -> None:
        self._context_menu = QtWidgets.QMenu(self)
        self._add_horizontal_guide_action = QAction("Add Horizontal Guide", self)
        self._add_horizontal_guide_action.triggered.connect(
            lambda: self._add_guide(True)
        )
        self._add_vertical_guide_action = QAction("Add Vertical Guide", self)
        self._add_vertical_guide_action.triggered.connect(
            lambda: self._add_guide(False)
        )
        self._remove_guide_action = QAction("Remove Guide", self)
        self._remove_guide_action.triggered.connect(self._remove_hovered_guide)
        self._remove_all_guides_action = QAction("Remove All Guides", self)
        self._remove_all_guides_action.triggered.connect(self._remove_all_guides)
        self._show_middle_rect_action = QAction("Show Hard Margins", self)
        self._show_middle_rect_action.setCheckable(True)
        self._show_middle_rect_action.setChecked(True)
        self._show_middle_rect_action.triggered.connect(self._toggle_middle_rect)

        menu = self._context_menu
        menu.addAction(self._add_horizontal_guide_action)
        menu.addAction(self._add_vertical_guide_action)
        menu.addSeparator()
        menu.addAction(self._remove_guide_action)
        menu.addAction(self._remove_all_guides_action)
        menu.addSeparator()
        menu.addAction(self._show_middle_rect_action)

    def set_rectangles(
        self, inner_rect: QRectF, middle_rect: QRectF, outer_rect: QRectF
    ) -> None:
        """Show the three rectangles.

        Args:
            inner_rect (QRectF): Content box in virtual coordinates.
            middle_rect (QRectF): Content plus hard margins.
            outer_rect (QRectF): Content plus hard and soft margins.
        """
        self._inner_rect = QRectF(inner_rect)
        self._middle_rect = QRectF(middle_rect)
        self._outer_rect = QRectF(outer_rect)
        self.set_display_extent(self._outer_rect.united(self._middle_rect))
        self.update()

    def set_mm_per_pixel(self, horizontal: float, vertical: float) -> None:
        """Set how many millimetres one virtual pixel measures on the page.

        Args:
            horizontal (float): Millimetres per pixel along x.
            vertical (float): Millimetres per pixel along y.
        """
        self._mm_per_pixel = (horizontal, vertical)

    def set_margin_links(self, left_right: bool, top_bottom: bool) -> None:
        """Choose which opposite margins move together.

        Args:
            left_right (bool): Dragging the left or right side moves both.
            top_bottom (bool): Dragging the top or bottom side moves both.
        """
        self._left_right_linked = left_right
        self._top_bottom_linked = top_bottom

    def margins_mm(self) -> Margins:
        """Return the hard margins the middle rectangle currently shows.

        Returns:
            Margins: Distance of the middle rectangle from the content box, in
            millimetres, never negative.
        """
        x_scale, y_scale = self._mm_per_pixel
        inner, middle = self._inner_rect, self._middle_rect
        return Margins(
            left=max(0.0, (inner.left() - middle.left()) * x_scale),
            right=max(0.0, (middle.right() - inner.right()) * x_scale),
            top=max(0.0, (inner.top() - middle.top()) * y_scale),
            bottom=max(0.0, (middle.bottom() - inner.bottom()) * y_scale),
        )

    def _edge_at(self, widget_pos: QPointF) -> Edge:
        """Return the corner or side of the middle rectangle under a position.

        Corners take priority over sides; a corner is two edges combined.
        """
        if not self._show_middle_rect or self._middle_rect.isEmpty():
            return Edge.NONE
        rect = self._middle_rect
        for corner in (
            Edge.LEFT | Edge.TOP,
            Edge.RIGHT | Edge.TOP,
            Edge.LEFT | Edge.BOTTOM,
            Edge.RIGHT | Edge.BOTTOM,
        ):
            point = self.virtual_to_widget_point(
                QPointF(
                    rect.left() if corner & Edge.LEFT else rect.right(),
                    rect.top() if corner & Edge.TOP else rect.bottom(),
                )
            )
            offset = widget_pos - point
            if offset.x() ** 2 + offset.y() ** 2 <= self.EDGE_HIT_DISTANCE**2:
                return corner
        sides = {
            Edge.LEFT: (rect.topLeft(), rect.bottomLeft()),
            Edge.RIGHT: (rect.topRight(), rect.bottomRight()),
            Edge.TOP: (rect.topLeft(), rect.topRight()),
            Edge.BOTTOM: (rect.bottomLeft(), rect.bottomRight()),
        }
        best, best_distance = Edge.NONE, self.EDGE_HIT_DISTANCE
        for edge in self._EDGES:
            start, end = sides[edge]
            distance = point_to_segment_distance(
                widget_pos,
                self.virtual_to_widget_point(start),
                self.virtual_to_widget_point(end),
            )
            if distance <= best_distance:
                best, best_distance = edge, distance
        return best

    def _grab_offset(self, edge: Edge, grab: QPointF) -> QPointF:
        """Return where a grab point lies relative to the sides it will move."""
        rect = self._middle_rect
        offset_x = offset_y = 0.0
        if edge & Edge.LEFT:
            offset_x = grab.x() - rect.left()
        elif edge & Edge.RIGHT:
            offset_x = grab.x() - rect.right()
        if edge & Edge.TOP:
            offset_y = grab.y() - rect.top()
        elif edge & Edge.BOTTOM:
            offset_y = grab.y() - rect.bottom()
        return QPointF(offset_x, offset_y)

    def _drag_edge_to(self, edge: Edge, pos: QPointF) -> None:
        """Move the sides of the middle rectangle, never inside the content box."""
        inner = self._inner_rect
        rect = QRectF(self._middle_rect)
        if edge & Edge.LEFT:
            left = min(pos.x(), inner.left())
            delta = rect.left() - left
            rect.setLeft(left)
            if self._left_right_linked:
                rect.setRight(rect.right() + delta)
        elif edge & Edge.RIGHT:
            right = max(pos.x(), inner.right())
            delta = right - rect.right()
            rect.setRight(right)
            if self._left_right_linked:
                rect.setLeft(rect.left() - delta)
        if edge & Edge.TOP:
            top = min(pos.y(), inner.top())
            delta = rect.top() - top
            rect.setTop(top)
            if self._top_bottom_linked:
                rect.setBottom(rect.bottom() + delta)
        elif edge & Edge.BOTTOM:
            bottom = max(pos.y(), inner.bottom())
            delta = bottom - rect.bottom()
            rect.setBottom(bottom)
            if self._top_bottom_linked:
                rect.setTop(rect.top() - delta)
        rect = rect.united(inner)
        if rect != self._middle_rect:
            self._middle_rect = rect
            self._drag_moved = True

    @staticmethod
    def _cursor_for(edge: Edge) -> Qt.CursorShape:
        if edge in (Edge.LEFT | Edge.TOP, Edge.RIGHT | Edge.BOTTOM):
            return Qt.CursorShape.SizeFDiagCursor
        if edge in (Edge.RIGHT | Edge.TOP, Edge.LEFT | Edge.BOTTOM):
            return Qt.CursorShape.SizeBDiagCursor
        if edge in (Edge.LEFT, Edge.RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if edge in (Edge.TOP, Edge.BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def _add_guide(self, horizontal: bool) -> None:
        virtual = self.widget_to_virtual_point(self._context_pos)
        position = virtual.y() if horizontal else virtual.x()
        self._guides.append(Guide(position, horizontal))
        self.update()

    def _remove_hovered_guide(self) -> None:
        if self._hovered_guide is not None and self._hovered_guide in self._guides:
            self._guides.remove(self._hovered_guide)
        self._hovered_guide = None
        self.update()

    def _remove_all_guides(self) -> None:
        self._guides.clear()
        self._hovered_guide = None
        self.update()

    def _toggle_middle_rect(self, checked: bool) -> None:
        self._show_middle_rect = checked
        self.update()

    def _guide_widget_position(self, guide: Guide) -> float:
        point = (
            QPointF(0, guide.position)
            if guide.is_horizontal
            else QPointF(guide.position, 0)
        )
        mapped = self.virtual_to_widget_point(point)
        return mapped.y() if guide.is_horizontal else mapped.x()

    def _guide_at(self, widget_pos: QPointF) -> Guide | None:
        for guide in self._guides:
            coordinate = widget_pos.y() if guide.is_horizontal else widget_pos.x()
            if (
                abs(coordinate - self._guide_widget_position(guide))
                <= self.GUIDE_HIT_DISTANCE
            ):
                return guide
        return None

    def _paint_overlay(self, painter: QPainter) -> None:
        if not self._outer_rect.isEmpty():
            self._paint_rect(painter, self._outer_rect, "margin_outer")
        if self._show_middle_rect and not self._middle_rect.isEmpty():
            self._paint_rect(painter, self._middle_rect, "margin_middle")
        if not self._inner_rect.isEmpty():
            self._paint_rect(painter, self._inner_rect, "margin_inner")
        self._paint_guides(painter)

    def _paint_rect(self, painter: QPainter, rect: QRectF, role: str) -> None:
        border, fill = view_color(f"{role}_border"), view_color(f"{role}_fill")
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(border)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(fill)
        painter.drawRect(self.virtual_to_widget_rect(rect))
        painter.restore()

    def _paint_guides(self, painter: QPainter) -> None:
        viewport = self.viewport()
        assert viewport is not None
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for guide in self._guides:
            hovered = guide is self._hovered_guide
            pen = QPen(view_color("guide_active" if hovered else "guide"))
            pen.setWidth(2 if hovered else 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            at = self._guide_widget_position(guide)
            painter.drawLine(
                QLineF(0, at, viewport.width(), at)
                if guide.is_horizontal
                else QLineF(at, 0, at, viewport.height())
            )
        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pick up a guide or a margin side.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            guide = self._guide_at(pos)
            if guide is not None:
                self._dragged_guide = guide
                self.setCursor(
                    Qt.CursorShape.SizeVerCursor
                    if guide.is_horizontal
                    else Qt.CursorShape.SizeHorCursor
                )
                event.accept()
                return
            edge = self._edge_at(pos)
            if edge != Edge.NONE:
                self._dragged_edge = edge
                self._drag_moved = False
                self._drag_offset = self._grab_offset(
                    edge, self.widget_to_virtual_point(pos)
                )
                self.setCursor(self._cursor_for(edge))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Move the picked guide or side, or update the hover state.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        pos = event.position()
        virtual = self.widget_to_virtual_point(pos)
        if self._dragged_guide is not None:
            self._dragged_guide.position = (
                virtual.y() if self._dragged_guide.is_horizontal else virtual.x()
            )
            self.update()
            event.accept()
            return
        if self._dragged_edge != Edge.NONE:
            self._drag_edge_to(self._dragged_edge, virtual - self._drag_offset)
            self.update()
            event.accept()
            return

        if not self._panning:
            self._hovered_guide = self._guide_at(pos)
            self._hovered_edge = self._edge_at(pos)
            if self._hovered_guide is not None:
                self.setCursor(
                    Qt.CursorShape.SizeVerCursor
                    if self._hovered_guide.is_horizontal
                    else Qt.CursorShape.SizeHorCursor
                )
            else:
                self.setCursor(self._cursor_for(self._hovered_edge))
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop what was picked up and report changed margins.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragged_guide is not None:
                self._dragged_guide = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
            if self._dragged_edge != Edge.NONE:
                self._dragged_edge = Edge.NONE
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                if self._drag_moved:
                    self.margins_set_locally.emit(self.margins_mm())
                return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show the guide menu.

        Args:
            event (QtGui.QContextMenuEvent): The context menu event.
        """
        self._context_pos = QPointF(event.pos())
        self._hovered_guide = self._guide_at(self._context_pos)
        self._remove_guide_action.setEnabled(self._hovered_guide is not None)
        self._remove_all_guides_action.setEnabled(bool(self._guides))
        self._context_menu.exec(event.globalPos())
