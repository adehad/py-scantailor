"""Select Content filter image view.

Shows the deskewed page with its content box and, when page detection is
enabled, its page box. Boxes are resized by their corners and edges and moved
by their body; a context menu and a double click create or remove the
content box.
"""

import math
from enum import Enum, auto
from typing import ClassVar

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPen

from scantailor.app.ui.image_views.base import (
    Edge,
    ImageViewBase,
    point_to_segment_distance,
)


class BoxKind(Enum):
    """The two boxes the view edits."""

    CONTENT = auto()
    PAGE = auto()


class SelectContentImageView(ImageViewBase):
    """Image view of the Select Content stage with editable boxes.

    Boxes are in virtual coordinates, the coordinates the stage stores.

    Signals:
        manual_content_rect_set: The user changed the content box. Args:
            rect (QRectF); empty when the box was removed.
        manual_page_rect_set: The user changed the page box. Args: rect (QRectF).
    """

    manual_content_rect_set = Signal(QRectF)
    manual_page_rect_set = Signal(QRectF)

    CORNER_HIT_RADIUS = 10
    EDGE_HIT_DISTANCE = 8
    MIN_BOX_SIZE = 10  # In virtual pixels
    NEW_BOX_FRACTION = 0.7  # Size of a new content box relative to the page

    BOX_COLORS: ClassVar[dict[BoxKind, tuple[QColor, QColor]]] = {
        BoxKind.CONTENT: (QColor(0, 0, 255, 180), QColor(0, 0, 255, 30)),
        BoxKind.PAGE: (QColor(255, 0, 0, 180), QColor(255, 0, 0, 20)),
    }
    HANDLE_COLOR = QColor(0, 0, 255, 200)

    _CORNERS = (
        Edge.LEFT | Edge.TOP,
        Edge.RIGHT | Edge.TOP,
        Edge.LEFT | Edge.BOTTOM,
        Edge.RIGHT | Edge.BOTTOM,
    )
    _CURSORS: ClassVar[dict[Edge, Qt.CursorShape]] = {
        Edge.LEFT | Edge.TOP: Qt.CursorShape.SizeFDiagCursor,
        Edge.RIGHT | Edge.BOTTOM: Qt.CursorShape.SizeFDiagCursor,
        Edge.RIGHT | Edge.TOP: Qt.CursorShape.SizeBDiagCursor,
        Edge.LEFT | Edge.BOTTOM: Qt.CursorShape.SizeBDiagCursor,
        Edge.LEFT: Qt.CursorShape.SizeHorCursor,
        Edge.RIGHT: Qt.CursorShape.SizeHorCursor,
        Edge.TOP: Qt.CursorShape.SizeVerCursor,
        Edge.BOTTOM: Qt.CursorShape.SizeVerCursor,
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._rects = {BoxKind.CONTENT: QRectF(), BoxKind.PAGE: QRectF()}
        self._page_rect_enabled = False
        # Box being dragged, the edges it is resized by (NONE moves the box) and
        # the grab point relative to the box origin.
        self._drag: tuple[BoxKind, Edge] | None = None
        self._drag_offset = QPointF()
        self._drag_moved = False
        self._page_rect_changed_by_content = False
        self._hover: tuple[BoxKind, Edge] | None = None
        self._build_context_menus()
        self.setMouseTracking(True)

    def _build_context_menus(self) -> None:
        self._no_content_menu = QtWidgets.QMenu(self)
        create = QAction("Create Content Box", self)
        create.setShortcut(QKeySequence("Ins"))
        create.triggered.connect(lambda: self._create_content_box(None))
        self._no_content_menu.addAction(create)
        self.addAction(create)

        self._have_content_menu = QtWidgets.QMenu(self)
        remove = QAction("Remove Content Box", self)
        remove.setShortcut(QKeySequence("Delete"))
        remove.triggered.connect(self._remove_content_box)
        self._have_content_menu.addAction(remove)
        self.addAction(remove)

    @property
    def content_rect(self) -> QRectF:
        """The content box in virtual coordinates; empty when there is none."""
        return QRectF(self._rects[BoxKind.CONTENT])

    def set_content_rect(self, rect: QRectF) -> None:
        """Show a content box.

        Args:
            rect (QRectF): Box in virtual coordinates; empty hides it.
        """
        self._rects[BoxKind.CONTENT] = QRectF(rect)
        self.update()

    @property
    def page_rect(self) -> QRectF:
        """The page box in virtual coordinates."""
        return QRectF(self._rects[BoxKind.PAGE])

    def set_page_rect(self, rect: QRectF) -> None:
        """Show a page box.

        Args:
            rect (QRectF): Box in virtual coordinates.
        """
        self._rects[BoxKind.PAGE] = QRectF(rect)
        self.update()

    def set_page_rect_enabled(self, enabled: bool) -> None:
        """Show and allow editing of the page box.

        Args:
            enabled (bool): True to show the page box.
        """
        self._page_rect_enabled = enabled
        self.update()

    def _editable_boxes(self) -> list[BoxKind]:
        boxes = [BoxKind.CONTENT]
        if self._page_rect_enabled:
            boxes.append(BoxKind.PAGE)
        return [kind for kind in boxes if not self._rects[kind].isEmpty()]

    def _create_content_box(self, center: QPointF | None) -> None:
        """Create a content box around a virtual point, or the page centre."""
        if self._drag is not None or not self._rects[BoxKind.CONTENT].isEmpty():
            return
        area = self._virtual_display_area
        width = area.width() * self.NEW_BOX_FRACTION
        height = area.height() * self.NEW_BOX_FRACTION
        centre = area.center() if center is None else center
        box = QRectF(
            centre.x() - width / 2, centre.y() - height / 2, width, height
        ).intersected(area)
        self._rects[BoxKind.CONTENT] = box
        self._describe_content_in_page()
        self.update()
        self.manual_content_rect_set.emit(box)
        self._report_page_rect_if_grown()

    def _describe_content_in_page(self) -> None:
        """Grow the page box to contain the content box."""
        if not self._page_rect_enabled or self._rects[BoxKind.CONTENT].isEmpty():
            return
        page = self._rects[BoxKind.PAGE]
        grown = page.united(self._rects[BoxKind.CONTENT])
        if grown != page:
            self._rects[BoxKind.PAGE] = grown
            self._page_rect_changed_by_content = True

    def _report_page_rect_if_grown(self) -> None:
        if self._page_rect_changed_by_content:
            self._page_rect_changed_by_content = False
            self.manual_page_rect_set.emit(QRectF(self._rects[BoxKind.PAGE]))

    def _remove_content_box(self) -> None:
        if self._drag is not None:
            return
        self._rects[BoxKind.CONTENT] = QRectF()
        self.update()
        self.manual_content_rect_set.emit(QRectF())

    def _hit_test(
        self, widget_pos: QPointF, move_boxes: bool = False
    ) -> tuple[BoxKind, Edge] | None:
        """Find the box handle under a widget position.

        Args:
            widget_pos (QPointF): Position in widget coordinates.
            move_boxes (bool): Also hit box bodies, which only move with Shift
                held; without it a drag in the body pans.

        Returns:
            The box and the edges a drag would move: two edges for a corner, one
            for a side, NONE for the body; None when nothing is hit.
        """
        boxes = self._editable_boxes()
        for kind in boxes:
            rect = self._rects[kind]
            for corner in self._CORNERS:
                point = self.virtual_to_widget_point(self._corner_point(rect, corner))
                offset = widget_pos - point
                if math.hypot(offset.x(), offset.y()) <= self.CORNER_HIT_RADIUS:
                    return kind, corner
        for kind in boxes:
            rect = self._rects[kind]
            for edge in (Edge.LEFT, Edge.RIGHT, Edge.TOP, Edge.BOTTOM):
                start, end = self._edge_points(rect, edge)
                distance = point_to_segment_distance(
                    widget_pos,
                    self.virtual_to_widget_point(start),
                    self.virtual_to_widget_point(end),
                )
                if distance <= self.EDGE_HIT_DISTANCE:
                    return kind, edge
        virtual = self.widget_to_virtual_point(widget_pos)
        if move_boxes:
            for kind in boxes:
                if self._rects[kind].contains(virtual):
                    return kind, Edge.NONE
        return None

    @staticmethod
    def _corner_point(rect: QRectF, corner: Edge) -> QPointF:
        return QPointF(
            rect.left() if corner & Edge.LEFT else rect.right(),
            rect.top() if corner & Edge.TOP else rect.bottom(),
        )

    @staticmethod
    def _edge_points(rect: QRectF, edge: Edge) -> tuple[QPointF, QPointF]:
        if edge == Edge.LEFT:
            return rect.topLeft(), rect.bottomLeft()
        if edge == Edge.RIGHT:
            return rect.topRight(), rect.bottomRight()
        if edge == Edge.TOP:
            return rect.topLeft(), rect.topRight()
        return rect.bottomLeft(), rect.bottomRight()

    @staticmethod
    def _grab_offset(rect: QRectF, edges: Edge, grab: QPointF) -> QPointF:
        """Return where a grab point lies relative to the edges it will move.

        Dragging then moves the edges by the pointer's travel instead of
        snapping them to the pointer. Moving the whole box measures from its
        top-left corner.
        """
        if edges == Edge.NONE:
            return QPointF(grab.x() - rect.x(), grab.y() - rect.y())
        x_edge = rect.left() if edges & Edge.LEFT else rect.right()
        y_edge = rect.top() if edges & Edge.TOP else rect.bottom()
        return QPointF(
            grab.x() - x_edge if edges & (Edge.LEFT | Edge.RIGHT) else 0.0,
            grab.y() - y_edge if edges & (Edge.TOP | Edge.BOTTOM) else 0.0,
        )

    def _resized(self, rect: QRectF, edges: Edge, pos: QPointF) -> QRectF:
        """Return the rectangle with the given edges moved towards a virtual point."""
        area = self._virtual_display_area
        x = max(area.left(), min(area.right(), pos.x()))
        y = max(area.top(), min(area.bottom(), pos.y()))
        result = QRectF(rect)
        if edges & Edge.LEFT:
            result.setLeft(min(x, rect.right() - self.MIN_BOX_SIZE))
        if edges & Edge.RIGHT:
            result.setRight(max(x, rect.left() + self.MIN_BOX_SIZE))
        if edges & Edge.TOP:
            result.setTop(min(y, rect.bottom() - self.MIN_BOX_SIZE))
        if edges & Edge.BOTTOM:
            result.setBottom(max(y, rect.top() + self.MIN_BOX_SIZE))
        return result

    def _moved(self, kind: BoxKind, rect: QRectF, origin: QPointF) -> QRectF:
        """Return the rectangle with its top-left corner moved.

        The content box stays inside the page; the page box keeps containing
        the content box.
        """
        moved = QRectF(origin.x(), origin.y(), rect.width(), rect.height())
        if kind == BoxKind.CONTENT:
            area = self._virtual_display_area
            x = max(area.left(), min(area.right() - rect.width(), origin.x()))
            y = max(area.top(), min(area.bottom() - rect.height(), origin.y()))
            return QRectF(x, y, rect.width(), rect.height())

        content = self._rects[BoxKind.CONTENT]
        if not content.isEmpty():
            moved.translate(min(0.0, content.left() - moved.left()), 0)
            moved.translate(max(0.0, content.right() - moved.right()), 0)
            moved.translate(0, min(0.0, content.top() - moved.top()))
            moved.translate(0, max(0.0, content.bottom() - moved.bottom()))
        return moved

    def _paint_overlay(self, painter: QPainter) -> None:
        boxes = self._editable_boxes()
        for kind in (BoxKind.PAGE, BoxKind.CONTENT):
            if kind in boxes:
                self._paint_box(painter, kind)

    def _paint_box(self, painter: QPainter, kind: BoxKind) -> None:
        border, fill = self.BOX_COLORS[kind]
        rect = self.virtual_to_widget_rect(self._rects[kind])
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(border)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(fill)
        painter.drawRect(rect)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.HANDLE_COLOR)
        size = 6
        for corner in (
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight(),
        ):
            painter.drawRect(
                QRectF(corner.x() - size, corner.y() - size, 2 * size, 2 * size)
            )
        painter.restore()

    @staticmethod
    def _shift_held(event: QtGui.QMouseEvent) -> bool:
        return bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pick up a box, a side or a corner.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(event.position(), self._shift_held(event))
            if hit is not None:
                kind, edges = hit
                self._drag = hit
                self._drag_moved = False
                grab = self.widget_to_virtual_point(event.position())
                self._drag_offset = self._grab_offset(self._rects[kind], edges, grab)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Resize or move the picked box, or update the cursor.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._drag is not None:
            kind, edges = self._drag
            virtual = self.widget_to_virtual_point(event.position())
            rect = self._rects[kind]
            if edges == Edge.NONE:
                new_rect = self._moved(kind, rect, virtual - self._drag_offset)
            else:
                new_rect = self._resized(rect, edges, virtual - self._drag_offset)
            if kind == BoxKind.PAGE:
                new_rect = new_rect.united(self._rects[BoxKind.CONTENT])
            if new_rect != rect:
                self._rects[kind] = new_rect
                if kind == BoxKind.CONTENT:
                    self._describe_content_in_page()
                self._drag_moved = True
                self.update()
            event.accept()
            return

        if not self._panning:
            hit = self._hit_test(event.position(), self._shift_held(event))
            if hit != self._hover:
                self._hover = hit
                edges = Edge.NONE if hit is None else hit[1]
                if hit is None:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                else:
                    self.setCursor(
                        self._CURSORS.get(edges, Qt.CursorShape.SizeAllCursor)
                    )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop the box and report it if it changed.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._drag is not None:
            kind, _edges = self._drag
            self._drag = None
            event.accept()
            self.update()
            if self._drag_moved:
                signal = (
                    self.manual_content_rect_set
                    if kind == BoxKind.CONTENT
                    else self.manual_page_rect_set
                )
                signal.emit(QRectF(self._rects[kind]))
                self._report_page_rect_if_grown()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Create a content box around the click when there is none.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._rects[BoxKind.CONTENT].isEmpty()
        ):
            self._create_content_box(self.widget_to_virtual_point(event.position()))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Offer to create or remove the content box.

        Args:
            event (QtGui.QContextMenuEvent): The context menu event.
        """
        menu = (
            self._no_content_menu
            if self._rects[BoxKind.CONTENT].isEmpty()
            else self._have_content_menu
        )
        menu.exec(event.globalPos())
