"""Select Content filter image view.

Provides an interactive image view for the Select Content filter with:
- Draggable content box corners and edges
- Optional page box display and editing
- Context menu for creating/removing content boxes
- Visual feedback for content and page boundaries
"""

import math
from enum import IntFlag, auto
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QLineF, QPointF, QRectF, QSizeF, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen

from scantailor.app.ui.image_views.base import ImageViewBase

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scantailor.core import ImageTransformation


class Edge(IntFlag):
    """Flags for rectangle edges."""

    NONE = 0
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


class SelectContentImageView(ImageViewBase):
    """Image view for Select Content filter with interactive box editing.

    Displays the image with content and page boxes that can be resized
    by dragging corners or edges.

    Signals:
        manual_content_rect_set: Emitted when content box is manually set.
        manual_page_rect_set: Emitted when page box is manually set.
        page_rect_size_changed: Emitted when page box size changes.
    """

    manual_content_rect_set = Signal(QRectF)
    manual_page_rect_set = Signal(QRectF)
    page_rect_size_changed = Signal(QSizeF)

    # Constants
    CORNER_HIT_RADIUS = 10
    EDGE_HIT_DISTANCE = 8
    MIN_BOX_SIZE = 10  # Minimum box dimension

    # Colors
    CONTENT_BOX_COLOR = QColor(0, 0, 255, 180)
    CONTENT_BOX_FILL = QColor(0, 0, 255, 30)
    PAGE_BOX_COLOR = QColor(255, 0, 0, 180)
    PAGE_BOX_FILL = QColor(255, 0, 0, 20)
    HANDLE_COLOR = QColor(0, 0, 255, 200)
    HANDLE_HIGHLIGHT = QColor(0, 128, 255, 255)

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        content_rect: QRectF | None = None,
        page_rect: QRectF | None = None,
        page_rect_enabled: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the Select Content image view.

        Args:
            image: The image to display.
            downscaled_image: Optional pre-downscaled version.
            transformation: Image transformation.
            content_rect: Initial content box in virtual image coordinates.
            page_rect: Initial page box in virtual image coordinates.
            page_rect_enabled: Whether page box editing is enabled.
            parent: Parent widget.
        """
        super().__init__(image, downscaled_image, transformation, parent=parent)

        # Content and page rectangles (in virtual image coordinates)
        self._content_rect = content_rect or QRectF()
        self._page_rect = page_rect or QRectF()
        self._page_rect_enabled = page_rect_enabled

        # Interaction state
        self._dragging_content_corner: Edge = Edge.NONE
        self._dragging_content_edge: Edge = Edge.NONE
        self._dragging_page_corner: Edge = Edge.NONE
        self._dragging_page_edge: Edge = Edge.NONE
        self._dragging_content_rect = False
        self._dragging_page_rect = False
        self._drag_offset = QPointF()

        # Hover state
        self._hovered_content_corner: Edge = Edge.NONE
        self._hovered_content_edge: Edge = Edge.NONE
        self._hovered_page_corner: Edge = Edge.NONE
        self._hovered_page_edge: Edge = Edge.NONE

        # Context menus
        self._setup_context_menus()

        self.setMouseTracking(True)

    def _setup_context_menus(self) -> None:
        """Set up context menus for content box operations."""
        # Menu when no content box exists
        self._no_content_menu = QtWidgets.QMenu(self)
        create_action = QAction("Create Content Box", self)
        create_action.triggered.connect(self._create_content_box)
        self._no_content_menu.addAction(create_action)

        # Menu when content box exists
        self._have_content_menu = QtWidgets.QMenu(self)
        remove_action = QAction("Remove Content Box", self)
        remove_action.triggered.connect(self._remove_content_box)
        self._have_content_menu.addAction(remove_action)

    @property
    def content_rect(self) -> QRectF:
        """Get the content box rectangle in virtual coordinates."""
        return self._content_rect

    def set_content_rect(self, rect: QRectF) -> None:
        """Set the content box rectangle.

        Args:
            rect: Rectangle in virtual image coordinates.
        """
        self._content_rect = rect
        self.update()

    @property
    def page_rect(self) -> QRectF:
        """Get the page box rectangle in virtual coordinates."""
        return self._page_rect

    def set_page_rect(self, rect: QRectF) -> None:
        """Set the page box rectangle.

        Args:
            rect: Rectangle in virtual image coordinates.
        """
        self._page_rect = rect
        self.update()

    @Slot(QRectF)
    def page_rect_set_externally(self, rect: QRectF) -> None:
        """Slot to set page rect from external source.

        Args:
            rect: Rectangle in virtual image coordinates.
        """
        self.set_page_rect(rect)

    @Slot(bool)
    def set_page_rect_enabled(self, enabled: bool) -> None:
        """Enable or disable page rect editing.

        Args:
            enabled: True to enable page rect editing.
        """
        self._page_rect_enabled = enabled
        self.update()

    def _create_content_box(self) -> None:
        """Create a default content box at the center of the image."""
        # Create a box at 25% margins from the image edges
        display_area = self._virtual_display_area
        margin_x = display_area.width() * 0.25
        margin_y = display_area.height() * 0.25

        self._content_rect = QRectF(
            display_area.x() + margin_x,
            display_area.y() + margin_y,
            display_area.width() - 2 * margin_x,
            display_area.height() - 2 * margin_y,
        )
        self.update()
        self.manual_content_rect_set.emit(self._content_rect)

    def _remove_content_box(self) -> None:
        """Remove the content box."""
        self._content_rect = QRectF()
        self.update()
        self.manual_content_rect_set.emit(self._content_rect)

    def _corner_position(self, rect: QRectF, edge_mask: Edge) -> QPointF:
        """Get the position of a corner of a rectangle.

        Args:
            rect: The rectangle.
            edge_mask: Edge flags indicating which corner.

        Returns:
            Corner position in virtual coordinates.
        """
        x = rect.left() if edge_mask & Edge.LEFT else rect.right()
        y = rect.top() if edge_mask & Edge.TOP else rect.bottom()
        return QPointF(x, y)

    def _edge_line(self, rect: QRectF, edge: Edge) -> QLineF:
        """Get a line representing an edge of a rectangle.

        Args:
            rect: The rectangle.
            edge: Which edge.

        Returns:
            Line segment in virtual coordinates.
        """
        if edge == Edge.LEFT:
            return QLineF(rect.topLeft(), rect.bottomLeft())
        elif edge == Edge.RIGHT:
            return QLineF(rect.topRight(), rect.bottomRight())
        elif edge == Edge.TOP:
            return QLineF(rect.topLeft(), rect.topRight())
        else:  # BOTTOM
            return QLineF(rect.bottomLeft(), rect.bottomRight())

    def _corner_at_position(self, rect: QRectF, widget_pos: QPointF) -> Edge:
        """Check if a position is over a corner of a rectangle.

        Args:
            rect: The rectangle in virtual coordinates.
            widget_pos: Position in widget coordinates.

        Returns:
            Edge flags for the corner, or Edge.NONE.
        """
        if rect.isEmpty():
            return Edge.NONE

        corners = [
            (Edge.LEFT | Edge.TOP, rect.topLeft()),
            (Edge.RIGHT | Edge.TOP, rect.topRight()),
            (Edge.LEFT | Edge.BOTTOM, rect.bottomLeft()),
            (Edge.RIGHT | Edge.BOTTOM, rect.bottomRight()),
        ]

        virtual_pos = self.widget_to_virtual_point(widget_pos)

        for edge_mask, corner in corners:
            corner_widget = self.virtual_to_widget_point(corner)
            dist = math.sqrt(
                (widget_pos.x() - corner_widget.x()) ** 2
                + (widget_pos.y() - corner_widget.y()) ** 2
            )
            if dist <= self.CORNER_HIT_RADIUS:
                return edge_mask

        return Edge.NONE

    def _edge_at_position(self, rect: QRectF, widget_pos: QPointF) -> Edge:
        """Check if a position is over an edge of a rectangle.

        Args:
            rect: The rectangle in virtual coordinates.
            widget_pos: Position in widget coordinates.

        Returns:
            Edge flag for the edge, or Edge.NONE.
        """
        if rect.isEmpty():
            return Edge.NONE

        virtual_pos = self.widget_to_virtual_point(widget_pos)

        # Check if inside rect first
        if not rect.contains(virtual_pos):
            return Edge.NONE

        edges = [Edge.LEFT, Edge.RIGHT, Edge.TOP, Edge.BOTTOM]

        for edge in edges:
            line = self._edge_line(rect, edge)
            line_widget = QLineF(
                self.virtual_to_widget_point(line.p1()),
                self.virtual_to_widget_point(line.p2()),
            )

            # Distance from point to line
            dist = self._point_to_line_distance(widget_pos, line_widget)
            if dist <= self.EDGE_HIT_DISTANCE:
                return edge

        return Edge.NONE

    def _rect_contains_point(self, rect: QRectF, widget_pos: QPointF) -> bool:
        """Check if a rectangle contains a point.

        Args:
            rect: Rectangle in virtual coordinates.
            widget_pos: Point in widget coordinates.

        Returns:
            True if the point is inside the rectangle.
        """
        if rect.isEmpty():
            return False
        virtual_pos = self.widget_to_virtual_point(widget_pos)
        return rect.contains(virtual_pos)

    def _point_to_line_distance(self, point: QPointF, line: QLineF) -> float:
        """Calculate the distance from a point to a line segment.

        Args:
            point: The point.
            line: The line segment.

        Returns:
            Distance from point to nearest point on line segment.
        """
        # Vector from line start to point
        dx = point.x() - line.x1()
        dy = point.y() - line.y1()

        # Vector along line
        line_dx = line.x2() - line.x1()
        line_dy = line.y2() - line.y1()

        line_len_sq = line_dx * line_dx + line_dy * line_dy
        if line_len_sq < 0.001:
            return math.sqrt(dx * dx + dy * dy)

        # Project point onto line
        t = max(0, min(1, (dx * line_dx + dy * line_dy) / line_len_sq))

        # Nearest point on line
        nearest_x = line.x1() + t * line_dx
        nearest_y = line.y1() + t * line_dy

        return math.sqrt((point.x() - nearest_x) ** 2 + (point.y() - nearest_y) ** 2)

    def _move_corner(self, rect: QRectF, edge_mask: Edge, new_pos: QPointF) -> QRectF:
        """Move a corner of a rectangle to a new position.

        Args:
            rect: The rectangle.
            edge_mask: Which corner to move.
            new_pos: New corner position in virtual coordinates.

        Returns:
            Updated rectangle.
        """
        new_rect = QRectF(rect)

        if edge_mask & Edge.LEFT:
            new_rect.setLeft(min(new_pos.x(), rect.right() - self.MIN_BOX_SIZE))
        if edge_mask & Edge.RIGHT:
            new_rect.setRight(max(new_pos.x(), rect.left() + self.MIN_BOX_SIZE))
        if edge_mask & Edge.TOP:
            new_rect.setTop(min(new_pos.y(), rect.bottom() - self.MIN_BOX_SIZE))
        if edge_mask & Edge.BOTTOM:
            new_rect.setBottom(max(new_pos.y(), rect.top() + self.MIN_BOX_SIZE))

        return new_rect.normalized()

    def _move_edge(self, rect: QRectF, edge: Edge, new_pos: QPointF) -> QRectF:
        """Move an edge of a rectangle to a new position.

        Args:
            rect: The rectangle.
            edge: Which edge to move.
            new_pos: New position in virtual coordinates.

        Returns:
            Updated rectangle.
        """
        new_rect = QRectF(rect)

        if edge == Edge.LEFT:
            new_rect.setLeft(min(new_pos.x(), rect.right() - self.MIN_BOX_SIZE))
        elif edge == Edge.RIGHT:
            new_rect.setRight(max(new_pos.x(), rect.left() + self.MIN_BOX_SIZE))
        elif edge == Edge.TOP:
            new_rect.setTop(min(new_pos.y(), rect.bottom() - self.MIN_BOX_SIZE))
        elif edge == Edge.BOTTOM:
            new_rect.setBottom(max(new_pos.y(), rect.top() + self.MIN_BOX_SIZE))

        return new_rect.normalized()

    def _get_cursor_for_edge(self, edge: Edge) -> Qt.CursorShape:
        """Get the appropriate cursor for an edge or corner.

        Args:
            edge: Edge flags.

        Returns:
            Cursor shape.
        """
        if edge == Edge.NONE:
            return Qt.CursorShape.ArrowCursor

        # Corner cursors
        if edge == (Edge.LEFT | Edge.TOP) or edge == (Edge.RIGHT | Edge.BOTTOM):
            return Qt.CursorShape.SizeFDiagCursor
        if edge == (Edge.RIGHT | Edge.TOP) or edge == (Edge.LEFT | Edge.BOTTOM):
            return Qt.CursorShape.SizeBDiagCursor

        # Edge cursors
        if edge in (Edge.LEFT, Edge.RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if edge in (Edge.TOP, Edge.BOTTOM):
            return Qt.CursorShape.SizeVerCursor

        return Qt.CursorShape.ArrowCursor

    def _paint_overlay(self, painter: QPainter) -> None:
        """Paint the content and page box overlays.

        Args:
            painter: The painter.
        """
        # Draw page box first (underneath content box)
        if self._page_rect_enabled and not self._page_rect.isEmpty():
            self._paint_box(
                painter,
                self._page_rect,
                self.PAGE_BOX_COLOR,
                self.PAGE_BOX_FILL,
                show_handles=(
                    self._hovered_page_corner != Edge.NONE
                    or self._hovered_page_edge != Edge.NONE
                    or self._dragging_page_corner != Edge.NONE
                    or self._dragging_page_edge != Edge.NONE
                ),
            )

        # Draw content box
        if not self._content_rect.isEmpty():
            self._paint_box(
                painter,
                self._content_rect,
                self.CONTENT_BOX_COLOR,
                self.CONTENT_BOX_FILL,
                show_handles=True,
            )

    def _paint_box(
        self,
        painter: QPainter,
        rect: QRectF,
        border_color: QColor,
        fill_color: QColor,
        show_handles: bool = True,
    ) -> None:
        """Paint a box with optional corner handles.

        Args:
            painter: The painter.
            rect: Rectangle in virtual coordinates.
            border_color: Color for the border.
            fill_color: Color for the fill.
            show_handles: Whether to draw corner handles.
        """
        widget_rect = self.virtual_to_widget_rect(rect)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Fill
        painter.setBrush(fill_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(widget_rect)

        # Border
        pen = QPen(border_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(widget_rect)

        # Corner handles
        if show_handles:
            handle_size = 6
            painter.setBrush(self.HANDLE_COLOR)

            corners = [
                widget_rect.topLeft(),
                widget_rect.topRight(),
                widget_rect.bottomLeft(),
                widget_rect.bottomRight(),
            ]

            for corner in corners:
                handle_rect = QRectF(
                    corner.x() - handle_size,
                    corner.y() - handle_size,
                    handle_size * 2,
                    handle_size * 2,
                )
                painter.drawRect(handle_rect)

        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press for box manipulation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()

            # Check content box corners first
            corner = self._corner_at_position(self._content_rect, pos)
            if corner != Edge.NONE:
                self._dragging_content_corner = corner
                event.accept()
                return

            # Check page box corners
            if self._page_rect_enabled:
                corner = self._corner_at_position(self._page_rect, pos)
                if corner != Edge.NONE:
                    self._dragging_page_corner = corner
                    event.accept()
                    return

            # Check content box edges
            edge = self._edge_at_position(self._content_rect, pos)
            if edge != Edge.NONE:
                self._dragging_content_edge = edge
                event.accept()
                return

            # Check page box edges
            if self._page_rect_enabled:
                edge = self._edge_at_position(self._page_rect, pos)
                if edge != Edge.NONE:
                    self._dragging_page_edge = edge
                    event.accept()
                    return

            # Check for rectangle dragging
            if self._rect_contains_point(self._content_rect, pos):
                self._dragging_content_rect = True
                virtual_pos = self.widget_to_virtual_point(pos)
                self._drag_offset = QPointF(
                    virtual_pos.x() - self._content_rect.x(),
                    virtual_pos.y() - self._content_rect.y(),
                )
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release after box manipulation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging_content_corner != Edge.NONE:
                self._dragging_content_corner = Edge.NONE
                self.manual_content_rect_set.emit(self._content_rect)
                event.accept()
                self.update()
                return

            if self._dragging_content_edge != Edge.NONE:
                self._dragging_content_edge = Edge.NONE
                self.manual_content_rect_set.emit(self._content_rect)
                event.accept()
                self.update()
                return

            if self._dragging_page_corner != Edge.NONE:
                self._dragging_page_corner = Edge.NONE
                self.manual_page_rect_set.emit(self._page_rect)
                self.page_rect_size_changed.emit(self._page_rect.size())
                event.accept()
                self.update()
                return

            if self._dragging_page_edge != Edge.NONE:
                self._dragging_page_edge = Edge.NONE
                self.manual_page_rect_set.emit(self._page_rect)
                self.page_rect_size_changed.emit(self._page_rect.size())
                event.accept()
                self.update()
                return

            if self._dragging_content_rect:
                self._dragging_content_rect = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.manual_content_rect_set.emit(self._content_rect)
                event.accept()
                self.update()
                return

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move for box manipulation and cursor updates.

        Args:
            event: The mouse event.
        """
        pos = event.position()
        virtual_pos = self.widget_to_virtual_point(pos)

        # Handle dragging
        if self._dragging_content_corner != Edge.NONE:
            self._content_rect = self._move_corner(
                self._content_rect, self._dragging_content_corner, virtual_pos
            )
            self.update()
            event.accept()
            return

        if self._dragging_content_edge != Edge.NONE:
            self._content_rect = self._move_edge(
                self._content_rect, self._dragging_content_edge, virtual_pos
            )
            self.update()
            event.accept()
            return

        if self._dragging_page_corner != Edge.NONE:
            self._page_rect = self._move_corner(
                self._page_rect, self._dragging_page_corner, virtual_pos
            )
            self.update()
            event.accept()
            return

        if self._dragging_page_edge != Edge.NONE:
            self._page_rect = self._move_edge(
                self._page_rect, self._dragging_page_edge, virtual_pos
            )
            self.update()
            event.accept()
            return

        if self._dragging_content_rect:
            new_pos = QPointF(
                virtual_pos.x() - self._drag_offset.x(),
                virtual_pos.y() - self._drag_offset.y(),
            )
            self._content_rect.moveTo(new_pos)
            self.update()
            event.accept()
            return

        # Update hover state and cursor
        self._hovered_content_corner = self._corner_at_position(self._content_rect, pos)
        if self._hovered_content_corner != Edge.NONE:
            self.setCursor(self._get_cursor_for_edge(self._hovered_content_corner))
            self.update()
            super().mouseMoveEvent(event)
            return

        if self._page_rect_enabled:
            self._hovered_page_corner = self._corner_at_position(self._page_rect, pos)
            if self._hovered_page_corner != Edge.NONE:
                self.setCursor(self._get_cursor_for_edge(self._hovered_page_corner))
                self.update()
                super().mouseMoveEvent(event)
                return

        self._hovered_content_edge = self._edge_at_position(self._content_rect, pos)
        if self._hovered_content_edge != Edge.NONE:
            self.setCursor(self._get_cursor_for_edge(self._hovered_content_edge))
            self.update()
            super().mouseMoveEvent(event)
            return

        if self._page_rect_enabled:
            self._hovered_page_edge = self._edge_at_position(self._page_rect, pos)
            if self._hovered_page_edge != Edge.NONE:
                self.setCursor(self._get_cursor_for_edge(self._hovered_page_edge))
                self.update()
                super().mouseMoveEvent(event)
                return

        # Reset cursor if not over anything
        if not self._dragging:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle double-click to create/remove content box.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()

            if self._content_rect.isEmpty():
                # Create content box centered at click position
                virtual_pos = self.widget_to_virtual_point(pos)
                display_area = self._virtual_display_area

                # Create a reasonably sized box
                width = display_area.width() * 0.5
                height = display_area.height() * 0.5

                self._content_rect = QRectF(
                    virtual_pos.x() - width / 2,
                    virtual_pos.y() - height / 2,
                    width,
                    height,
                )

                # Clamp to display area
                self._content_rect = self._content_rect.intersected(display_area)

                self.update()
                self.manual_content_rect_set.emit(self._content_rect)
                event.accept()
                return

        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show context menu for content box operations.

        Args:
            event: The context menu event.
        """
        if self._content_rect.isEmpty():
            self._no_content_menu.exec(event.globalPos())
        else:
            self._have_content_menu.exec(event.globalPos())
