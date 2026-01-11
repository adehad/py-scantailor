"""Page Layout filter image view.

Provides an interactive image view for the Page Layout filter with:
- Draggable margin edges
- Visual display of inner (content), middle (hard margins), and outer (soft margins) rectangles
- Alignment guides
- Context menu for guide management
"""

import math
from enum import IntFlag, auto
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QLineF, QPointF, QRectF, QSizeF, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.core import Margins

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scantailor.core import ImageTransformation
    from scantailor.filters.page_layout import Alignment


class Edge(IntFlag):
    """Flags for rectangle edges."""

    NONE = 0
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


class Guide:
    """A horizontal or vertical alignment guide."""

    def __init__(self, position: float, is_horizontal: bool, index: int) -> None:
        """Initialize a guide.

        Args:
            position: Position in virtual coordinates.
            is_horizontal: True for horizontal guide, False for vertical.
            index: Unique index for this guide.
        """
        self.position = position
        self.is_horizontal = is_horizontal
        self.index = index


class PageLayoutImageView(ImageViewBase):
    """Image view for Page Layout filter with interactive margin editing.

    Displays the image with three nested rectangles:
    - Inner rect: Content box (from Select Content)
    - Middle rect: Content + hard margins (user-defined)
    - Outer rect: Content + hard + soft margins (for alignment with other pages)

    Users can drag the edges of the middle rect to adjust hard margins.

    Signals:
        margins_set_locally: Emitted when margins are set by user (Margins in mm).
        invalidate_thumbnail: Emitted when thumbnail needs refresh.
    """

    margins_set_locally = Signal(Margins)
    invalidate_thumbnail = Signal()

    # Constants
    EDGE_HIT_DISTANCE = 10
    GUIDE_HIT_DISTANCE = 6

    # Colors
    INNER_RECT_COLOR = QColor(0, 0, 255, 200)
    INNER_RECT_FILL = QColor(0, 0, 255, 20)
    MIDDLE_RECT_COLOR = QColor(0, 180, 0, 200)
    MIDDLE_RECT_FILL = QColor(0, 180, 0, 20)
    OUTER_RECT_COLOR = QColor(180, 0, 180, 150)
    OUTER_RECT_FILL = QColor(180, 0, 180, 10)
    GUIDE_COLOR = QColor(255, 165, 0, 180)
    GUIDE_HIGHLIGHT = QColor(255, 200, 0, 255)

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        inner_rect: QRectF | None = None,
        middle_rect: QRectF | None = None,
        outer_rect: QRectF | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the Page Layout image view.

        Args:
            image: The image to display.
            downscaled_image: Optional pre-downscaled version.
            transformation: Image transformation.
            inner_rect: Content box in virtual coordinates.
            middle_rect: Content + hard margins in virtual coordinates.
            outer_rect: Content + hard + soft margins in virtual coordinates.
            parent: Parent widget.
        """
        super().__init__(image, downscaled_image, transformation, parent=parent)

        # Rectangles (in virtual image coordinates)
        self._inner_rect = inner_rect or QRectF()  # Content box
        self._middle_rect = middle_rect or QRectF()  # Content + hard margins
        self._outer_rect = outer_rect or QRectF()  # Content + all margins

        # Margin linking state
        self._left_right_linked = True
        self._top_bottom_linked = True

        # Pixels to mm conversion (will be set from transformation)
        self._pixels_to_mm = 1.0 / 10.0  # Default ~0.1mm per pixel
        self._mm_to_pixels = 10.0

        # Interaction state
        self._dragging_inner_edge: Edge = Edge.NONE
        self._dragging_middle_edge: Edge = Edge.NONE
        self._dragging_inner_rect = False
        self._drag_offset = QPointF()

        # Hover state
        self._hovered_inner_edge: Edge = Edge.NONE
        self._hovered_middle_edge: Edge = Edge.NONE

        # Guides
        self._guides: dict[int, Guide] = {}
        self._guide_index_counter = 0
        self._dragging_guide: int | None = None
        self._hovered_guide: int | None = None
        self._last_context_menu_pos = QPointF()

        # Show/hide options
        self._show_middle_rect = True

        # Context menu
        self._setup_context_menu()

        self.setMouseTracking(True)

    def _setup_context_menu(self) -> None:
        """Set up the context menu."""
        self._context_menu = QtWidgets.QMenu(self)

        self._add_horizontal_guide_action = QAction("Add Horizontal Guide", self)
        self._add_horizontal_guide_action.triggered.connect(self._add_horizontal_guide)
        self._context_menu.addAction(self._add_horizontal_guide_action)

        self._add_vertical_guide_action = QAction("Add Vertical Guide", self)
        self._add_vertical_guide_action.triggered.connect(self._add_vertical_guide)
        self._context_menu.addAction(self._add_vertical_guide_action)

        self._context_menu.addSeparator()

        self._remove_guide_action = QAction("Remove Guide", self)
        self._remove_guide_action.triggered.connect(self._remove_guide_under_mouse)
        self._context_menu.addAction(self._remove_guide_action)

        self._remove_all_guides_action = QAction("Remove All Guides", self)
        self._remove_all_guides_action.triggered.connect(self._remove_all_guides)
        self._context_menu.addAction(self._remove_all_guides_action)

        self._context_menu.addSeparator()

        self._show_middle_rect_action = QAction("Show Hard Margins", self)
        self._show_middle_rect_action.setCheckable(True)
        self._show_middle_rect_action.setChecked(True)
        self._show_middle_rect_action.triggered.connect(self._toggle_middle_rect)
        self._context_menu.addAction(self._show_middle_rect_action)

    def set_rectangles(
        self,
        inner_rect: QRectF,
        middle_rect: QRectF,
        outer_rect: QRectF,
    ) -> None:
        """Set all three rectangles.

        Args:
            inner_rect: Content box.
            middle_rect: Content + hard margins.
            outer_rect: Content + all margins.
        """
        self._inner_rect = inner_rect
        self._middle_rect = middle_rect
        self._outer_rect = outer_rect
        self.update()

    def set_pixels_to_mm(self, scale: float) -> None:
        """Set the pixels-to-mm conversion factor.

        Args:
            scale: Pixels per mm.
        """
        if scale > 0:
            self._pixels_to_mm = 1.0 / scale
            self._mm_to_pixels = scale

    @Slot(Margins)
    def margins_set_externally(self, margins: Margins) -> None:
        """Slot to set margins from external source.

        Args:
            margins: Margins in mm.
        """
        # Convert mm to pixels and update middle rect
        left_px = margins.left * self._mm_to_pixels
        right_px = margins.right * self._mm_to_pixels
        top_px = margins.top * self._mm_to_pixels
        bottom_px = margins.bottom * self._mm_to_pixels

        self._middle_rect = QRectF(
            self._inner_rect.left() - left_px,
            self._inner_rect.top() - top_px,
            self._inner_rect.width() + left_px + right_px,
            self._inner_rect.height() + top_px + bottom_px,
        )
        self.update()

    @Slot(bool)
    def left_right_link_toggled(self, linked: bool) -> None:
        """Handle left-right margin linking change.

        Args:
            linked: Whether margins are linked.
        """
        self._left_right_linked = linked

    @Slot(bool)
    def top_bottom_link_toggled(self, linked: bool) -> None:
        """Handle top-bottom margin linking change.

        Args:
            linked: Whether margins are linked.
        """
        self._top_bottom_linked = linked

    def _calc_margins_mm(self) -> Margins:
        """Calculate current margins in mm.

        Returns:
            Margins in mm.
        """
        left = (self._inner_rect.left() - self._middle_rect.left()) * self._pixels_to_mm
        right = (
            self._middle_rect.right() - self._inner_rect.right()
        ) * self._pixels_to_mm
        top = (self._inner_rect.top() - self._middle_rect.top()) * self._pixels_to_mm
        bottom = (
            self._middle_rect.bottom() - self._inner_rect.bottom()
        ) * self._pixels_to_mm

        return Margins(
            left=max(0, left),
            right=max(0, right),
            top=max(0, top),
            bottom=max(0, bottom),
        )

    def _edge_at_position(
        self, rect: QRectF, widget_pos: QPointF, exclude_inner: bool = False
    ) -> Edge:
        """Check if a position is near an edge of a rectangle.

        Args:
            rect: The rectangle in virtual coordinates.
            widget_pos: Position in widget coordinates.
            exclude_inner: If True, only detect edges on the outside of the inner rect.

        Returns:
            Edge flag, or Edge.NONE.
        """
        if rect.isEmpty():
            return Edge.NONE

        virtual_pos = self.widget_to_virtual_point(widget_pos)

        # For middle rect, we want to detect edges between inner and middle
        edges = [Edge.LEFT, Edge.RIGHT, Edge.TOP, Edge.BOTTOM]
        closest_edge = Edge.NONE
        closest_dist = float("inf")

        for edge in edges:
            # Get edge line
            if edge == Edge.LEFT:
                line = QLineF(rect.topLeft(), rect.bottomLeft())
            elif edge == Edge.RIGHT:
                line = QLineF(rect.topRight(), rect.bottomRight())
            elif edge == Edge.TOP:
                line = QLineF(rect.topLeft(), rect.topRight())
            else:  # BOTTOM
                line = QLineF(rect.bottomLeft(), rect.bottomRight())

            # Convert to widget coordinates
            line_widget = QLineF(
                self.virtual_to_widget_point(line.p1()),
                self.virtual_to_widget_point(line.p2()),
            )

            # Distance from point to line
            dist = self._point_to_line_distance(widget_pos, line_widget)
            if dist < closest_dist and dist <= self.EDGE_HIT_DISTANCE:
                closest_dist = dist
                closest_edge = edge

        return closest_edge

    def _point_to_line_distance(self, point: QPointF, line: QLineF) -> float:
        """Calculate distance from point to line segment."""
        dx = point.x() - line.x1()
        dy = point.y() - line.y1()
        line_dx = line.x2() - line.x1()
        line_dy = line.y2() - line.y1()

        line_len_sq = line_dx * line_dx + line_dy * line_dy
        if line_len_sq < 0.001:
            return math.sqrt(dx * dx + dy * dy)

        t = max(0, min(1, (dx * line_dx + dy * line_dy) / line_len_sq))
        nearest_x = line.x1() + t * line_dx
        nearest_y = line.y1() + t * line_dy

        return math.sqrt((point.x() - nearest_x) ** 2 + (point.y() - nearest_y) ** 2)

    def _move_middle_edge(self, edge: Edge, virtual_pos: QPointF) -> None:
        """Move an edge of the middle rectangle.

        Args:
            edge: Which edge to move.
            virtual_pos: New position in virtual coordinates.
        """
        new_rect = QRectF(self._middle_rect)

        if edge == Edge.LEFT:
            new_left = min(virtual_pos.x(), self._inner_rect.left())
            delta = new_rect.left() - new_left
            new_rect.setLeft(new_left)
            if self._left_right_linked:
                new_rect.setRight(new_rect.right() + delta)

        elif edge == Edge.RIGHT:
            new_right = max(virtual_pos.x(), self._inner_rect.right())
            delta = new_right - new_rect.right()
            new_rect.setRight(new_right)
            if self._left_right_linked:
                new_rect.setLeft(new_rect.left() - delta)

        elif edge == Edge.TOP:
            new_top = min(virtual_pos.y(), self._inner_rect.top())
            delta = new_rect.top() - new_top
            new_rect.setTop(new_top)
            if self._top_bottom_linked:
                new_rect.setBottom(new_rect.bottom() + delta)

        elif edge == Edge.BOTTOM:
            new_bottom = max(virtual_pos.y(), self._inner_rect.bottom())
            delta = new_bottom - new_rect.bottom()
            new_rect.setBottom(new_bottom)
            if self._top_bottom_linked:
                new_rect.setTop(new_rect.top() - delta)

        # Ensure middle rect contains inner rect
        if new_rect.contains(self._inner_rect):
            self._middle_rect = new_rect

    def _get_cursor_for_edge(self, edge: Edge) -> Qt.CursorShape:
        """Get cursor shape for an edge."""
        if edge in (Edge.LEFT, Edge.RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if edge in (Edge.TOP, Edge.BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    # Guide methods

    def _add_horizontal_guide(self) -> None:
        """Add a horizontal guide at the last context menu position."""
        virtual_pos = self.widget_to_virtual_point(self._last_context_menu_pos)
        index = self._guide_index_counter
        self._guide_index_counter += 1
        self._guides[index] = Guide(virtual_pos.y(), True, index)
        self.update()

    def _add_vertical_guide(self) -> None:
        """Add a vertical guide at the last context menu position."""
        virtual_pos = self.widget_to_virtual_point(self._last_context_menu_pos)
        index = self._guide_index_counter
        self._guide_index_counter += 1
        self._guides[index] = Guide(virtual_pos.x(), False, index)
        self.update()

    def _remove_guide_under_mouse(self) -> None:
        """Remove the guide under the mouse cursor."""
        if self._hovered_guide is not None and self._hovered_guide in self._guides:
            del self._guides[self._hovered_guide]
            self._hovered_guide = None
            self.update()

    def _remove_all_guides(self) -> None:
        """Remove all guides."""
        self._guides.clear()
        self._hovered_guide = None
        self.update()

    def _toggle_middle_rect(self, checked: bool) -> None:
        """Toggle visibility of middle rect."""
        self._show_middle_rect = checked
        self.update()

    def _guide_at_position(self, widget_pos: QPointF) -> int | None:
        """Find a guide at the given position.

        Args:
            widget_pos: Position in widget coordinates.

        Returns:
            Guide index or None.
        """
        viewport = self.viewport()
        if viewport is None:
            return None

        for index, guide in self._guides.items():
            if guide.is_horizontal:
                guide_widget_y = self.virtual_to_widget_point(
                    QPointF(0, guide.position)
                ).y()
                if abs(widget_pos.y() - guide_widget_y) <= self.GUIDE_HIT_DISTANCE:
                    return index
            else:
                guide_widget_x = self.virtual_to_widget_point(
                    QPointF(guide.position, 0)
                ).x()
                if abs(widget_pos.x() - guide_widget_x) <= self.GUIDE_HIT_DISTANCE:
                    return index

        return None

    def _paint_overlay(self, painter: QPainter) -> None:
        """Paint the rectangle overlays and guides.

        Args:
            painter: The painter.
        """
        # Draw outer rect (soft margins)
        if not self._outer_rect.isEmpty():
            self._paint_rect(
                painter, self._outer_rect, self.OUTER_RECT_COLOR, self.OUTER_RECT_FILL
            )

        # Draw middle rect (hard margins)
        if self._show_middle_rect and not self._middle_rect.isEmpty():
            self._paint_rect(
                painter,
                self._middle_rect,
                self.MIDDLE_RECT_COLOR,
                self.MIDDLE_RECT_FILL,
            )

        # Draw inner rect (content)
        if not self._inner_rect.isEmpty():
            self._paint_rect(
                painter, self._inner_rect, self.INNER_RECT_COLOR, self.INNER_RECT_FILL
            )

        # Draw guides
        self._paint_guides(painter)

    def _paint_rect(
        self,
        painter: QPainter,
        rect: QRectF,
        border_color: QColor,
        fill_color: QColor,
    ) -> None:
        """Paint a rectangle.

        Args:
            painter: The painter.
            rect: Rectangle in virtual coordinates.
            border_color: Border color.
            fill_color: Fill color.
        """
        widget_rect = self.virtual_to_widget_rect(rect)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

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

        painter.restore()

    def _paint_guides(self, painter: QPainter) -> None:
        """Paint alignment guides.

        Args:
            painter: The painter.
        """
        viewport = self.viewport()
        if viewport is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        for index, guide in self._guides.items():
            is_hovered = index == self._hovered_guide
            color = self.GUIDE_HIGHLIGHT if is_hovered else self.GUIDE_COLOR

            pen = QPen(color)
            pen.setWidth(2 if is_hovered else 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)

            if guide.is_horizontal:
                y = self.virtual_to_widget_point(QPointF(0, guide.position)).y()
                painter.drawLine(QLineF(0, y, viewport.width(), y))
            else:
                x = self.virtual_to_widget_point(QPointF(guide.position, 0)).x()
                painter.drawLine(QLineF(x, 0, x, viewport.height()))

        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press for margin and guide manipulation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()

            # Check guides first
            guide = self._guide_at_position(pos)
            if guide is not None:
                self._dragging_guide = guide
                if self._guides[guide].is_horizontal:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return

            # Check middle rect edges (margin edges)
            if self._show_middle_rect:
                edge = self._edge_at_position(self._middle_rect, pos)
                if edge != Edge.NONE:
                    self._dragging_middle_edge = edge
                    self.setCursor(self._get_cursor_for_edge(edge))
                    event.accept()
                    return

            # Check inner rect edges (content box)
            edge = self._edge_at_position(self._inner_rect, pos)
            if edge != Edge.NONE:
                self._dragging_inner_edge = edge
                self.setCursor(self._get_cursor_for_edge(edge))
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release after manipulation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging_guide is not None:
                self._dragging_guide = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return

            if self._dragging_middle_edge != Edge.NONE:
                self._dragging_middle_edge = Edge.NONE
                self.setCursor(Qt.CursorShape.ArrowCursor)
                margins = self._calc_margins_mm()
                self.margins_set_locally.emit(margins)
                self.invalidate_thumbnail.emit()
                event.accept()
                return

            if self._dragging_inner_edge != Edge.NONE:
                self._dragging_inner_edge = Edge.NONE
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move for manipulation and cursor updates.

        Args:
            event: The mouse event.
        """
        pos = event.position()
        virtual_pos = self.widget_to_virtual_point(pos)

        # Handle guide dragging
        if self._dragging_guide is not None:
            guide = self._guides.get(self._dragging_guide)
            if guide is not None:
                if guide.is_horizontal:
                    guide.position = virtual_pos.y()
                else:
                    guide.position = virtual_pos.x()
                self.update()
            event.accept()
            return

        # Handle middle rect edge dragging
        if self._dragging_middle_edge != Edge.NONE:
            self._move_middle_edge(self._dragging_middle_edge, virtual_pos)
            self.update()
            event.accept()
            return

        # Update hover state
        self._hovered_guide = self._guide_at_position(pos)
        if self._hovered_guide is not None:
            guide = self._guides[self._hovered_guide]
            if guide.is_horizontal:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.update()
            super().mouseMoveEvent(event)
            return

        if self._show_middle_rect:
            self._hovered_middle_edge = self._edge_at_position(self._middle_rect, pos)
            if self._hovered_middle_edge != Edge.NONE:
                self.setCursor(self._get_cursor_for_edge(self._hovered_middle_edge))
                self.update()
                super().mouseMoveEvent(event)
                return

        self._hovered_inner_edge = self._edge_at_position(self._inner_rect, pos)
        if self._hovered_inner_edge != Edge.NONE:
            self.setCursor(self._get_cursor_for_edge(self._hovered_inner_edge))
            self.update()
            super().mouseMoveEvent(event)
            return

        # Reset cursor
        if not self._dragging:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show context menu.

        Args:
            event: The context menu event.
        """
        self._last_context_menu_pos = QPointF(event.pos())

        # Enable/disable remove guide action based on hover
        self._hovered_guide = self._guide_at_position(self._last_context_menu_pos)
        self._remove_guide_action.setEnabled(self._hovered_guide is not None)
        self._remove_all_guides_action.setEnabled(len(self._guides) > 0)

        self._context_menu.exec(event.globalPos())
