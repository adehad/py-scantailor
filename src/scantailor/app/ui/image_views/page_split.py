"""Page Split filter image view.

Provides an interactive image view for the Page Split filter with:
- Draggable split line(s) with endpoint handles
- Visual feedback for page layout (single, two pages, etc.)
- Buttons to restore removed page halves
"""

import math

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.core import ImageTransformation
from scantailor.filters.page_split import PageLayout


class PageSplitImageView(ImageViewBase):
    """Image view for Page Split filter with interactive split line editing.

    Displays the image with one or two draggable split lines that define
    how the page is divided. Supports single page and two-page layouts.

    Signals:
        page_layout_set_locally: Emitted when layout is changed by user.
        invalidate_thumbnail: Emitted when thumbnail needs refresh.
    """

    page_layout_set_locally = Signal(object)  # PageLayout
    invalidate_thumbnail = Signal()

    # Constants
    HANDLE_RADIUS = 8
    HANDLE_HIT_RADIUS = 15
    LINE_HIT_DISTANCE = 10

    # Colors
    SPLIT_LINE_COLOR = QColor(0, 0, 255, 200)
    SPLIT_LINE_HIGHLIGHT = QColor(0, 128, 255, 255)
    HANDLE_COLOR = QColor(0, 0, 255, 180)
    HANDLE_HIGHLIGHT = QColor(0, 128, 255, 255)
    LEFT_PAGE_COLOR = QColor(0, 200, 0, 50)
    RIGHT_PAGE_COLOR = QColor(0, 0, 200, 50)
    REMOVED_PAGE_COLOR = QColor(200, 0, 0, 80)

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        split_line: QLineF | None = None,
        left_page_removed: bool = False,
        right_page_removed: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the Page Split image view.

        Args:
            image: The image to display.
            downscaled_image: Optional pre-downscaled version.
            transformation: Image transformation.
            split_line: Initial split line in virtual coordinates.
            left_page_removed: Whether left page is removed.
            right_page_removed: Whether right page is removed.
            parent: Parent widget.
        """
        super().__init__(image, downscaled_image, transformation, parent=parent)

        # Split line (in virtual image coordinates)
        # For a two-page layout, this is the dividing line
        self._split_line = split_line or QLineF()

        # Page removal state
        self._left_page_removed = left_page_removed
        self._right_page_removed = right_page_removed

        # Interaction state
        self._dragging_handle = (
            -1
        )  # -1 = not dragging, 0 = top handle, 1 = bottom handle
        self._dragging_line = False
        self._line_drag_offset = 0.0

        # Hover state
        self._hovered_handle = -1
        self._hovered_line = False

        # Unremove buttons
        self._left_unremove_rect = QRectF()
        self._right_unremove_rect = QRectF()
        self._hovered_left_unremove = False
        self._hovered_right_unremove = False

        self.setMouseTracking(True)

    def set_split_line(self, line: QLineF) -> None:
        """Set the split line.

        Args:
            line: Split line in virtual image coordinates.
        """
        self._split_line = line
        self.update()

    @Slot(PageLayout)
    def page_layout_set_externally(self, layout: PageLayout) -> None:
        """Slot to set page layout from external source.

        Args:
            layout: The page layout.
        """
        # Extract split line from layout
        # This would depend on the actual PageLayout implementation
        self.update()

    def set_page_removal_state(self, left_removed: bool, right_removed: bool) -> None:
        """Set which page halves are removed.

        Args:
            left_removed: Whether left page is removed.
            right_removed: Whether right page is removed.
        """
        self._left_page_removed = left_removed
        self._right_page_removed = right_removed
        self.update()

    def _get_widget_split_line(self) -> QLineF:
        """Get the split line in widget coordinates.

        Returns:
            Split line in widget coordinates.
        """
        if self._split_line.isNull():
            return QLineF()

        return QLineF(
            self.virtual_to_widget_point(self._split_line.p1()),
            self.virtual_to_widget_point(self._split_line.p2()),
        )

    def _get_handle_positions(self) -> tuple[QPointF, QPointF]:
        """Get positions of the split line handles.

        Returns:
            Tuple of (top_handle, bottom_handle) positions in widget coordinates.
        """
        widget_line = self._get_widget_split_line()
        return widget_line.p1(), widget_line.p2()

    def _handle_at_position(self, pos: QPointF) -> int:
        """Check if a position is over a handle.

        Args:
            pos: Position in widget coordinates.

        Returns:
            0 for top handle, 1 for bottom handle, -1 if not over a handle.
        """
        if self._split_line.isNull():
            return -1

        top_handle, bottom_handle = self._get_handle_positions()

        # Check top handle
        dist = math.sqrt(
            (pos.x() - top_handle.x()) ** 2 + (pos.y() - top_handle.y()) ** 2
        )
        if dist <= self.HANDLE_HIT_RADIUS:
            return 0

        # Check bottom handle
        dist = math.sqrt(
            (pos.x() - bottom_handle.x()) ** 2 + (pos.y() - bottom_handle.y()) ** 2
        )
        if dist <= self.HANDLE_HIT_RADIUS:
            return 1

        return -1

    def _line_at_position(self, pos: QPointF) -> bool:
        """Check if a position is near the split line.

        Args:
            pos: Position in widget coordinates.

        Returns:
            True if near the line.
        """
        if self._split_line.isNull():
            return False

        widget_line = self._get_widget_split_line()
        dist = self._point_to_line_distance(pos, widget_line)
        return dist <= self.LINE_HIT_DISTANCE

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

    def _move_handle(self, handle_idx: int, pos: QPointF) -> None:
        """Move a handle to a new position.

        Args:
            handle_idx: Which handle (0 = top, 1 = bottom).
            pos: New position in widget coordinates.
        """
        virtual_pos = self.widget_to_virtual_point(pos)

        if handle_idx == 0:
            # Move top endpoint, keep bottom fixed
            self._split_line.setP1(virtual_pos)
        else:
            # Move bottom endpoint, keep top fixed
            self._split_line.setP2(virtual_pos)

    def _move_line(self, delta_x: float) -> None:
        """Move the entire split line horizontally.

        Args:
            delta_x: Horizontal movement in virtual coordinates.
        """
        new_line = QLineF(
            self._split_line.x1() + delta_x,
            self._split_line.y1(),
            self._split_line.x2() + delta_x,
            self._split_line.y2(),
        )
        self._split_line = new_line

    def _get_left_page_center(self) -> QPointF:
        """Get the center of the left page area.

        Returns:
            Center point in widget coordinates.
        """
        display_rect = self.virtual_to_widget_rect(self._virtual_display_area)
        widget_line = self._get_widget_split_line()

        if widget_line.isNull():
            return display_rect.center()

        left_center_x = (display_rect.left() + widget_line.x1()) / 2
        center_y = display_rect.center().y()
        return QPointF(left_center_x, center_y)

    def _get_right_page_center(self) -> QPointF:
        """Get the center of the right page area.

        Returns:
            Center point in widget coordinates.
        """
        display_rect = self.virtual_to_widget_rect(self._virtual_display_area)
        widget_line = self._get_widget_split_line()

        if widget_line.isNull():
            return display_rect.center()

        right_center_x = (widget_line.x1() + display_rect.right()) / 2
        center_y = display_rect.center().y()
        return QPointF(right_center_x, center_y)

    def _update_unremove_button_rects(self) -> None:
        """Update the unremove button rectangles."""
        button_size = 40

        if self._left_page_removed:
            center = self._get_left_page_center()
            self._left_unremove_rect = QRectF(
                center.x() - button_size / 2,
                center.y() - button_size / 2,
                button_size,
                button_size,
            )
        else:
            self._left_unremove_rect = QRectF()

        if self._right_page_removed:
            center = self._get_right_page_center()
            self._right_unremove_rect = QRectF(
                center.x() - button_size / 2,
                center.y() - button_size / 2,
                button_size,
                button_size,
            )
        else:
            self._right_unremove_rect = QRectF()

    def _paint_overlay(self, painter: QPainter) -> None:
        """Paint the split line overlay and page regions.

        Args:
            painter: The painter.
        """
        self._update_unremove_button_rects()

        # Draw page regions
        self._paint_page_regions(painter)

        # Draw split line
        if not self._split_line.isNull():
            self._paint_split_line(painter)

        # Draw unremove buttons
        self._paint_unremove_buttons(painter)

    def _paint_page_regions(self, painter: QPainter) -> None:
        """Paint the page region overlays.

        Args:
            painter: The painter.
        """
        display_rect = self.virtual_to_widget_rect(self._virtual_display_area)
        widget_line = self._get_widget_split_line()

        if widget_line.isNull():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Left page region
        left_rect = QRectF(
            display_rect.left(),
            display_rect.top(),
            widget_line.x1() - display_rect.left(),
            display_rect.height(),
        )
        if self._left_page_removed:
            painter.fillRect(left_rect, self.REMOVED_PAGE_COLOR)
        else:
            painter.fillRect(left_rect, self.LEFT_PAGE_COLOR)

        # Right page region
        right_rect = QRectF(
            widget_line.x1(),
            display_rect.top(),
            display_rect.right() - widget_line.x1(),
            display_rect.height(),
        )
        if self._right_page_removed:
            painter.fillRect(right_rect, self.REMOVED_PAGE_COLOR)
        else:
            painter.fillRect(right_rect, self.RIGHT_PAGE_COLOR)

        painter.restore()

    def _paint_split_line(self, painter: QPainter) -> None:
        """Paint the split line and handles.

        Args:
            painter: The painter.
        """
        widget_line = self._get_widget_split_line()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Draw line
        is_line_active = self._dragging_line or self._hovered_line
        line_color = (
            self.SPLIT_LINE_HIGHLIGHT if is_line_active else self.SPLIT_LINE_COLOR
        )
        pen = QPen(line_color)
        pen.setWidth(3 if is_line_active else 2)
        painter.setPen(pen)
        painter.drawLine(widget_line)

        # Draw handles
        top_handle, bottom_handle = self._get_handle_positions()

        for i, handle_pos in enumerate([top_handle, bottom_handle]):
            is_active = self._dragging_handle == i or self._hovered_handle == i
            color = self.HANDLE_HIGHLIGHT if is_active else self.HANDLE_COLOR
            radius = self.HANDLE_RADIUS + 2 if is_active else self.HANDLE_RADIUS

            painter.setBrush(color)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(handle_pos, radius, radius)

        painter.restore()

    def _paint_unremove_buttons(self, painter: QPainter) -> None:
        """Paint the unremove buttons for removed pages.

        Args:
            painter: The painter.
        """
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for rect, is_hovered, label in [
            (self._left_unremove_rect, self._hovered_left_unremove, "L"),
            (self._right_unremove_rect, self._hovered_right_unremove, "R"),
        ]:
            if rect.isEmpty():
                continue

            # Button background
            bg_color = QColor(0, 150, 0, 200) if is_hovered else QColor(0, 150, 0, 150)
            painter.setBrush(bg_color)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawRoundedRect(rect, 5, 5)

            # Plus icon
            painter.setPen(QPen(Qt.GlobalColor.white, 3))
            center = rect.center()
            size = rect.width() * 0.3
            painter.drawLine(
                QPointF(center.x() - size, center.y()),
                QPointF(center.x() + size, center.y()),
            )
            painter.drawLine(
                QPointF(center.x(), center.y() - size),
                QPointF(center.x(), center.y() + size),
            )

        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press for line and handle manipulation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()

            # Check unremove buttons first
            if (
                not self._left_unremove_rect.isEmpty()
                and self._left_unremove_rect.contains(pos)
            ):
                self._left_page_removed = False
                self.update()
                self.page_layout_set_locally.emit(None)  # Would emit actual layout
                event.accept()
                return

            if (
                not self._right_unremove_rect.isEmpty()
                and self._right_unremove_rect.contains(pos)
            ):
                self._right_page_removed = False
                self.update()
                self.page_layout_set_locally.emit(None)  # Would emit actual layout
                event.accept()
                return

            # Check handles
            handle = self._handle_at_position(pos)
            if handle >= 0:
                self._dragging_handle = handle
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                event.accept()
                return

            # Check line
            if self._line_at_position(pos):
                self._dragging_line = True
                virtual_pos = self.widget_to_virtual_point(pos)
                self._line_drag_offset = virtual_pos.x() - self._split_line.x1()
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release after manipulation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging_handle >= 0:
                self._dragging_handle = -1
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.page_layout_set_locally.emit(None)  # Would emit actual layout
                self.invalidate_thumbnail.emit()
                event.accept()
                self.update()
                return

            if self._dragging_line:
                self._dragging_line = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.page_layout_set_locally.emit(None)  # Would emit actual layout
                self.invalidate_thumbnail.emit()
                event.accept()
                self.update()
                return

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move for manipulation and cursor updates.

        Args:
            event: The mouse event.
        """
        pos = event.position()
        virtual_pos = self.widget_to_virtual_point(pos)

        # Handle dragging
        if self._dragging_handle >= 0:
            self._move_handle(self._dragging_handle, pos)
            self.update()
            event.accept()
            return

        if self._dragging_line:
            new_x = virtual_pos.x() - self._line_drag_offset
            delta_x = new_x - self._split_line.x1()
            self._move_line(delta_x)
            self.update()
            event.accept()
            return

        # Update hover state
        self._hovered_handle = self._handle_at_position(pos)
        self._hovered_line = self._line_at_position(pos)
        self._hovered_left_unremove = (
            not self._left_unremove_rect.isEmpty()
            and self._left_unremove_rect.contains(pos)
        )
        self._hovered_right_unremove = (
            not self._right_unremove_rect.isEmpty()
            and self._right_unremove_rect.contains(pos)
        )

        # Update cursor
        if self._hovered_handle >= 0:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._hovered_line:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self._hovered_left_unremove or self._hovered_right_unremove:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif not self._dragging:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self.update()
        super().mouseMoveEvent(event)
