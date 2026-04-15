"""Deskew filter image view.

Provides an interactive image view for the Deskew filter with:
- Grid overlay to visualize alignment
- Draggable rotation handles
- Ctrl+Wheel rotation with fine adjustment using Ctrl+Shift
"""

import math

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.core.transformation import ImageTransformation


class DeskewImageView(ImageViewBase):
    """Image view for the Deskew filter with interactive angle adjustment.

    Provides visual feedback with a grid overlay and draggable rotation
    handles on rotation arcs at the left and right sides of the image.

    Signals:
        manual_deskew_angle_set: Emitted when user manually sets deskew angle (degrees).
    """

    manual_deskew_angle_set = Signal(float)

    # Constants
    MAX_ROTATION_DEG = 45.0
    MAX_ROTATION_SIN = math.sin(math.radians(MAX_ROTATION_DEG))
    CELL_SIZE = 20  # Grid cell size in pixels
    HANDLE_RADIUS = 8  # Rotation handle radius
    HANDLE_HIT_RADIUS = 15  # Hit detection radius for handles

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the Deskew image view.

        Args:
            image: The image to display.
            downscaled_image: Optional pre-downscaled version.
            transformation: Image transformation.
            parent: Parent widget.
        """
        super().__init__(image, downscaled_image, transformation, parent=parent)

        self._deskew_angle = 0.0  # Current deskew angle in degrees
        self._dragging_handle = (
            -1
        )  # -1 = not dragging, 0 = left handle, 1 = right handle
        self._show_grid = True

        # Set up keyboard shortcuts
        self._setup_shortcuts()

        # Status tip
        self.setStatusTip(
            "Use Ctrl+Wheel to rotate or Ctrl+Shift+Wheel for finer rotation."
        )

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts for rotation."""
        # Rotate left with comma key
        rotate_left = QtGui.QAction(self)
        rotate_left.setShortcut(QtGui.QKeySequence(","))
        rotate_left.triggered.connect(self._do_rotate_left)
        self.addAction(rotate_left)

        # Rotate right with period key
        rotate_right = QtGui.QAction(self)
        rotate_right.setShortcut(QtGui.QKeySequence("."))
        rotate_right.triggered.connect(self._do_rotate_right)
        self.addAction(rotate_right)

    def _do_rotate_left(self) -> None:
        """Rotate left (counter-clockwise) by a small amount."""
        self._do_rotate(-0.10)

    def _do_rotate_right(self) -> None:
        """Rotate right (clockwise) by a small amount."""
        self._do_rotate(0.10)

    def _do_rotate(self, delta_deg: float) -> None:
        """Rotate by the given delta.

        Args:
            delta_deg: Rotation delta in degrees.
        """
        new_angle = self._deskew_angle + delta_deg
        new_angle = max(-self.MAX_ROTATION_DEG, min(self.MAX_ROTATION_DEG, new_angle))
        if abs(new_angle - self._deskew_angle) > 0.001:
            self._deskew_angle = new_angle
            self.update()
            self.manual_deskew_angle_set.emit(self._deskew_angle)

    @property
    def deskew_angle(self) -> float:
        """Get the current deskew angle in degrees."""
        return self._deskew_angle

    def set_deskew_angle(self, degrees: float) -> None:
        """Set the deskew angle.

        Args:
            degrees: The deskew angle in degrees.
        """
        degrees = max(-self.MAX_ROTATION_DEG, min(self.MAX_ROTATION_DEG, degrees))
        if abs(degrees - self._deskew_angle) > 0.001:
            self._deskew_angle = degrees
            self.update()

    @Slot(float)
    def manual_deskew_angle_set_externally(self, degrees: float) -> None:
        """Slot to set deskew angle from external source.

        Args:
            degrees: The deskew angle in degrees.
        """
        self.set_deskew_angle(degrees)

    def set_show_grid(self, show: bool) -> None:
        """Set whether to show the alignment grid.

        Args:
            show: True to show the grid.
        """
        self._show_grid = show
        self.update()

    def _get_rotation_origin(self) -> QPointF:
        """Get the rotation origin point in widget coordinates.

        Returns:
            Center point of the viewport.
        """
        viewport = self.viewport()
        if viewport is None:
            return QPointF()

        w = viewport.width()
        h = viewport.height()
        return QPointF(math.floor(0.5 * w) + 0.5, math.floor(0.5 * h) + 0.5)

    def _get_rotation_arc_square(self) -> QRectF:
        """Get the square where rotation arcs are drawn.

        Returns:
            Rectangle for drawing the rotation arcs.
        """
        viewport = self.viewport()
        if viewport is None:
            return QRectF()

        # Account for handle size and scrollbar margins
        h_margin = self.HANDLE_RADIUS + 20
        v_margin = self.HANDLE_RADIUS + 20

        reduced_rect = QRectF(
            h_margin,
            v_margin,
            viewport.width() - 2 * h_margin,
            viewport.height() - 2 * v_margin,
        )

        if reduced_rect.width() <= 0 or reduced_rect.height() <= 0:
            return QRectF()

        # Scale to fit while maintaining aspect ratio for the arc
        arc_width = min(
            reduced_rect.width(), reduced_rect.height() / self.MAX_ROTATION_SIN
        )
        arc_height = arc_width  # Square for circular arcs

        arc_rect = QRectF(0, 0, arc_width, arc_height)
        arc_rect.moveCenter(reduced_rect.center())

        return arc_rect

    def _get_rotation_handles(self, arc_square: QRectF) -> tuple[QPointF, QPointF]:
        """Get the positions of the two rotation handles.

        Args:
            arc_square: The arc square rectangle.

        Returns:
            Tuple of (left_handle, right_handle) positions.
        """
        if arc_square.isEmpty():
            return QPointF(), QPointF()

        rot_rad = math.radians(self._deskew_angle)
        rot_sin = math.sin(rot_rad)
        rot_cos = math.cos(rot_rad)

        arc_radius = 0.5 * arc_square.width()
        arc_center = arc_square.center()

        # Left handle (on left arc)
        left_handle = QPointF(
            -rot_cos * arc_radius + arc_center.x(),
            -rot_sin * arc_radius + arc_center.y(),
        )

        # Right handle (on right arc)
        right_handle = QPointF(
            rot_cos * arc_radius + arc_center.x(),
            rot_sin * arc_radius + arc_center.y(),
        )

        return left_handle, right_handle

    def _paint_overlay(self, painter: QPainter) -> None:
        """Paint the grid overlay and rotation handles.

        Args:
            painter: The painter to draw with.
        """
        viewport = self.viewport()
        if viewport is None:
            return

        w = viewport.width()
        h = viewport.height()
        center = self._get_rotation_origin()

        # Draw grid
        if self._show_grid:
            self._paint_grid(painter, w, h, center)

        # Draw rotation arcs
        self._paint_rotation_arcs(painter)

    def _paint_grid(
        self, painter: QPainter, w: float, h: float, center: QPointF
    ) -> None:
        """Paint the alignment grid.

        Args:
            painter: The painter.
            w: Viewport width.
            h: Viewport height.
            center: Grid center point.
        """
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Semi-transparent grid lines
        grid_pen = QPen(QColor(0, 0, 0xD1, 90))
        grid_pen.setCosmetic(True)
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)

        lines: list[QLineF] = []

        # Horizontal lines
        y = center.y()
        while (y := y - self.CELL_SIZE) > 0:
            lines.append(QLineF(0.5, y, w - 0.5, y))
        y = center.y()
        while (y := y + self.CELL_SIZE) < h:
            lines.append(QLineF(0.5, y, w - 0.5, y))

        # Vertical lines
        x = center.x()
        while (x := x - self.CELL_SIZE) > 0:
            lines.append(QLineF(x, 0.5, x, h - 0.5))
        x = center.x()
        while (x := x + self.CELL_SIZE) < w:
            lines.append(QLineF(x, 0.5, x, h - 0.5))

        for line in lines:
            painter.drawLine(line)

        # Center crosshair (more visible)
        center_pen = QPen(QColor(0, 0, 0xD1))
        center_pen.setCosmetic(True)
        center_pen.setWidth(1)
        painter.setPen(center_pen)

        painter.drawLine(QPointF(0.5, center.y()), QPointF(w - 0.5, center.y()))
        painter.drawLine(QPointF(center.x(), 0.5), QPointF(center.x(), h - 0.5))

        painter.restore()

    def _paint_rotation_arcs(self, painter: QPainter) -> None:
        """Paint the rotation arcs and handles.

        Args:
            painter: The painter.
        """
        arc_square = self._get_rotation_arc_square()
        if arc_square.isEmpty():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Draw arcs
        arc_pen = QPen(QColor(0, 0, 0xD1))
        arc_pen.setWidthF(1.5)
        painter.setPen(arc_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Left arc (180 +/- max_rotation degrees)
        start_angle = int(16 * (180 - self.MAX_ROTATION_DEG))
        span_angle = int(16 * 2 * self.MAX_ROTATION_DEG)
        painter.drawArc(arc_square, start_angle, span_angle)

        # Right arc (0 +/- max_rotation degrees)
        start_angle = int(16 * (-self.MAX_ROTATION_DEG))
        span_angle = int(16 * 2 * self.MAX_ROTATION_DEG)
        painter.drawArc(arc_square, start_angle, span_angle)

        # Draw handles
        left_handle, right_handle = self._get_rotation_handles(arc_square)

        # Handle appearance
        handle_pen = QPen(QColor(0, 0, 0xD1))
        handle_pen.setWidth(2)
        painter.setPen(handle_pen)

        # Highlight if dragging
        if self._dragging_handle == 0:
            painter.setBrush(QColor(0, 0x99, 0xFF, 200))
        else:
            painter.setBrush(QColor(0, 0x99, 0xFF, 150))
        painter.drawEllipse(left_handle, self.HANDLE_RADIUS, self.HANDLE_RADIUS)

        if self._dragging_handle == 1:
            painter.setBrush(QColor(0, 0x99, 0xFF, 200))
        else:
            painter.setBrush(QColor(0, 0x99, 0xFF, 150))
        painter.drawEllipse(right_handle, self.HANDLE_RADIUS, self.HANDLE_RADIUS)

        # Draw current angle text
        angle_text = f"{self._deskew_angle:.2f}°"
        painter.setPen(QColor(0, 0, 0))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        text_rect = QRectF(
            arc_square.center().x() - 30,
            arc_square.bottom() + 5,
            60,
            20,
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, angle_text)

        painter.restore()

    def _handle_at_position(self, pos: QPointF) -> int:
        """Check if a position is over a rotation handle.

        Args:
            pos: Position in widget coordinates.

        Returns:
            0 for left handle, 1 for right handle, -1 if not over a handle.
        """
        arc_square = self._get_rotation_arc_square()
        if arc_square.isEmpty():
            return -1

        left_handle, right_handle = self._get_rotation_handles(arc_square)

        # Check left handle
        dist = math.sqrt(
            (pos.x() - left_handle.x()) ** 2 + (pos.y() - left_handle.y()) ** 2
        )
        if dist <= self.HANDLE_HIT_RADIUS:
            return 0

        # Check right handle
        dist = math.sqrt(
            (pos.x() - right_handle.x()) ** 2 + (pos.y() - right_handle.y()) ** 2
        )
        if dist <= self.HANDLE_HIT_RADIUS:
            return 1

        return -1

    def _handle_move_request(self, handle_idx: int, pos: QPointF) -> None:
        """Handle a request to move a rotation handle.

        Args:
            handle_idx: Which handle (0 = left, 1 = right).
            pos: New position in widget coordinates.
        """
        arc_square = self._get_rotation_arc_square()
        if arc_square.isEmpty():
            return

        arc_radius = 0.5 * arc_square.width()
        arc_center = arc_square.center()

        # Calculate angle from handle position
        rel_y = pos.y() - arc_center.y()
        rel_y = max(-arc_radius, min(arc_radius, rel_y))

        # Convert to angle
        if arc_radius > 0:
            angle_rad = math.asin(rel_y / arc_radius)
        else:
            angle_rad = 0

        # Left handle has inverted angle
        if handle_idx == 0:
            angle_rad = -angle_rad

        angle_deg = math.degrees(angle_rad)
        angle_deg = max(-self.MAX_ROTATION_DEG, min(self.MAX_ROTATION_DEG, angle_deg))

        if abs(angle_deg - self._deskew_angle) > 0.001:
            self._deskew_angle = angle_deg
            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press for handle dragging.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._handle_at_position(event.position())
            if handle >= 0:
                self._dragging_handle = handle
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                self.update()
                return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release for handle dragging.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_handle >= 0:
            self._dragging_handle = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.manual_deskew_angle_set.emit(self._deskew_angle)
            event.accept()
            self.update()
            return

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move for handle dragging and cursor updates.

        Args:
            event: The mouse event.
        """
        if self._dragging_handle >= 0:
            self._handle_move_request(self._dragging_handle, event.position())
            event.accept()
            return

        # Update cursor based on handle proximity
        handle = self._handle_at_position(event.position())
        if handle >= 0:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif not self._dragging:  # Don't override pan cursor
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Handle wheel event for rotation with Ctrl modifier.

        Args:
            event: The wheel event.
        """
        modifiers = event.modifiers()

        # Ctrl+Wheel for rotation
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            degree_fraction = 0.1
        elif modifiers == (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        ):
            degree_fraction = 0.05
        else:
            # Pass to parent for zoom
            super().wheelEvent(event)
            return

        event.accept()

        delta = degree_fraction * event.angleDelta().y() / 120
        new_angle = self._deskew_angle - delta
        new_angle = max(-self.MAX_ROTATION_DEG, min(self.MAX_ROTATION_DEG, new_angle))

        if abs(new_angle - self._deskew_angle) > 0.001:
            self._deskew_angle = new_angle
            self.update()
            self.manual_deskew_angle_set.emit(self._deskew_angle)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press for rotation shortcuts.

        Args:
            event: The key event.
        """
        if event.key() == Qt.Key.Key_Left:
            self._do_rotate(-0.5)
            event.accept()
        elif event.key() == Qt.Key.Key_Right:
            self._do_rotate(0.5)
            event.accept()
        else:
            super().keyPressEvent(event)
