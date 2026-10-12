"""Deskew filter image view.

Shows the page rotated by the current deskew angle under an alignment grid.
Two handles on arcs at the left and right of the window set the angle; the
page follows them while they are dragged. Ctrl+Wheel and the arrow keys
rotate in small steps.
"""

import copy
import math

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QLineF, QLocale, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.transformation import ImageTransformation


def format_angle(degrees: float) -> str:
    """Format an angle as the Deskew spin box shows it.

    Args:
        degrees (float): Angle in degrees.

    Returns:
        str: Two decimals and a degree sign, rounded by Qt's number formatting so
        the view label and the spin box never disagree on ties such as 0.125;
        a value that rounds to zero carries no minus sign.
    """
    locale = QLocale()
    text = locale.toString(degrees, ord("f"), 2)
    if not any(digit in text for digit in "123456789"):
        text = text.removeprefix(locale.negativeSign())
    return f"{text}°"


class DeskewImageView(ImageViewBase):
    """Image view of the Deskew stage with interactive angle adjustment.

    The transformation given to the view is the page without rotation; the
    view adds the deskew angle on top of it.

    Signals:
        manual_deskew_angle_set: The user set an angle. Args: degrees (float),
            clockwise positive.
    """

    manual_deskew_angle_set = Signal(float)

    wheel_zoom_at_center = True

    MAX_ROTATION_DEG = 45.0
    MAX_ROTATION_SIN = math.sin(math.radians(MAX_ROTATION_DEG))
    CELL_SIZE = 20  # Grid cell size in widget pixels
    HANDLE_RADIUS = 8
    HANDLE_HIT_RADIUS = 15
    WHEEL_STEP_DEG = 0.1
    WHEEL_FINE_STEP_DEG = 0.05
    ARROW_STEP_DEG = 0.5
    SHORTCUT_STEP_DEG = 0.1
    # Wheel notch size in QWheelEvent angle units.
    WHEEL_NOTCH = 120

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._deskew_angle = 0.0
        self._base_transformation: ImageTransformation | None = None
        self._dragging_handle = -1
        self._drag_offset_y = 0.0
        self._show_grid = True

        for key, step in (
            (",", -self.SHORTCUT_STEP_DEG),
            (".", self.SHORTCUT_STEP_DEG),
        ):
            action = QtGui.QAction(self)
            action.setShortcut(QtGui.QKeySequence(key))
            action.triggered.connect(lambda _checked=False, s=step: self._rotate_by(s))
            self.addAction(action)

        self.setStatusTip(
            "Use Ctrl+Wheel to rotate or Ctrl+Shift+Wheel for finer rotation."
        )

    def set_image(
        self,
        image: QImage | NDArray[np.uint8],
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        keep_view: bool = False,
    ) -> None:
        """Set the image and the page's unrotated transformation.

        Args:
            image (QImage | NDArray[np.uint8]): The image to display.
            downscaled_image (QImage | None): Optional pre-downscaled version.
            transformation (ImageTransformation | None): Maps image to virtual
                coordinates without the deskew rotation.
            keep_view (bool): Keep the zoom and the image point in view.
        """
        self._base_transformation = transformation
        rotated = None if transformation is None else self._rotated(transformation)
        super().set_image(image, downscaled_image, rotated, keep_view)

    def set_transformation(
        self, transformation: ImageTransformation, keep_view: bool = True
    ) -> None:
        """Change the page's unrotated transformation.

        Args:
            transformation (ImageTransformation): Maps image to virtual
                coordinates without the deskew rotation.
            keep_view (bool): Keep the zoom and the image point in view.
        """
        self._base_transformation = transformation
        super().set_transformation(self._rotated(transformation), keep_view)

    def _rotated(self, transformation: ImageTransformation) -> ImageTransformation:
        rotated = copy.deepcopy(transformation)
        rotated.set_post_rotation(self._deskew_angle)
        return rotated

    @property
    def deskew_angle(self) -> float:
        """The deskew angle in degrees, clockwise positive."""
        return self._deskew_angle

    def set_deskew_angle(self, degrees: float) -> None:
        """Rotate the page to an angle without reporting it as a user edit.

        Args:
            degrees (float): The deskew angle in degrees.
        """
        degrees = max(-self.MAX_ROTATION_DEG, min(self.MAX_ROTATION_DEG, degrees))
        if abs(degrees - self._deskew_angle) < 1e-6:
            return
        self._deskew_angle = degrees
        if self._base_transformation is not None:
            ImageViewBase.set_transformation(
                self, self._rotated(self._base_transformation), keep_view=True
            )
        self.update()

    def _rotate_by(self, delta_deg: float) -> None:
        """Rotate by a step and report it."""
        before = self._deskew_angle
        self.set_deskew_angle(before + delta_deg)
        if self._deskew_angle != before:
            self.manual_deskew_angle_set.emit(self._deskew_angle)

    def set_show_grid(self, show: bool) -> None:
        """Show or hide the alignment grid.

        Args:
            show (bool): True to show the grid.
        """
        self._show_grid = show
        self.update()

    def _rotation_origin(self) -> QPointF:
        viewport = self.viewport()
        assert viewport is not None
        return QPointF(
            math.floor(0.5 * viewport.width()) + 0.5,
            math.floor(0.5 * viewport.height()) + 0.5,
        )

    def rotation_arc_square(self) -> QRectF:
        """Return the square whose left and right arcs carry the handles.

        Returns:
            QRectF: Square in widget coordinates; empty when the window is too
            small.
        """
        viewport = self.viewport()
        assert viewport is not None
        margin = self.HANDLE_RADIUS + 20
        reduced = QRectF(
            margin,
            margin,
            viewport.width() - 2 * margin,
            viewport.height() - 2 * margin,
        )
        if reduced.width() <= 0 or reduced.height() <= 0:
            return QRectF()

        size = min(reduced.width(), reduced.height() / self.MAX_ROTATION_SIN)
        square = QRectF(0, 0, size, size)
        square.moveCenter(reduced.center())
        return square

    def rotation_handles(self) -> tuple[QPointF, QPointF]:
        """Return the positions of the two rotation handles.

        Returns:
            tuple[QPointF, QPointF]: Left and right handle in widget
            coordinates; null points when the window is too small.
        """
        square = self.rotation_arc_square()
        if square.isEmpty():
            return QPointF(), QPointF()

        radians = math.radians(self._deskew_angle)
        radius = 0.5 * square.width()
        center = square.center()
        sin, cos = math.sin(radians), math.cos(radians)
        return (
            QPointF(-cos * radius + center.x(), -sin * radius + center.y()),
            QPointF(cos * radius + center.x(), sin * radius + center.y()),
        )

    def _paint_overlay(self, painter: QPainter) -> None:
        viewport = self.viewport()
        assert viewport is not None
        if self._show_grid:
            self._paint_grid(
                painter, viewport.width(), viewport.height(), self._rotation_origin()
            )
        self._paint_rotation_arcs(painter)

    def _paint_grid(
        self, painter: QPainter, width: float, height: float, center: QPointF
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        pen = QPen(view_color("deskew_grid"))
        pen.setCosmetic(True)
        painter.setPen(pen)

        step = self.CELL_SIZE
        ys = [
            *np.arange(center.y() - step, 0, -step),
            *np.arange(center.y() + step, height, step),
        ]
        xs = [
            *np.arange(center.x() - step, 0, -step),
            *np.arange(center.x() + step, width, step),
        ]
        for y in ys:
            painter.drawLine(QLineF(0.5, y, width - 0.5, y))
        for x in xs:
            painter.drawLine(QLineF(x, 0.5, x, height - 0.5))

        axis_pen = QPen(view_color("deskew_axis"))
        axis_pen.setCosmetic(True)
        painter.setPen(axis_pen)
        painter.drawLine(QPointF(0.5, center.y()), QPointF(width - 0.5, center.y()))
        painter.drawLine(QPointF(center.x(), 0.5), QPointF(center.x(), height - 0.5))
        painter.restore()

    def _paint_rotation_arcs(self, painter: QPainter) -> None:
        square = self.rotation_arc_square()
        if square.isEmpty():
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        accent = self.palette().color(QPalette.ColorRole.Accent)
        arc_pen = QPen(accent)
        arc_pen.setWidthF(1.5)
        painter.setPen(arc_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        span = int(16 * 2 * self.MAX_ROTATION_DEG)
        painter.drawArc(square, int(16 * (180 - self.MAX_ROTATION_DEG)), span)
        painter.drawArc(square, int(16 * -self.MAX_ROTATION_DEG), span)

        handle_pen = QPen(accent)
        handle_pen.setWidth(2)
        painter.setPen(handle_pen)
        for index, handle in enumerate(self.rotation_handles()):
            fill = QColor(accent)
            fill.setAlpha(200 if self._dragging_handle == index else 150)
            painter.setBrush(fill)
            painter.drawEllipse(handle, self.HANDLE_RADIUS, self.HANDLE_RADIUS)

        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            QRectF(square.center().x() - 30, square.bottom() + 5, 60, 20),
            Qt.AlignmentFlag.AlignCenter,
            format_angle(self._deskew_angle),
        )
        painter.restore()

    def _handle_at(self, pos: QPointF) -> int:
        """Return 0 for the left handle, 1 for the right one, -1 for neither."""
        for index, handle in enumerate(self.rotation_handles()):
            if math.hypot(pos.x() - handle.x(), pos.y() - handle.y()) <= (
                self.HANDLE_HIT_RADIUS
            ):
                return index
        return -1

    def _angle_for_handle_position(self, handle: int, pos: QPointF) -> float:
        square = self.rotation_arc_square()
        radius = 0.5 * square.width()
        if radius <= 0:
            return self._deskew_angle
        rel_y = max(-radius, min(radius, pos.y() - square.center().y()))
        angle = math.degrees(math.asin(rel_y / radius))
        return -angle if handle == 0 else angle

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pick up a handle.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._handle_at(event.position())
            if handle >= 0:
                self._dragging_handle = handle
                self._drag_offset_y = (
                    event.position().y() - self.rotation_handles()[handle].y()
                )
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Rotate while a handle is dragged; show the hand cursor over handles.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._dragging_handle >= 0:
            self.set_deskew_angle(
                self._angle_for_handle_position(
                    self._dragging_handle,
                    event.position() - QPointF(0, self._drag_offset_y),
                )
            )
            event.accept()
            return
        if not self._panning:
            over_handle = self._handle_at(event.position()) >= 0
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if over_handle
                else Qt.CursorShape.ArrowCursor
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop the handle and report the angle.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_handle >= 0:
            self._dragging_handle = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.manual_deskew_angle_set.emit(self._deskew_angle)
            event.accept()
            self.update()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Rotate with Ctrl held, zoom otherwise.

        Args:
            event (QtGui.QWheelEvent): The wheel event.
        """
        modifiers = event.modifiers()
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            step = self.WHEEL_STEP_DEG
        elif modifiers == (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        ):
            step = self.WHEEL_FINE_STEP_DEG
        else:
            super().wheelEvent(event)
            return
        event.accept()
        self._rotate_by(-step * event.angleDelta().y() / self.WHEEL_NOTCH)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Rotate with the left and right arrow keys.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        if event.key() == Qt.Key.Key_Left:
            self._rotate_by(-self.ARROW_STEP_DEG)
        elif event.key() == Qt.Key.Key_Right:
            self._rotate_by(self.ARROW_STEP_DEG)
        else:
            super().keyPressEvent(event)
            return
        event.accept()
