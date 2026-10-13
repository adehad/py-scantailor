"""Picking a fill colour from what an image view shows.

The picker samples a small circle around the cursor and takes the median of its
colours. The view owns the mouse and paint events, so it forwards them to the
interaction, which reports whether it consumed them.
"""

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen

PICKUP_RADIUS = 7  # The sampled square is 15 by 15 pixels, odd so it has a centre.
_BITS_PER_CHANNEL = 8
_CHANNELS = 3


def _spread_bits_table(channel: int) -> NDArray[np.uint32]:
    """Map each channel value to its bits at their places in the mixed value."""
    values = np.arange(1 << _BITS_PER_CHANNEL, dtype=np.uint32)
    spread = np.zeros_like(values)
    for bit in range(_BITS_PER_CHANNEL):
        source = _BITS_PER_CHANNEL - 1 - bit
        target = 3 * (_BITS_PER_CHANNEL - bit) - 1 - channel
        spread |= ((values >> source) & 1) << target
    return spread


_SPREAD_BITS = tuple(_spread_bits_table(channel) for channel in range(_CHANNELS))


def bit_mix(rgb: NDArray[np.uint8]) -> NDArray[np.uint32]:
    """Interleave the bits of the colour channels, most significant first.

    Sorting by the mixed value orders colours by their coarse tone before their
    fine detail, so the median of the mixed values is a colour that is really
    present and typical, unlike a median taken per channel.

    Args:
        rgb (NDArray[np.uint8]): Colours as rows of (red, green, blue), any
            leading shape.

    Returns:
        NDArray[np.uint32]: One 24 bit value per colour, with the red bit of
            each triple first.
    """
    red, green, blue = (
        _SPREAD_BITS[channel][rgb[..., channel]] for channel in range(3)
    )
    return red | green | blue


def bit_unmix(mixed: int) -> tuple[int, int, int]:
    """Undo ``bit_mix`` for one value.

    Args:
        mixed (int): A bit mixed colour.

    Returns:
        tuple[int, int, int]: Red, green and blue.
    """
    channels = [0, 0, 0]
    for bit in range(_BITS_PER_CHANNEL):
        target = _BITS_PER_CHANNEL - 1 - bit
        for channel in range(_CHANNELS):
            source = 3 * (_BITS_PER_CHANNEL - bit) - 1 - channel
            if mixed >> source & 1:
                channels[channel] |= 1 << target
    return channels[0], channels[1], channels[2]


def median_color(rgb: NDArray[np.uint8]) -> tuple[int, int, int] | None:
    """Return the median colour of the pixels inside the pickup circle.

    Args:
        rgb (NDArray[np.uint8]): Square patch of shape (size, size, 3) with an
            odd size; the circle is centred on it.

    Returns:
        tuple[int, int, int] | None: Red, green and blue of the median, None
            when the patch holds no pixel.
    """
    height, width = rgb.shape[:2]
    y_center, x_center = height // 2, width // 2
    rows, columns = np.ogrid[:height, :width]
    inside = (rows - y_center) ** 2 + (columns - x_center) ** 2 <= x_center * y_center
    mixed = bit_mix(rgb[inside])
    if mixed.size == 0:
        return None
    middle = mixed.size // 2
    return bit_unmix(int(np.partition(mixed, middle)[middle]))


class ColorPickupInteraction(QtCore.QObject):
    """Lets the user click a colour in an image view.

    While active the cursor is a cross and a red circle marks the sampled area.
    A left click picks the median colour under the circle; Escape cancels.

    Signals:
        color_picked: A colour was picked (QColor).
        cancelled: The pickup ended without a colour.
    """

    color_picked = Signal(QColor)
    cancelled = Signal()

    def __init__(
        self, view: QtWidgets.QAbstractScrollArea, parent: QtCore.QObject | None = None
    ) -> None:
        """Create the interaction.

        Args:
            view (QtWidgets.QAbstractScrollArea): The view whose viewport is
                sampled and drawn on.
            parent (QtCore.QObject | None): Parent object.
        """
        super().__init__(parent or view)
        self._view = view
        self._active = False
        self._sampling = False
        self._cursor = QPoint()

    def is_active(self) -> bool:
        """Return whether a colour is being picked.

        Returns:
            bool: True between ``start`` and the pick or cancel.
        """
        return self._active

    def start(self, cursor: QPoint | None = None) -> None:
        """Begin picking.

        Args:
            cursor (QPoint | None): Where the circle starts, in viewport
                coordinates; the current cursor position when None.
        """
        viewport = self._viewport()
        self._cursor = (
            cursor
            if cursor is not None
            else viewport.mapFromGlobal(QtGui.QCursor.pos())
        )
        self._active = True
        viewport.setCursor(Qt.CursorShape.CrossCursor)
        viewport.update()

    def cancel(self) -> None:
        """Stop without a colour."""
        if self._active:
            self._finish()
            self.cancelled.emit()

    def pick_at(self, center: QPoint) -> QColor | None:
        """Sample the viewport around a point and report the median colour.

        Args:
            center (QPoint): Centre of the circle in viewport coordinates.

        Returns:
            QColor | None: The colour, also sent with ``color_picked``; None when
                nothing could be sampled, in which case the pickup goes on.
        """
        rgb = self._sample(center)
        picked = None if rgb is None else median_color(rgb)
        if picked is None:
            return None
        color = QColor(*picked)
        self._finish()
        self.color_picked.emit(color)
        return color

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> bool:
        """Pick on a left click.

        Args:
            event (QtGui.QMouseEvent): The press.

        Returns:
            bool: True when the interaction used the event.
        """
        if not self._active:
            return False
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.pick_at(event.position().toPoint())
        event.accept()
        return True

    def handle_mouse_move(self, event: QtGui.QMouseEvent) -> bool:
        """Move the circle with the cursor.

        Args:
            event (QtGui.QMouseEvent): The move.

        Returns:
            bool: True when the interaction used the event.
        """
        if not self._active:
            return False
        self._cursor = event.position().toPoint()
        self._viewport().update()
        event.accept()
        return True

    def handle_key_press(self, event: QtGui.QKeyEvent) -> bool:
        """Cancel on Escape.

        Args:
            event (QtGui.QKeyEvent): The key press.

        Returns:
            bool: True when the interaction used the event.
        """
        if self._active and event.key() == Qt.Key.Key_Escape:
            self.cancel()
            event.accept()
            return True
        return False

    def paint_overlay(self, painter: QPainter) -> None:
        """Draw the circle; call it last in the view's paint.

        Args:
            painter (QPainter): Painter on the viewport.
        """
        if not self._active or self._sampling:
            return
        painter.save()
        painter.setWorldTransform(QtGui.QTransform())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(self._target_rect(self._cursor))
        painter.restore()

    def _viewport(self) -> QtWidgets.QWidget:
        viewport = self._view.viewport()
        assert viewport is not None
        return viewport

    def _finish(self) -> None:
        self._active = False
        viewport = self._viewport()
        viewport.setCursor(Qt.CursorShape.ArrowCursor)
        viewport.update()

    @staticmethod
    def _target_rect(center: QPoint) -> QRect:
        rect = QRect(0, 0, 2 * PICKUP_RADIUS + 1, 2 * PICKUP_RADIUS + 1)
        rect.moveCenter(center)
        return rect

    def _sample(self, center: QPoint) -> NDArray[np.uint8] | None:
        """Grab what the viewport shows around a point, without the circle."""
        rect = self._target_rect(center)
        self._sampling = True
        try:
            pixmap = self._viewport().grab(rect)
        finally:
            self._sampling = False
        if pixmap.isNull():
            return None
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        width, height = image.width(), image.height()
        if width != rect.width() or height != rect.height():
            return None
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8)
        rows = pixels.reshape(height, image.bytesPerLine())[:, : width * _CHANNELS]
        return rows.reshape(height, width, _CHANNELS).copy()
