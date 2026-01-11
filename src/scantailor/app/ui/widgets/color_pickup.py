"""Color pickup interaction widget for fill zone editing.

Provides a tool for picking colors from the image to use as fill colors
in fill zones. The picker shows a circular area around the cursor and
takes the median color from pixels within that area.
"""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap


class ColorPickupInteraction(QtCore.QObject):
    """Interactive color picker for fill zones.

    When activated, shows a circular pickup area that follows the cursor.
    Clicking picks the median color from pixels within the circle.

    The bit-mixing technique is used to get a perceptually better median
    color by interleaving the bits of RGB channels before sorting.

    Signals:
        color_picked: Emitted when a color is picked (QColor).
        cancelled: Emitted when the interaction is cancelled.
    """

    color_picked = Signal(QColor)
    cancelled = Signal()

    # Pickup circle size (odd for symmetry)
    PICKUP_RADIUS = 7  # 15x15 pixel area

    def __init__(
        self,
        image_view: QtWidgets.QWidget,
        parent: QtCore.QObject | None = None,
    ) -> None:
        """Initialize the color pickup interaction.

        Args:
            image_view: The image view widget to pick colors from.
            parent: Parent QObject.
        """
        super().__init__(parent)

        self._image_view = image_view
        self._active = False
        self._dont_draw_circle = False

        # Install event filter on the image view
        self._image_view.installEventFilter(self)

        # Set up escape shortcut
        self._escape_shortcut = QtGui.QShortcut(Qt.Key.Key_Escape, self._image_view)
        self._escape_shortcut.setAutoRepeat(False)
        self._escape_shortcut.setEnabled(False)
        self._escape_shortcut.activated.connect(self._cancel)

    def start(self) -> None:
        """Start the color pickup interaction."""
        self._active = True
        self._escape_shortcut.setEnabled(True)
        self._image_view.setCursor(Qt.CursorShape.CrossCursor)
        self._image_view.setMouseTracking(True)
        self._image_view.update()

    def stop(self) -> None:
        """Stop the color pickup interaction."""
        self._active = False
        self._escape_shortcut.setEnabled(False)
        self._image_view.setCursor(Qt.CursorShape.ArrowCursor)
        self._image_view.update()

    def is_active(self) -> bool:
        """Check if the interaction is active.

        Returns:
            True if active.
        """
        return self._active

    def _cancel(self) -> None:
        """Cancel the interaction."""
        self.stop()
        self.cancelled.emit()

    def _get_target_rect(self) -> QRect:
        """Get the pickup circle bounding rectangle.

        Returns:
            Rectangle centered on the cursor position.
        """
        mouse_pos = self._image_view.mapFromGlobal(QCursor.pos())
        size = self.PICKUP_RADIUS * 2 + 1  # Odd size for symmetry
        rect = QRect(0, 0, size, size)
        rect.moveCenter(mouse_pos)
        return rect

    def _take_color(self) -> None:
        """Pick the color from the current cursor position."""
        self._dont_draw_circle = True

        rect = self._get_target_rect()

        # Grab the pixels from the image view
        pixmap = self._image_view.grab(rect)
        if pixmap.isNull():
            self._dont_draw_circle = False
            return

        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)

        width = rect.width()
        height = rect.height()
        x_center = width // 2
        y_center = height // 2
        sqdist_threshold = x_center * y_center

        # Extract pixel data
        ptr = image.constBits()
        if ptr is None:
            self._dont_draw_circle = False
            return

        # Convert to numpy array for easier manipulation
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))

        # Collect colors from pixels within the circle
        bitmixed_colors: list[int] = []

        for y in range(height):
            dy = y - y_center
            dy_sq = dy * dy
            for x in range(width):
                dx = x - x_center
                dx_sq = dx * dx
                sqdist = dy_sq + dx_sq
                if sqdist <= sqdist_threshold:
                    # BGRA format in Qt
                    b, g, r = arr[y, x, 0], arr[y, x, 1], arr[y, x, 2]
                    color = (r << 16) | (g << 8) | b
                    bitmixed_colors.append(self._bit_mix_color(color))

        self._dont_draw_circle = False

        if not bitmixed_colors:
            return

        # Find median using bit-mixed values
        bitmixed_colors.sort()
        median_mixed = bitmixed_colors[len(bitmixed_colors) // 2]
        unmixed = self._bit_unmix_color(median_mixed)

        r = (unmixed >> 16) & 0xFF
        g = (unmixed >> 8) & 0xFF
        b = unmixed & 0xFF

        color = QColor(r, g, b)
        self.stop()
        self.color_picked.emit(color)

    @staticmethod
    def _bit_mix_color(color: int) -> int:
        """Mix the bits of RGB channels for better median calculation.

        This interleaves the bits of R, G, B channels so that sorting
        by the mixed value gives a perceptually better median.

        Args:
            color: RGB color as 0x00RRGGBB.

        Returns:
            Bit-mixed value.
        """
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF

        result = 0
        for bit in range(8):
            bit_mask = 1 << (7 - bit)
            if r & bit_mask:
                result |= 1 << (23 - bit * 3)
            if g & bit_mask:
                result |= 1 << (22 - bit * 3)
            if b & bit_mask:
                result |= 1 << (21 - bit * 3)

        return result

    @staticmethod
    def _bit_unmix_color(mixed: int) -> int:
        """Unmix the bits back to RGB channels.

        Args:
            mixed: Bit-mixed value.

        Returns:
            RGB color as 0x00RRGGBB.
        """
        r = 0
        g = 0
        b = 0

        for bit in range(8):
            r_bit = 23 - bit * 3
            g_bit = 22 - bit * 3
            b_bit = 21 - bit * 3
            out_bit = 7 - bit

            if mixed & (1 << r_bit):
                r |= 1 << out_bit
            if mixed & (1 << g_bit):
                g |= 1 << out_bit
            if mixed & (1 << b_bit):
                b |= 1 << out_bit

        return (r << 16) | (g << 8) | b

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Filter events on the image view.

        Args:
            watched: The watched object.
            event: The event.

        Returns:
            True if the event was handled.
        """
        if not self._active or watched is not self._image_view:
            return False

        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            mouse_event = event  # type: QtGui.QMouseEvent
            if mouse_event.button() == Qt.MouseButton.LeftButton:
                self._take_color()
                return True

        elif event.type() == QtCore.QEvent.Type.MouseMove:
            self._image_view.update()
            return False  # Don't consume, allow view to update

        elif event.type() == QtCore.QEvent.Type.Paint:
            # We need to draw our overlay after the view paints
            # This is handled differently - we call paint_overlay from the view
            pass

        return False

    def paint_overlay(self, painter: QPainter) -> None:
        """Paint the pickup circle overlay.

        This should be called from the image view's paint event.

        Args:
            painter: The painter to draw with.
        """
        if not self._active or self._dont_draw_circle:
            return

        painter.save()
        painter.setWorldTransform(QtGui.QTransform())  # Reset transform
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pen = QPen(Qt.GlobalColor.red)
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        rect = self._get_target_rect()
        painter.drawEllipse(rect)

        painter.restore()


class ColorPickerButton(QtWidgets.QPushButton):
    """Button that shows the current color and opens a color picker dialog.

    Also supports color pickup from the image via ColorPickupInteraction.

    Signals:
        color_changed: Emitted when the color changes (QColor).
    """

    color_changed = Signal(QColor)

    def __init__(
        self,
        initial_color: QColor = QColor(255, 255, 255),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the color picker button.

        Args:
            initial_color: Initial color to display.
            parent: Parent widget.
        """
        super().__init__(parent)

        self._color = initial_color
        self._update_button_style()

        self.clicked.connect(self._show_color_dialog)

    @property
    def color(self) -> QColor:
        """Get the current color."""
        return self._color

    def set_color(self, color: QColor) -> None:
        """Set the current color.

        Args:
            color: The new color.
        """
        if color != self._color:
            self._color = color
            self._update_button_style()
            self.color_changed.emit(color)

    def _update_button_style(self) -> None:
        """Update the button appearance based on current color."""
        # Calculate a contrasting text color
        luminance = (
            0.299 * self._color.red()
            + 0.587 * self._color.green()
            + 0.114 * self._color.blue()
        )
        text_color = "black" if luminance > 128 else "white"

        self.setStyleSheet(
            f"background-color: {self._color.name()}; "
            f"color: {text_color}; "
            f"border: 1px solid gray; "
            f"padding: 4px 8px; "
            f"min-width: 60px;"
        )
        self.setText(self._color.name().upper())

    def _show_color_dialog(self) -> None:
        """Show the color picker dialog."""
        color = QtWidgets.QColorDialog.getColor(
            self._color,
            self,
            "Select Color",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self.set_color(color)
