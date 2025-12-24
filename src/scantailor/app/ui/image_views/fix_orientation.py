"""Fix Orientation filter image view.

Provides a simple image preview for the Fix Orientation filter stage.
This is the simplest image view - it just displays the image with the
current rotation applied, supporting zoom and pan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtGui
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPen, QTransform

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.core import OrthogonalRotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scantailor.core import ImageTransformation


class FixOrientationImageView(ImageViewBase):
    """Image view for the Fix Orientation filter.

    Displays the image with the current rotation applied. Supports
    zoom, pan, and visual feedback for the rotation state.

    Signals:
        rotation_requested: Emitted when user requests rotation change.
    """

    rotation_requested = Signal(OrthogonalRotation)

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        parent: QtGui.QWidget | None = None,
    ) -> None:
        """Initialize the Fix Orientation image view.

        Args:
            image: The image to display.
            downscaled_image: Optional pre-downscaled version.
            transformation: Image transformation.
            parent: Parent widget.
        """
        super().__init__(image, downscaled_image, transformation, parent=parent)

        self._rotation = OrthogonalRotation(degrees=0)
        self._show_rotation_indicator = True

    def set_rotation(self, rotation: OrthogonalRotation) -> None:
        """Set the current rotation.

        Args:
            rotation: The rotation to apply.
        """
        self._rotation = rotation
        self.update()

    @Slot(OrthogonalRotation)
    def set_pre_rotation(self, rotation: OrthogonalRotation) -> None:
        """Slot to set rotation from external source.

        Args:
            rotation: The rotation to apply.
        """
        self.set_rotation(rotation)

    def set_show_rotation_indicator(self, show: bool) -> None:
        """Set whether to show the rotation indicator.

        Args:
            show: True to show the indicator.
        """
        self._show_rotation_indicator = show
        self.update()

    def _paint_overlay(self, painter: QPainter) -> None:
        """Paint the rotation indicator overlay.

        Args:
            painter: The painter to draw with.
        """
        if not self._show_rotation_indicator:
            return

        # Draw rotation indicator in bottom-right corner
        viewport = self.viewport()
        if viewport is None:
            return

        # Indicator size and position
        size = 60
        margin = 10
        x = viewport.width() - size - margin
        y = viewport.height() - size - margin
        center = QPointF(x + size / 2, y + size / 2)

        # Draw background circle
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg_color = QtGui.QColor(0, 0, 0, 128)
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, size / 2, size / 2)

        # Draw rotation arrow
        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.GlobalColor.white)

        # Apply rotation to the arrow
        painter.translate(center)
        painter.rotate(self._rotation.degrees)

        # Draw an arrow pointing up (which rotates with the current rotation)
        arrow_length = size / 2 - 10
        arrow_width = 8

        # Arrow shaft
        painter.drawLine(
            QPointF(0, arrow_length / 2),
            QPointF(0, -arrow_length / 2 + arrow_width),
        )

        # Arrow head
        arrow_head = QtGui.QPolygonF(
            [
                QPointF(0, -arrow_length / 2),
                QPointF(-arrow_width, -arrow_length / 2 + arrow_width * 1.5),
                QPointF(arrow_width, -arrow_length / 2 + arrow_width * 1.5),
            ]
        )
        painter.drawPolygon(arrow_head)

        # Draw rotation text
        painter.resetTransform()
        text_color = QtGui.QColor(255, 255, 255, 200)
        painter.setPen(text_color)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        text = f"{self._rotation.degrees}"
        text_rect = QRectF(x, y + size + 2, size, 16)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)

        painter.restore()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press for rotation shortcuts.

        Args:
            event: The key event.
        """
        if event.key() == Qt.Key.Key_Left or event.key() == Qt.Key.Key_L:
            # Rotate left (counter-clockwise)
            new_rotation = OrthogonalRotation(
                degrees=(self._rotation.degrees - 90) % 360
            )
            self.rotation_requested.emit(new_rotation)
            event.accept()
        elif event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_R:
            # Rotate right (clockwise)
            new_rotation = OrthogonalRotation(
                degrees=(self._rotation.degrees + 90) % 360
            )
            self.rotation_requested.emit(new_rotation)
            event.accept()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle double-click to rotate.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            # Double-click to rotate clockwise
            new_rotation = OrthogonalRotation(
                degrees=(self._rotation.degrees + 90) % 360
            )
            self.rotation_requested.emit(new_rotation)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
