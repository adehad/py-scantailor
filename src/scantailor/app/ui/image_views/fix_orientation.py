"""Fix Orientation filter image view.

Shows the unrotated source image under the stored rotation, with an arrow in
a corner that points where the top of the page now is. Only the part inside the
manual trim is shown. Left and right arrow keys, L and R, and a double click
ask for a quarter turn.
"""

import copy

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.models import OrthogonalRotation
from scantailor.core.transformation import ImageTransformation, Rect
from scantailor.filters.fix_orientation.image_trim import ImageTrim


class FixOrientationImageView(ImageViewBase):
    """Image view of the Fix Orientation stage.

    The transformation given to the view maps the source image without
    rotation; the view adds the rotation on top of it.

    Signals:
        rotation_requested: The user asked for a quarter turn. Args: rotation
            (OrthogonalRotation), the total rotation wanted.
    """

    rotation_requested = Signal(OrthogonalRotation)

    INDICATOR_SIZE = 60
    INDICATOR_MARGIN = 10

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._rotation = OrthogonalRotation(degrees=0)
        self._base_transformation: ImageTransformation | None = None
        self._trim = ImageTrim()

    @property
    def rotation(self) -> OrthogonalRotation:
        """The rotation the page is shown with."""
        return self._rotation

    def set_image(
        self,
        image: QImage | NDArray[np.uint8],
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        keep_view: bool = False,
    ) -> None:
        """Set the unrotated image and its transformation.

        Args:
            image (QImage | NDArray[np.uint8]): The source image.
            downscaled_image (QImage | None): Optional pre-downscaled version.
            transformation (ImageTransformation | None): Maps image to virtual
                coordinates without the orthogonal rotation.
            keep_view (bool): Keep the zoom and the image point in view.
        """
        self._base_transformation = transformation
        rotated = None if transformation is None else self._rotated(transformation)
        super().set_image(image, downscaled_image, rotated, keep_view)

    def _rotated(self, transformation: ImageTransformation) -> ImageTransformation:
        rotated = copy.deepcopy(transformation)
        rotated.set_pre_rotation(self._rotation)
        if self._trim.enabled:
            width, height = (
                transformation.orig_rect.width,
                transformation.orig_rect.height,
            )
            x, y, inner_width, inner_height = self._trim.to_inner_rect(
                round(width), round(height)
            )
            rotated.set_pre_crop_area(
                rotated.orig_rect_to_pre_crop_space(
                    Rect(x=x, y=y, width=inner_width, height=inner_height)
                )
            )
        return rotated

    def set_trim(self, trim: ImageTrim) -> None:
        """Show only the part of the page that a trim keeps.

        Args:
            trim (ImageTrim): The trim of the shown image; its edges are those
                of the unrotated image.
        """
        if trim == self._trim:
            return
        self._trim = trim
        if self._base_transformation is not None:
            ImageViewBase.set_transformation(
                self, self._rotated(self._base_transformation), keep_view=True
            )
        self.update()

    def set_rotation(self, rotation: OrthogonalRotation) -> None:
        """Show the page under a rotation.

        Args:
            rotation (OrthogonalRotation): The rotation to show.
        """
        if rotation == self._rotation:
            return
        self._rotation = rotation
        if self._base_transformation is not None:
            # A quarter turn swaps the page's width and height, so the old zoom
            # no longer fits.
            ImageViewBase.set_transformation(
                self, self._rotated(self._base_transformation), keep_view=False
            )
        self.update()

    def _paint_overlay(self, painter: QPainter) -> None:
        viewport = self.viewport()
        assert viewport is not None
        size = self.INDICATOR_SIZE
        left = viewport.width() - size - self.INDICATOR_MARGIN
        top = viewport.height() - size - self.INDICATOR_MARGIN
        center = QPointF(left + size / 2, top + size / 2)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(view_color("indicator_scrim"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, size / 2, size / 2)

        pen = QPen(view_color("indicator_glyph"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(view_color("indicator_glyph"))
        painter.translate(center)
        painter.rotate(self._rotation.degrees)
        length = size / 2 - 10
        head = 8
        painter.drawLine(QPointF(0, length / 2), QPointF(0, -length / 2 + head))
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QPointF(0, -length / 2),
                    QPointF(-head, -length / 2 + head * 1.5),
                    QPointF(head, -length / 2 + head * 1.5),
                ]
            )
        )

        painter.resetTransform()
        painter.setPen(view_color("indicator_label"))
        painter.drawText(
            QRectF(left, top + size + 2, size, 16),
            Qt.AlignmentFlag.AlignCenter,
            str(self._rotation.degrees),
        )
        painter.restore()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Turn the page with the arrow keys, L and R.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_L):
            self.rotation_requested.emit(self._rotation.rotate_counter_clockwise())
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_R):
            self.rotation_requested.emit(self._rotation.rotate_clockwise())
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Turn the page clockwise.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.rotation_requested.emit(self._rotation.rotate_clockwise())
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
