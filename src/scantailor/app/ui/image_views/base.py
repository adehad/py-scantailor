"""Base image view widget for displaying and interacting with images.

The view shows an image through an ImageTransformation. Three coordinate
systems are involved:

- image: pixel corners of the displayed image;
- virtual: coordinates after the ImageTransformation, in which filter
  geometry (split lines, content boxes, margins) is expressed;
- widget: viewport pixels.

Zoom, wheel zoom, drag panning, scroll bars and fit-to-window live here;
subclasses draw overlays and handle their own handles.
"""

import math
from enum import IntFlag, auto
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import (
    QT_TRANSLATE_NOOP,
    QCoreApplication,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QImage,
    QPainter,
    QPainterPath,
    QPixmap,
    QPolygonF,
    QStatusTipEvent,
    QTransform,
)

from scantailor.core.transformation import ImageTransformation

# Marked, not translated: these are read at import, before the translator is
# installed, so they are translated where they are shown.
DEFAULT_STATUS_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ImageViewBase",
        "Use the mouse wheel or +/- to zoom.  When zoomed, dragging is possible.",
    ),
)
PAN_STATUS_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "DragHandler",
        "Unrestricted dragging is possible by holding down the Shift key.",
    ),
)
_MODIFIER_OF_KEY: dict[int, Qt.KeyboardModifier] = {
    Qt.Key.Key_Shift.value: Qt.KeyboardModifier.ShiftModifier,
    Qt.Key.Key_Control.value: Qt.KeyboardModifier.ControlModifier,
    Qt.Key.Key_Alt.value: Qt.KeyboardModifier.AltModifier,
}


class Edge(IntFlag):
    """Sides of a rectangle; two flags together name a corner."""

    NONE = 0
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


def point_to_segment_distance(point: QPointF, start: QPointF, end: QPointF) -> float:
    """Return the distance from a point to a line segment.

    Args:
        point (QPointF): The point.
        start (QPointF): One end of the segment.
        end (QPointF): The other end of the segment.

    Returns:
        float: Distance to the nearest point of the segment, in the points' units.
    """
    seg_x = end.x() - start.x()
    seg_y = end.y() - start.y()
    length_sq = seg_x * seg_x + seg_y * seg_y
    if length_sq < 1e-9:
        return math.hypot(point.x() - start.x(), point.y() - start.y())
    t = ((point.x() - start.x()) * seg_x + (point.y() - start.y()) * seg_y) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(
        point.x() - (start.x() + t * seg_x), point.y() - (start.y() + t * seg_y)
    )


def matrix_to_qtransform(matrix: NDArray[np.float64]) -> QTransform:
    """Convert a 3x3 column-vector affine matrix to a QTransform.

    Args:
        matrix (NDArray[np.float64]): Matrix with ``x' = m00 x + m01 y + m02``.

    Returns:
        QTransform: The same mapping; Qt stores the transpose of the linear part.
    """
    return QTransform(
        matrix[0, 0],
        matrix[1, 0],
        matrix[0, 1],
        matrix[1, 1],
        matrix[0, 2],
        matrix[1, 2],
    )


def numpy_to_qimage(array: NDArray[np.uint8]) -> QImage:
    """Copy an RGB, RGBA or gray array into a QImage.

    Args:
        array (NDArray[np.uint8]): ``(H, W)`` gray, ``(H, W, 3)`` RGB or
            ``(H, W, 4)`` RGBA pixels.

    Returns:
        QImage: A detached image that does not reference ``array``.

    Raises:
        ValueError: If the array has another shape.
    """
    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        height, width = array.shape
        return QImage(
            array.data, width, height, width, QImage.Format.Format_Grayscale8
        ).copy()
    if array.ndim == 3 and array.shape[2] in (3, 4):
        height, width, channels = array.shape
        pixel_format = (
            QImage.Format.Format_RGB888
            if channels == 3
            else QImage.Format.Format_RGBA8888
        )
        return QImage(array.data, width, height, width * channels, pixel_format).copy()
    raise ValueError(f"Unsupported array shape: {array.shape}")


class ImageViewBase(QtWidgets.QAbstractScrollArea):
    """Base class for widgets that display and manipulate images.

    Signals:
        zoom_changed: The zoom level changed (float zoom_level).
        transform_changed: The virtual-to-widget transform changed.
        status_tip_changed: The hint for what the pointer is over changed (str).
            It is also sent to the window's status bar.
    """

    zoom_changed = Signal(float)
    transform_changed = Signal()
    status_tip_changed = Signal(str)

    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0
    ZOOM_STEP = 2 ** (1 / 6)  # Zoom factor of one +/- key press
    WHEEL_ZOOM_STEP = 2**0.25  # Zoom factor of one wheel notch (15 degrees)
    wheel_zoom_at_center = False  # Zoom about the view centre, not the cursor
    default_status_tip = (
        DEFAULT_STATUS_TIP  # Shown while nothing under the pointer has a hint
    )
    default_status_tip_context = "ImageViewBase"
    HQ_TRANSFORM_DELAY_MS = 150  # Idle time before drawing the full-size image
    DOWNSCALED_MAX_SIZE = 1024

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        margins: tuple[float, float, float, float] = (0, 0, 0, 0),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the image view.

        Args:
            image (QImage | NDArray[np.uint8] | None): The image to display.
            downscaled_image (QImage | None): Optional smaller copy drawn while
                the view is changing.
            transformation (ImageTransformation | None): Maps image to virtual
                coordinates; identity when None.
            margins (tuple[float, float, float, float]): Left, top, right and
                bottom space kept free around the image, in widget pixels.
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent)

        self.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        viewport = self.viewport()
        assert viewport is not None
        viewport.setBackgroundRole(QtGui.QPalette.ColorRole.Dark)
        viewport.setAutoFillBackground(True)
        viewport.setMouseTracking(True)
        default_tip = self._translated_default_status_tip()
        viewport.setStatusTip(default_tip)
        self._status_tip = default_tip

        self._image: QImage | None = None
        self._pixmap: QPixmap | None = None
        self._pixmap_to_image = QTransform()
        self._hq_ready = False

        self._image_to_virtual = QTransform()
        self._virtual_to_image = QTransform()
        self._virtual_to_widget = QTransform()
        self._widget_to_virtual = QTransform()

        self._virtual_crop_area = QPolygonF()
        self._virtual_display_area = QRectF()
        self._page_area = QRectF()
        self._extent = QRectF()
        self._margins = margins

        self._zoom = 1.0
        self._min_zoom = self.MIN_ZOOM
        self._fit_mode = True
        self._widget_focal_point = QPointF()
        self._virtual_focal_point = QPointF()

        self._panning = False
        self._last_mouse_pos = QPointF(-1, -1)
        self._syncing_scroll_bars = False

        self._hq_timer = QTimer(self)
        self._hq_timer.setSingleShot(True)
        self._hq_timer.timeout.connect(self._on_hq_timeout)

        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            assert bar is not None
            bar.valueChanged.connect(self._on_scroll_bar_changed)

        if image is not None:
            self.set_image(image, downscaled_image, transformation)

    def set_image(
        self,
        image: QImage | NDArray[np.uint8],
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        keep_view: bool = False,
    ) -> None:
        """Set the image to display.

        Args:
            image (QImage | NDArray[np.uint8]): The image to display.
            downscaled_image (QImage | None): Optional pre-downscaled version.
            transformation (ImageTransformation | None): Maps image to virtual
                coordinates; identity when None.
            keep_view (bool): Keep the zoom and the part of the image in view
                instead of fitting the whole image into the window.
        """
        if isinstance(image, np.ndarray):
            image = numpy_to_qimage(image)

        self._image = image
        self._hq_ready = False
        if downscaled_image is not None:
            self._pixmap = QPixmap.fromImage(downscaled_image)
            scale = image.width() / max(1, downscaled_image.width())
            self._pixmap_to_image = QTransform.fromScale(scale, scale)
        else:
            self._pixmap = self._create_downscaled_pixmap(image)

        self._apply_transformation(transformation, keep_view)

    def set_transformation(
        self, transformation: ImageTransformation, keep_view: bool = True
    ) -> None:
        """Change how the current image maps to virtual coordinates.

        Args:
            transformation (ImageTransformation): The new mapping.
            keep_view (bool): Keep the zoom and the image point in view.
        """
        self._apply_transformation(transformation, keep_view)

    def _create_downscaled_pixmap(self, image: QImage) -> QPixmap:
        """Build the pixmap that is drawn while the view is changing."""
        limit = self.DOWNSCALED_MAX_SIZE
        width, height = image.width(), image.height()
        if max(width, height) <= limit:
            self._pixmap_to_image = QTransform()
            return QPixmap.fromImage(image)

        scale = limit / max(width, height)
        scaled = image.scaled(
            max(1, round(width * scale)),
            max(1, round(height * scale)),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._pixmap_to_image = QTransform.fromScale(
            width / scaled.width(), height / scaled.height()
        )
        return QPixmap.fromImage(scaled)

    def _apply_transformation(
        self, transformation: ImageTransformation | None, keep_view: bool
    ) -> None:
        focal_image = self._virtual_to_image.map(self._virtual_focal_point)

        if transformation is not None:
            self._image_to_virtual = matrix_to_qtransform(transformation.transform)
            inverted, invertible = self._image_to_virtual.inverted()
            self._virtual_to_image = inverted if invertible else QTransform()
            rect = transformation.resulting_rect
            self._page_area = QRectF(rect.x, rect.y, rect.width, rect.height)
            self._virtual_crop_area = QPolygonF(
                [QPointF(x, y) for x, y in transformation.resulting_pre_crop_area]
            )
        else:
            self._image_to_virtual = QTransform()
            self._virtual_to_image = QTransform()
            if self._image is None or self._image.isNull():
                self._page_area = QRectF()
            else:
                self._page_area = QRectF(
                    0, 0, self._image.width(), self._image.height()
                )
            self._virtual_crop_area = QPolygonF(self._page_area)
        self._virtual_display_area = self._page_area.united(self._extent)

        if keep_view and not self._fit_mode and self._image is not None:
            self._virtual_focal_point = self._image_to_virtual.map(focal_image)
            self._update_widget_transform()
        else:
            self._fit_to_widget()

    def set_display_extent(self, extent: QRectF) -> None:
        """Make the view cover a virtual rectangle besides the page.

        Overlays such as margins reach outside the page; without this they could
        not be reached by panning or fitting.

        Args:
            extent (QRectF): Rectangle in virtual coordinates; empty for none.
        """
        self._extent = QRectF(extent)
        self._virtual_display_area = self._page_area.united(self._extent)
        if self._fit_mode:
            self._fit_to_widget()
        else:
            self._update_widget_transform()

    def _available_rect(self) -> QRectF:
        viewport = self.viewport()
        assert viewport is not None
        left, top, right, bottom = self._margins
        return QRectF(
            left,
            top,
            viewport.width() - left - right,
            viewport.height() - top - bottom,
        )

    def _fit_to_widget(self) -> None:
        """Show the whole display area, centred, never magnified beyond 100%."""
        available = self._available_rect()
        area = self._virtual_display_area
        if area.isEmpty() or available.width() <= 0 or available.height() <= 0:
            return

        fit_zoom = min(
            available.width() / area.width(), available.height() / area.height()
        )
        self._zoom = min(fit_zoom, 1.0)
        self._min_zoom = self._zoom
        self._fit_mode = True
        self._widget_focal_point = available.center()
        self._virtual_focal_point = area.center()
        self._update_widget_transform()

    def _build_virtual_to_widget(self) -> QTransform:
        zoom = self._zoom
        focal_v = self._virtual_focal_point
        focal_w = self._widget_focal_point
        return QTransform(
            zoom,
            0,
            0,
            zoom,
            focal_w.x() - zoom * focal_v.x(),
            focal_w.y() - zoom * focal_v.y(),
        )

    @staticmethod
    def _axis_shift(
        low: float, high: float, avail_low: float, avail_high: float
    ) -> float:
        """Shift that keeps an extent inside the window, or centres it if it fits."""
        if high - low <= avail_high - avail_low:
            return (avail_low + avail_high) / 2 - (low + high) / 2
        if low > avail_low:
            return avail_low - low
        if high < avail_high:
            return avail_high - high
        return 0.0

    def _update_widget_transform(self) -> None:
        """Rebuild the virtual-to-widget transform from zoom and focal points."""
        self._virtual_to_widget = self._build_virtual_to_widget()
        if not self._virtual_display_area.isEmpty():
            shown = self._virtual_to_widget.mapRect(self._virtual_display_area)
            avail = self._available_rect()
            self._widget_focal_point += QPointF(
                self._axis_shift(
                    shown.left(), shown.right(), avail.left(), avail.right()
                ),
                self._axis_shift(
                    shown.top(), shown.bottom(), avail.top(), avail.bottom()
                ),
            )
            self._virtual_to_widget = self._build_virtual_to_widget()

        inverted, invertible = self._virtual_to_widget.inverted()
        if invertible:
            self._widget_to_virtual = inverted

        self._sync_scroll_bars()
        self._hq_ready = False
        self._hq_timer.start(self.HQ_TRANSFORM_DELAY_MS)
        self.transform_changed.emit()
        self.update()

    def _sync_scroll_bars(self) -> None:
        if self._virtual_display_area.isEmpty():
            return
        shown = self._virtual_to_widget.mapRect(self._virtual_display_area)
        avail = self._available_rect()
        horizontal = self.horizontalScrollBar()
        vertical = self.verticalScrollBar()
        assert horizontal is not None and vertical is not None

        self._syncing_scroll_bars = True
        try:
            for bar, extent, avail_size, offset in (
                (horizontal, shown.width(), avail.width(), avail.left() - shown.left()),
                (vertical, shown.height(), avail.height(), avail.top() - shown.top()),
            ):
                maximum = max(0, math.ceil(extent - avail_size))
                bar.setRange(0, maximum)
                bar.setPageStep(max(1, int(avail_size)))
                bar.setValue(max(0, min(maximum, round(offset))))
        finally:
            self._syncing_scroll_bars = False

    def _on_scroll_bar_changed(self) -> None:
        if self._syncing_scroll_bars or self._virtual_display_area.isEmpty():
            return
        shown = self._virtual_to_widget.mapRect(self._virtual_display_area)
        avail = self._available_rect()
        horizontal = self.horizontalScrollBar()
        vertical = self.verticalScrollBar()
        assert horizontal is not None and vertical is not None
        self._widget_focal_point += QPointF(
            avail.left() - horizontal.value() - shown.left(),
            avail.top() - vertical.value() - shown.top(),
        )
        self._fit_mode = False
        self._update_widget_transform()

    def _on_hq_timeout(self) -> None:
        self._hq_ready = True
        self.update()

    def set_zoom(self, zoom: float, center_point: QPointF | None = None) -> None:
        """Set the zoom level.

        Args:
            zoom (float): Zoom level; 1.0 shows one image pixel per widget pixel.
            center_point (QPointF | None): Widget point that stays fixed; the
                current focal point when None.
        """
        zoom = max(self._min_zoom, min(self.MAX_ZOOM, zoom))
        if abs(zoom - self._zoom) < 1e-6:
            return

        if center_point is not None:
            self._virtual_focal_point = self._widget_to_virtual.map(center_point)
            self._widget_focal_point = QPointF(center_point)

        self._zoom = zoom
        self._fit_mode = False
        self._update_widget_transform()
        self.zoom_changed.emit(zoom)

    def zoom_in(self, center_point: QPointF | None = None) -> None:
        """Zoom in by one step.

        Args:
            center_point (QPointF | None): Widget point to zoom around.
        """
        self.set_zoom(self._zoom * self.ZOOM_STEP, center_point)

    def zoom_out(self, center_point: QPointF | None = None) -> None:
        """Zoom out by one step.

        Args:
            center_point (QPointF | None): Widget point to zoom around.
        """
        self.set_zoom(self._zoom / self.ZOOM_STEP, center_point)

    def zoom_to_fit(self) -> None:
        """Fit the whole display area into the widget."""
        self._fit_to_widget()
        self.zoom_changed.emit(self._zoom)

    @property
    def image(self) -> QImage | None:
        """The displayed image, or None before one was set."""
        return self._image

    @property
    def zoom_level(self) -> float:
        """The current zoom level."""
        return self._zoom

    @property
    def virtual_display_area(self) -> QRectF:
        """Bounds of the displayed page in virtual coordinates."""
        return QRectF(self._virtual_display_area)

    @property
    def image_to_virtual(self) -> QTransform:
        """Image-to-virtual transformation."""
        return self._image_to_virtual

    @property
    def virtual_to_widget(self) -> QTransform:
        """Virtual-to-widget transformation."""
        return self._virtual_to_widget

    @property
    def widget_to_virtual(self) -> QTransform:
        """Widget-to-virtual transformation."""
        return self._widget_to_virtual

    def image_to_widget(self) -> QTransform:
        """Return the composite image-to-widget transformation."""
        return self._image_to_virtual * self._virtual_to_widget

    def widget_to_image(self) -> QTransform:
        """Return the composite widget-to-image transformation."""
        return self._widget_to_virtual * self._virtual_to_image

    def update(self, *args: Any) -> None:
        """Schedule a repaint of the image and its overlay.

        The view draws on its viewport, a child widget that the scroll area's
        own ``update()`` does not repaint, so every overload goes to the viewport.

        Args:
            *args (Any): Nothing, or the rectangle or region to repaint, in
                viewport coordinates, as for ``QWidget.update``.
        """
        viewport = self.viewport()
        assert viewport is not None
        viewport.update(*args)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draw the image, then the overlay of the subclass.

        Args:
            event (QtGui.QPaintEvent): The paint event.
        """
        viewport = self.viewport()
        assert viewport is not None
        painter = QPainter(viewport)
        self._paint_image(painter)
        self._paint_overlay(painter)
        painter.end()

    def _paint_image(self, painter: QPainter) -> None:
        if self._image is None or self._pixmap is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self._virtual_crop_area.isEmpty():
            clip = QPainterPath()
            clip.addPolygon(self._virtual_to_widget.map(self._virtual_crop_area))
            clip.closeSubpath()
            painter.setClipPath(clip)

        if self._hq_ready:
            painter.setTransform(self.image_to_widget())
            painter.drawImage(0, 0, self._image)
        else:
            painter.setTransform(self._pixmap_to_image * self.image_to_widget())
            painter.drawPixmap(0, 0, self._pixmap)
        painter.restore()

    def _translated_default_status_tip(self) -> str:
        """Return the idle hint in the active language.

        Returns:
            str: The default status tip, translated in its own context.
        """
        return QCoreApplication.translate(
            self.default_status_tip_context, self.default_status_tip
        )

    def set_status_tip(self, tip: str) -> None:
        """Show a hint in the window's status bar.

        Args:
            tip (str): The hint; the default one when empty.
        """
        tip = tip or self._translated_default_status_tip()
        if tip == self._status_tip:
            return
        self._status_tip = tip
        viewport = self.viewport()
        assert viewport is not None
        viewport.setStatusTip(tip)
        QtWidgets.QApplication.sendEvent(viewport, QStatusTipEvent(tip))
        self.status_tip_changed.emit(tip)

    @property
    def status_tip_text(self) -> str:
        """The hint currently offered for the status bar."""
        return self._status_tip

    def _status_tip_at(
        self, widget_pos: QPointF, modifiers: Qt.KeyboardModifier
    ) -> str:
        """Return the hint for what lies under a widget position.

        Subclasses override this; an empty string means the default hint.

        Args:
            widget_pos (QPointF): Pointer position in widget coordinates.
            modifiers (Qt.KeyboardModifier): Keys held, which can change what a
                drag would do.

        Returns:
            str: The hint.
        """
        return ""

    def _refresh_status_tip(self, modifiers: Qt.KeyboardModifier) -> None:
        """Show the hint for the last pointer position with other keys held."""
        self.set_status_tip(self._status_tip_at(self._last_mouse_pos, modifiers))

    def _paint_overlay(self, painter: QPainter) -> None:
        """Draw graphics on top of the image; subclasses override this.

        Args:
            painter (QPainter): Painter on the viewport, in widget coordinates.
        """

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Keep the image fitted, or the view position valid, after a resize.

        Args:
            event (QtGui.QResizeEvent): The resize event.
        """
        super().resizeEvent(event)
        if self._fit_mode:
            self._fit_to_widget()
        elif not self._virtual_display_area.isEmpty():
            self._update_widget_transform()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Start panning on a left or middle button press nothing else handled.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.set_status_tip(
                QCoreApplication.translate("DragHandler", PAN_STATUS_TIP)
            )
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Stop panning.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._panning and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.set_status_tip(
                self._status_tip_at(event.position(), event.modifiers())
            )
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pan while dragging.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._panning:
            self._widget_focal_point += event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self._fit_mode = False
            self._update_widget_transform()
            event.accept()
        else:
            self._last_mouse_pos = event.position()
            self.set_status_tip(
                self._status_tip_at(event.position(), event.modifiers())
            )
            super().mouseMoveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Zoom around the cursor, or the view centre when so configured.

        Args:
            event (QtGui.QWheelEvent): The wheel event.
        """
        notches = event.angleDelta().y() / 120.0
        if notches:
            focus = (
                QPointF(self.viewport().rect().center())
                if self.wheel_zoom_at_center
                else event.position()
            )
            self.set_zoom(self._zoom * self.WHEEL_ZOOM_STEP**notches, focus)
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle the zoom shortcuts.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        key = event.key()
        if key in _MODIFIER_OF_KEY:
            self._refresh_status_tip(event.modifiers())
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
        elif key == Qt.Key.Key_Minus:
            self.zoom_out()
        elif key == Qt.Key.Key_0:
            self.zoom_to_fit()
        elif key == Qt.Key.Key_1:
            self.set_zoom(1.0)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        """Update the hint when a modifier key is let go.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        released = _MODIFIER_OF_KEY.get(event.key())
        if released is not None:
            # Some platforms still list the key that was let go.
            self._refresh_status_tip(event.modifiers() & ~released)
        super().keyReleaseEvent(event)

    def virtual_to_widget_point(self, point: QPointF) -> QPointF:
        """Convert a point from virtual to widget coordinates.

        Args:
            point (QPointF): Point in virtual coordinates.

        Returns:
            QPointF: The point in widget coordinates.
        """
        return self._virtual_to_widget.map(point)

    def widget_to_virtual_point(self, point: QPointF) -> QPointF:
        """Convert a point from widget to virtual coordinates.

        Args:
            point (QPointF): Point in widget coordinates.

        Returns:
            QPointF: The point in virtual coordinates.
        """
        return self._widget_to_virtual.map(point)

    def virtual_to_widget_rect(self, rect: QRectF) -> QRectF:
        """Convert a rectangle from virtual to widget coordinates.

        Args:
            rect (QRectF): Rectangle in virtual coordinates.

        Returns:
            QRectF: Bounding rectangle in widget coordinates.
        """
        return self._virtual_to_widget.mapRect(rect)

    def widget_to_virtual_rect(self, rect: QRectF) -> QRectF:
        """Convert a rectangle from widget to virtual coordinates.

        Args:
            rect (QRectF): Rectangle in widget coordinates.

        Returns:
            QRectF: Bounding rectangle in virtual coordinates.
        """
        return self._widget_to_virtual.mapRect(rect)
