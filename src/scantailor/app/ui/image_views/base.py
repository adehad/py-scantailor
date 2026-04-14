"""Base image view widget for displaying and interacting with images.

This module provides the base class for all filter-specific image views,
handling common functionality like:
- Image display with zoom and pan
- Coordinate transformations (image, virtual, widget)
- High-quality delayed rendering for large images
- Mouse and keyboard interaction
"""

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform

from scantailor.core import ImageTransformation


class FocalPointMode(Enum):
    """Mode for handling focal point during transform updates."""

    CENTER_IF_FITS = auto()  # Center image if it fits in widget
    DONT_CENTER = auto()  # Keep current focal point


class ImageViewBase(QtWidgets.QAbstractScrollArea):
    """Base class for widgets that display and manipulate images.

    This class operates with multiple coordinate systems:
    - Image coordinates: Original pixel coordinates of the source image
    - Virtual coordinates: Transformed coordinates after applying ImageTransformation
    - Widget coordinates: Screen coordinates within this widget

    Signals:
        zoom_changed: Emitted when zoom level changes (float zoom_level)
        transform_changed: Emitted when the display transform changes
    """

    zoom_changed = Signal(float)
    transform_changed = Signal()

    # Constants
    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0
    ZOOM_STEP = 1.1  # Multiply/divide by this for zoom in/out
    HQ_TRANSFORM_DELAY_MS = 150  # Delay before building high-quality version

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
            image: The image to display (QImage or numpy array).
            downscaled_image: Optional pre-downscaled version for faster rendering.
            transformation: Image transformation defining virtual coordinates.
            margins: Widget margins (left, top, right, bottom) in pixels.
            parent: Parent widget.
        """
        super().__init__(parent)

        # Viewport setup
        self.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        viewport = self.viewport()
        if viewport:
            viewport.setBackgroundRole(QtGui.QPalette.ColorRole.Dark)
            viewport.setAutoFillBackground(True)
            viewport.setMouseTracking(True)

        # Image data
        self._image: QImage | None = None
        self._pixmap: QPixmap | None = None  # Display pixmap (possibly downscaled)
        self._hq_pixmap: QPixmap | None = None  # High-quality transformed pixmap
        self._hq_pixmap_pos: QtCore.QPoint = QtCore.QPoint(0, 0)
        self._hq_xform: QTransform = QTransform()
        self._hq_enabled = True

        # Transformation matrices
        self._pixmap_to_image = QTransform()
        self._image_to_virtual = QTransform()
        self._virtual_to_image = QTransform()
        self._virtual_to_widget = QTransform()
        self._widget_to_virtual = QTransform()

        # Display settings
        self._virtual_crop_area: QtGui.QPolygonF = QtGui.QPolygonF()
        self._virtual_display_area = QRectF()
        self._margins = margins  # left, top, right, bottom

        # Zoom and pan state
        self._zoom = 1.0
        self._widget_focal_point = QPointF()
        self._pixmap_focal_point = QPointF()

        # Interaction state
        self._dragging = False
        self._drag_start_pos = QPointF()
        self._last_mouse_pos = QPointF()

        # Timer for delayed high-quality rendering
        self._hq_timer = QTimer(self)
        self._hq_timer.setSingleShot(True)
        self._hq_timer.timeout.connect(self._build_hq_version)

        # Set up scroll bars
        h_scroll = self.horizontalScrollBar()
        v_scroll = self.verticalScrollBar()
        if h_scroll:
            h_scroll.valueChanged.connect(self._on_scroll_bar_changed)
        if v_scroll:
            v_scroll.valueChanged.connect(self._on_scroll_bar_changed)

        # Initialize with provided image
        if image is not None:
            self.set_image(image, downscaled_image, transformation)

    def set_image(
        self,
        image: QImage | NDArray[np.uint8],
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
    ) -> None:
        """Set the image to display.

        Args:
            image: The image to display (QImage or numpy array).
            downscaled_image: Optional pre-downscaled version.
            transformation: Optional image transformation.
        """
        # Convert numpy array to QImage if needed
        if isinstance(image, np.ndarray):
            image = self._numpy_to_qimage(image)

        self._image = image
        self._hq_pixmap = None

        # Create or use downscaled pixmap for display
        if downscaled_image is not None:
            self._pixmap = QPixmap.fromImage(downscaled_image)
            # Compute pixmap-to-image transformation
            if image.width() > 0 and downscaled_image.width() > 0:
                scale = image.width() / downscaled_image.width()
                self._pixmap_to_image = QTransform.fromScale(scale, scale)
        else:
            # Downscale if image is large
            self._pixmap = self._create_downscaled_pixmap(image)

        # Set up transformation
        if transformation is not None:
            self._setup_transformation(transformation)
        else:
            # Identity transformation
            self._image_to_virtual = QTransform()
            self._virtual_to_image = QTransform()
            if image.isNull():
                self._virtual_display_area = QRectF()
                self._virtual_crop_area = QtGui.QPolygonF()
            else:
                self._virtual_display_area = QRectF(0, 0, image.width(), image.height())
                self._virtual_crop_area = QtGui.QPolygonF(self._virtual_display_area)

        # Reset view
        self._fit_to_widget()
        self.update()

    def _numpy_to_qimage(self, array: NDArray[np.uint8]) -> QImage:
        """Convert a numpy array to QImage.

        Args:
            array: Image as numpy array (H, W) or (H, W, C).

        Returns:
            QImage representation of the array.
        """
        if array.ndim == 2:
            # Grayscale
            h, w = array.shape
            bytes_per_line = w
            return QImage(
                array.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8
            ).copy()
        elif array.ndim == 3:
            h, w, c = array.shape
            if c == 3:
                # RGB - convert to RGBA for Qt
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[:, :, :3] = array
                rgba[:, :, 3] = 255
                bytes_per_line = w * 4
                return QImage(
                    rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
                ).copy()
            elif c == 4:
                # RGBA
                bytes_per_line = w * 4
                return QImage(
                    array.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
                ).copy()
        raise ValueError(f"Unsupported array shape: {array.shape}")

    def _create_downscaled_pixmap(self, image: QImage) -> QPixmap:
        """Create a downscaled pixmap for faster display.

        Args:
            image: The source image.

        Returns:
            Downscaled pixmap (or original if already small).
        """
        max_size = 1024
        w, h = image.width(), image.height()

        if w <= max_size and h <= max_size:
            self._pixmap_to_image = QTransform()
            return QPixmap.fromImage(image)

        # Compute scale factor
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        scaled_image = image.scaled(
            new_w,
            new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Store inverse scale for coordinate mapping
        self._pixmap_to_image = QTransform.fromScale(1.0 / scale, 1.0 / scale)

        return QPixmap.fromImage(scaled_image)

    def _setup_transformation(self, transformation: ImageTransformation) -> None:
        """Set up coordinate transformations from ImageTransformation.

        Args:
            transformation: The image transformation to apply.
        """
        # Get the transformation matrix
        matrix = transformation.transform
        self._image_to_virtual = QTransform(
            matrix[0, 0],
            matrix[0, 1],
            matrix[1, 0],
            matrix[1, 1],
            matrix[0, 2],
            matrix[1, 2],
        )

        # Compute inverse
        inverted, invertible = self._image_to_virtual.inverted()
        if invertible:
            self._virtual_to_image = inverted
        else:
            self._virtual_to_image = QTransform()

        # Set display area from transformation's result rect
        result_rect = transformation.resulting_rect
        self._virtual_display_area = QRectF(
            result_rect.x, result_rect.y, result_rect.width, result_rect.height
        )
        self._virtual_crop_area = QtGui.QPolygonF(self._virtual_display_area)

    def _fit_to_widget(self) -> None:
        """Fit the image to the widget size."""
        viewport = self.viewport()
        if viewport is None or self._virtual_display_area.isEmpty():
            return

        # Available area (accounting for margins)
        available = QRectF(
            self._margins[0],
            self._margins[1],
            viewport.width() - self._margins[0] - self._margins[2],
            viewport.height() - self._margins[1] - self._margins[3],
        )

        if available.width() <= 0 or available.height() <= 0:
            return

        # Compute zoom to fit
        display_rect = self._virtual_display_area
        zoom_x = available.width() / display_rect.width()
        zoom_y = available.height() / display_rect.height()
        self._zoom = min(zoom_x, zoom_y, 1.0)  # Don't zoom in beyond 100%

        # Center the image
        self._widget_focal_point = available.center()
        self._pixmap_focal_point = display_rect.center()

        self._update_widget_transform()

    def _update_widget_transform(self) -> None:
        """Update the virtual-to-widget transformation based on current state."""
        if self._pixmap is None:
            return

        # Build transformation: scale around focal point
        t = QTransform()

        # Translate so pixmap focal point is at origin
        t.translate(-self._pixmap_focal_point.x(), -self._pixmap_focal_point.y())

        # Scale
        t.scale(self._zoom, self._zoom)

        # Translate to widget focal point
        t2 = QTransform()
        t2.translate(self._widget_focal_point.x(), self._widget_focal_point.y())

        self._virtual_to_widget = t * t2
        inverted, invertible = self._virtual_to_widget.inverted()
        if invertible:
            self._widget_to_virtual = inverted

        self._update_scroll_bars()
        self._schedule_hq_rebuild()
        self.transform_changed.emit()

    def _update_scroll_bars(self) -> None:
        """Update scroll bar ranges based on current transform."""
        viewport = self.viewport()
        if viewport is None or self._virtual_display_area.isEmpty():
            return

        # Get the displayed image rect in widget coordinates
        display_rect = self._virtual_to_widget.mapRect(self._virtual_display_area)

        h_scroll = self.horizontalScrollBar()
        v_scroll = self.verticalScrollBar()

        # Compute scroll ranges
        h_range = max(0, int(display_rect.width() - viewport.width()))
        v_range = max(0, int(display_rect.height() - viewport.height()))

        if h_scroll:
            h_scroll.setRange(0, h_range)
            h_scroll.setPageStep(viewport.width())
        if v_scroll:
            v_scroll.setRange(0, v_range)
            v_scroll.setPageStep(viewport.height())

    def _on_scroll_bar_changed(self) -> None:
        """Handle scroll bar value changes."""
        # TODO: Implement scroll-based panning
        pass

    def _schedule_hq_rebuild(self) -> None:
        """Schedule rebuild of high-quality pixmap."""
        if self._hq_enabled:
            self._hq_timer.start(self.HQ_TRANSFORM_DELAY_MS)

    def _build_hq_version(self) -> None:
        """Build high-quality transformed version of the image."""
        if self._image is None or self._image.isNull():
            return

        viewport = self.viewport()
        if viewport is None:
            return

        # Only build HQ version if zoomed in enough to see the difference
        if self._zoom < 0.5:
            self._hq_pixmap = None
            return

        # Get visible area in image coordinates
        visible_rect = QRectF(viewport.rect())
        virtual_rect = self._widget_to_virtual.mapRect(visible_rect)
        image_rect = self._virtual_to_image.mapRect(virtual_rect)

        # Clamp to image bounds
        image_rect = image_rect.intersected(QRectF(self._image.rect()))
        if image_rect.isEmpty():
            self._hq_pixmap = None
            return

        # Extract and transform the visible portion
        source_rect = image_rect.toAlignedRect()
        cropped = self._image.copy(source_rect)

        if cropped.isNull():
            return

        # Compute the transformation for this crop
        # Transform: crop origin -> virtual -> widget
        crop_xform = QTransform()
        crop_xform.translate(source_rect.x(), source_rect.y())

        full_xform = crop_xform * self._image_to_virtual * self._virtual_to_widget

        # Get the destination size
        dest_rect = full_xform.mapRect(QRectF(0, 0, cropped.width(), cropped.height()))
        dest_size = dest_rect.size().toSize()

        if dest_size.width() <= 0 or dest_size.height() <= 0:
            return

        # Create high-quality transformed image
        hq_image = QImage(dest_size, QImage.Format.Format_ARGB32_Premultiplied)
        hq_image.fill(Qt.GlobalColor.transparent)

        painter = QPainter(hq_image)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Apply transformation relative to the destination rect origin
        painter.translate(-dest_rect.x(), -dest_rect.y())
        painter.setTransform(full_xform, True)
        painter.drawImage(0, 0, cropped)
        painter.end()

        self._hq_pixmap = QPixmap.fromImage(hq_image)
        self._hq_pixmap_pos = dest_rect.topLeft().toPoint()
        self._hq_xform = self._virtual_to_widget

        self.update()

    def set_zoom(self, zoom: float, center_point: QPointF | None = None) -> None:
        """Set the zoom level.

        Args:
            zoom: The zoom level (1.0 = 100%).
            center_point: Point in widget coordinates to zoom around.
        """
        zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        if abs(zoom - self._zoom) < 0.001:
            return

        # Zoom around the specified point or current focal point
        if center_point is not None:
            # Convert widget point to virtual coordinates before zoom
            virtual_point = self._widget_to_virtual.map(center_point)
            self._pixmap_focal_point = virtual_point
            self._widget_focal_point = center_point

        self._zoom = zoom
        self._update_widget_transform()
        self.zoom_changed.emit(zoom)
        self.update()

    def zoom_in(self, center_point: QPointF | None = None) -> None:
        """Zoom in by one step.

        Args:
            center_point: Point to zoom around.
        """
        self.set_zoom(self._zoom * self.ZOOM_STEP, center_point)

    def zoom_out(self, center_point: QPointF | None = None) -> None:
        """Zoom out by one step.

        Args:
            center_point: Point to zoom around.
        """
        self.set_zoom(self._zoom / self.ZOOM_STEP, center_point)

    def zoom_to_fit(self) -> None:
        """Fit the entire image in the widget."""
        self._fit_to_widget()
        self.zoom_changed.emit(self._zoom)
        self.update()

    @property
    def zoom_level(self) -> float:
        """Get the current zoom level."""
        return self._zoom

    @property
    def image_to_virtual(self) -> QTransform:
        """Get the image-to-virtual transformation."""
        return self._image_to_virtual

    @property
    def virtual_to_widget(self) -> QTransform:
        """Get the virtual-to-widget transformation."""
        return self._virtual_to_widget

    @property
    def widget_to_virtual(self) -> QTransform:
        """Get the widget-to-virtual transformation."""
        return self._widget_to_virtual

    def image_to_widget(self) -> QTransform:
        """Get the composite image-to-widget transformation."""
        return self._image_to_virtual * self._virtual_to_widget

    def widget_to_image(self) -> QTransform:
        """Get the composite widget-to-image transformation."""
        return self._widget_to_virtual * self._virtual_to_image

    # Event handlers

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Handle paint event.

        Args:
            event: The paint event.
        """
        viewport = self.viewport()
        if viewport is None:
            return

        painter = QPainter(viewport)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Draw the image
        self._paint_image(painter)

        # Let subclasses draw overlays
        self._paint_overlay(painter)

        painter.end()

    def _paint_image(self, painter: QPainter) -> None:
        """Paint the image to the viewport.

        Args:
            painter: The painter to draw with.
        """
        if self._pixmap is None:
            return

        # Use HQ pixmap if available and transform hasn't changed
        if self._hq_pixmap is not None and self._hq_xform == self._virtual_to_widget:
            # Draw the high-quality version
            painter.drawPixmap(self._hq_pixmap_pos, self._hq_pixmap)
        else:
            # Draw using the downscaled pixmap with transformation
            painter.save()

            # Apply pixmap-to-image, then image-to-virtual, then virtual-to-widget
            full_xform = (
                self._pixmap_to_image * self._image_to_virtual * self._virtual_to_widget
            )
            painter.setTransform(full_xform)
            painter.drawPixmap(0, 0, self._pixmap)

            painter.restore()

    def _paint_overlay(self, painter: QPainter) -> None:
        """Paint overlay graphics (to be overridden by subclasses).

        Args:
            painter: The painter to draw with.
        """
        pass

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handle resize event.

        Args:
            event: The resize event.
        """
        super().resizeEvent(event)
        self._fit_to_widget()
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press event.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self._dragging = True
            self._drag_start_pos = event.position()
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release event.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.MiddleButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move event.

        Args:
            event: The mouse event.
        """
        if self._dragging:
            # Pan the image
            delta = event.position() - self._last_mouse_pos
            self._widget_focal_point += delta
            self._last_mouse_pos = event.position()
            self._update_widget_transform()
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Handle mouse wheel event for zooming.

        Args:
            event: The wheel event.
        """
        # Zoom in/out with scroll wheel
        delta = event.angleDelta().y()
        center = event.position()

        if delta > 0:
            self.zoom_in(center)
        elif delta < 0:
            self.zoom_out(center)

        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press event.

        Args:
            event: The key event.
        """
        if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.zoom_in()
            event.accept()
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
        elif event.key() == Qt.Key.Key_0:
            self.zoom_to_fit()
            event.accept()
        elif event.key() == Qt.Key.Key_1:
            self.set_zoom(1.0)
            event.accept()
        else:
            super().keyPressEvent(event)

    # Utility methods for subclasses

    def virtual_to_widget_point(self, point: QPointF) -> QPointF:
        """Convert a point from virtual to widget coordinates.

        Args:
            point: Point in virtual coordinates.

        Returns:
            Point in widget coordinates.
        """
        return self._virtual_to_widget.map(point)

    def widget_to_virtual_point(self, point: QPointF) -> QPointF:
        """Convert a point from widget to virtual coordinates.

        Args:
            point: Point in widget coordinates.

        Returns:
            Point in virtual coordinates.
        """
        return self._widget_to_virtual.map(point)

    def virtual_to_widget_rect(self, rect: QRectF) -> QRectF:
        """Convert a rect from virtual to widget coordinates.

        Args:
            rect: Rectangle in virtual coordinates.

        Returns:
            Rectangle in widget coordinates.
        """
        return self._virtual_to_widget.mapRect(rect)

    def widget_to_virtual_rect(self, rect: QRectF) -> QRectF:
        """Convert a rect from widget to virtual coordinates.

        Args:
            rect: Rectangle in widget coordinates.

        Returns:
            Rectangle in virtual coordinates.
        """
        return self._widget_to_virtual.mapRect(rect)
