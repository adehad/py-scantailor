"""Output filter image views.

Provides image views for the Output filter including:
- TabbedImageView: Multi-tab view for different output modes
- OutputImageView: Base view for output with zone editing support
- ZoneEditor: Interactive zone editing for picture and fill zones
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import QTabWidget

from scantailor.app.ui.image_views.base import ImageViewBase

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scantailor.core import ImageTransformation


class ImageViewTab(Enum):
    """Tabs available in the Output tabbed image view."""

    ORIGINAL = auto()
    OUTPUT = auto()
    PICTURE_ZONES = auto()
    FILL_ZONES = auto()
    DEWARPING = auto()


class TabbedImageView(QTabWidget):
    """Tabbed container for multiple output image views.

    Provides tabs for viewing different aspects of the output:
    - Original image
    - Processed output
    - Picture zones editor
    - Fill zones editor
    - Dewarping view

    The view synchronizes zoom and pan position between tabs.

    Signals:
        tab_changed: Emitted when the active tab changes (ImageViewTab).
    """

    tab_changed = Signal(ImageViewTab)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the tabbed image view.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._registry: dict[QtWidgets.QWidget, ImageViewTab] = {}
        self._tab_image_rects: dict[ImageViewTab, QRectF] = {}
        self._prev_tab_index = -1

        # Connect tab change signal
        self.currentChanged.connect(self._on_tab_changed)

        # Set up keyboard shortcuts for tab switching (1-5)
        self._shortcuts: list[QShortcut] = []
        for i in range(5):
            shortcut = QShortcut(QtGui.QKeySequence(str(i + 1)), self)
            shortcut.activated.connect(lambda idx=i: self._switch_to_tab(idx))
            self._shortcuts.append(shortcut)

    def add_tab(
        self, widget: QtWidgets.QWidget, label: str, tab: ImageViewTab
    ) -> None:
        """Add a tab with an associated ImageViewTab enum.

        Args:
            widget: The widget to add as a tab.
            label: Tab label text.
            tab: The ImageViewTab enum value for this tab.
        """
        self.addTab(widget, label)
        self._registry[widget] = tab

    def set_image_rect_map(self, rects: dict[ImageViewTab, QRectF]) -> None:
        """Set the image rectangles for each tab.

        Used for synchronizing zoom and position between tabs.

        Args:
            rects: Map of tab to image rectangle.
        """
        self._tab_image_rects = rects

    @Slot(ImageViewTab)
    def set_current_tab(self, tab: ImageViewTab) -> None:
        """Switch to the specified tab.

        Args:
            tab: The tab to switch to.
        """
        for i in range(self.count()):
            widget = self.widget(i)
            if widget in self._registry and self._registry[widget] == tab:
                self.setCurrentIndex(i)
                break

    def _switch_to_tab(self, index: int) -> None:
        """Switch to a tab by index.

        Args:
            index: Tab index (0-based).
        """
        if 0 <= index < self.count():
            self.setCurrentIndex(index)

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change.

        Args:
            index: New tab index.
        """
        # Copy zoom and position from previous tab
        if self._prev_tab_index >= 0 and self._prev_tab_index != index:
            self._copy_view_zoom_and_pos(self._prev_tab_index, index)

        self._prev_tab_index = index

        # Emit the tab changed signal
        widget = self.widget(index)
        if widget in self._registry:
            self.tab_changed.emit(self._registry[widget])

    def _copy_view_zoom_and_pos(self, old_idx: int, new_idx: int) -> None:
        """Copy zoom level and position between image views.

        Args:
            old_idx: Source tab index.
            new_idx: Destination tab index.
        """
        old_widget = self.widget(old_idx)
        new_widget = self.widget(new_idx)

        if not isinstance(old_widget, ImageViewBase) or not isinstance(
            new_widget, ImageViewBase
        ):
            return

        # Copy zoom level
        new_widget.set_zoom(old_widget.zoom_level)

        # TODO: Copy scroll position based on image rect mapping


class OutputImageView(ImageViewBase):
    """Image view for Output filter with zone editing support.

    Extends ImageViewBase with zone overlay and editing capabilities.
    Used as the base for picture zone and fill zone editors.

    Signals:
        zones_changed: Emitted when zones are modified.
    """

    zones_changed = Signal()

    # Colors
    ZONE_BORDER_COLOR = QColor(0, 100, 255, 200)
    ZONE_FILL_COLOR = QColor(0, 100, 255, 50)
    ZONE_SELECTED_BORDER = QColor(255, 165, 0, 255)
    ZONE_SELECTED_FILL = QColor(255, 165, 0, 80)
    ZONE_HANDLE_COLOR = QColor(0, 100, 255, 255)

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the Output image view.

        Args:
            image: The image to display.
            downscaled_image: Optional pre-downscaled version.
            transformation: Image transformation.
            parent: Parent widget.
        """
        super().__init__(image, downscaled_image, transformation, parent=parent)

        # Zone data (list of polygons in virtual coordinates)
        self._zones: list[QPolygonF] = []
        self._selected_zone_index: int = -1

        # Zone editing state
        self._creating_zone = False
        self._current_zone_points: list[QPointF] = []

        self.setMouseTracking(True)

    def set_zones(self, zones: list[QPolygonF]) -> None:
        """Set the zones to display.

        Args:
            zones: List of zone polygons in virtual coordinates.
        """
        self._zones = zones
        self._selected_zone_index = -1
        self.update()

    def get_zones(self) -> list[QPolygonF]:
        """Get the current zones.

        Returns:
            List of zone polygons.
        """
        return self._zones.copy()

    def add_zone(self, zone: QPolygonF) -> None:
        """Add a new zone.

        Args:
            zone: Zone polygon in virtual coordinates.
        """
        self._zones.append(zone)
        self.zones_changed.emit()
        self.update()

    def remove_zone(self, index: int) -> None:
        """Remove a zone by index.

        Args:
            index: Zone index to remove.
        """
        if 0 <= index < len(self._zones):
            del self._zones[index]
            if self._selected_zone_index == index:
                self._selected_zone_index = -1
            elif self._selected_zone_index > index:
                self._selected_zone_index -= 1
            self.zones_changed.emit()
            self.update()

    def clear_zones(self) -> None:
        """Remove all zones."""
        self._zones.clear()
        self._selected_zone_index = -1
        self.zones_changed.emit()
        self.update()

    def start_zone_creation(self) -> None:
        """Start creating a new zone."""
        self._creating_zone = True
        self._current_zone_points = []
        self.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_zone_creation(self) -> None:
        """Cancel the current zone creation."""
        self._creating_zone = False
        self._current_zone_points = []
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _finish_zone_creation(self) -> None:
        """Finish creating the current zone."""
        if len(self._current_zone_points) >= 3:
            polygon = QPolygonF(self._current_zone_points)
            self.add_zone(polygon)

        self._creating_zone = False
        self._current_zone_points = []
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _zone_at_position(self, widget_pos: QPointF) -> int:
        """Find the zone at a widget position.

        Args:
            widget_pos: Position in widget coordinates.

        Returns:
            Zone index or -1 if no zone at position.
        """
        virtual_pos = self.widget_to_virtual_point(widget_pos)

        for i, zone in enumerate(self._zones):
            if zone.containsPoint(virtual_pos, Qt.FillRule.OddEvenFill):
                return i

        return -1

    def _paint_overlay(self, painter: QPainter) -> None:
        """Paint zone overlays.

        Args:
            painter: The painter.
        """
        # Paint existing zones
        for i, zone in enumerate(self._zones):
            self._paint_zone(
                painter,
                zone,
                is_selected=(i == self._selected_zone_index),
            )

        # Paint zone being created
        if self._creating_zone and self._current_zone_points:
            self._paint_zone_preview(painter)

    def _paint_zone(
        self, painter: QPainter, zone: QPolygonF, is_selected: bool = False
    ) -> None:
        """Paint a zone polygon.

        Args:
            painter: The painter.
            zone: Zone polygon in virtual coordinates.
            is_selected: Whether the zone is selected.
        """
        # Convert to widget coordinates
        widget_polygon = QPolygonF()
        for point in zone:
            widget_polygon.append(self.virtual_to_widget_point(point))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Fill
        fill_color = self.ZONE_SELECTED_FILL if is_selected else self.ZONE_FILL_COLOR
        painter.setBrush(fill_color)

        # Border
        border_color = (
            self.ZONE_SELECTED_BORDER if is_selected else self.ZONE_BORDER_COLOR
        )
        pen = QPen(border_color)
        pen.setWidth(2 if is_selected else 1)
        painter.setPen(pen)

        painter.drawPolygon(widget_polygon)

        # Draw handles if selected
        if is_selected:
            painter.setBrush(self.ZONE_HANDLE_COLOR)
            painter.setPen(Qt.PenStyle.NoPen)
            for point in widget_polygon:
                painter.drawEllipse(point, 5, 5)

        painter.restore()

    def _paint_zone_preview(self, painter: QPainter) -> None:
        """Paint the zone being created.

        Args:
            painter: The painter.
        """
        if not self._current_zone_points:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Draw lines between points
        pen = QPen(self.ZONE_BORDER_COLOR)
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)

        widget_points = [
            self.virtual_to_widget_point(p) for p in self._current_zone_points
        ]

        for i in range(len(widget_points) - 1):
            painter.drawLine(widget_points[i], widget_points[i + 1])

        # Draw closing line to first point (preview)
        if len(widget_points) >= 2:
            painter.setPen(QPen(self.ZONE_BORDER_COLOR.lighter(), 1, Qt.PenStyle.DotLine))
            painter.drawLine(widget_points[-1], widget_points[0])

        # Draw points
        painter.setBrush(self.ZONE_HANDLE_COLOR)
        painter.setPen(Qt.PenStyle.NoPen)
        for point in widget_points:
            painter.drawEllipse(point, 5, 5)

        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press for zone interaction.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()

            if self._creating_zone:
                # Add point to zone being created
                virtual_pos = self.widget_to_virtual_point(pos)
                self._current_zone_points.append(virtual_pos)
                self.update()
                event.accept()
                return

            # Select zone at click position
            zone_idx = self._zone_at_position(pos)
            if zone_idx >= 0:
                self._selected_zone_index = zone_idx
                self.update()
                event.accept()
                return

            # Deselect if clicking outside zones
            if self._selected_zone_index >= 0:
                self._selected_zone_index = -1
                self.update()

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle double-click to finish zone creation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._creating_zone:
            self._finish_zone_creation()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press for zone editing.

        Args:
            event: The key event.
        """
        if event.key() == Qt.Key.Key_Escape:
            if self._creating_zone:
                self.cancel_zone_creation()
                event.accept()
                return
            elif self._selected_zone_index >= 0:
                self._selected_zone_index = -1
                self.update()
                event.accept()
                return

        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self._selected_zone_index >= 0:
                self.remove_zone(self._selected_zone_index)
                event.accept()
                return

        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self._creating_zone:
                self._finish_zone_creation()
                event.accept()
                return

        super().keyPressEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show context menu for zone operations.

        Args:
            event: The context menu event.
        """
        menu = QtWidgets.QMenu(self)

        if self._creating_zone:
            finish_action = menu.addAction("Finish Zone")
            finish_action.triggered.connect(self._finish_zone_creation)
            cancel_action = menu.addAction("Cancel")
            cancel_action.triggered.connect(self.cancel_zone_creation)
        else:
            create_action = menu.addAction("Create Zone")
            create_action.triggered.connect(self.start_zone_creation)

            if self._selected_zone_index >= 0:
                menu.addSeparator()
                delete_action = menu.addAction("Delete Zone")
                delete_action.triggered.connect(
                    lambda: self.remove_zone(self._selected_zone_index)
                )

            if self._zones:
                menu.addSeparator()
                clear_action = menu.addAction("Clear All Zones")
                clear_action.triggered.connect(self.clear_zones)

        menu.exec(event.globalPos())


class PictureZoneEditor(OutputImageView):
    """Zone editor specialized for picture zones.

    Picture zones are areas that should be preserved as images
    (photographs, illustrations) rather than being binarized.

    Signals:
        invalidate_thumbnail: Emitted when zones change.
    """

    invalidate_thumbnail = Signal()

    # Picture mask overlay colors
    MASK_COLOR = QColor(255, 0, 0, 100)

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        picture_mask: NDArray[np.uint8] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the picture zone editor.

        Args:
            image: The image to display.
            downscaled_image: Optional pre-downscaled version.
            transformation: Image transformation.
            picture_mask: Binary mask of detected picture areas.
            parent: Parent widget.
        """
        super().__init__(image, downscaled_image, transformation, parent)

        self._picture_mask = picture_mask
        self._show_mask = True
        self._mask_animation_phase = 0

        # Animation timer for mask border
        self._mask_animation_timer = QtCore.QTimer(self)
        self._mask_animation_timer.timeout.connect(self._advance_mask_animation)
        self._mask_animation_timer.start(100)

        # Emit invalidate when zones change
        self.zones_changed.connect(self.invalidate_thumbnail.emit)

    def set_picture_mask(self, mask: NDArray[np.uint8] | None) -> None:
        """Set the picture mask.

        Args:
            mask: Binary mask of picture areas.
        """
        self._picture_mask = mask
        self.update()

    def set_show_mask(self, show: bool) -> None:
        """Set whether to show the picture mask overlay.

        Args:
            show: True to show the mask.
        """
        self._show_mask = show
        self.update()

    def _advance_mask_animation(self) -> None:
        """Advance the mask border animation."""
        self._mask_animation_phase = (self._mask_animation_phase + 10) % 360
        if self._show_mask and self._picture_mask is not None:
            self.update()

    def _paint_overlay(self, painter: QPainter) -> None:
        """Paint picture mask and zone overlays.

        Args:
            painter: The painter.
        """
        # Paint picture mask first
        if self._show_mask and self._picture_mask is not None:
            self._paint_picture_mask(painter)

        # Paint zones on top
        super()._paint_overlay(painter)

    def _paint_picture_mask(self, painter: QPainter) -> None:
        """Paint the picture mask overlay with animated border.

        Args:
            painter: The painter.
        """
        # This is a simplified version - the actual implementation would
        # transform the mask to widget coordinates and draw it with
        # an animated dashed border
        pass


class FillZoneEditor(OutputImageView):
    """Zone editor specialized for fill zones.

    Fill zones are areas that should be filled with a solid color
    (typically white for removing unwanted marks or artifacts).

    Signals:
        invalidate_thumbnail: Emitted when zones change.
    """

    invalidate_thumbnail = Signal()

    def __init__(
        self,
        image: QImage | NDArray[np.uint8] | None = None,
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the fill zone editor.

        Args:
            image: The image to display.
            downscaled_image: Optional pre-downscaled version.
            transformation: Image transformation.
            parent: Parent widget.
        """
        super().__init__(image, downscaled_image, transformation, parent)

        # Fill zones have associated colors
        self._zone_colors: list[QColor] = []

        # Default fill color
        self._default_fill_color = QColor(255, 255, 255)  # White

        # Emit invalidate when zones change
        self.zones_changed.connect(self.invalidate_thumbnail.emit)

    def add_zone_with_color(self, zone: QPolygonF, color: QColor) -> None:
        """Add a fill zone with a specific color.

        Args:
            zone: Zone polygon in virtual coordinates.
            color: Fill color for the zone.
        """
        self._zones.append(zone)
        self._zone_colors.append(color)
        self.zones_changed.emit()
        self.update()

    def set_zone_color(self, index: int, color: QColor) -> None:
        """Set the color of a fill zone.

        Args:
            index: Zone index.
            color: New fill color.
        """
        if 0 <= index < len(self._zone_colors):
            self._zone_colors[index] = color
            self.zones_changed.emit()
            self.update()

    def remove_zone(self, index: int) -> None:
        """Remove a fill zone.

        Args:
            index: Zone index to remove.
        """
        if 0 <= index < len(self._zone_colors):
            del self._zone_colors[index]
        super().remove_zone(index)

    def clear_zones(self) -> None:
        """Remove all fill zones."""
        self._zone_colors.clear()
        super().clear_zones()

    def _paint_zone(
        self, painter: QPainter, zone: QPolygonF, is_selected: bool = False
    ) -> None:
        """Paint a fill zone with its assigned color.

        Args:
            painter: The painter.
            zone: Zone polygon in virtual coordinates.
            is_selected: Whether the zone is selected.
        """
        # Find the index of this zone
        try:
            zone_idx = self._zones.index(zone)
            fill_color = (
                self._zone_colors[zone_idx]
                if zone_idx < len(self._zone_colors)
                else self._default_fill_color
            )
        except ValueError:
            fill_color = self._default_fill_color

        # Convert to widget coordinates
        widget_polygon = QPolygonF()
        for point in zone:
            widget_polygon.append(self.virtual_to_widget_point(point))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Fill with zone color (semi-transparent for preview)
        preview_color = QColor(fill_color)
        preview_color.setAlpha(128)
        painter.setBrush(preview_color)

        # Border
        border_color = (
            self.ZONE_SELECTED_BORDER if is_selected else self.ZONE_BORDER_COLOR
        )
        pen = QPen(border_color)
        pen.setWidth(2 if is_selected else 1)
        painter.setPen(pen)

        painter.drawPolygon(widget_polygon)

        # Draw handles if selected
        if is_selected:
            painter.setBrush(self.ZONE_HANDLE_COLOR)
            painter.setPen(Qt.PenStyle.NoPen)
            for point in widget_polygon:
                painter.drawEllipse(point, 5, 5)

        painter.restore()
