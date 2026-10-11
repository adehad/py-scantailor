"""Output filter image views.

Provides image views for the Output filter including:
- TabbedImageView: Multi-tab view for different output modes
- OutputImageView: Output view with picture and fill zone editing
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from itertools import pairwise
from typing import ClassVar

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import QTabWidget

from scantailor.app.ui.image_views.base import ImageViewBase


class ZoneKind(Enum):
    """The two kinds of zone the Output view edits."""

    PICTURE = auto()
    FILL = auto()


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

    def add_tab(self, widget: QtWidgets.QWidget, label: str, tab: ImageViewTab) -> None:
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


@dataclass
class ViewZone:
    """A zone as the view shows it.

    Attributes:
        kind: Picture or fill zone.
        polygon: Vertices in virtual coordinates.
        fill_color: RGB colour a fill zone paints; unused for picture zones.
        properties: Opaque data the view carries along unchanged, so that
            properties it does not edit survive a round trip.
    """

    kind: ZoneKind
    polygon: QPolygonF
    fill_color: tuple[int, int, int] = (255, 255, 255)
    properties: object | None = None

    def copy(self) -> "ViewZone":
        """Return a copy with its own polygon."""
        return ViewZone(
            self.kind, QPolygonF(self.polygon), self.fill_color, self.properties
        )


class OutputImageView(ImageViewBase):
    """Image view of the Output stage with picture and fill zone editing.

    Zones are polygons in virtual coordinates. A zone is created by clicking
    its corners and finishing with a double click or Enter, selected by
    clicking it, reshaped by dragging its corners and deleted with Delete.

    Signals:
        zones_changed: The user created, reshaped or removed a zone.
    """

    zones_changed = Signal()

    VERTEX_HIT_RADIUS = 8
    MIN_ZONE_VERTICES = 3
    ZONE_COLORS: ClassVar[dict[ZoneKind, tuple[QColor, QColor]]] = {
        ZoneKind.PICTURE: (QColor(0, 100, 255, 200), QColor(0, 100, 255, 50)),
        ZoneKind.FILL: (QColor(0, 160, 60, 200), QColor(0, 160, 60, 50)),
    }
    SELECTED_BORDER = QColor(255, 165, 0, 255)
    SELECTED_FILL_ALPHA = 80
    VERTEX_COLOR = QColor(0, 100, 255, 255)
    FILL_PREVIEW_ALPHA = 128

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._zones: list[ViewZone] = []
        self._selected = -1
        self._creating: ZoneKind | None = None
        self._new_points: list[QPointF] = []
        self._dragged_vertex = -1
        self._drag_offset = QPointF()
        self._drag_moved = False
        self.setMouseTracking(True)

    def set_zones(self, zones: list[ViewZone]) -> None:
        """Show zones without reporting a change.

        Args:
            zones (list[ViewZone]): Zones in virtual coordinates.
        """
        self._zones = [zone.copy() for zone in zones]
        self._selected = -1
        self._dragged_vertex = -1
        self.cancel_zone_creation()

    def zones(self) -> list[ViewZone]:
        """Return the zones as shown.

        Returns:
            list[ViewZone]: Copies of the zones in virtual coordinates.
        """
        return [zone.copy() for zone in self._zones]

    def start_zone_creation(self, kind: ZoneKind) -> None:
        """Start collecting the corners of a new zone.

        Args:
            kind (ZoneKind): Kind of zone to create.
        """
        self._creating = kind
        self._new_points = []
        self.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_zone_creation(self) -> None:
        """Drop the zone being created."""
        self._creating = None
        self._new_points = []
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _finish_zone_creation(self) -> None:
        kind = self._creating
        points = self._new_points
        self.cancel_zone_creation()
        if kind is not None and len(points) >= self.MIN_ZONE_VERTICES:
            self._zones.append(ViewZone(kind, QPolygonF(points)))
            self._selected = len(self._zones) - 1
            self.zones_changed.emit()

    def remove_selected_zone(self) -> None:
        """Delete the selected zone."""
        if 0 <= self._selected < len(self._zones):
            del self._zones[self._selected]
            self._selected = -1
            self.update()
            self.zones_changed.emit()

    def clear_zones(self) -> None:
        """Delete every zone."""
        if self._zones:
            self._zones.clear()
            self._selected = -1
            self.update()
            self.zones_changed.emit()

    def _zone_at(self, widget_pos: QPointF) -> int:
        virtual = self.widget_to_virtual_point(widget_pos)
        for index in reversed(range(len(self._zones))):
            if self._zones[index].polygon.containsPoint(
                virtual, Qt.FillRule.OddEvenFill
            ):
                return index
        return -1

    def _vertex_at(self, widget_pos: QPointF) -> int:
        """Return the index of a selected-zone vertex under a position, or -1."""
        if not 0 <= self._selected < len(self._zones):
            return -1
        for index, point in enumerate(self._zones[self._selected].polygon.toList()):
            offset = self.virtual_to_widget_point(point) - widget_pos
            if math.hypot(offset.x(), offset.y()) <= self.VERTEX_HIT_RADIUS:
                return index
        return -1

    def _paint_overlay(self, painter: QPainter) -> None:
        for index, zone in enumerate(self._zones):
            self._paint_zone(painter, zone, index == self._selected)
        if self._creating is not None and self._new_points:
            self._paint_new_zone(painter)

    def _paint_zone(self, painter: QPainter, zone: ViewZone, selected: bool) -> None:
        border, fill = self.ZONE_COLORS[zone.kind]
        if zone.kind == ZoneKind.FILL:
            fill = QColor(*zone.fill_color, self.FILL_PREVIEW_ALPHA)
        if selected:
            border = self.SELECTED_BORDER
            fill = QColor(fill)
            fill.setAlpha(self.SELECTED_FILL_ALPHA)
        polygon = self._virtual_to_widget.map(zone.polygon)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(fill)
        painter.setPen(QPen(border, 2 if selected else 1))
        painter.drawPolygon(polygon)
        if selected:
            painter.setBrush(self.VERTEX_COLOR)
            painter.setPen(Qt.PenStyle.NoPen)
            for point in polygon.toList():
                painter.drawEllipse(point, 5, 5)
        painter.restore()

    def _paint_new_zone(self, painter: QPainter) -> None:
        assert self._creating is not None
        border = self.ZONE_COLORS[self._creating][0]
        points = [self.virtual_to_widget_point(p) for p in self._new_points]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(border, 2, Qt.PenStyle.DashLine))
        for start, end in pairwise(points):
            painter.drawLine(start, end)
        if len(points) >= 2:
            painter.setPen(QPen(border.lighter(), 1, Qt.PenStyle.DotLine))
            painter.drawLine(points[-1], points[0])
        painter.setBrush(self.VERTEX_COLOR)
        painter.setPen(Qt.PenStyle.NoPen)
        for point in points:
            painter.drawEllipse(point, 5, 5)
        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Add a corner, pick a corner of the selected zone, or select a zone.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if self._creating is not None:
                self._new_points.append(self.widget_to_virtual_point(pos))
                self.update()
                event.accept()
                return

            vertex = self._vertex_at(pos)
            if vertex >= 0:
                self._dragged_vertex = vertex
                self._drag_moved = False
                corner = self._zones[self._selected].polygon.at(vertex)
                self._drag_offset = self.widget_to_virtual_point(pos) - corner
                event.accept()
                return

            zone = self._zone_at(pos)
            if zone >= 0:
                self._selected = zone
                self.update()
                event.accept()
                return
            if self._selected >= 0:
                self._selected = -1
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Move the picked corner.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._dragged_vertex >= 0:
            point = self.widget_to_virtual_point(event.position()) - self._drag_offset
            polygon = self._zones[self._selected].polygon
            if polygon.at(self._dragged_vertex) != point:
                corners = polygon.toList()
                corners[self._dragged_vertex] = point
                self._zones[self._selected].polygon = QPolygonF(corners)
                self._drag_moved = True
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop the corner and report the new shape.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._dragged_vertex >= 0:
            self._dragged_vertex = -1
            event.accept()
            if self._drag_moved:
                self.zones_changed.emit()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Finish the zone being created.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._creating is not None:
            self._finish_zone_creation()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Finish, cancel, deselect or delete with Enter, Escape and Delete.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        key = event.key()
        if key == Qt.Key.Key_Escape and self._creating is not None:
            self.cancel_zone_creation()
        elif key == Qt.Key.Key_Escape and self._selected >= 0:
            self._selected = -1
            self.update()
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._selected >= 0:
            self.remove_selected_zone()
        elif (
            key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._creating is not None
        ):
            self._finish_zone_creation()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show the zone menu.

        Args:
            event (QtGui.QContextMenuEvent): The context menu event.
        """
        menu = QtWidgets.QMenu(self)
        if self._creating is not None:
            menu.addAction("Finish Zone", self._finish_zone_creation)
            menu.addAction("Cancel", self.cancel_zone_creation)
        else:
            menu.addAction(
                "Create Picture Zone",
                lambda: self.start_zone_creation(ZoneKind.PICTURE),
            )
            menu.addAction(
                "Create Fill Zone", lambda: self.start_zone_creation(ZoneKind.FILL)
            )
            if self._selected >= 0:
                menu.addSeparator()
                menu.addAction("Delete Zone", self.remove_selected_zone)
            if self._zones:
                menu.addSeparator()
                menu.addAction("Clear All Zones", self.clear_zones)
        menu.exec(event.globalPos())
