"""Output filter image views.

Provides image views for the Output filter including:
- TabbedImageView: Multi-tab view for different output modes
- OutputImageView: Output view with picture and fill zone editing
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QLineF, QPointF, QRectF, QSettings, Qt, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import QTabWidget

from scantailor.app.ui.icons import icon
from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.settings import ApplicationSettings, ZoneCreationMode
from scantailor.dewarping.page_dewarper import dewarper_from_polylines
from scantailor.filters.output.params import DistortionModelSpec
from scantailor.math.spline import XSpline


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


_TAB_ICONS: dict[ImageViewTab, str] = {
    ImageViewTab.PICTURE_ZONES: "image",
    ImageViewTab.FILL_ZONES: "paint-bucket",
    ImageViewTab.DEWARPING: "dewarp-grid",
}


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
        glyph = _TAB_ICONS.get(tab)
        if glyph is None:
            self.addTab(widget, label)
        else:
            self.addTab(widget, icon(glyph), label)
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

    Zones are polygons in virtual coordinates. A zone is created in one of
    three modes, chosen with Z (polygonal), X (lasso) and C (rectangular): by
    clicking its corners and finishing with a double click or Enter, by
    dragging a freehand outline, or by dragging a rectangle. It is selected by
    clicking it, reshaped by dragging its corners and deleted with Delete. The
    mode a new view starts in is the application's default zone creation mode.

    Optional guides, a dashed cross through the centre of the view, help to
    judge the alignment of the output. The choice is kept in the application
    settings under ``SHOW_GUIDES_KEY``.

    Signals:
        zones_changed: The user created, reshaped or removed a zone.
    """

    zones_changed = Signal()

    SHOW_GUIDES_KEY = "output/showGuides"

    VERTEX_HIT_RADIUS = 8
    MIN_ZONE_VERTICES = 3
    LASSO_STEP = 4
    SELECTED_FILL_ALPHA = 80
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
        self._creation_mode = (
            ApplicationSettings.get_instance().default_zone_creation_mode
        )
        self._outlining = False
        self._outline_anchor = QPointF()
        self._dragged_vertex = -1
        self._drag_offset = QPointF()
        self._drag_moved = False
        self._show_guides = bool(
            QSettings().value(self.SHOW_GUIDES_KEY, False, type=bool)
        )
        self.setMouseTracking(True)

    def guides_shown(self) -> bool:
        """Return whether the centre guides are drawn.

        Returns:
            bool: True while the guides show.
        """
        return self._show_guides

    def set_guides_shown(self, shown: bool) -> None:
        """Show or hide the centre guides and remember the choice.

        Args:
            shown (bool): True to draw the guides.
        """
        self._show_guides = shown
        QSettings().setValue(self.SHOW_GUIDES_KEY, shown)
        self.update()

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

    def creation_mode(self) -> ZoneCreationMode:
        """Return how new zones are drawn.

        Returns:
            ZoneCreationMode: The mode the next zone is created in.
        """
        return self._creation_mode

    def set_creation_mode(self, mode: ZoneCreationMode) -> None:
        """Choose how new zones are drawn and make it the application default.

        A zone being drawn in another mode is dropped.

        Args:
            mode (ZoneCreationMode): The mode for the zones created from now on.
        """
        if mode is self._creation_mode:
            return
        self._creation_mode = mode
        if self._creating is not None:
            self._new_points = []
            self._outlining = False
            self.update()
        settings = ApplicationSettings.get_instance()
        settings.default_zone_creation_mode = mode
        settings.save()

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
        self._outlining = False
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
        if self._show_guides:
            self._paint_guides(painter)

    def _paint_guides(self, painter: QPainter) -> None:
        viewport = self.viewport()
        assert viewport is not None
        centre = QPointF(viewport.width() / 2, viewport.height() / 2)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(view_color("guide"))
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QLineF(centre.x(), 0, centre.x(), viewport.height()))
        painter.drawLine(QLineF(0, centre.y(), viewport.width(), centre.y()))
        painter.restore()

    @staticmethod
    def _zone_colors(kind: ZoneKind) -> tuple[QColor, QColor]:
        role = "zone_picture" if kind == ZoneKind.PICTURE else "zone_fill"
        return view_color(f"{role}_border"), view_color(f"{role}_fill")

    def _paint_zone(self, painter: QPainter, zone: ViewZone, selected: bool) -> None:
        border, fill = self._zone_colors(zone.kind)
        if zone.kind == ZoneKind.FILL:
            fill = QColor(*zone.fill_color, self.FILL_PREVIEW_ALPHA)
        if selected:
            border = view_color("zone_selected")
            fill = QColor(fill)
            fill.setAlpha(self.SELECTED_FILL_ALPHA)
        polygon = self._virtual_to_widget.map(zone.polygon)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(fill)
        painter.setPen(QPen(border, 2 if selected else 1))
        painter.drawPolygon(polygon)
        if selected:
            painter.setBrush(view_color("zone_vertex"))
            painter.setPen(Qt.PenStyle.NoPen)
            for point in polygon.toList():
                painter.drawEllipse(point, 5, 5)
        painter.restore()

    def _paint_new_zone(self, painter: QPainter) -> None:
        assert self._creating is not None
        border = self._zone_colors(self._creating)[0]
        points = [self.virtual_to_widget_point(p) for p in self._new_points]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(border, 2, Qt.PenStyle.DashLine))
        for start, end in pairwise(points):
            painter.drawLine(start, end)
        if len(points) >= 2:
            painter.setPen(QPen(border.lighter(), 1, Qt.PenStyle.DotLine))
            painter.drawLine(points[-1], points[0])
        if self._creation_mode is not ZoneCreationMode.LASSO:
            painter.setBrush(view_color("zone_vertex"))
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
                point = self.widget_to_virtual_point(pos)
                if self._creation_mode is ZoneCreationMode.POLYGONAL:
                    self._new_points.append(point)
                else:
                    self._outlining = True
                    self._outline_anchor = point
                    self._new_points = [point]
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
        """Extend the outline being drawn, or move the picked corner.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._outlining:
            self._extend_outline(self.widget_to_virtual_point(event.position()))
            event.accept()
            return
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

    def _extend_outline(self, point: QPointF) -> None:
        if self._creation_mode is ZoneCreationMode.RECTANGULAR:
            anchor = self._outline_anchor
            self._new_points = [
                anchor,
                QPointF(point.x(), anchor.y()),
                point,
                QPointF(anchor.x(), point.y()),
            ]
        else:
            last = self.virtual_to_widget_point(self._new_points[-1])
            here = self.virtual_to_widget_point(point)
            if math.hypot(here.x() - last.x(), here.y() - last.y()) >= self.LASSO_STEP:
                self._new_points.append(point)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Finish a dragged outline, or drop the corner and report the shape.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._outlining:
            self._outlining = False
            if self._creation_mode is ZoneCreationMode.RECTANGULAR:
                corners = self._new_points
                if len(corners) == 4 and (
                    corners[0].x() == corners[2].x() or corners[0].y() == corners[2].y()
                ):
                    self._new_points = []
            self._finish_zone_creation()
            event.accept()
            return
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

        Z, X and C choose the polygonal, lasso and rectangular creation modes.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        key = event.key()
        if key in _MODE_KEYS and not event.modifiers():
            self.set_creation_mode(_MODE_KEYS[key])
        elif key == Qt.Key.Key_Escape and self._creating is not None:
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
        menu.addSeparator()
        guides = menu.addAction(icon("crosshair"), "Show guides")
        guides.setCheckable(True)
        guides.setChecked(self._show_guides)
        guides.toggled.connect(self.set_guides_shown)
        menu.exec(event.globalPos())


_MODE_KEYS = {
    Qt.Key.Key_Z: ZoneCreationMode.POLYGONAL,
    Qt.Key.Key_X: ZoneCreationMode.LASSO,
    Qt.Key.Key_C: ZoneCreationMode.RECTANGULAR,
}


def _nearest_on_segment(
    point: QPointF, start: QPointF, end: QPointF
) -> tuple[QPointF, float]:
    """Return the point of a segment closest to a point, and its distance."""
    edge = end - start
    length_squared = edge.x() ** 2 + edge.y() ** 2
    if length_squared == 0:
        nearest = start
    else:
        along = (point.x() - start.x()) * edge.x() + (point.y() - start.y()) * edge.y()
        nearest = start + edge * min(1.0, max(0.0, along / length_squared))
    offset = point - nearest
    return nearest, math.hypot(offset.x(), offset.y())


class DewarpingImageView(ImageViewBase):
    """Shows the distortion model of a page as a grid and edits its edges.

    The model is two edges, top and bottom, each a curve through control points
    that run left to right. Dragging a control point reshapes the curve, pressing
    on a curve between points adds a point, and D removes the point under the
    cursor. Everything is in the pixel coordinates of the view's image.

    Signals:
        distortion_model_changed: The user reshaped an edge; carries the new
            DistortionModelSpec, whose edges are sampled from the curves.
    """

    distortion_model_changed = Signal(object)

    HIT_RADIUS = 8
    MAX_CONTROL_POINTS = 5
    GRID_LINES = 30
    _DEWARPER_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._controls: list[list[QPointF]] = [[], []]
        self._edges: list[NDArray[np.float64]] = [np.empty((0, 2)), np.empty((0, 2))]
        self._grid: NDArray[np.float64] | None = None
        self._depth_perception = 2.0
        self._shown_model: DistortionModelSpec | None = None
        self._dragged: tuple[int, int] | None = None
        self._drag_changed = False
        self._hover = QPointF()
        self.setMouseTracking(True)

    def set_distortion_model(
        self,
        model: DistortionModelSpec,
        depth_perception: float,
        default_area: QRectF | None = None,
    ) -> None:
        """Show a model without reporting a change.

        Args:
            model (DistortionModelSpec): Edges in image pixels. When it is not
                defined the view starts from straight edges along the top and
                bottom of ``default_area``.
            depth_perception (float): Distance to the page plane, 1 to 3.
            default_area (QRectF | None): Where straight edges start; the whole
                image when None.
        """
        self._depth_perception = depth_perception
        self._dragged = None
        if model != self._shown_model or not self._controls[0]:
            self._shown_model = model
            if model.is_defined():
                self._controls = [
                    self._pick_control_points(model.top),
                    self._pick_control_points(model.bottom),
                ]
            else:
                area = default_area
                if area is None:
                    area = QRectF(self._image.rect()) if self._image else QRectF()
                self._controls = [
                    [area.topLeft(), area.topRight()],
                    [area.bottomLeft(), area.bottomRight()],
                ]
        self._rebuild()

    def set_depth_perception(self, depth_perception: float) -> None:
        """Redraw the grid for another depth perception.

        Args:
            depth_perception (float): Distance to the page plane, 1 to 3.
        """
        self._depth_perception = depth_perception
        self._rebuild()

    def control_points(self) -> tuple[list[QPointF], list[QPointF]]:
        """Return the control points of the top and bottom edge.

        Returns:
            tuple[list[QPointF], list[QPointF]]: Copies, in image pixels.
        """
        return list(self._controls[0]), list(self._controls[1])

    def distortion_model(self) -> DistortionModelSpec:
        """Return the model as the edges are now.

        Returns:
            DistortionModelSpec: Edges sampled from the curves, in image pixels.
        """
        return DistortionModelSpec(
            top=tuple((float(x), float(y)) for x, y in self._edges[0]),
            bottom=tuple((float(x), float(y)) for x, y in self._edges[1]),
        )

    def _pick_control_points(
        self, polyline: tuple[tuple[float, float], ...]
    ) -> list[QPointF]:
        """Pick the control points of an edge: the ends and a few between them."""
        if len(polyline) <= self.MAX_CONTROL_POINTS:
            picked = list(range(len(polyline)))
        else:
            spread = np.linspace(0, len(polyline) - 1, self.MAX_CONTROL_POINTS)
            picked = np.round(spread).astype(int).tolist()
        return [QPointF(*polyline[i]) for i in picked]

    @staticmethod
    def _sample_curve(points: list[QPointF]) -> NDArray[np.float64]:
        """Sample a curve passing through the control points."""
        spline = XSpline()
        last = len(points) - 1
        for index, point in enumerate(points):
            tension = 0.0 if index in (0, last) else -1.0
            spline.append_control_point((point.x(), point.y()), tension)
        return spline.to_polyline()

    def _rebuild(self) -> None:
        self._edges = [self._sample_curve(points) for points in self._controls]
        self._grid = self._compute_grid()
        self.update()

    def _compute_grid(self) -> NDArray[np.float64] | None:
        """Return the grid as ``(columns, rows, 2)`` image points, or None."""
        dewarper = dewarper_from_polylines(
            self._edges[0], self._edges[1], self._depth_perception
        )
        if dewarper is None:
            return None
        try:
            batch = dewarper.map_generatrices(np.linspace(0.0, 1.0, self.GRID_LINES))
        except self._DEWARPER_ERRORS:
            return None
        rows = np.linspace(0.0, 1.0, self.GRID_LINES)
        matrices = batch.pln2img_1d
        numerator = matrices[:, 0, 0, None] * rows + matrices[:, 0, 1, None]
        denominator = matrices[:, 1, 0, None] * rows + matrices[:, 1, 1, None]
        degenerate = np.abs(denominator) < 1e-10
        along = np.where(
            degenerate, 0.0, numerator / np.where(degenerate, 1.0, denominator)
        )
        span = batch.end - batch.start
        return batch.start[:, None, :] + along[..., None] * span[:, None, :]

    def _paint_overlay(self, painter: QPainter) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        grid_pen = QPen(view_color("dewarp_grid"), 1.2)
        grid_pen.setCosmetic(True)
        painter.setPen(grid_pen)
        painter.setTransform(self.image_to_widget(), True)
        if self._grid is not None:
            for line in (*self._grid, *self._grid.transpose(1, 0, 2)):
                painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in line]))
        else:
            self._paint_frame(painter)
        painter.restore()

        painter.save()
        painter.setPen(QPen(view_color("dewarp_handle"), 4))
        to_widget = self.image_to_widget()
        for points in self._controls:
            for point in points:
                painter.drawPoint(to_widget.map(point))
        painter.restore()

    def _paint_frame(self, painter: QPainter) -> None:
        """Draw the edges and the sides when the model has no valid grid."""
        top, bottom = self._edges
        if len(top) < 2 or len(bottom) < 2:
            return
        for edge in (top, bottom):
            painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in edge]))
        painter.drawLine(QPointF(*top[0]), QPointF(*bottom[0]))
        painter.drawLine(QPointF(*top[-1]), QPointF(*bottom[-1]))

    def _handle_at(self, widget_pos: QPointF) -> tuple[int, int] | None:
        """Return the (edge, control point) under a widget position."""
        to_widget = self.image_to_widget()
        for edge, points in enumerate(self._controls):
            for index, point in enumerate(points):
                offset = to_widget.map(point) - widget_pos
                if math.hypot(offset.x(), offset.y()) <= self.HIT_RADIUS:
                    return edge, index
        return None

    def _insert_at(self, widget_pos: QPointF) -> tuple[int, int] | None:
        """Add a control point on the edge near a widget position, if any."""
        to_widget = self.image_to_widget()
        for edge, polyline in enumerate(self._edges):
            for start, end in pairwise(polyline):
                nearest, distance = _nearest_on_segment(
                    widget_pos,
                    to_widget.map(QPointF(*start)),
                    to_widget.map(QPointF(*end)),
                )
                if distance <= self.HIT_RADIUS:
                    point = self.widget_to_image().map(nearest)
                    points = self._controls[edge]
                    index = sum(1 for p in points if p.x() < point.x())
                    index = min(max(index, 1), len(points) - 1)
                    points.insert(index, point)
                    self._rebuild()
                    return edge, index
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pick a control point, or add one when pressing on an edge.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._controls[0]:
            pos = event.position()
            handle = self._handle_at(pos)
            self._drag_changed = False
            if handle is None:
                handle = self._insert_at(pos)
                self._drag_changed = handle is not None
            if handle is not None:
                self._dragged = handle
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Move the picked control point.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        self._hover = event.position()
        if self._dragged is not None:
            edge, index = self._dragged
            point = self.widget_to_image().map(event.position())
            if self._controls[edge][index] != point:
                self._controls[edge][index] = point
                self._drag_changed = True
                self._rebuild()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop the control point and report the new model.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._dragged is not None:
            self._dragged = None
            event.accept()
            if self._drag_changed:
                self._report_change()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Remove the control point under the cursor with D.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        if event.key() == Qt.Key.Key_D and self._remove_control_point_under_cursor():
            event.accept()
            return
        super().keyPressEvent(event)

    def _remove_control_point_under_cursor(self) -> bool:
        handle = self._handle_at(self._hover)
        if handle is None:
            return False
        edge, index = handle
        points = self._controls[edge]
        if index in (0, len(points) - 1):
            return False
        del points[index]
        self._rebuild()
        self._report_change()
        return True

    def _report_change(self) -> None:
        self._shown_model = self.distortion_model()
        self.distortion_model_changed.emit(self._shown_model)
