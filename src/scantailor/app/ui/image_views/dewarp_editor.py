"""Distortion model editing shared by the dewarping views.

A model is two edges, top and bottom, each a curve through control points that
run left to right. The editor draws the grid the model implies and turns mouse
and key events of a host view into edits. It is not a widget: the host forwards
the events and calls ``paint`` from its overlay.
"""

import math
from collections.abc import Callable
from itertools import pairwise
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui
from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication, QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPen, QPolygonF

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.image_views.zone_geometry import nearest_on_segment
from scantailor.app.ui.theme.tokens import view_color
from scantailor.dewarping.page_dewarper import dewarper_from_polylines
from scantailor.filters.output.params import DistortionModelSpec
from scantailor.math.spline import XSpline

_END_POINT_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "InteractiveXSpline",
        "This point can be dragged. Hold Ctrl or Shift to drag along axes.",
    ),
)
_POINT_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "InteractiveXSpline", "Drag this point or delete it by pressing D key."
    ),
)
_CURVE_TIP = cast(
    str, QT_TRANSLATE_NOOP("InteractiveXSpline", "Click to create a new control point.")
)


class DewarpEditor:
    """State and interaction of the manual dewarping grid.

    Everything is in the pixel coordinates of the host view's image.
    """

    HIT_RADIUS = 8
    MAX_CONTROL_POINTS = 5
    GRID_LINES = 30
    _DEWARPER_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError)

    def __init__(
        self, host: ImageViewBase, on_change: Callable[[DistortionModelSpec], None]
    ) -> None:
        """Create the editor.

        Args:
            host (ImageViewBase): The view it draws on and reads the transform from.
            on_change (Callable[[DistortionModelSpec], None]): Called with the new
                model after the user reshaped an edge.
        """
        self._host = host
        self._on_change = on_change
        self._controls: list[list[QPointF]] = [[], []]
        self._edges: list[NDArray[np.float64]] = [np.empty((0, 2)), np.empty((0, 2))]
        self._grid: NDArray[np.float64] | None = None
        self._depth_perception = 2.0
        self._shown_model: DistortionModelSpec | None = None
        self._dragged: tuple[int, int] | None = None
        self._drag_changed = False
        self._hover = QPointF()

    @property
    def grid(self) -> NDArray[np.float64] | None:
        """The grid as ``(columns, rows, 2)`` image points; None without a valid one."""
        return self._grid

    @property
    def is_dragging(self) -> bool:
        """True while a control point is being dragged."""
        return self._dragged is not None

    def set_distortion_model(
        self,
        model: DistortionModelSpec,
        depth_perception: float,
        default_area: QRectF | None = None,
    ) -> None:
        """Show a model without reporting a change.

        Args:
            model (DistortionModelSpec): Edges in image pixels. When it is not
                defined the editor starts from straight edges along the top and
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
                    image = self._host.image
                    area = QRectF(image.rect()) if image else QRectF()
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

    def has_model(self) -> bool:
        """Return whether a model was shown, so that there are edges to edit."""
        return bool(self._controls[0])

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
        self._host.update()

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

    def paint(self, painter: QPainter) -> None:
        """Draw the grid and the control points.

        Args:
            painter (QPainter): Painter on the host's viewport.
        """
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        grid_pen = QPen(view_color("dewarp_grid"), 1.2)
        grid_pen.setCosmetic(True)
        painter.setPen(grid_pen)
        painter.setTransform(self._host.image_to_widget(), True)
        if self._grid is not None:
            for line in (*self._grid, *self._grid.transpose(1, 0, 2)):
                painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in line]))
        else:
            self._paint_frame(painter)
        painter.restore()

        painter.save()
        painter.setPen(QPen(view_color("dewarp_handle"), 4))
        to_widget = self._host.image_to_widget()
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
        to_widget = self._host.image_to_widget()
        for edge, points in enumerate(self._controls):
            for index, point in enumerate(points):
                offset = to_widget.map(point) - widget_pos
                if math.hypot(offset.x(), offset.y()) <= self.HIT_RADIUS:
                    return edge, index
        return None

    def status_tip_at(self, widget_pos: QPointF) -> str:
        """Return the hint for what lies under a widget position.

        Args:
            widget_pos (QPointF): Position in the host's viewport.

        Returns:
            str: The hint; empty when there is nothing to edit there.
        """
        if not self._controls[0]:
            return ""
        handle = self._handle_at(widget_pos)
        if handle is not None:
            edge, index = handle
            is_end = index in (0, len(self._controls[edge]) - 1)
            tip = _END_POINT_TIP if is_end else _POINT_TIP
            return QCoreApplication.translate("InteractiveXSpline", tip)
        to_widget = self._host.image_to_widget()
        for polyline in self._edges:
            for start, end in pairwise(polyline):
                _nearest, distance = nearest_on_segment(
                    widget_pos,
                    to_widget.map(QPointF(*start)),
                    to_widget.map(QPointF(*end)),
                )
                if distance <= self.HIT_RADIUS:
                    return QCoreApplication.translate("InteractiveXSpline", _CURVE_TIP)
        return ""

    def _insert_at(self, widget_pos: QPointF) -> tuple[int, int] | None:
        """Add a control point on the edge near a widget position, if any."""
        to_widget = self._host.image_to_widget()
        for edge, polyline in enumerate(self._edges):
            for start, end in pairwise(polyline):
                nearest, distance = nearest_on_segment(
                    widget_pos,
                    to_widget.map(QPointF(*start)),
                    to_widget.map(QPointF(*end)),
                )
                if distance <= self.HIT_RADIUS:
                    point = self._host.widget_to_image().map(nearest)
                    points = self._controls[edge]
                    index = sum(1 for p in points if p.x() < point.x())
                    index = min(max(index, 1), len(points) - 1)
                    points.insert(index, point)
                    self._rebuild()
                    return edge, index
        return None

    def mouse_press(self, event: QtGui.QMouseEvent) -> bool:
        """Pick a control point, or add one when pressing on an edge.

        Args:
            event (QtGui.QMouseEvent): The press.

        Returns:
            bool: True when the editor took the event.
        """
        if event.button() != Qt.MouseButton.LeftButton or not self._controls[0]:
            return False
        pos = event.position()
        handle = self._handle_at(pos)
        self._drag_changed = False
        if handle is None:
            handle = self._insert_at(pos)
            self._drag_changed = handle is not None
        if handle is None:
            return False
        self._dragged = handle
        event.accept()
        return True

    def mouse_move(self, event: QtGui.QMouseEvent) -> bool:
        """Move the picked control point.

        Args:
            event (QtGui.QMouseEvent): The move.

        Returns:
            bool: True when a control point is being dragged.
        """
        self._hover = event.position()
        if self._dragged is None:
            return False
        edge, index = self._dragged
        point = self._host.widget_to_image().map(event.position())
        points = self._controls[edge]
        if index in (0, len(points) - 1):
            shifted = self._shifted_edge(
                points, points[index], point, event.modifiers()
            )
            if shifted is not None:
                self._controls[edge] = shifted
                self._drag_changed = True
                self._rebuild()
                event.accept()
                return True
        if points[index] != point:
            points[index] = point
            self._drag_changed = True
            self._rebuild()
        event.accept()
        return True

    @staticmethod
    def _shifted_edge(
        points: list[QPointF],
        grabbed: QPointF,
        target: QPointF,
        modifiers: Qt.KeyboardModifier,
    ) -> list[QPointF] | None:
        """Move a whole edge along one axis while an end is dragged with a key held.

        Shift moves it along x and Ctrl along y.

        Returns:
            list[QPointF] | None: The moved points; None when neither key is held.
        """
        along_x = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        along_y = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if not along_x and not along_y:
            return None
        shift = target - grabbed
        offset = QPointF(shift.x(), 0.0) if along_x else QPointF(0.0, shift.y())
        return [point + offset for point in points]

    def mouse_release(self, event: QtGui.QMouseEvent) -> bool:
        """Drop the control point and report the new model.

        Args:
            event (QtGui.QMouseEvent): The release.

        Returns:
            bool: True when a control point was being dragged.
        """
        if event.button() != Qt.MouseButton.LeftButton or self._dragged is None:
            return False
        self._dragged = None
        event.accept()
        if self._drag_changed:
            self._report_change()
        return True

    def key_press(self, event: QtGui.QKeyEvent) -> bool:
        """Remove the control point under the cursor with D.

        Args:
            event (QtGui.QKeyEvent): The key press.

        Returns:
            bool: True when a point was removed.
        """
        if event.key() == Qt.Key.Key_D and self._remove_control_point_under_cursor():
            event.accept()
            return True
        return False

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
        self._on_change(self._shown_model)
