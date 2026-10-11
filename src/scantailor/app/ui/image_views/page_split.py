"""Page Split filter image view.

Shows the oriented source image with the cutting lines of the page layout:
one line splits a spread into two pages, two lines cut the offcut away from
a single page. Lines are vertical, as PageLayout can only describe them so;
dragging a handle or the line moves it sideways.
"""

import math
from itertools import pairwise

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF

from scantailor.app.ui.image_views.base import ImageViewBase, point_to_segment_distance
from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType

# Smallest distance, in image pixels, kept between cutting lines and the outline.
_MIN_GAP_PX = 1.0


class PageSplitImageView(ImageViewBase):
    """Image view of the Page Split stage with draggable cutting lines.

    Layout coordinates are image coordinates of the displayed image.

    Signals:
        page_layout_set_locally: The user moved a line. Args: layout (PageLayout).
    """

    page_layout_set_locally = Signal(object)

    HANDLE_RADIUS = 8
    HANDLE_HIT_RADIUS = 15
    LINE_HIT_DISTANCE = 10

    LINE_COLOR = QColor(0, 0, 255, 200)
    LINE_HIGHLIGHT = QColor(0, 128, 255, 255)
    HANDLE_COLOR = QColor(0, 0, 255, 180)
    PAGE_COLORS = (QColor(0, 200, 0, 50), QColor(0, 0, 200, 50))
    OFFCUT_COLOR = QColor(200, 0, 0, 80)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._layout: PageLayout | None = None
        self._line_xs: list[float] = []
        self._dragged_line = -1
        self._drag_offset = 0.0
        self._drag_moved = False
        self._hovered_line = -1
        self.setMouseTracking(True)

    @property
    def page_layout(self) -> PageLayout | None:
        """The layout being shown, including any unreleased drag."""
        return self._current_layout()

    def set_page_layout(self, layout: PageLayout | None) -> None:
        """Show a layout.

        Args:
            layout (PageLayout | None): Layout in image coordinates; None shows
                no lines.
        """
        self._layout = layout
        self._line_xs = self._cutter_positions(layout)
        self._dragged_line = -1
        self.update()

    @staticmethod
    def _cutter_positions(layout: PageLayout | None) -> list[float]:
        if layout is None or layout.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT:
            return []
        cutters = [layout.cutter1]
        if layout.layout_type == PageLayoutType.SINGLE_PAGE_CUT:
            cutters.append(layout.cutter2)
        return sorted(
            (line[0][0] + line[1][0]) / 2 for line in cutters if line is not None
        )

    def _current_layout(self) -> PageLayout | None:
        layout = self._layout
        if layout is None or not self._line_xs:
            return layout
        if layout.layout_type == PageLayoutType.TWO_PAGES:
            return layout.with_split_line(self._line_xs[0])
        if len(self._line_xs) == 2:
            return layout.with_cutters(self._line_xs[0], self._line_xs[1])
        return layout

    def line_handle_positions(self) -> list[tuple[QPointF, QPointF]]:
        """Return the end handles of every line.

        Returns:
            list[tuple[QPointF, QPointF]]: Top and bottom handle of each line in
            widget coordinates, left line first.
        """
        if self._layout is None:
            return []
        _x, top, _w, height = self._layout.outline
        to_widget = self.image_to_widget()
        return [
            (
                to_widget.map(QPointF(line_x, top)),
                to_widget.map(QPointF(line_x, top + height)),
            )
            for line_x in self._line_xs
        ]

    def _line_at(self, pos: QPointF) -> int:
        """Return the index of the line or handle under a widget position, or -1."""
        lines = self.line_handle_positions()
        for index, (top, bottom) in enumerate(lines):
            for handle in (top, bottom):
                if math.hypot(pos.x() - handle.x(), pos.y() - handle.y()) <= (
                    self.HANDLE_HIT_RADIUS
                ):
                    return index
        for index, (top, bottom) in enumerate(lines):
            if point_to_segment_distance(pos, top, bottom) <= self.LINE_HIT_DISTANCE:
                return index
        return -1

    def _image_x(self, widget_pos: QPointF) -> float:
        return self.widget_to_image().map(widget_pos).x()

    def _clamp_line(self, index: int, line_x: float) -> float:
        assert self._layout is not None
        outline_x, _y, width, _h = self._layout.outline
        low = outline_x + _MIN_GAP_PX
        high = outline_x + width - _MIN_GAP_PX
        if index > 0:
            low = max(low, self._line_xs[index - 1] + _MIN_GAP_PX)
        if index + 1 < len(self._line_xs):
            high = min(high, self._line_xs[index + 1] - _MIN_GAP_PX)
        return max(low, min(high, line_x))

    def _paint_overlay(self, painter: QPainter) -> None:
        if self._layout is None:
            return
        outline_x, top, width, height = self._layout.outline
        to_widget = self.image_to_widget()

        def region(left: float, right: float) -> QPolygonF:
            return to_widget.map(QPolygonF(QRectF(left, top, right - left, height)))

        painter.save()
        right_edge = outline_x + width
        bounds = [outline_x, *self._line_xs, right_edge]
        cut = self._layout.layout_type == PageLayoutType.SINGLE_PAGE_CUT
        for index, (left, right) in enumerate(pairwise(bounds)):
            if cut:
                color = self.PAGE_COLORS[0] if index == 1 else self.OFFCUT_COLOR
            elif self._line_xs:
                color = self.PAGE_COLORS[index]
            else:
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(region(left, right))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, (handle_top, handle_bottom) in enumerate(
            self.line_handle_positions()
        ):
            active = index in (self._dragged_line, self._hovered_line)
            painter.setPen(
                QPen(
                    self.LINE_HIGHLIGHT if active else self.LINE_COLOR,
                    3 if active else 2,
                )
            )
            painter.drawLine(handle_top, handle_bottom)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.setBrush(self.LINE_HIGHLIGHT if active else self.HANDLE_COLOR)
            for handle in (handle_top, handle_bottom):
                painter.drawEllipse(handle, self.HANDLE_RADIUS, self.HANDLE_RADIUS)
        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pick up a line or handle.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._line_at(event.position())
            if index >= 0:
                self._dragged_line = index
                self._drag_offset = (
                    self._image_x(event.position()) - self._line_xs[index]
                )
                self._drag_moved = False
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Move the picked line or update the hover state.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._dragged_line >= 0:
            new_x = self._clamp_line(
                self._dragged_line, self._image_x(event.position()) - self._drag_offset
            )
            if new_x != self._line_xs[self._dragged_line]:
                self._line_xs[self._dragged_line] = new_x
                self._drag_moved = True
                self.update()
            event.accept()
            return

        if not self._panning:
            hovered = self._line_at(event.position())
            if hovered != self._hovered_line:
                self._hovered_line = hovered
                self.setCursor(
                    Qt.CursorShape.SizeHorCursor
                    if hovered >= 0
                    else Qt.CursorShape.ArrowCursor
                )
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop the line and report the new layout if it moved.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._dragged_line >= 0:
            self._dragged_line = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            self.update()
            layout = self._current_layout()
            if self._drag_moved and layout is not None:
                self._layout = layout
                self.page_layout_set_locally.emit(layout)
            return
        super().mouseReleaseEvent(event)
