"""Page Layout filter image view.

Shows the deskewed page with three nested rectangles: the content box, the
content plus hard margins, and that plus the soft margins that bring every
aligned page to one size. Dragging a side of the middle rectangle sets the hard
margins. Dragging a side of the content box does the same from the other end:
the margin changes and the page moves so that the side stays under the pointer.
Holding Ctrl and/or Shift moves the content within the page. Alignment guides
can be added from the context menu; a double click on content snaps it to the
nearest guide. The context menu also turns on a millimetre grid over the page.
"""

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import cast

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import (
    QT_TRANSLATE_NOOP,
    QCoreApplication,
    QLineF,
    QPointF,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QAction, QPainter, QPen, QTransform

from scantailor.app.ui.icons import icon
from scantailor.app.ui.image_views.base import (
    Edge,
    ImageViewBase,
    matrix_to_qtransform,
    point_to_segment_distance,
)
from scantailor.app.ui.image_views.page_grid import paint_page_grid
from scantailor.app.ui.image_views.page_layout_state import Guide, PageLayoutState
from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.content_mask import ContentMask
from scantailor.core.models import Margins
from scantailor.core.settings import ApplicationSettings

_MARGINS_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "page_layout::ImageView", "Resize margins by dragging any of the solid lines."
    ),
)
_MOVE_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "page_layout::ImageView", "Hold left mouse button to drag the page content."
    ),
)
_FINISH_DRAG_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "page_layout::ImageView", "Release left mouse button to finish dragging."
    ),
)
_GUIDE_TIP = cast(str, QT_TRANSLATE_NOOP("page_layout::ImageView", "Drag the guide."))

_VERTICAL_MOVE = Qt.KeyboardModifier.ControlModifier
_HORIZONTAL_MOVE = Qt.KeyboardModifier.ShiftModifier
_MOVE_MODIFIERS = (
    _VERTICAL_MOVE,
    _HORIZONTAL_MOVE,
    _VERTICAL_MOVE | _HORIZONTAL_MOVE,
)
_GUIDE_MODIFIERS = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
# Half the side, in mask pixels, of the square searched for content around a click.
_SNAP_SEARCH_RADIUS = 15


class _Target(Enum):
    """The rectangle a drag works on."""

    INNER = auto()
    MIDDLE = auto()


@dataclass
class _Grab:
    """A rectangle side, corner or body picked up by the pointer.

    Attributes:
        target: The rectangle.
        edges: The sides it moves; NONE moves the content within the page.
        start_pos: Pointer position at the press, widget coordinates.
        middle_widget_rect: The hard margins rectangle at the press, widget
            coordinates.
        virtual_to_widget: Mapping at the press; the page moves while the
            content box is dragged.
        widget_to_virtual: Inverse of ``virtual_to_widget``.
        focal_point: Widget point the view is centred on at the press.
        was_fitting: Whether the view fitted the page at the press.
        offset: Where the pointer grabbed a middle rectangle side relative to
            it, in virtual coordinates.
        moved: Whether the rectangle changed since the press.
    """

    target: _Target
    edges: Edge
    start_pos: QPointF
    middle_widget_rect: QRectF
    virtual_to_widget: QTransform
    widget_to_virtual: QTransform
    focal_point: QPointF
    was_fitting: bool
    offset: QPointF = field(default_factory=QPointF)
    moved: bool = False


class PageLayoutImageView(ImageViewBase):
    """Image view of the Page Layout stage with draggable margins.

    Signals:
        margins_set_locally: The user changed the hard margins. Args: margins
            (Margins) in millimetres.
    """

    margins_set_locally = Signal(Margins)

    default_status_tip = _MARGINS_TIP
    default_status_tip_context = "page_layout::ImageView"
    EDGE_HIT_DISTANCE = 10
    GUIDE_HIT_DISTANCE = 6

    _SIDES = (Edge.LEFT, Edge.RIGHT, Edge.TOP, Edge.BOTTOM)
    _CORNERS = (
        Edge.LEFT | Edge.TOP,
        Edge.RIGHT | Edge.TOP,
        Edge.LEFT | Edge.BOTTOM,
        Edge.RIGHT | Edge.BOTTOM,
    )

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        state: PageLayoutState | None = None,
    ) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
            state (PageLayoutState | None): Link and guide state to share with
                the options widget; a private one when None.
        """
        super().__init__(parent=parent)
        self._inner_rect = QRectF()
        self._middle_rect = QRectF()
        self._outer_rect = QRectF()
        self._mm_per_pixel = (1.0, 1.0)
        self._aligned = True
        self._null_content = True

        self._grab: _Grab | None = None
        self._hovered_guide: Guide | None = None
        self._dragged_guide: Guide | None = None
        self._context_pos = QPointF()
        self._mask_provider: Callable[[], ContentMask | None] | None = None
        self._grid_shown = ApplicationSettings.get_instance().margins_grid_shown

        self._state = state if state is not None else PageLayoutState(self)
        self._seen_links = (False, False)
        self._connect_state(self._state)
        self._build_context_menu()
        self.setMouseTracking(True)

    @property
    def state(self) -> PageLayoutState:
        """The link and guide state in use."""
        return self._state

    def set_state(self, state: PageLayoutState) -> None:
        """Use link and guide state that is shared with an options widget.

        Args:
            state (PageLayoutState): The state to follow from now on.
        """
        if state is self._state:
            return
        self._disconnect_state(self._state)
        self._state = state
        self._connect_state(state)
        self._sync_context_menu()
        self.update()

    def _connect_state(self, state: PageLayoutState) -> None:
        self._seen_links = (state.left_right_linked, state.top_bottom_linked)
        state.links_changed.connect(self._on_links_changed)
        state.guides_changed.connect(self.update)
        state.middle_rect_shown_changed.connect(self._on_middle_rect_shown_changed)

    def _disconnect_state(self, state: PageLayoutState) -> None:
        state.links_changed.disconnect(self._on_links_changed)
        state.guides_changed.disconnect(self.update)
        state.middle_rect_shown_changed.disconnect(self._on_middle_rect_shown_changed)

    def _build_context_menu(self) -> None:
        self._context_menu = QtWidgets.QMenu(self)
        self._add_horizontal_guide_action = QAction(
            QCoreApplication.translate(
                "page_layout::ImageView", "Add a horizontal guide"
            ),
            self,
        )
        self._add_horizontal_guide_action.triggered.connect(
            lambda: self._add_guide(True)
        )
        self._add_vertical_guide_action = QAction(
            QCoreApplication.translate(
                "page_layout::ImageView", "Add a vertical guide"
            ),
            self,
        )
        self._add_vertical_guide_action.triggered.connect(
            lambda: self._add_guide(False)
        )
        self._remove_all_guides_action = QAction(
            QCoreApplication.translate(
                "page_layout::ImageView", "Remove all the guides"
            ),
            self,
        )
        self._remove_all_guides_action.triggered.connect(self._clear_guides)
        self._remove_guide_action = QAction(
            QCoreApplication.translate("page_layout::ImageView", "Remove this guide"),
            self,
        )
        self._remove_guide_action.triggered.connect(self._remove_hovered_guide)
        self._show_middle_rect_action = QAction(
            QCoreApplication.translate(
                "page_layout::ImageView", "Show hard margins rectangle"
            ),
            self,
        )
        self._show_middle_rect_action.setCheckable(True)
        self._show_middle_rect_action.toggled.connect(self._toggle_middle_rect)
        self._show_grid_action = QAction(
            icon("layout-grid"),
            QCoreApplication.translate("page_layout::ImageView", "Show grid"),
            self,
        )
        self._show_grid_action.setToolTip(
            QCoreApplication.translate(
                "page_layout::ImageView",
                "Draw a line every 10 mm across the page, through its centre, "
                "to compare positions between pages.",
            )
        )
        self._show_grid_action.setCheckable(True)
        self._show_grid_action.toggled.connect(self.set_grid_shown)

        menu = self._context_menu
        menu.setToolTipsVisible(True)
        menu.addAction(self._add_horizontal_guide_action)
        menu.addAction(self._add_vertical_guide_action)
        menu.addAction(self._remove_all_guides_action)
        menu.addAction(self._remove_guide_action)
        self._guide_actions_separator = menu.addSeparator()
        menu.addAction(self._show_middle_rect_action)
        menu.addAction(self._show_grid_action)
        self._sync_context_menu()

    def _sync_context_menu(self) -> None:
        for action, checked in (
            (self._show_middle_rect_action, self._state.middle_rect_shown),
            (self._show_grid_action, self._grid_shown),
        ):
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def grid_shown(self) -> bool:
        """Return whether the millimetre grid is drawn over the page.

        Returns:
            bool: True while the grid shows.
        """
        return self._grid_shown

    def set_grid_shown(self, shown: bool) -> None:
        """Show or hide the millimetre grid and remember the choice.

        Args:
            shown (bool): True to draw the grid.
        """
        self._grid_shown = shown
        self._sync_context_menu()
        settings = ApplicationSettings.get_instance()
        settings.margins_grid_shown = shown
        # save() has logged the failure; the view still follows the choice.
        with contextlib.suppress(OSError):
            settings.save()
        self.update()

    def set_rectangles(
        self, inner_rect: QRectF, middle_rect: QRectF, outer_rect: QRectF
    ) -> None:
        """Show the three rectangles.

        Args:
            inner_rect (QRectF): Content box in virtual coordinates.
            middle_rect (QRectF): Content plus hard margins.
            outer_rect (QRectF): Content plus hard and soft margins.
        """
        self._inner_rect = QRectF(inner_rect)
        self._middle_rect = QRectF(middle_rect)
        self._outer_rect = QRectF(outer_rect)
        self._null_content = inner_rect.width() < 1 and inner_rect.height() < 1
        self.set_display_extent(self._extent_rect())
        self._keep_guides_inside()
        self.update()

    def _extent_rect(self) -> QRectF:
        if self._aligned:
            return self._outer_rect.united(self._middle_rect)
        return QRectF(self._middle_rect)

    def set_aligned(self, aligned: bool) -> None:
        """Say whether the page is matched in size with the other pages.

        Only an aligned page has soft margins and guides.

        Args:
            aligned (bool): True if the page is aligned with others.
        """
        if aligned == self._aligned:
            return
        self._aligned = aligned
        self.set_display_extent(self._extent_rect())
        self.update()

    def set_mm_per_pixel(self, horizontal: float, vertical: float) -> None:
        """Set how many millimetres one virtual pixel measures on the page.

        Args:
            horizontal (float): Millimetres per pixel along x.
            vertical (float): Millimetres per pixel along y.
        """
        self._mm_per_pixel = (horizontal, vertical)

    def set_content_mask_provider(
        self, provider: Callable[[], ContentMask | None] | None
    ) -> None:
        """Say where to find out what is content when the user double clicks.

        Args:
            provider (Callable[[], ContentMask | None] | None): Called on the
                first double click that needs the mask; may return None while
                there is none. None turns the double click off.
        """
        self._mask_provider = provider

    def set_margin_links(self, left_right: bool, top_bottom: bool) -> None:
        """Choose which opposite margins move together.

        Args:
            left_right (bool): Dragging the left or right side moves both.
            top_bottom (bool): Dragging the top or bottom side moves both.
        """
        self._state.left_right_linked = left_right
        self._state.top_bottom_linked = top_bottom

    def margins_mm(self) -> Margins:
        """Return the hard margins the middle rectangle currently shows.

        Returns:
            Margins: Distance of the middle rectangle from the content box, in
            millimetres, never negative.
        """
        x_scale, y_scale = self._mm_per_pixel
        inner, middle = self._inner_rect, self._middle_rect
        return Margins(
            left=max(0.0, (inner.left() - middle.left()) * x_scale),
            right=max(0.0, (middle.right() - inner.right()) * x_scale),
            top=max(0.0, (inner.top() - middle.top()) * y_scale),
            bottom=max(0.0, (middle.bottom() - inner.bottom()) * y_scale),
        )

    def _middle_rect_for(self, margins: Margins) -> QRectF:
        x_scale, y_scale = self._mm_per_pixel
        inner = self._inner_rect
        return QRectF(
            inner.left() - margins.left / x_scale,
            inner.top() - margins.top / y_scale,
            inner.width() + (margins.left + margins.right) / x_scale,
            inner.height() + (margins.top + margins.bottom) / y_scale,
        )

    def _on_links_changed(self) -> None:
        """Make a margin pair equal at once when it was just linked."""
        state = self._state
        linked = (state.left_right_linked, state.top_bottom_linked)
        newly_linked = tuple(
            now and not before for now, before in zip(linked, self._seen_links)
        )
        self._seen_links = linked
        if not any(newly_linked) or self._inner_rect.isEmpty():
            return
        margins = self.margins_mm()
        updates: dict[str, float] = {}
        if newly_linked[0] and margins.left != margins.right:
            updates["left"] = updates["right"] = min(margins.left, margins.right)
        if newly_linked[1] and margins.top != margins.bottom:
            updates["top"] = updates["bottom"] = min(margins.top, margins.bottom)
        if not updates:
            return
        margins = margins.model_copy(update=updates)
        self._middle_rect = self._middle_rect_for(margins)
        self.update()
        self.margins_set_locally.emit(margins)

    def _on_middle_rect_shown_changed(self) -> None:
        self._sync_context_menu()
        self.update()

    def _middle_rect_active(self) -> bool:
        """Whether the hard margins rectangle is drawn and can be dragged."""
        shown = not self._null_content and self._state.middle_rect_shown
        return (shown or not self._aligned) and not self._middle_rect.isEmpty()

    def _inner_rect_active(self) -> bool:
        return not self._null_content and not self._inner_rect.isEmpty()

    def _corner_at(self, rect: QRectF, pos: QPointF) -> Edge:
        for corner in self._CORNERS:
            point = self.virtual_to_widget_point(
                QPointF(
                    rect.left() if corner & Edge.LEFT else rect.right(),
                    rect.top() if corner & Edge.TOP else rect.bottom(),
                )
            )
            offset = pos - point
            if offset.x() ** 2 + offset.y() ** 2 <= self.EDGE_HIT_DISTANCE**2:
                return corner
        return Edge.NONE

    def _side_at(self, rect: QRectF, pos: QPointF) -> Edge:
        sides = {
            Edge.LEFT: (rect.topLeft(), rect.bottomLeft()),
            Edge.RIGHT: (rect.topRight(), rect.bottomRight()),
            Edge.TOP: (rect.topLeft(), rect.topRight()),
            Edge.BOTTOM: (rect.bottomLeft(), rect.bottomRight()),
        }
        best, best_distance = Edge.NONE, self.EDGE_HIT_DISTANCE
        for edge in self._SIDES:
            start, end = sides[edge]
            distance = point_to_segment_distance(
                pos,
                self.virtual_to_widget_point(start),
                self.virtual_to_widget_point(end),
            )
            if distance <= best_distance:
                best, best_distance = edge, distance
        return best

    def _hit_test(
        self, pos: QPointF, modifiers: Qt.KeyboardModifier
    ) -> tuple[_Target, Edge] | None:
        """Find what a drag started at a widget position would pick up.

        Corners come before sides and the content box before the hard margins
        rectangle; the body of the content box only moves with Ctrl or Shift.

        Returns:
            The rectangle and the sides a drag moves, NONE for the content box
            body; None when nothing is hit.
        """
        rects: list[tuple[_Target, QRectF]] = []
        if self._inner_rect_active():
            rects.append((_Target.INNER, self._inner_rect))
        if self._middle_rect_active():
            rects.append((_Target.MIDDLE, self._middle_rect))
        for finder in (self._corner_at, self._side_at):
            for target, rect in rects:
                edge = finder(rect, pos)
                if edge != Edge.NONE:
                    return target, edge
        if (
            self._inner_rect_active()
            and modifiers in _MOVE_MODIFIERS
            and self.virtual_to_widget_rect(self._inner_rect).contains(pos)
        ):
            return _Target.INNER, Edge.NONE
        return None

    def _grab_offset(self, edge: Edge, grab: QPointF) -> QPointF:
        """Return where a grab point lies relative to the sides it will move."""
        rect = self._middle_rect
        offset_x = offset_y = 0.0
        if edge & Edge.LEFT:
            offset_x = grab.x() - rect.left()
        elif edge & Edge.RIGHT:
            offset_x = grab.x() - rect.right()
        if edge & Edge.TOP:
            offset_y = grab.y() - rect.top()
        elif edge & Edge.BOTTOM:
            offset_y = grab.y() - rect.bottom()
        return QPointF(offset_x, offset_y)

    def _begin_grab(self, target: _Target, edges: Edge, pos: QPointF) -> _Grab:
        grab = _Grab(
            target,
            edges,
            QPointF(pos),
            self.virtual_to_widget_rect(self._middle_rect),
            self._virtual_to_widget,
            self._widget_to_virtual,
            QPointF(self._widget_focal_point),
            self._fit_mode,
        )
        if target == _Target.MIDDLE:
            grab.offset = self._grab_offset(edges, self.widget_to_virtual_point(pos))
        else:
            # The page moves with the pointer, which a refit would undo.
            self._fit_mode = False
        return grab

    def _force_non_negative_margins(self, rect: QRectF) -> QRectF:
        """Return the rectangle grown to contain the content box."""
        return rect.united(self._inner_rect)

    def _drag_middle_to(self, grab: _Grab, pos: QPointF) -> None:
        """Move the sides of the middle rectangle, never inside the content box."""
        edge = grab.edges
        target = self.widget_to_virtual_point(pos) - grab.offset
        inner = self._inner_rect
        rect = QRectF(self._middle_rect)
        links = self._state
        if edge & Edge.LEFT:
            left = min(target.x(), inner.left())
            delta = rect.left() - left
            rect.setLeft(left)
            if links.left_right_linked:
                rect.setRight(rect.right() + delta)
        elif edge & Edge.RIGHT:
            right = max(target.x(), inner.right())
            delta = right - rect.right()
            rect.setRight(right)
            if links.left_right_linked:
                rect.setLeft(rect.left() - delta)
        if edge & Edge.TOP:
            top = min(target.y(), inner.top())
            delta = rect.top() - top
            rect.setTop(top)
            if links.top_bottom_linked:
                rect.setBottom(rect.bottom() + delta)
        elif edge & Edge.BOTTOM:
            bottom = max(target.y(), inner.bottom())
            delta = bottom - rect.bottom()
            rect.setBottom(bottom)
            if links.top_bottom_linked:
                rect.setTop(rect.top() - delta)
        rect = self._force_non_negative_margins(rect)
        if rect != self._middle_rect:
            self._middle_rect = rect
            grab.moved = True

    def _drag_inner_edge(self, grab: _Grab, pos: QPointF) -> None:
        """Resize the margins from a content box side, keeping it under the pointer.

        The content box does not change; the hard margins rectangle grows or
        shrinks the other way and the page is shifted by what the rectangle
        could not follow, so the side appears to follow the pointer.
        """
        edges = grab.edges
        delta = pos - grab.start_pos
        links = self._state
        left = right = top = bottom = 0.0
        if edges & Edge.LEFT:
            left = delta.x()
            right = -left if links.left_right_linked else 0.0
        elif edges & Edge.RIGHT:
            right = delta.x()
            left = -right if links.left_right_linked else 0.0
        if edges & Edge.TOP:
            top = delta.y()
            bottom = -top if links.top_bottom_linked else 0.0
        elif edges & Edge.BOTTOM:
            bottom = delta.y()
            top = -bottom if links.top_bottom_linked else 0.0

        widget_rect = QRectF(grab.middle_widget_rect)
        widget_rect.adjust(-left, -top, -right, -bottom)
        middle = self._force_non_negative_margins(
            grab.widget_to_virtual.mapRect(widget_rect)
        )
        widget_rect = grab.virtual_to_widget.mapRect(middle)

        before = grab.middle_widget_rect
        shift_x = shift_y = 0.0
        if edges & Edge.LEFT:
            shift_x = before.left() - widget_rect.left()
        elif edges & Edge.RIGHT:
            shift_x = before.right() - widget_rect.right()
        if edges & Edge.TOP:
            shift_y = before.top() - widget_rect.top()
        elif edges & Edge.BOTTOM:
            shift_y = before.bottom() - widget_rect.bottom()
        self._set_middle_while_dragging(
            grab, middle, grab.focal_point + QPointF(shift_x, shift_y)
        )

    def _move_content(
        self, grab: _Grab, pos: QPointF, modifiers: Qt.KeyboardModifier
    ) -> None:
        """Move the content within the page by changing the margins oppositely."""
        delta = pos - grab.start_pos
        if modifiers == _VERTICAL_MOVE:
            delta.setX(0)
        elif modifiers == _HORIZONTAL_MOVE:
            delta.setY(0)
        widget_rect = grab.middle_widget_rect.translated(-delta)
        middle = self._force_non_negative_margins(
            grab.widget_to_virtual.mapRect(widget_rect)
        )
        self._set_middle_while_dragging(grab, middle, grab.focal_point + delta)

    def _set_middle_while_dragging(
        self, grab: _Grab, middle: QRectF, focal_point: QPointF
    ) -> None:
        if middle != self._middle_rect or focal_point != self._widget_focal_point:
            grab.moved = True
        self._middle_rect = middle
        self._widget_focal_point = focal_point
        self._virtual_display_area = self._page_area.united(self._extent_rect())
        self._update_widget_transform()

    def _finish_grab(self, grab: _Grab) -> None:
        if grab.target == _Target.INNER:
            self._fit_mode = grab.was_fitting
        if not grab.moved:
            return
        margins = self.margins_mm()
        self.margins_set_locally.emit(margins)
        if grab.target == _Target.INNER and grab.was_fitting:
            self._fit_to_widget()

    @staticmethod
    def _cursor_for(edge: Edge) -> Qt.CursorShape:
        if edge in (Edge.LEFT | Edge.TOP, Edge.RIGHT | Edge.BOTTOM):
            return Qt.CursorShape.SizeFDiagCursor
        if edge in (Edge.RIGHT | Edge.TOP, Edge.LEFT | Edge.BOTTOM):
            return Qt.CursorShape.SizeBDiagCursor
        if edge in (Edge.LEFT, Edge.RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if edge in (Edge.TOP, Edge.BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.DragMoveCursor

    @staticmethod
    def _guide_cursor(guide: Guide) -> Qt.CursorShape:
        return (
            Qt.CursorShape.SplitVCursor
            if guide.is_horizontal
            else Qt.CursorShape.SplitHCursor
        )

    def _axis_scale(self, horizontal: bool) -> float:
        """Return millimetres per virtual pixel across a guide."""
        x_scale, y_scale = self._mm_per_pixel
        return y_scale if horizontal else x_scale

    def _centre_along(self, horizontal: bool) -> float:
        centre = self._outer_rect.center()
        return centre.y() if horizontal else centre.x()

    def _guide_virtual_position(self, guide: Guide) -> float:
        horizontal = guide.is_horizontal
        return self._centre_along(horizontal) + guide.position_mm / self._axis_scale(
            horizontal
        )

    def _position_mm(self, horizontal: bool, virtual: float) -> float:
        """Convert a virtual coordinate across a guide to millimetres from centre."""
        return (virtual - self._centre_along(horizontal)) * self._axis_scale(horizontal)

    def _guide_limit_mm(self, guide: Guide) -> float:
        """Return how far from the centre a guide may go: half the outer rectangle."""
        horizontal = guide.is_horizontal
        size = self._outer_rect.height() if horizontal else self._outer_rect.width()
        return size / 2 * self._axis_scale(horizontal)

    def _keep_guides_inside(self) -> None:
        if self._outer_rect.isEmpty():
            return
        for guide in self._state.guides:
            limit = self._guide_limit_mm(guide)
            clamped = max(-limit, min(limit, guide.position_mm))
            self._state.move_guide(guide, clamped)

    def _add_guide(self, horizontal: bool) -> None:
        if not self._aligned:
            return
        virtual = self.widget_to_virtual_point(self._context_pos)
        self._state.add_guide(
            horizontal,
            self._position_mm(horizontal, virtual.y() if horizontal else virtual.x()),
        )

    def _remove_hovered_guide(self) -> None:
        if self._hovered_guide is not None:
            self._state.remove_guide(self._hovered_guide)
        self._hovered_guide = None

    def _clear_guides(self) -> None:
        self._state.clear_guides()
        self._hovered_guide = None

    def _toggle_middle_rect(self, checked: bool) -> None:
        if self._aligned and not self._null_content:
            self._state.middle_rect_shown = checked

    def _guide_widget_position(self, guide: Guide) -> float:
        position = self._guide_virtual_position(guide)
        point = QPointF(0, position) if guide.is_horizontal else QPointF(position, 0)
        mapped = self.virtual_to_widget_point(point)
        return mapped.y() if guide.is_horizontal else mapped.x()

    def _guide_at(self, widget_pos: QPointF) -> Guide | None:
        if not self._aligned:
            return None
        for guide in self._state.guides:
            coordinate = widget_pos.y() if guide.is_horizontal else widget_pos.x()
            if (
                abs(coordinate - self._guide_widget_position(guide))
                <= self.GUIDE_HIT_DISTANCE
            ):
                return guide
        return None

    def _status_tip_at(
        self, widget_pos: QPointF, modifiers: Qt.KeyboardModifier
    ) -> str:
        if self._guide_at(widget_pos) is not None:
            return QCoreApplication.translate("page_layout::ImageView", _GUIDE_TIP)
        hit = self._hit_test(widget_pos, modifiers)
        if hit is not None and hit == (_Target.INNER, Edge.NONE):
            return QCoreApplication.translate("page_layout::ImageView", _MOVE_TIP)
        return ""

    def _paint_overlay(self, painter: QPainter) -> None:
        if self._grid_shown:
            paint_page_grid(
                painter, self._extent_rect(), self._mm_per_pixel, self.virtual_to_widget
            )
        if self._aligned and not self._outer_rect.isEmpty():
            self._paint_rect(painter, self._outer_rect, "margin_outer")
        if self._middle_rect_active():
            self._paint_rect(painter, self._middle_rect, "margin_middle")
        if self._inner_rect_active():
            self._paint_rect(painter, self._inner_rect, "margin_inner")
        if self._aligned:
            self._paint_guides(painter)

    def _paint_rect(self, painter: QPainter, rect: QRectF, role: str) -> None:
        border, fill = view_color(f"{role}_border"), view_color(f"{role}_fill")
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(border)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(fill)
        painter.drawRect(self.virtual_to_widget_rect(rect))
        painter.restore()

    def _paint_guides(self, painter: QPainter) -> None:
        viewport = self.viewport()
        assert viewport is not None
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for guide in self._state.guides:
            hovered = guide is self._hovered_guide
            pen = QPen(view_color("guide_active" if hovered else "guide"))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            at = self._guide_widget_position(guide)
            painter.drawLine(
                QLineF(0, at, viewport.width(), at)
                if guide.is_horizontal
                else QLineF(at, 0, at, viewport.height())
            )
        painter.restore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pick up a guide, a rectangle side or the content.

        Guides are only picked up with Ctrl and Alt held, as plain drags near
        them belong to the margins.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            modifiers = event.modifiers()
            guide = self._guide_at(pos) if modifiers == _GUIDE_MODIFIERS else None
            if guide is not None:
                self._dragged_guide = guide
                self.setCursor(self._guide_cursor(guide))
                event.accept()
                return
            hit = self._hit_test(pos, modifiers)
            if hit is not None:
                target, edges = hit
                self._grab = self._begin_grab(target, edges, pos)
                self.setCursor(self._cursor_for(edges))
                if edges == Edge.NONE:
                    self.set_status_tip(
                        QCoreApplication.translate(
                            "page_layout::ImageView", _FINISH_DRAG_TIP
                        )
                    )
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Move the picked guide, side or content, or update the hover state.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        pos = event.position()
        if self._dragged_guide is not None:
            guide = self._dragged_guide
            virtual = self.widget_to_virtual_point(pos)
            limit = self._guide_limit_mm(guide)
            position = self._position_mm(
                guide.is_horizontal, virtual.y() if guide.is_horizontal else virtual.x()
            )
            self._state.move_guide(guide, max(-limit, min(limit, position)))
            event.accept()
            return
        grab = self._grab
        if grab is not None:
            if grab.target == _Target.MIDDLE:
                self._drag_middle_to(grab, pos)
                self.update()
            elif grab.edges == Edge.NONE:
                self._move_content(grab, pos, event.modifiers())
            else:
                self._drag_inner_edge(grab, pos)
            event.accept()
            return

        if not self._panning:
            self._hovered_guide = self._guide_at(pos)
            if self._hovered_guide is not None:
                self.setCursor(self._guide_cursor(self._hovered_guide))
            else:
                hit = self._hit_test(pos, event.modifiers())
                self.setCursor(
                    Qt.CursorShape.ArrowCursor
                    if hit is None
                    else self._cursor_for(hit[1])
                )
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop what was picked up and report changed margins.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragged_guide is not None:
                self._dragged_guide = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
            grab = self._grab
            if grab is not None:
                self._grab = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.set_status_tip(
                    self._status_tip_at(event.position(), event.modifiers())
                )
                event.accept()
                self._finish_grab(grab)
                return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Move the content so that a side of what is under the click meets a guide.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._aligned
            and self._state.guides
        ):
            self._attach_content_to_nearest_guide(event.position(), event.modifiers())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _content_under(self, widget_pos: QPointF) -> QRectF | None:
        """Return the content around a widget position inside the content box."""
        provider = self._mask_provider
        mask = None if provider is None else provider()
        if mask is None or mask.is_null():
            return None
        to_content = self.widget_to_image() * matrix_to_qtransform(
            mask.original_to_content
        )
        centre = to_content.map(widget_pos).toPoint()
        found = mask.find_content_in_area(
            centre.x() - _SNAP_SEARCH_RADIUS,
            centre.y() - _SNAP_SEARCH_RADIUS,
            2 * _SNAP_SEARCH_RADIUS,
            2 * _SNAP_SEARCH_RADIUS,
        )
        if found is None:
            return None
        to_virtual = (
            matrix_to_qtransform(mask.content_to_original) * self._image_to_virtual
        )
        area = to_virtual.mapRect(QRectF(*found)).intersected(self._inner_rect)
        return None if area.isEmpty() else area

    def _nearest_guide_delta(
        self, found: QRectF, modifiers: Qt.KeyboardModifier
    ) -> QPointF:
        """Return the shift that puts a side of ``found`` on the nearest guide."""
        only_horizontal = modifiers == _HORIZONTAL_MOVE
        only_vertical = modifiers == _VERTICAL_MOVE
        both = modifiers == (_VERTICAL_MOVE | _HORIZONTAL_MOVE)
        nearest_x = nearest_y = float(2**31 - 1)
        delta = QPointF()
        for guide in self._state.guides:
            position = self._guide_virtual_position(guide)
            if not guide.is_horizontal:
                if only_vertical:
                    continue
                diff = self._closer_of(
                    position - found.left(), position - found.right()
                )
                limit = nearest_x if both else min(nearest_x, nearest_y)
                if abs(diff) < limit:
                    nearest_x = abs(diff)
                    delta.setX(diff)
                    if not both:
                        delta.setY(0.0)
            else:
                if only_horizontal:
                    continue
                diff = self._closer_of(
                    position - found.top(), position - found.bottom()
                )
                limit = nearest_y if both else min(nearest_x, nearest_y)
                if abs(diff) < limit:
                    nearest_y = abs(diff)
                    delta.setY(diff)
                    if not both:
                        delta.setX(0.0)
        return delta

    @staticmethod
    def _closer_of(first: float, second: float) -> float:
        return first if abs(first) <= abs(second) else second

    def _limited_content_shift(self, delta: QPointF) -> QPointF:
        """Shorten a shift so the hard margins rectangle stays within the outer one."""
        inner, middle, outer = self._inner_rect, self._middle_rect, self._outer_rect
        moved = QRectF(middle).translated(-delta).united(inner)
        # Growing the rectangle moves its centre, which the guides are measured from.
        grow_x = math.copysign(max(0.0, moved.width() - middle.width()), delta.x())
        grow_y = math.copysign(max(0.0, moved.height() - middle.height()), delta.y())
        delta = QPointF(delta.x() + grow_x, delta.y() + grow_y)
        moved = moved.translated(-grow_x, -grow_y).united(inner)
        excess_x = math.copysign(max(0.0, moved.width() - outer.width()), delta.x())
        excess_y = math.copysign(max(0.0, moved.height() - outer.height()), delta.y())
        return QPointF(delta.x() - excess_x, delta.y() - excess_y)

    def _attach_content_to_nearest_guide(
        self, widget_pos: QPointF, modifiers: Qt.KeyboardModifier
    ) -> None:
        if self._grab is not None or not self._inner_rect_active():
            return
        found = self._content_under(widget_pos)
        if found is None:
            return
        delta = self._nearest_guide_delta(found, modifiers)
        if delta.isNull():
            return
        delta = self._limited_content_shift(delta)

        origin = self.virtual_to_widget_point(QPointF(0, 0))
        grab = self._begin_grab(_Target.INNER, Edge.NONE, origin)
        self._move_content(
            grab,
            self.virtual_to_widget_point(delta),
            Qt.KeyboardModifier.NoModifier,
        )
        self._finish_grab(grab)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show the guide and grid menu.

        A page that is not aligned has no guides, so its menu only has the grid.

        Args:
            event (QtGui.QContextMenuEvent): The context menu event.
        """
        if self._grab is not None:
            return
        self._context_pos = QPointF(event.pos()) + QPointF(0.5, 0.5)
        if not self._extent_rect().contains(
            self.widget_to_virtual_point(self._context_pos)
        ):
            return
        self._hovered_guide = (
            self._guide_at(self._context_pos) if self._aligned else None
        )
        on_guide = self._hovered_guide is not None
        guide_items = self._aligned and not on_guide
        self._add_horizontal_guide_action.setVisible(guide_items)
        self._add_vertical_guide_action.setVisible(guide_items)
        self._remove_all_guides_action.setVisible(
            guide_items and bool(self._state.guides)
        )
        self._remove_guide_action.setVisible(on_guide)
        self._guide_actions_separator.setVisible(guide_items)
        self._show_middle_rect_action.setVisible(guide_items and not self._null_content)
        self._show_grid_action.setVisible(not on_guide)
        self._context_menu.popup(event.globalPos())
