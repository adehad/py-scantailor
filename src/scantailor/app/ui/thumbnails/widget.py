"""Standalone page thumbnail list with sort order and selection handling."""

import contextlib
import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from PySide6 import QtCore
from PySide6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QLineF,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QListView,
    QMenu,
    QMessageBox,
    QRubberBand,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui.icons import icon
from scantailor.app.ui.theme.fonts import small
from scantailor.app.ui.theme.tokens import view_color
from scantailor.app.ui.thumbnails.cache import ThumbnailCache
from scantailor.app.ui.thumbnails.deviation import deviation_provider
from scantailor.app.ui.thumbnails.model import (
    DeviantPagesProvider,
    DeviantRole,
    IncompletenessProvider,
    IncompleteRole,
    IsCurrentRole,
    PageOrderOption,
    PageRole,
    ThumbnailModel,
    ThumbnailRenderer,
    natural_order,
    order_by_height,
    order_by_width,
)
from scantailor.app.ui.thumbnails.stage_orders import (
    completeness_provider,
    stage_orders_provider,
)
from scantailor.core.models import PageId
from scantailor.core.project import Project
from scantailor.core.settings import ApplicationSettings
from scantailor.core.stage_sequence import StageSequence

_PADDING = 6
_CURRENT_BORDER = 2
_FRAME_MARGIN = 3
_FRAME_RADIUS = 4
_CAPTION_LEADING = 4
# Fraction of the picture the question mark of an incomplete page fills.
_QUESTION_MARK_FILL = 0.9
_QUESTION_MARK_ALPHA = 160
_QUESTION_MARK_PEN = 2
# The deviation asterisk spans this fraction of the picture's shorter side.
_ASTERISK_SPAN = 0.25
_ASTERISK_SPOKES = 3


def _question_mark() -> QPainterPath:
    """Return the question mark outline, in its own units with y pointing down."""
    path = QPainterPath()
    path.moveTo(4.42188, -2.40625)
    for control1, control2, end in (
        ((4.42188, -3.20312), (4.51562, -3.32812), (5.23438, -3.84375)),
        ((6.34375, -4.625), (6.67188, -5.15625), (6.67188, -6.17188)),
        ((6.67188, -7.79688), (5.4375, -8.92188), (3.6875, -8.92188)),
        ((2.65625, -8.92188), (1.84375, -8.5625), (1.32812, -7.85938)),
        ((0.9375, -7.32812), (0.78125, -6.75), (0.765625, -5.76562)),
    ):
        path.cubicTo(*control1, *control2, *end)
    path.lineTo(2.40625, -5.76562)
    path.lineTo(2.40625, -5.79688)
    for control1, control2, end in (
        ((2.34375, -6.76562), (2.92188, -7.51562), (3.71875, -7.51562)),
        ((4.4375, -7.51562), (4.98438, -6.90625), (4.98438, -6.125)),
        ((4.98438, -5.59375), (4.82812, -5.35938), (4.125, -4.78125)),
        ((3.17188, -3.96875), (2.90625, -3.4375), (2.9375, -2.40625)),
    ):
        path.cubicTo(*control1, *control2, *end)
    path.lineTo(4.42188, -2.40625)
    path.moveTo(4.625, -1.75)
    path.lineTo(2.8125, -1.75)
    path.lineTo(2.8125, 0.0)
    path.lineTo(4.625, 0.0)
    path.lineTo(4.625, -1.75)
    return path


_QUESTION_MARK = _question_mark()


def _caption_height() -> int:
    return QFontMetrics(small(QApplication.font())).height() + _CAPTION_LEADING


class ThumbnailDelegate(QStyledItemDelegate):
    """Paints a thumbnail, its caption and the current-page highlight."""

    def __init__(self, thumbnail_size: QSize, parent: QWidget | None = None) -> None:
        """Initialise; arguments are described on the class."""
        super().__init__(parent)
        self._thumbnail_size = thumbnail_size

    def set_thumbnail_size(self, size: QSize) -> None:
        """Change the bounding size a thumbnail is drawn at.

        Args:
            size (QSize): The new bounding size; callers must lay the view out again.
        """
        self._thumbnail_size = QSize(size)

    @property
    def thumbnail_size(self) -> QSize:
        """Bounding size a thumbnail is drawn at."""
        return self._thumbnail_size

    def cell_size(self) -> QSize:
        """Return the size of one cell: the thumbnail, its caption and padding."""
        return QSize(
            self._thumbnail_size.width() + 2 * _PADDING,
            self._thumbnail_size.height() + _caption_height() + 2 * _PADDING,
        )

    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> QSize:
        """Return the fixed cell size."""
        return self.cell_size()

    def paint(
        self,
        painter: QPainter,
        item_option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Draw one cell; a placeholder frame stands in until the render lands."""
        # PySide6 stubs omit the public QStyleOption fields (rect, state, palette).
        option = cast(Any, item_option)
        palette = option.palette
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Thumbnails are kept at the render resolution and drawn at the cell size.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        picture_area = QRect(
            option.rect.left() + _PADDING,
            option.rect.top() + _PADDING,
            self._thumbnail_size.width(),
            self._thumbnail_size.height(),
        )
        frame_area = picture_area.adjusted(
            -_FRAME_MARGIN, -_FRAME_MARGIN, _FRAME_MARGIN, _FRAME_MARGIN
        )
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(palette.highlight())
            painter.drawRoundedRect(frame_area, _FRAME_RADIUS, _FRAME_RADIUS)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        image = index.data(Qt.ItemDataRole.DecorationRole)
        if image is None:
            painter.setPen(palette.mid().color())
            painter.drawRect(picture_area.adjusted(0, 0, -1, -1))
        else:
            size = image.size().scaled(
                picture_area.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
            target = QRect(QPoint(0, 0), size)
            target.moveCenter(picture_area.center())
            painter.drawImage(target, image)
        if index.data(IncompleteRole):
            self._paint_question_mark(painter, picture_area)
        elif index.data(DeviantRole):
            self._paint_asterisk(painter, picture_area)

        # The caption lies below the inset selection fill, on the plain background.
        painter.setPen(palette.text().color())
        painter.setFont(small(QApplication.font()))
        caption_area = QRect(
            option.rect.left() + _PADDING,
            picture_area.bottom() + 1 + _FRAME_MARGIN,
            self._thumbnail_size.width(),
            _caption_height(),
        )
        painter.drawText(
            caption_area,
            Qt.AlignmentFlag.AlignCenter,
            painter.fontMetrics().elidedText(
                str(index.data(Qt.ItemDataRole.DisplayRole)),
                Qt.TextElideMode.ElideMiddle,
                caption_area.width(),
            ),
        )

        if index.data(IsCurrentRole):
            accent = palette.color(QPalette.ColorRole.Accent)
            painter.setPen(QPen(accent, _CURRENT_BORDER))
            inset = _CURRENT_BORDER / 2
            painter.drawRoundedRect(
                QRectF(frame_area).adjusted(-inset, -inset, inset, inset),
                _FRAME_RADIUS,
                _FRAME_RADIUS,
            )
        painter.restore()

    @staticmethod
    def _paint_asterisk(painter: QPainter, area: QRect) -> None:
        """Draw the mark that says the page stands out from the other pages."""
        radius = min(area.width(), area.height()) * _ASTERISK_SPAN / 2
        center = QRectF(area).center()
        pen = QPen(view_color("deviant_mark"), max(2.0, radius / 3.5))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.save()
        painter.setPen(pen)
        for spoke in range(_ASTERISK_SPOKES):
            angle = math.pi / 2 + spoke * math.pi / _ASTERISK_SPOKES
            offset = QPointF(radius * math.cos(angle), radius * math.sin(angle))
            painter.drawLine(QLineF(center - offset, center + offset))
        painter.restore()

    @staticmethod
    def _paint_question_mark(painter: QPainter, area: QRect) -> None:
        """Draw the mark that says the page has no result at this stage yet."""
        bounds = _QUESTION_MARK.boundingRect()
        scale = min(
            area.width() * _QUESTION_MARK_FILL / bounds.width(),
            area.height() * _QUESTION_MARK_FILL / bounds.height(),
        )
        color = view_color("split_offcut")
        color.setAlpha(_QUESTION_MARK_ALPHA)
        painter.save()
        painter.translate(area.center())
        painter.scale(scale, scale)
        painter.translate(-bounds.center())
        pen = QPen(color, _QUESTION_MARK_PEN)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(_QUESTION_MARK)
        painter.restore()


class _RubberBandSelector(QObject):
    """Selects every thumbnail a Ctrl+drag over empty space touches.

    Dragging over a thumbnail is left to the view, so Ctrl+click keeps toggling
    single pages. The selection is extended, never replaced.

    Args:
        view (QListView): The view whose viewport is watched.
    """

    _MIN_EXTENT = 3

    def __init__(self, view: QListView) -> None:
        """Initialise; arguments are described on the class."""
        super().__init__(view)
        self._view = view
        self._band = QRubberBand(QRubberBand.Shape.Rectangle, view.viewport())
        self._origin = QPoint()
        self._active = False
        view.viewport().installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind not in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonRelease,
        ):
            return False
        mouse = cast(QMouseEvent, event)
        position = mouse.position().toPoint()
        if kind == QEvent.Type.MouseButtonPress:
            return self._begin(mouse, position)
        if not self._active:
            return False
        if kind == QEvent.Type.MouseMove:
            self._band.setGeometry(QRect(self._origin, position).normalized())
            return True
        if mouse.button() != Qt.MouseButton.LeftButton:
            return False
        self._finish()
        return True

    def _begin(self, mouse: QMouseEvent, position: QPoint) -> bool:
        view = self._view
        starts_here = (
            mouse.button() == Qt.MouseButton.LeftButton
            and bool(mouse.modifiers() & Qt.KeyboardModifier.ControlModifier)
            and not view.indexAt(position).isValid()
        )
        if not starts_here:
            return False
        self._origin = position
        self._active = True
        self._band.setGeometry(QRect(position, QSize()))
        self._band.show()
        return True

    def _finish(self) -> None:
        self._active = False
        self._band.hide()
        area = self._band.geometry().normalized()
        if area.width() > self._MIN_EXTENT and area.height() > self._MIN_EXTENT:
            self._select_touching(area)

    def _select_touching(self, area: QRect) -> None:
        view = self._view
        model = view.model()
        selection = view.selectionModel()
        touched = [
            index
            for row in range(model.rowCount())
            if view.visualRect(index := model.index(row, 0)).intersects(area)
        ]
        if not touched:
            return
        for index in touched:
            selection.select(index, QItemSelectionModel.SelectionFlag.Select)
        current = selection.currentIndex()
        leader_still_valid = (
            current.isValid()
            and selection.isSelected(current)
            and view.visualRect(current).intersects(area)
        )
        if not leader_still_valid:
            selection.setCurrentIndex(
                touched[0], QItemSelectionModel.SelectionFlag.NoUpdate
            )


@dataclass
class _DragSelection:
    """A press-and-drag over the thumbnails in progress.

    Attributes:
        adds: True when the rectangle selects what it touches, False when it
            deselects.
        before: Rows selected when the drag began; a row the rectangle stops
            covering returns to this state.
        touched: Rows the rectangle covered at the last move.
    """

    adds: bool
    before: set[int]
    touched: set[int] = field(default_factory=set)


class _ThumbnailView(QListView):
    """List view whose left clicks select thumbnails as upstream's sequence does.

    A plain click selects only the clicked thumbnail, Ctrl+click adds or removes
    one, and Shift+click adds every thumbnail between the current one and the
    clicked one. With ``toggle_on_click`` set, a plain click acts as Ctrl+click.
    A plain click goes ahead only when ``reset_accepted`` returns True.
    The last selected thumbnail cannot be removed; removing the current one makes
    the nearest selected thumbnail current, the one before it first.

    Dragging on from the press adds every thumbnail the rectangle from the press
    point to the cursor touches, or removes them when the press removed the
    pressed thumbnail. The rectangle replaces its own earlier extent, so moving
    back shrinks it, and it never removes the last selected thumbnail.

    These clicks bypass Qt's own press handling. Right clicks, keys and the
    empty area keep Qt's behaviour.

    Signals:
        selection_reset (): A plain click replaced the selection with one
            thumbnail.
    """

    selection_reset = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise; arguments are described on the class."""
        super().__init__(parent)
        self._drag: _DragSelection | None = None
        self._press_point = QPoint()
        self.toggle_on_click = False
        self.reset_accepted: Callable[[], bool] = lambda: True
        self._press_handled = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        index = self.indexAt(event.position().toPoint())
        if event.button() != Qt.MouseButton.LeftButton or not index.isValid():
            super().mousePressEvent(event)
            return
        self._press_handled = True
        self._drag = None
        self._press_point = event.position().toPoint() + self._scroll_offset()
        event.accept()
        modifiers = event.modifiers()
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        # As in Qt's own press handling, the clicked thumbnail must not scroll
        # away from under the pointer; a drag from it would start elsewhere.
        auto_scroll = self.hasAutoScroll()
        self.setAutoScroll(False)
        try:
            if modifiers & Qt.KeyboardModifier.ControlModifier or (
                self.toggle_on_click and not shift
            ):
                self._toggle(index)
            elif shift:
                self._extend_to(index)
            else:
                self._select_only(index)
        finally:
            self.setAutoScroll(auto_scroll)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press_handled and event.buttons() & Qt.MouseButton.LeftButton:
            event.accept()
            if self._drag is not None:
                self._drag_to(event.position().toPoint())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press_handled and event.button() == Qt.MouseButton.LeftButton:
            self._press_handled = False
            self._drag = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _scroll_offset(self) -> QPoint:
        return QPoint(self.horizontalOffset(), self.verticalOffset())

    def _begin_drag(self, adds: bool) -> None:
        rows = {index.row() for index in self.selectionModel().selectedRows()}
        self._drag = _DragSelection(adds=adds, before=rows)

    def _drag_to(self, position: QPoint) -> None:
        drag = self._drag
        assert drag is not None
        area = QRect(self._press_point - self._scroll_offset(), position).normalized()
        model = self.model()
        selection = self.selectionModel()
        touched = {
            row
            for row in range(model.rowCount())
            if self.visualRect(model.index(row, 0)).intersects(area)
        }
        if not drag.adds and not drag.before - touched:
            return
        flags = QItemSelectionModel.SelectionFlag
        for row in drag.touched - touched:
            selection.select(
                model.index(row, 0),
                flags.Select if row in drag.before else flags.Deselect,
            )
        for row in touched - drag.touched:
            selection.select(
                model.index(row, 0), flags.Select if drag.adds else flags.Deselect
            )
        drag.touched = touched

    def _select_only(self, index: QModelIndex) -> None:
        if not self.reset_accepted():
            return
        self.selectionModel().setCurrentIndex(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
        self._begin_drag(adds=True)
        self.selection_reset.emit()

    def _toggle(self, index: QModelIndex) -> None:
        selection = self.selectionModel()
        flags = QItemSelectionModel.SelectionFlag
        if not selection.isSelected(index):
            selection.setCurrentIndex(index, flags.Select)
            self._begin_drag(adds=True)
            return
        if len(selection.selectedRows()) < 2:
            selection.setCurrentIndex(index, flags.NoUpdate)
            self._begin_drag(adds=True)
            return
        was_current = index == selection.currentIndex()
        selection.select(index, flags.Deselect)
        self._begin_drag(adds=False)
        if was_current:
            selection.setCurrentIndex(
                self._nearest_selected(index.row()), flags.NoUpdate
            )

    def _nearest_selected(self, row: int) -> QModelIndex:
        """Return the selected row closest to ``row``, looking before it first."""
        model = self.model()
        selection = self.selectionModel()
        for distance in range(1, model.rowCount()):
            for candidate in (row - distance, row + distance):
                index = model.index(candidate, 0)
                if index.isValid() and selection.isSelected(index):
                    return index
        return QModelIndex()

    def _extend_to(self, index: QModelIndex) -> None:
        selection = self.selectionModel()
        current = selection.currentIndex()
        if not current.isValid():
            self._select_only(index)
            return
        first, last = sorted((current.row(), index.row()))
        model = self.model()
        for row in range(first, last + 1):
            selection.select(
                model.index(row, 0), QItemSelectionModel.SelectionFlag.Select
            )
        selection.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)


class ThumbnailListWidget(QWidget):
    """Scrollable list of page thumbnails.

    Previews come from ``renderer`` on worker threads and are cached, so the
    GUI thread never renders. The widget does not know about the main window;
    connect ``page_selected`` to navigation.

    Args:
        renderer (ThumbnailRenderer): Renders a small QImage for (page, stage).
        project (Project | None): Pages to list.
        stage (str): Stage key whose previews are shown.
        params_hash (Callable[[PageId, str], Hashable] | None): Digest of the
            parameters affecting a preview; a changed digest forces a re-render.
        cache_directory (Path | None): Directory for on-disk thumbnails, or
            None for memory only.
        cache_capacity (int): Thumbnails kept in memory.
        thumbnail_size (QSize | None): Bounding size of a thumbnail as drawn,
            the cell size; defaults to 160x160.
        render_size (QSize | None): Resolution thumbnails are rendered and cached
            at; defaults to ``thumbnail_size``.
        single_column (bool): Show one thumbnail per row instead of filling the
            width with as many as fit.
        extra_orders (Sequence[PageOrderOption]): Further sort options, for
            orders that need stage data, shown after the built-in ones.
        thread_count (int): Worker threads.
        parent (QWidget | None): Parent widget.

    Signals:
        page_selected (PageId): The current page changed.
        selection_changed (list): All selected pages, in display order.
        insert_before_requested (PageId): "Insert before ..." was chosen for a page.
        insert_after_requested (PageId): "Insert after ..." was chosen for a page.
        remove_pages_requested (list): "Remove from project ..." was chosen; the
            selected pages, in display order.
        insert_here_requested (): "Insert here ..." was chosen in the empty area.
        orders_changed (): The sort options were replaced, as the shown stage
            has others.
        selection_reset (): The selection was replaced with the one current
            page, by a plain click or by ``set_current_page``.
    """

    page_selected = Signal(object)
    selection_changed = Signal(object)
    insert_before_requested = Signal(object)
    insert_after_requested = Signal(object)
    remove_pages_requested = Signal(object)
    insert_here_requested = Signal()
    orders_changed = Signal()
    selection_reset = Signal()

    def __init__(
        self,
        renderer: ThumbnailRenderer,
        project: Project | None = None,
        stage: str = "",
        params_hash: Callable[[PageId, str], Hashable] | None = None,
        cache_directory: Path | None = None,
        cache_capacity: int = 256,
        thumbnail_size: QSize | None = None,
        render_size: QSize | None = None,
        single_column: bool = False,
        extra_orders: Sequence[PageOrderOption] = (),
        thread_count: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise; arguments are described on the class."""
        super().__init__(parent)
        size = thumbnail_size or QSize(160, 160)
        self._model = ThumbnailModel(
            renderer,
            cache=ThumbnailCache(cache_capacity, cache_directory),
            params_hash=params_hash,
            render_size=render_size or size,
            thread_count=thread_count,
            parent=self,
        )
        self._model.set_stage(stage)
        self._default_orders = [
            natural_order(),
            order_by_width(),
            order_by_height(),
            *extra_orders,
        ]
        self._orders = list(self._default_orders)
        self._stage_orders: Callable[[str], Sequence[PageOrderOption]] | None = None
        # Position of the sort option last chosen at each stage key.
        self._chosen_orders: dict[str, int] = {}
        self._restoring = False
        self._sorting_paused = False
        self._follow_current = True
        self._has_project = project is not None
        self._context_menu_enabled = True

        self._order_combo = QComboBox(self)
        for order in self._orders:
            self._order_combo.addItem(order.name)

        self._view = _ThumbnailView(self)
        self._view.selection_reset.connect(self.selection_reset)
        self._view.reset_accepted = self.canceling_selection_accepted
        self._view.setModel(self._model)
        self._delegate = ThumbnailDelegate(size, self._view)
        self._view.setItemDelegate(self._delegate)
        self._view.setUniformItemSizes(True)
        self._view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._on_context_menu_requested)
        self._rubber_band = _RubberBandSelector(self._view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._order_combo)
        layout.addWidget(self._view)

        self._order_combo.currentIndexChanged.connect(self._on_order_chosen)
        selection = self._view.selectionModel()
        selection.currentChanged.connect(self._on_current_changed)
        selection.selectionChanged.connect(self._on_selection_changed)

        self.set_single_column(single_column)
        self._model.set_project(project)

    @property
    def model(self) -> ThumbnailModel:
        """The underlying model."""
        return self._model

    @property
    def view(self) -> QListView:
        """The list view."""
        return self._view

    @property
    def order_combo(self) -> QComboBox:
        """The sort order combo box."""
        return self._order_combo

    @property
    def render_size(self) -> QSize:
        """Resolution thumbnails are rendered at; safe to read from workers."""
        return self._model.render_size

    @property
    def cell_size(self) -> QSize:
        """Bounding size a thumbnail is drawn at."""
        return self._delegate.thumbnail_size

    def set_cell_size(self, size: QSize) -> None:
        """Draw thumbnails at another size, laying the list out again.

        Args:
            size (QSize): The new bounding size of a thumbnail as drawn.
        """
        self._delegate.set_thumbnail_size(size)
        self._view.setMinimumWidth(self._minimum_view_width())
        self._view.doItemsLayout()
        self._view.viewport().update()

    def set_render_size(self, size: QSize) -> None:
        """Render thumbnails at another resolution, rendering them again.

        Args:
            size (QSize): The new bounding size of rendered thumbnails.
        """
        self._model.set_render_size(size)

    def set_single_column(self, single_column: bool) -> None:
        """Show one thumbnail per row, or as many per row as the width allows.

        Args:
            single_column (bool): True for one column.
        """
        view = self._view
        if single_column:
            view.setViewMode(QListView.ViewMode.ListMode)
            view.setFlow(QListView.Flow.TopToBottom)
            view.setWrapping(False)
        else:
            view.setViewMode(QListView.ViewMode.IconMode)
            view.setFlow(QListView.Flow.LeftToRight)
            view.setWrapping(True)
        view.setResizeMode(QListView.ResizeMode.Adjust)
        view.setMovement(QListView.Movement.Static)
        view.setSpacing(0)
        view.setUniformItemSizes(True)
        view.setMinimumWidth(self._minimum_view_width())

    def apply_settings(self, settings: ApplicationSettings) -> None:
        """Take the thumbnail options of the application settings.

        Args:
            settings (ApplicationSettings): Source of the render resolution, the
                cell size and the column mode.
        """
        quality = settings.thumbnail_quality
        max_width, max_height = settings.max_thumbnail_size
        self.set_single_column(settings.single_column_thumbnail_display)
        self.set_cell_size(QSize(round(max_width), round(max_height)))
        self.set_render_size(QSize(quality.width, quality.height))

    def _minimum_view_width(self) -> int:
        """Width that shows one whole cell next to a scroll bar."""
        style = self._view.style()
        scroll_bar = style.pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        frame = 2 * self._view.frameWidth()
        return self._delegate.cell_size().width() + scroll_bar + frame

    def set_project(self, project: Project | None) -> None:
        """Show the pages of a project, keeping the chosen sort order.

        Args:
            project (Project | None): The project, or None to clear the list.
        """
        self._has_project = project is not None
        self._model.set_project(project)

    def set_context_menu_enabled(self, enabled: bool) -> None:
        """Allow or suppress the page menu, for example while a batch runs.

        Args:
            enabled (bool): Whether right-clicking shows a menu.
        """
        self._context_menu_enabled = enabled

    def page_menu(self, page: PageId) -> QMenu:
        """Build the menu for a page; its actions emit the request signals.

        Args:
            page (PageId): The page the menu is for.

        Returns:
            QMenu: Insert before/after and remove, parented to the list.
        """
        menu = QMenu(self)
        insert_before = menu.addAction(
            icon("insert-before"),
            QtCore.QCoreApplication.translate("MainWindow", "Insert before ..."),
        )
        insert_after = menu.addAction(
            icon("insert-after"),
            QtCore.QCoreApplication.translate("MainWindow", "Insert after ..."),
        )
        menu.addSeparator()
        remove = menu.addAction(
            icon("trash-2", tint="danger"),
            QtCore.QCoreApplication.translate("MainWindow", "Remove from project ..."),
        )
        insert_before.triggered.connect(lambda: self.insert_before_requested.emit(page))
        insert_after.triggered.connect(lambda: self.insert_after_requested.emit(page))
        remove.triggered.connect(self._request_removal)
        return menu

    def empty_area_menu(self) -> QMenu:
        """Build the menu for the space past the last page.

        Returns:
            QMenu: A single "Insert here ..." action, parented to the list.
        """
        menu = QMenu(self)
        insert_here = menu.addAction(
            icon("insert-here"),
            QtCore.QCoreApplication.translate("MainWindow", "Insert here ..."),
        )
        insert_here.triggered.connect(self.insert_here_requested)
        return menu

    def _request_removal(self) -> None:
        self.remove_pages_requested.emit(self._selected_project_pages())

    def _selected_project_pages(self) -> list[PageId]:
        """Return every page the selected rows stand for, in display order."""
        return [
            page
            for selected in self.selected_pages()
            for page in self._model.pages_of_row(selected)
        ]

    def _on_context_menu_requested(self, position: QPoint) -> None:
        if not self._context_menu_enabled:
            return
        index = self._view.indexAt(position)
        if index.isValid():
            page = index.data(PageRole)
            if index not in self._view.selectionModel().selectedRows():
                self._view.setCurrentIndex(index)
            menu = self.page_menu(page)
        elif self._has_project:
            menu = self.empty_area_menu()
        else:
            return
        self._show_menu(menu, self._view.viewport().mapToGlobal(position))
        menu.deleteLater()

    def _show_menu(self, menu: QMenu, global_position: QPoint) -> None:
        menu.exec(global_position)

    def set_stage(self, stage: str, lists_images: bool = False) -> None:
        """Show previews for a different stage.

        The current page and the selection carry over when the list switches
        between pages and images: an image is selected when any of its pages
        was, and selecting an image selects all its pages.

        Args:
            stage (str): Stage key passed to the renderer.
            lists_images (bool): Show one thumbnail per image instead of one per
                page, as for the stages that work on whole images.
        """
        if lists_images == self._model.lists_images:
            self._model.set_stage(stage, lists_images)
        else:
            selected = self._selected_project_pages()
            current = self._model.current_page
            self._restoring = True
            try:
                self._model.set_stage(stage, lists_images)
                self._restore(selected, current)
            finally:
                self._restoring = False
            self.selection_changed.emit(self.selected_pages())
        if self._stage_orders is not None:
            self.set_orders(
                self._stage_orders(stage), self._chosen_orders.get(stage, 0)
            )

    def invalidate(self, page: PageId | None = None) -> None:
        """Refresh previews after stage parameters changed.

        Args:
            page (PageId | None): The page to refresh, or None for all.
        """
        self._model.invalidate(page)

    def pages(self) -> list[PageId]:
        """Return the page of each thumbnail in display order.

        Where the list shows images, each is the first page of its image.
        """
        return self._model.pages()

    def selected_pages(self) -> list[PageId]:
        """Return the page of each selected thumbnail in display order."""
        rows = sorted(
            index.row() for index in self._view.selectionModel().selectedRows()
        )
        pages = self._model.pages()
        return [pages[row] for row in rows]

    def current_page(self) -> PageId | None:
        """Return the current page, or None."""
        return self._model.current_page

    def set_current_page(self, page: PageId) -> None:
        """Make a page current and select only it.

        Where the list shows images, the page's image becomes current and the
        page itself is remembered, so it is current again once pages are shown.

        Scrolls it into view only while following the current page. Emits
        ``page_selected`` if the current page changes, and ``selection_reset``.

        Args:
            page (PageId): The page to make current.
        """
        row = self._model.row_of(page)
        if row < 0:
            return
        # QAbstractItemView scrolls to a new current index on its own; only
        # the follow setting may decide that for programmatic navigation.
        self._view.setAutoScroll(False)
        try:
            # QListView.setCurrentIndex would read the keyboard modifiers held
            # right now and extend the selection while Ctrl or Shift is down.
            self._view.selectionModel().setCurrentIndex(
                self._model.index(row), QItemSelectionModel.SelectionFlag.ClearAndSelect
            )
        finally:
            self._view.setAutoScroll(True)
        self._model.set_current_page(page)
        if self._follow_current:
            self.scroll_to_current()
        self.selection_reset.emit()

    def scroll_to_current(self) -> None:
        """Scroll so the current page is visible."""
        page = self._model.current_page
        row = self._model.row_of(page) if page is not None else -1
        if row >= 0:
            self._view.scrollTo(
                self._model.index(row), QAbstractItemView.ScrollHint.EnsureVisible
            )

    def set_follow_current(self, enabled: bool) -> None:
        """Scroll to the current page whenever it changes.

        Args:
            enabled (bool): Whether to follow the current page.
        """
        self._follow_current = enabled

    def canceling_selection_accepted(self) -> bool:
        """Ask whether a selection of several pages may be replaced by one page.

        Nothing is asked when the settings turn the question off, or when the
        selection is one page or the two pages of one image. Ticking "Don't
        show this message again" and answering OK turns the question off.

        Returns:
            bool: False when the user cancelled, True otherwise.
        """
        settings = ApplicationSettings.get_instance()
        if not settings.show_canceling_selection_question:
            return True
        selected = self.selected_pages()
        if len(selected) < 2 or (
            len(selected) == 2 and selected[0].image_id == selected[1].image_id
        ):
            return True
        box = QMessageBox(self)
        box.setWindowTitle(
            QtCore.QCoreApplication.translate(
                "QObject", "Canceling multi page selection"
            )
        )
        box.setText(
            QtCore.QCoreApplication.translate(
                "QObject", "%1 pages selection are going to be canceled. Continue?"
            ).replace("%1", str(len(selected)))
        )
        box.setIcon(QMessageBox.Icon.Question)
        box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Ok)
        box.setCheckBox(
            QCheckBox(
                QtCore.QCoreApplication.translate(
                    "QObject", "Don't show this message again."
                ),
                box,
            )
        )
        try:
            accepted = box.exec() == QMessageBox.StandardButton.Ok
            if accepted and box.checkBox().isChecked():
                settings.show_canceling_selection_question = False
                # save() has logged the failure; the answer holds for this session.
                with contextlib.suppress(OSError):
                    settings.save()
        finally:
            box.deleteLater()
        return accepted

    def set_selection_mode_enabled(self, enabled: bool) -> None:
        """Make a plain click add or remove a page, as Ctrl+click does.

        Args:
            enabled (bool): True to toggle pages with a plain click, False for a
                plain click to select only the clicked page.
        """
        self._view.toggle_on_click = enabled

    def set_order(self, index: int) -> None:
        """Choose a sort order by its position in the combo box.

        Args:
            index (int): Index of the option.
        """
        self._order_combo.setCurrentIndex(index)

    def order_names(self) -> list[str]:
        """Return the names of the sort options on offer, in order."""
        return [order.name for order in self._orders]

    def set_orders(
        self, orders: Sequence[PageOrderOption] | None, chosen: int = 0
    ) -> None:
        """Replace the sort options and choose one of them.

        Args:
            orders (Sequence[PageOrderOption] | None): The options; None for the
                built-in ones. Empty leaves the pages in natural order.
            chosen (int): Position of the option to choose; the first one when
                out of range.
        """
        self._orders = list(self._default_orders if orders is None else orders)
        self._order_combo.blockSignals(True)
        self._order_combo.clear()
        for order in self._orders:
            self._order_combo.addItem(order.name)
        if 0 <= chosen < len(self._orders):
            self._order_combo.setCurrentIndex(chosen)
        self._order_combo.blockSignals(False)
        self._apply_order()
        self.orders_changed.emit()

    def set_stage_orders_provider(
        self, provider: Callable[[str], Sequence[PageOrderOption]] | None
    ) -> None:
        """Offer the sort options of whichever stage is shown.

        Each stage starts on its first option, and the options chosen at each
        stage before are forgotten.

        Args:
            provider (Callable[[str], Sequence[PageOrderOption]] | None): Returns
                the options of a stage key; None goes back to the same options
                for every stage.
        """
        self._stage_orders = provider
        self._chosen_orders.clear()
        self.set_orders(None if provider is None else provider(self._model.stage))

    def set_incompleteness_provider(
        self, provider: IncompletenessProvider | None
    ) -> None:
        """Choose how pages with no result at the shown stage are found.

        Args:
            provider (IncompletenessProvider | None): The check; None marks no
                page.
        """
        self._model.set_incompleteness_provider(provider)

    def set_deviation_provider(self, provider: DeviantPagesProvider | None) -> None:
        """Choose how the pages that stand out at the shown stage are found.

        Args:
            provider (DeviantPagesProvider | None): The finder; None marks no page.
        """
        self._model.set_deviation_provider(provider)

    def refresh_marks(self) -> None:
        """Look again at which pages are incomplete or deviant, rendering nothing."""
        self._model.refresh()

    def bind_stage_sequence(self, sequence: StageSequence) -> None:
        """Take the sort options and the marked pages from a project's stages.

        Args:
            sequence (StageSequence): The stages, whose settings are only read.
        """
        self.set_incompleteness_provider(completeness_provider(sequence))
        self.set_deviation_provider(deviation_provider(sequence))
        self.set_stage_orders_provider(stage_orders_provider(sequence))

    @property
    def sort_reversed(self) -> bool:
        """Whether the pages are in the opposite of the chosen order."""
        return self._model.reversed_order

    def set_sort_reversed(self, reverse: bool) -> None:
        """Put the pages in the opposite of the chosen order, or back.

        Args:
            reverse (bool): True for the opposite order.
        """
        if reverse == self._model.reversed_order:
            return
        self._apply_order(reverse)

    def select_pages(self, pages: Sequence[PageId]) -> None:
        """Add pages to the selection without changing the current page.

        Args:
            pages (Sequence[PageId]): The pages to select; those not listed are ignored.
        """
        self._restoring = True
        try:
            self._restore(list(pages), None)
        finally:
            self._restoring = False
        self.selection_changed.emit(self.selected_pages())

    def set_sorting_paused(self, paused: bool) -> None:
        """Show the pages in the project's order, or go back to the chosen order.

        Args:
            paused (bool): True to ignore the chosen sort order for now.
        """
        if paused == self._sorting_paused:
            return
        self._sorting_paused = paused
        self._apply_order()

    def item_rect(self, page: PageId) -> QRect:
        """Return a page's rectangle in viewport coordinates."""
        return self._view.visualRect(self._model.index(self._model.row_of(page)))

    def _on_order_chosen(self, option_index: int) -> None:
        self._chosen_orders[self._model.stage] = option_index
        self._apply_order()

    def _current_order(self) -> PageOrderOption:
        index = self._order_combo.currentIndex()
        if self._sorting_paused or not 0 <= index < len(self._orders):
            return natural_order()
        return self._orders[index]

    def _apply_order(self, reverse: bool | None = None) -> None:
        selected = self.selected_pages()
        current = self._model.current_page
        self._restoring = True
        try:
            self._model.set_order(self._current_order(), reverse)
            self._restore(selected, current)
        finally:
            self._restoring = False
        self.selection_changed.emit(self.selected_pages())

    def _restore(self, selected: list[PageId], current: PageId | None) -> None:
        selection = self._view.selectionModel()
        for page in selected:
            selection.select(
                self._model.index(self._model.row_of(page)),
                selection.SelectionFlag.Select,
            )
        if current is not None and self._model.row_of(current) >= 0:
            self._model.set_current_page(current)
            selection.setCurrentIndex(
                self._model.index(self._model.row_of(current)),
                selection.SelectionFlag.NoUpdate,
            )

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        page = current.data(PageRole) if current.isValid() else None
        shown = self._model.current_page
        if page is not None and shown is not None:
            if self._model.row_of(shown) == current.row():
                # The row stands for the image of the remembered page.
                return
        self._model.set_current_page(page)
        if page is None or self._restoring:
            return
        if self._follow_current:
            self.scroll_to_current()
        self.page_selected.emit(page)

    def _on_selection_changed(self) -> None:
        if not self._restoring:
            self.selection_changed.emit(self.selected_pages())
