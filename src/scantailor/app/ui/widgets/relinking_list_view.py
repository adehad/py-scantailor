"""Custom list view for the relinking dialog with status indicators.

This list view displays a visual status layer showing which files exist (green)
and which are missing (red) using colored rounded rectangles on the right side.
"""

from dataclasses import dataclass

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QRect, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QListView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)


# Status constants matching the C++ RelinkingModel
class RelinkingStatus:
    """Status values for relinking items."""

    EXISTS = 0
    MISSING = 1
    UPDATE_PENDING = 2


# Role constants
UNCOMMITTED_STATUS_ROLE = Qt.ItemDataRole.UserRole + 2


@dataclass
class IndicationGroup:
    """A group of adjacent items with the same status."""

    rect: QRect
    status: int


class RelinkingListViewDelegate(QStyledItemDelegate):
    """Custom delegate that allows the list view to draw status indicators."""

    def __init__(self, owner: "RelinkingListView") -> None:
        """Initialize the delegate.

        Args:
            owner: The RelinkingListView that owns this delegate.
        """
        super().__init__(owner)
        self._owner = owner

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Paint the item and maybe draw status layer.

        Args:
            painter: The painter to use.
            option: Style options for the item.
            index: The model index of the item.
        """
        # QStyleOption.rect missing from PySide6 stubs (PYSIDE-3034)
        # pyrefly: ignore[missing-attribute]
        self._owner._maybe_draw_status_layer(painter, index, option.rect)
        super().paint(painter, option, index)


class GroupAggregator:
    """Aggregates adjacent items with the same status into groups."""

    def __init__(self) -> None:
        """Initialize the aggregator."""
        self._groups: list[IndicationGroup] = []

    def process(self, rect: QRect, status: int) -> None:
        """Process an item rectangle and status.

        Args:
            rect: The rectangle for the item.
            status: The status of the item.
        """
        if not self._groups or self._groups[-1].status != status:
            self._groups.append(IndicationGroup(QRect(rect), status))
        else:
            # Merge with previous group
            self._groups[-1].rect = self._groups[-1].rect.united(rect)

    @property
    def groups(self) -> list[IndicationGroup]:
        """Get the aggregated groups.

        Returns:
            List of indication groups.
        """
        return self._groups


class RelinkingListView(QListView):
    """Custom list view for the relinking dialog.

    This view displays a status indicator layer on the right side showing
    which files exist (green rounded rectangles) and which are missing
    (red rounded rectangles).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the relinking list view.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._status_layer_drawn = False
        self.setItemDelegate(RelinkingListViewDelegate(self))

    def paintEvent(self, e: QPaintEvent) -> None:
        """Handle paint event.

        Args:
            e: The paint event.
        """
        self._status_layer_drawn = False
        super().paintEvent(e)

    def _maybe_draw_status_layer(
        self,
        painter: QPainter,
        item_index: QModelIndex | QPersistentModelIndex,
        item_paint_rect: QRect,
    ) -> None:
        """Draw the status layer once during painting.

        Args:
            painter: The painter to use.
            item_index: The index of the item being painted.
            item_paint_rect: The rectangle where the item is painted.
        """
        if self._status_layer_drawn:
            return

        painter.save()
        try:
            # Translate painter to viewport coordinates
            painter.translate(
                item_paint_rect.topLeft() - self.visualRect(item_index).topLeft()
            )
            painter.setClipRect(self.viewport().rect())
            self._draw_status_layer(painter)
        finally:
            painter.restore()

        self._status_layer_drawn = True

    def _draw_status_layer(self, painter: QPainter) -> None:
        """Draw the status indicator layer.

        Args:
            painter: The painter to use (in viewport coordinates).
        """
        drawing_rect = self.viewport().rect()
        top_index = self.indexAt(drawing_rect.topLeft())

        if not top_index.isValid():
            # No visible elements
            return

        # Start one row above the visible area if possible
        # (appearance depends on neighbors)
        if top_index.row() > 0:
            top_index = top_index.sibling(top_index.row() - 1, 0)

        # Aggregate items into groups by status
        group_aggregator = GroupAggregator()
        model = top_index.model()
        if model is None:
            return

        rows = model.rowCount(top_index.parent())

        for row in range(top_index.row(), rows):
            index = top_index.sibling(row, 0)
            item_rect = self.visualRect(index)

            # Create status indicator rectangle on the right side
            rect = QRect(drawing_rect)
            rect.setTop(item_rect.top())
            rect.setBottom(item_rect.bottom())
            rect.setWidth(item_rect.height())  # Square indicator
            rect.moveRight(drawing_rect.right())

            # Get status from model
            status = index.data(UNCOMMITTED_STATUS_ROLE)
            if status is None:
                status = RelinkingStatus.EXISTS

            group_aggregator.process(rect, status)

            # Break after first invisible item (but process it for neighbor)
            if row != top_index.row() and not item_rect.intersects(drawing_rect):
                break

        # Draw the status indicators
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw existing items (green), then missing items (red)
        for status, pen_color, brush_color in [
            (
                RelinkingStatus.EXISTS,
                QColor(0x3A, 0x58, 0x27),
                QColor(0x89, 0xE7, 0x4A),
            ),
            (
                RelinkingStatus.MISSING,
                QColor(0x6F, 0x27, 0x19),
                QColor(0xFF, 0x67, 0x4B),
            ),
        ]:
            pen = QPen(pen_color)
            pen.setWidthF(1.5)
            brush = QBrush(brush_color)

            painter.setPen(pen)
            painter.setBrush(brush)

            for group in group_aggregator.groups:
                if group.status == status:
                    radius = 0.5 * group.rect.width()
                    rect = group.rect.toRectF()
                    pen_width = pen.widthF()
                    rect.adjust(pen_width, pen_width, -pen_width, -pen_width)
                    painter.drawRoundedRect(rect, radius, radius)
