"""Custom table view for displaying filter stages in the main window.

This view displays the 6 filter stages as a table with stage names and
a batch processing launch button. Each name carries its stage number in the
same cell, so the number shares the font, colour and selection of the name.
"""

from PySide6.QtCore import (
    QAbstractTableModel,
    QCoreApplication,
    QEvent,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
    Signal,
)
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QHeaderView,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QToolButton,
    QWidget,
)

from scantailor.app.ui.icons import icon
from scantailor.app.ui.theme.fonts import tabular
from scantailor.app.ui.theme.stylesheet import SPACE_S
from scantailor.core.stage_sequence import StageSequence

# Vertical space around the stage name: 36 px rows at 13 px Inter.
_ROW_LEADING = 20
# Figure space: as wide as a digit, so the names line up behind the numbers.
_NUMBER_GAP = "\u2007\u2007"


class StageListModel(QAbstractTableModel):
    """Model for the stage list table."""

    def __init__(
        self, stages: StageSequence | None = None, parent: QWidget | None = None
    ) -> None:
        """Initialize the model.

        Args:
            stages: The stage sequence.
            parent: Parent object.
        """
        super().__init__(parent)
        self._stages = stages
        self._show_animation = False
        self._selected_row = -1

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        """Get row count.

        Args:
            parent: Parent index (unused for table).

        Returns:
            Number of stages (6).
        """
        if parent.isValid():
            return 0
        return 6  # Always 6 stages

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        """Get column count.

        Args:
            parent: Parent index (unused for table).

        Returns:
            Always 2 (name column + button column).
        """
        if parent.isValid():
            return 0
        return 2

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        """Get data for a cell.

        Args:
            index: Cell index.
            role: Data role.

        Returns:
            Data for the requested role.
        """
        if not index.isValid() or not self._stages:
            return None

        row = index.row()
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole and col == 0:
            # Get stage name
            stage_names = [
                QCoreApplication.translate(
                    "fix_orientation::Filter", "Fix Orientation"
                ),
                QCoreApplication.translate("page_split::Filter", "Split Pages"),
                QCoreApplication.translate("deskew::Filter", "Deskew"),
                QCoreApplication.translate("select_content::Filter", "Select Content"),
                QCoreApplication.translate("page_layout::Filter", "Margins"),
                QCoreApplication.translate("output::Filter", "Output"),
            ]
            if 0 <= row < len(stage_names):
                return stage_names[row]

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col == 0:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            else:
                return Qt.AlignmentFlag.AlignCenter

        return None

    def set_batch_animation(self, selected_row: int, show: bool) -> None:
        """Set batch processing animation state.

        Args:
            selected_row: The selected row to show animation for.
            show: Whether to show the animation.
        """
        self._selected_row = selected_row
        self._show_animation = show


class StageListDelegate(QStyledItemDelegate):
    """Delegate for stage list items."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the delegate.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

    def initStyleOption(
        self,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Prefix the stage name with its number, in tabular figures.

        Args:
            option: Style options to fill.
            index: Model index.
        """
        super().initStyleOption(option, index)
        if index.column() == 0 and option.text:
            # QStyleOptionViewItem fields are typed as `...` in the PySide6 stubs
            # (PYSIDE-3034).
            # pyrefly: ignore[bad-assignment]
            option.text = f"{index.row() + 1}{_NUMBER_GAP}{option.text}"
            # pyrefly: ignore[bad-assignment, bad-argument-type]
            option.font = tabular(option.font)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Paint the item without focus indicator.

        Args:
            painter: The painter.
            option: Style options.
            index: Model index.
        """
        # Remove focus state to prevent focus rectangle
        opt = QStyleOptionViewItem(option)
        # QStyleOption.state is typed as `...` in the PySide6 stubs (PYSIDE-3034)
        # pyrefly: ignore[bad-assignment, unsupported-operation]
        opt.state &= ~QStyle.StateFlag.State_HasFocus
        super().paint(painter, opt, index)


class StageListView(QTableView):
    """Custom table view for displaying filter stages.

    This view shows the 6 processing stages and provides a batch
    processing launch button.

    Signals:
        launch_batch_processing: Emitted when batch processing is requested.
    """

    launch_batch_processing = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the stage list view.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._model: StageListModel | None = None
        self._launch_btn: QToolButton | None = None
        self._batch_processing_possible = False
        self._batch_processing_in_progress = False

        # Configure view
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.SingleSelection)

        # Set up delegates to remove focus indicators
        self.setItemDelegateForColumn(0, StageListDelegate(self))
        self.setItemDelegateForColumn(1, StageListDelegate(self))

        # Configure headers
        h_header = self.horizontalHeader()
        h_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        h_header.hide()

        v_header = self.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        v_header.setSectionsMovable(False)
        v_header.hide()

        # Create launch button (initially hidden)
        self._launch_btn = QToolButton(self.viewport())
        self._launch_btn.setIcon(icon("play"))
        self._launch_btn.setToolTip(self.tr("Launch batch processing"))
        self._launch_btn.hide()
        self._launch_btn.clicked.connect(self.launch_batch_processing.emit)

        # Set size policy
        size_policy = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        size_policy.setVerticalStretch(1)
        self.setSizePolicy(size_policy)
        self._apply_row_height()

    def row_height(self) -> int:
        """Return the height of one stage row, which grows with the font.

        Returns:
            int: Height in logical pixels.
        """
        return QFontMetrics(self.font()).height() + _ROW_LEADING

    def _apply_row_height(self) -> None:
        row = self.row_height()
        self.verticalHeader().setDefaultSectionSize(row)
        if self._launch_btn is not None:
            self._launch_btn.setFixedSize(row - SPACE_S, row - SPACE_S)
        if self._model is not None:
            self.horizontalHeader().resizeSection(1, row)
            height = self.verticalHeader().length()
            height += self.height() - self.viewport().height()
            self.setFixedHeight(height)
        self._update_launch_button()

    def changeEvent(self, event: QEvent) -> None:
        """Resize the rows when the font changes.

        Args:
            event: The change event.
        """
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._apply_row_height()

    def set_stages(self, stages: StageSequence) -> None:
        """Set the stage sequence to display.

        Args:
            stages: The stage sequence.
        """
        # Delete old model if exists
        if self._model:
            self._model.deleteLater()

        # Create and set new model
        self._model = StageListModel(stages, self)
        self.setModel(self._model)

        # Configure column widths
        h_header = self.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)

        self._apply_row_height()
        self.updateGeometry()

    def set_batch_processing_possible(self, possible: bool) -> None:
        """Set whether batch processing is possible.

        Args:
            possible: True if batch processing can be launched.
        """
        if self._batch_processing_possible == possible:
            return

        self._batch_processing_possible = possible
        self._update_launch_button()

    def set_batch_processing_in_progress(self, in_progress: bool) -> None:
        """Set whether batch processing is in progress.

        Args:
            in_progress: True if batch processing is running.
        """
        if self._batch_processing_in_progress == in_progress:
            return

        self._batch_processing_in_progress = in_progress
        self._update_launch_button()

    def selectionChanged(self, selected, deselected) -> None:
        """Handle selection changes.

        Args:
            selected: Newly selected items.
            deselected: Newly deselected items.
        """
        super().selectionChanged(selected, deselected)
        self._update_launch_button()

    def _update_launch_button(self) -> None:
        """Update launch button visibility and position."""
        if not self._launch_btn:
            return

        # Show button if batch processing is possible and not in progress
        show_button = (
            self._batch_processing_possible and not self._batch_processing_in_progress
        )

        # Get selected row
        selection = self.selectionModel()
        if not selection or not selection.hasSelection():
            show_button = False
        else:
            selected_row = selection.currentIndex().row()
            if selected_row < 0:
                show_button = False

        if show_button:
            # Position button in second column of selected row
            selected_row = selection.currentIndex().row()
            index = self._model.index(selected_row, 1) if self._model else QModelIndex()

            if index.isValid():
                rect = self.visualRect(index)
                # Center button in cell
                btn_x = rect.x() + (rect.width() - self._launch_btn.width()) // 2
                btn_y = rect.y() + (rect.height() - self._launch_btn.height()) // 2
                self._launch_btn.move(btn_x, btn_y)
                self._launch_btn.raise_()
                self._launch_btn.show()
        else:
            self._launch_btn.hide()

    def resizeEvent(self, event) -> None:
        """Handle resize event.

        Args:
            event: Resize event.
        """
        super().resizeEvent(event)
        self._update_launch_button()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        """Handle scrolling.

        Args:
            dx: Horizontal scroll delta.
            dy: Vertical scroll delta.
        """
        super().scrollContentsBy(dx, dy)
        self._update_launch_button()
