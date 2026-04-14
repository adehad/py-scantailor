"""Relinking dialog for fixing broken file paths in projects."""

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor


class RelinkingStatus:
    """Status of a file that may need relinking."""

    MISSING = "missing"
    RELINKED = "relinked"
    OK = "ok"


class RelinkingItem:
    """Item representing a file path that may need relinking."""

    def __init__(self, original_path: Path, status: str = RelinkingStatus.MISSING):
        """Initialize a relinking item.

        Args:
            original_path: The original file path from the project.
            status: Current status (missing, relinked, ok).
        """
        self.original_path = original_path
        self.new_path: Path | None = None
        self.status = status

    @property
    def display_path(self) -> Path:
        """Get the path to display (new path if relinked, otherwise original)."""
        return self.new_path if self.new_path else self.original_path

    @property
    def is_missing(self) -> bool:
        """Check if the file is missing."""
        return self.status == RelinkingStatus.MISSING


class RelinkingModel(QAbstractListModel):
    """Model for the list of files that may need relinking."""

    def __init__(self, items: list[RelinkingItem] | None = None, parent=None):
        """Initialize the model.

        Args:
            items: List of relinking items.
            parent: Parent object.
        """
        super().__init__(parent)
        self._items = items or []

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        """Get the number of rows."""
        return len(self._items)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        """Get data for a cell."""
        if not index.isValid() or index.row() >= len(self._items):
            return None

        item = self._items[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return str(item.display_path)
        elif role == Qt.ItemDataRole.ToolTipRole:
            if item.new_path:
                return f"Original: {item.original_path}\nRelinked to: {item.new_path}"
            return str(item.original_path)
        elif role == Qt.ItemDataRole.ForegroundRole:
            if item.status == RelinkingStatus.MISSING:
                return QColor(Qt.GlobalColor.red)
            elif item.status == RelinkingStatus.RELINKED:
                return QColor(Qt.GlobalColor.blue)
        elif role == Qt.ItemDataRole.UserRole:
            return item

        return None

    def get_item(self, row: int) -> RelinkingItem | None:
        """Get an item by row index."""
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def update_item(self, row: int, new_path: Path) -> None:
        """Update an item with a new path.

        Args:
            row: Row index.
            new_path: New file path.
        """
        if 0 <= row < len(self._items):
            self._items[row].new_path = new_path
            self._items[row].status = RelinkingStatus.RELINKED
            index = self.index(row)
            self.dataChanged.emit(index, index)

    def get_items(self) -> list[RelinkingItem]:
        """Get all items."""
        return self._items


class RelinkingDialog(QtWidgets.QDialog):
    """Dialog for relinking missing files in a project.

    When project files are moved or renamed, this dialog allows the user
    to fix the broken paths by selecting new locations for the files.

    Signals:
        relinking_completed: Emitted when relinking is completed.
            Args: items (list[RelinkingItem]) - the updated items
    """

    relinking_completed = Signal(list)

    def __init__(
        self,
        missing_files: list[Path] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the relinking dialog.

        Args:
            missing_files: List of missing file paths.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Relinking")
        self.resize(500, 400)

        # Create items from missing files
        items = [RelinkingItem(p) for p in (missing_files or [])]
        self._model = RelinkingModel(items)
        self._undo_stack: list[tuple[int, Path | None, str]] = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Path visualization and controls
        top_layout = QtWidgets.QHBoxLayout()

        self._path_label = QtWidgets.QLabel()
        self._path_label.setWordWrap(True)
        top_layout.addWidget(self._path_label, 1)

        self._error_label = QtWidgets.QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setWordWrap(True)
        top_layout.addWidget(self._error_label, 1)

        self._undo_btn = QtWidgets.QToolButton()
        self._undo_btn.setText("Undo")
        self._undo_btn.setToolTip("Undo last relinking")
        self._undo_btn.setEnabled(False)
        top_layout.addWidget(self._undo_btn)

        layout.addLayout(top_layout)

        # File list
        self._list_view = QtWidgets.QListView()
        self._list_view.setModel(self._model)
        self._list_view.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        layout.addWidget(self._list_view)

        # Relink buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self._relink_btn = QtWidgets.QPushButton("Relink Selected...")
        btn_layout.addWidget(self._relink_btn)

        self._relink_all_btn = QtWidgets.QPushButton("Relink All in Folder...")
        btn_layout.addWidget(self._relink_all_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Dialog buttons
        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self._button_box)

        self._update_ui()

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._list_view.selectionModel().selectionChanged.connect(self._update_ui)
        self._relink_btn.clicked.connect(self._relink_selected)
        self._relink_all_btn.clicked.connect(self._relink_all_in_folder)
        self._undo_btn.clicked.connect(self._undo)
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)

    def _update_ui(self) -> None:
        """Update UI state based on selection."""
        has_selection = self._list_view.selectionModel().hasSelection()
        self._relink_btn.setEnabled(has_selection)

        # Update path display for selected item
        indexes = self._list_view.selectionModel().selectedIndexes()
        if indexes:
            item = self._model.get_item(indexes[0].row())
            if item:
                self._path_label.setText(str(item.original_path))
                if item.is_missing:
                    self._error_label.setText("File not found")
                else:
                    self._error_label.clear()
        else:
            self._path_label.clear()
            self._error_label.clear()

        self._undo_btn.setEnabled(len(self._undo_stack) > 0)

    def _relink_selected(self) -> None:
        """Relink selected files."""
        indexes = self._list_view.selectionModel().selectedIndexes()
        if not indexes:
            return

        # Get a file to relink to
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files (*.*)",
        )

        if file_path:
            new_path = Path(file_path)
            for index in indexes:
                row = index.row()
                item = self._model.get_item(row)
                if item:
                    # Save for undo
                    self._undo_stack.append((row, item.new_path, item.status))
                    self._model.update_item(row, new_path)

        self._update_ui()

    def _relink_all_in_folder(self) -> None:
        """Relink all missing files using a folder."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Files",
            "",
        )

        if folder:
            folder_path = Path(folder)
            for row, item in enumerate(self._model.get_items()):
                if item.is_missing:
                    # Try to find the file by name in the folder
                    potential_path = folder_path / item.original_path.name
                    if potential_path.exists():
                        self._undo_stack.append((row, item.new_path, item.status))
                        self._model.update_item(row, potential_path)

        self._update_ui()

    def _undo(self) -> None:
        """Undo the last relinking operation."""
        if not self._undo_stack:
            return

        row, old_path, old_status = self._undo_stack.pop()
        item = self._model.get_item(row)
        if item:
            item.new_path = old_path
            item.status = old_status
            index = self._model.index(row)
            self._model.dataChanged.emit(index, index)

        self._update_ui()

    def _on_accept(self) -> None:
        """Handle accept - emit results and close."""
        self.relinking_completed.emit(self._model.get_items())
        self.accept()

    def get_relinked_paths(self) -> dict[Path, Path]:
        """Get a mapping of original paths to new paths.

        Returns:
            Dictionary mapping original paths to relinked paths.
        """
        result = {}
        for item in self._model.get_items():
            if item.new_path:
                result[item.original_path] = item.new_path
        return result

    def has_missing_files(self) -> bool:
        """Check if there are still missing files.

        Returns:
            True if any files are still missing.
        """
        return any(item.is_missing for item in self._model.get_items())
