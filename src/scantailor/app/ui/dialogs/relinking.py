"""Relinking dialog: point the image and output folders of a project elsewhere."""

from pathlib import Path

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QDir, Qt

from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.icons import icon
from scantailor.app.ui.utils import get_cwidget
from scantailor.app.ui.widgets.relinkable_path_visualization import (
    PathType,
    RelinkablePathVisualization,
)
from scantailor.app.ui.widgets.relinking_list_view import (
    UNCOMMITTED_STATUS_ROLE,
    RelinkingListView,
    RelinkingStatus,
)
from scantailor.core.relinker import Relinker

_UI_FOLDER = Path(__file__).parent.parent
_ORIGINAL_PATH_ROLE = Qt.ItemDataRole.UserRole

_Change = list[tuple[Path, Path | None]]


class RelinkingDialog(QtWidgets.QDialog):
    """Lets the user choose a new location for each image folder and the output folder.

    Selecting a folder shows its path as clickable components; clicking one asks
    for the folder that replaces it, which also relinks every listed folder below
    it. The result is a ``Relinker`` that the caller applies to the project.
    """

    def __init__(
        self,
        directories: list[Path],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the dialog.

        Args:
            directories (list[Path]): Folders to offer for relinking, as stored
                in the project.
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self._relinker = Relinker()
        self._history: list[_Change] = []

        self.ui = load_ui_widget(_UI_FOLDER / "RelinkingDialog.ui", self)
        embed_ui(self, self.ui)
        self._path_view = get_cwidget(
            self.ui, RelinkablePathVisualization, "pathVisualization"
        )
        self._error_label = get_cwidget(self.ui, QtWidgets.QLabel, "errorLabel")
        self._undo_button = get_cwidget(self.ui, QtWidgets.QToolButton, "undoButton")
        self._list_view = get_cwidget(self.ui, RelinkingListView, "listView")
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")

        self._model = QtGui.QStandardItemModel(self)
        for directory in directories:
            item = QtGui.QStandardItem()
            item.setEditable(False)
            item.setData(directory, _ORIGINAL_PATH_ROLE)
            self._model.appendRow(item)
        self._list_view.setModel(self._model)
        self._list_view.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._undo_button.setIcon(icon("undo-2"))
        self._error_label.setVisible(False)
        self._undo_button.setVisible(False)
        self._path_view.setVisible(False)

        self._list_view.selectionModel().selectionChanged.connect(
            self._selection_changed
        )
        self._path_view.clicked.connect(self._path_component_clicked)
        self._undo_button.clicked.connect(self.undo)
        self._button_box.accepted.connect(self.accept)
        self._refresh_items()

    def relinker(self) -> Relinker:
        """Return the mapping the user chose."""
        return self._relinker

    def directories(self) -> list[Path]:
        """Return the folders listed in the dialog."""
        return [
            self._model.item(row).data(_ORIGINAL_PATH_ROLE)
            for row in range(self._model.rowCount())
        ]

    def relink_directory(self, old: Path, new: Path) -> None:
        """Point a listed folder at a new location.

        Args:
            old (Path): A folder as listed in the dialog.
            new (Path): Where its files are now.
        """
        self._apply_change({old: new})

    def undo(self) -> None:
        """Revert the most recent relinking."""
        if not self._history:
            return
        self._revert(self._history.pop())
        self._error_label.setVisible(False)
        self._refresh_items()

    def has_missing_directories(self) -> bool:
        """Whether a listed folder still does not exist after relinking."""
        return any(
            not self._relinker.substitute(directory).is_dir()
            for directory in self.directories()
        )

    def _apply_change(self, targets: dict[Path, Path]) -> bool:
        """Relink several listed folders as one undoable step.

        Args:
            targets (dict[Path, Path]): New location of each listed folder to change.

        Returns:
            bool: False, with nothing changed, when two listed folders would end
                up in the same place.
        """
        change: _Change = [(old, self._relinker.target_of(old)) for old in targets]
        for old, new in targets.items():
            self._relinker.relink(old, new)
        if self._merges_folders():
            self._revert(change)
            self._error_label.setText(
                self.tr("This change would merge several files into one.")
            )
            self._error_label.setVisible(True)
            self._path_view.clear()
            self._path_view.setVisible(False)
            return False
        self._history.append(change)
        self._refresh_items()
        return True

    def _revert(self, change: _Change) -> None:
        """Restore the mapping recorded by ``_apply_change``."""
        for old, previous in reversed(change):
            if previous is None:
                self._relinker.unlink(old)
            else:
                self._relinker.relink(old, previous)

    def _merges_folders(self) -> bool:
        """Whether two distinct listed folders now resolve to the same folder."""
        resolved = [self._relinker.substitute(d) for d in self.directories()]
        return len(set(resolved)) < len(resolved)

    def _selected_directory(self) -> Path | None:
        """Return the listed folder that is selected, as stored in the project."""
        indexes = self._list_view.selectionModel().selectedIndexes()
        return indexes[0].data(_ORIGINAL_PATH_ROLE) if indexes else None

    def _selection_changed(self) -> None:
        """Show the clickable path of the selected folder."""
        self._error_label.setVisible(False)
        self._show_selected_path()

    def _show_selected_path(self) -> None:
        """Fill the path components from the selection, or hide them."""
        selected = self._selected_directory()
        if selected is None:
            self._path_view.clear()
            self._path_view.setVisible(False)
            return
        current = self._relinker.substitute(selected)
        self._path_view.set_path(str(current), PathType.DIR, clickable=True)
        self._path_view.setVisible(True)

    def _path_component_clicked(
        self, prefix_path: str, _suffix_path: str, _path_type: int
    ) -> None:
        """Ask for the folder replacing a clicked component and relink below it.

        Args:
            prefix_path (str): The path up to and including the clicked component.
            _suffix_path (str): The rest of the selected path.
            _path_type (int): ``PathType`` of the clicked component.
        """
        prefix = Path(prefix_path)
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Substitution Directory for %1").replace(
                "%1", QDir.toNativeSeparators(prefix_path)
            ),
            prefix_path if prefix.is_dir() else "",
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if not chosen or Path(chosen) == prefix:
            return
        self._relink_below(prefix, Path(chosen))

    def _relink_below(self, prefix: Path, replacement: Path) -> None:
        """Relink every listed folder at or below ``prefix`` onto ``replacement``.

        Args:
            prefix (Path): The path component that was replaced.
            replacement (Path): The folder chosen to stand in for it.
        """
        targets: dict[Path, Path] = {}
        for directory in self.directories():
            current = self._relinker.substitute(directory)
            if current.is_relative_to(prefix):
                targets[directory] = replacement / current.relative_to(prefix)
        self._apply_change(targets)

    def _refresh_items(self) -> None:
        """Show the substituted path of each row and whether it exists."""
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            original: Path = item.data(_ORIGINAL_PATH_ROLE)
            current = self._relinker.substitute(original)
            item.setText(QDir.toNativeSeparators(str(current)))
            item.setToolTip(str(original))
            item.setData(
                RelinkingStatus.EXISTS if current.is_dir() else RelinkingStatus.MISSING,
                UNCOMMITTED_STATUS_ROLE,
            )
        self._undo_button.setVisible(bool(self._history))
        self._show_selected_path()
        self._list_view.viewport().update()
