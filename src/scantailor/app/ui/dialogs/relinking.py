"""Relinking dialog: point the image and output folders of a project elsewhere."""

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor

from scantailor.core.relinker import Relinker

_PATH_ROLE = Qt.ItemDataRole.UserRole


class RelinkingDialog(QtWidgets.QDialog):
    """Lets the user choose a new folder for each image folder and the output folder.

    Relinking a folder also relinks everything below it. The result is a
    ``Relinker`` that the caller applies to the project.
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
        self.setWindowTitle(self.tr("Relinking"))
        self.resize(560, 400)
        self._relinker = Relinker()
        self._history: list[tuple[Path, Path | None]] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                self.tr(
                    "Select a folder and point it to where its files are now. "
                    "Folders shown in red could not be found."
                )
            )
        )
        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        for directory in directories:
            item = QtWidgets.QListWidgetItem()
            item.setData(_PATH_ROLE, directory)
            self._list.addItem(item)
        layout.addWidget(self._list)

        buttons = QtWidgets.QHBoxLayout()
        self._relink_btn = QtWidgets.QPushButton(self.tr("Relink..."))
        self._undo_btn = QtWidgets.QPushButton(self.tr("Undo"))
        buttons.addWidget(self._relink_btn)
        buttons.addWidget(self._undo_btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self._button_box)

        self._relink_btn.clicked.connect(self._relink_selected)
        self._undo_btn.clicked.connect(self.undo)
        self._list.itemSelectionChanged.connect(self._update_controls)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        self._refresh_items()

    def relinker(self) -> Relinker:
        """Return the mapping the user chose."""
        return self._relinker

    def directories(self) -> list[Path]:
        """Return the folders listed in the dialog."""
        return [
            self._list.item(row).data(_PATH_ROLE) for row in range(self._list.count())
        ]

    def relink_directory(self, old: Path, new: Path) -> None:
        """Point a listed folder at a new location.

        Args:
            old (Path): A folder as listed in the dialog.
            new (Path): Where its files are now.
        """
        self._history.append((old, self._relinker.target_of(old)))
        self._relinker.relink(old, new)
        self._refresh_items()

    def undo(self) -> None:
        """Revert the most recent relinking."""
        if not self._history:
            return
        old, previous = self._history.pop()
        if previous is None:
            self._relinker.unlink(old)
        else:
            self._relinker.relink(old, previous)
        self._refresh_items()

    def has_missing_directories(self) -> bool:
        """Whether a listed folder still does not exist after relinking."""
        return any(
            not self._relinker.substitute(directory).is_dir()
            for directory in self.directories()
        )

    def _relink_selected(self) -> None:
        """Ask for a folder and apply it to the selected rows."""
        selected = [item.data(_PATH_ROLE) for item in self._list.selectedItems()]
        if not selected:
            return
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select the folder of the images"), str(selected[0])
        )
        if not chosen:
            return
        for directory in selected:
            self.relink_directory(directory, Path(chosen))

    def _refresh_items(self) -> None:
        """Show the substituted path of each row, red while it is still missing."""
        for row in range(self._list.count()):
            item = self._list.item(row)
            original: Path = item.data(_PATH_ROLE)
            current = self._relinker.substitute(original)
            item.setText(str(current))
            item.setToolTip(str(original))
            missing = QBrush(QColor("red")) if not current.is_dir() else None
            item.setData(Qt.ItemDataRole.ForegroundRole, missing)
        self._update_controls()

    def _update_controls(self) -> None:
        """Enable the buttons that currently make sense."""
        self._relink_btn.setEnabled(bool(self._list.selectedItems()))
        self._undo_btn.setEnabled(bool(self._history))
