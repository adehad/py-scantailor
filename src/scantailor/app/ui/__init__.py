"""UI Elements & Screens."""
from __future__ import annotations

import pathlib

from PySide6 import QtCore, QtUiTools, QtWidgets

UI_FOLDER = pathlib.Path(__file__).parent


def load_ui_widget(ui_filename: pathlib.Path, parent=None) -> QtWidgets.QWidget:
    """Load UI Widget."""
    loader = QtUiTools.QUiLoader()
    ui_file = QtCore.QFile(str(ui_filename))
    ui_file.open(QtCore.QFile.ReadOnly)  # type: ignore[attr-defined]
    ui = loader.load(ui_file, parent)
    ui_file.close()
    return ui
