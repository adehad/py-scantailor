"""UI Elements & Screens."""

import pathlib
from collections.abc import Iterable

from PySide6 import QtCore, QtUiTools, QtWidgets

UI_FOLDER = pathlib.Path(__file__).parent


def load_ui_widget(
    ui_filename: pathlib.Path,
    parent: QtWidgets.QWidget | None = None,
    extra_widgets: Iterable[type[QtWidgets.QWidget]] = (),
) -> QtWidgets.QWidget:
    """Load a ``.ui`` file.

    Args:
        ui_filename (pathlib.Path): The Qt Designer file.
        parent (QtWidgets.QWidget | None): Parent of the loaded widget.
        extra_widgets (Iterable[type[QtWidgets.QWidget]]): Python widget classes the
            file promotes by class name. Without registration QUiLoader silently
            substitutes the plain base widget.

    Returns:
        QtWidgets.QWidget: The loaded top-level widget.
    """
    loader = QtUiTools.QUiLoader()
    for widget_class in extra_widgets:
        loader.registerCustomWidget(widget_class)
    ui_file = QtCore.QFile(str(ui_filename))
    ui_file.open(QtCore.QFile.ReadOnly)  # type: ignore[attr-defined]
    try:
        ui = loader.load(ui_file, parent)
    finally:
        ui_file.close()
    return ui
