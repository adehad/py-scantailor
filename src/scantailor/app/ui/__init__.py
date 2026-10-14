"""UI Elements & Screens."""

import pathlib
from collections.abc import Iterable

from PySide6 import QtCore, QtUiTools, QtWidgets

UI_FOLDER = pathlib.Path(__file__).parent


def register_promoted_widgets(loader: QtUiTools.QUiLoader) -> None:
    """Register the Python widgets that ``.ui`` files promote by class name.

    Without registration QUiLoader silently substitutes the plain base widget.

    Args:
        loader (QtUiTools.QUiLoader): The loader to register the classes with.
    """
    # Imported here because these modules import this package.
    from scantailor.app.ui.panels.system_load import SystemLoadWidget
    from scantailor.app.ui.widgets.relinkable_path_visualization import (
        RelinkablePathVisualization,
    )
    from scantailor.app.ui.widgets.relinking_list_view import RelinkingListView
    from scantailor.app.ui.widgets.stage_list_view import StageListView

    for widget_class in (
        StageListView,
        SystemLoadWidget,
        RelinkingListView,
        RelinkablePathVisualization,
    ):
        loader.registerCustomWidget(widget_class)


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
            file promotes by class name, beyond those of ``register_promoted_widgets``.

    Returns:
        QtWidgets.QWidget: The loaded top-level widget.
    """
    loader = QtUiTools.QUiLoader()
    register_promoted_widgets(loader)
    for widget_class in extra_widgets:
        loader.registerCustomWidget(widget_class)
    ui_file = QtCore.QFile(str(ui_filename))
    ui_file.open(QtCore.QFile.ReadOnly)  # type: ignore[attr-defined]
    try:
        ui = loader.load(ui_file, parent)
    finally:
        ui_file.close()
    return ui


def embed_ui(dialog: QtWidgets.QDialog, ui: QtWidgets.QWidget) -> None:
    """Show a loaded form inside ``dialog`` instead of as a separate window.

    A ``.ui`` whose root is a QDialog loads as a hidden top-level window, so a
    wrapper that is run with ``exec()`` would open empty. The form is demoted to a
    plain child widget filling the wrapper, which takes over its title and size.
    The form stays reachable through ``findChild`` by object name, and its own
    accept and reject end the wrapper.

    Args:
        dialog (QtWidgets.QDialog): The wrapper that is shown to the user.
        ui (QtWidgets.QWidget): The form loaded with ``dialog`` as its parent.
    """
    ui.setWindowFlags(QtCore.Qt.WindowType.Widget)
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(ui)
    dialog.setWindowTitle(ui.windowTitle())
    dialog.resize(ui.size())
    if isinstance(ui, QtWidgets.QDialog):
        ui.accepted.connect(dialog.accept)
        ui.rejected.connect(dialog.reject)
