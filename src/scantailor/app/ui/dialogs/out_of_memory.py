"""Out of memory dialog for handling OOM situations."""

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent


class OutOfMemoryDialog(QtWidgets.QDialog):
    """Dialog displayed when an out-of-memory condition is detected.

    Provides information about possible causes and solutions, and offers
    the user the ability to save their project before the application
    potentially crashes.

    Signals:
        save_project_requested: Emitted when user clicks "Save Project"
        save_project_as_requested: Emitted when user clicks "Save Project As"
    """

    save_project_requested = Signal()
    save_project_as_requested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the out-of-memory dialog.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "OutOfMemoryDialog.ui", self)
        self._setup_widgets()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._top_level_stack = get_cwidget(
            self.ui, QtWidgets.QStackedWidget, "topLevelStack"
        )
        self._tab_widget = get_cwidget(self.ui, QtWidgets.QTabWidget, "tabWidget")

        # Main page buttons
        self._save_project_btn = get_cwidget(
            self.ui, QtWidgets.QPushButton, "saveProjectBtn"
        )
        self._save_project_as_btn = get_cwidget(
            self.ui, QtWidgets.QPushButton, "saveProjectAsBtn"
        )
        self._dont_save_btn = get_cwidget(self.ui, QtWidgets.QPushButton, "dontSaveBtn")

        # Success page
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")

        # Start on the main page
        self._top_level_stack.setCurrentIndex(0)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._save_project_btn.clicked.connect(self._on_save_project)
        self._save_project_as_btn.clicked.connect(self._on_save_project_as)
        self._dont_save_btn.clicked.connect(self.reject)
        self._button_box.rejected.connect(self.reject)

    def _on_save_project(self) -> None:
        """Handle save project button click."""
        self.save_project_requested.emit()

    def _on_save_project_as(self) -> None:
        """Handle save project as button click."""
        self.save_project_as_requested.emit()

    def show_save_success(self) -> None:
        """Show the success page after project is saved."""
        self._top_level_stack.setCurrentIndex(1)

    def show(self) -> None:
        """Show the dialog."""
        self.ui.show()
