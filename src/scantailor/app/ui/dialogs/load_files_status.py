"""Load files status dialog showing results of file loading."""

from pathlib import Path

from PySide6 import QtWidgets

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent


class LoadFilesStatusDialog(QtWidgets.QDialog):
    """Dialog showing the results of loading files.

    Displays two tabs:
    - Successfully loaded files (in green)
    - Failed to load files (in red)
    """

    def __init__(
        self,
        loaded_files: list[Path] | None = None,
        failed_files: list[tuple[Path, str]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the dialog.

        Args:
            loaded_files: List of successfully loaded file paths.
            failed_files: List of tuples (path, error_message) for failed files.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._loaded_files = loaded_files or []
        self._failed_files = failed_files or []

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "LoadFilesStatusDialog.ui", self)
        self._setup_widgets()
        self._populate_lists()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._tab_widget = get_cwidget(self.ui, QtWidgets.QTabWidget, "tabWidget")
        self._loaded_text = get_cwidget(
            self.ui, QtWidgets.QPlainTextEdit, "loadedFiles"
        )
        self._failed_text = get_cwidget(
            self.ui, QtWidgets.QPlainTextEdit, "failedFiles"
        )
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")

        # Update tab titles with counts
        loaded_count = len(self._loaded_files)
        failed_count = len(self._failed_files)

        self._tab_widget.setTabText(0, f"Loaded successfully: {loaded_count}")
        self._tab_widget.setTabText(1, f"Failed to load: {failed_count}")

        # Focus on failed tab if there are failures
        if failed_count > 0:
            self._tab_widget.setCurrentIndex(1)
        else:
            self._tab_widget.setCurrentIndex(0)

    def _populate_lists(self) -> None:
        """Populate the text areas with file lists."""
        # Loaded files
        loaded_text = "\n".join(str(f) for f in self._loaded_files)
        self._loaded_text.setPlainText(loaded_text)

        # Failed files with error messages
        failed_lines = []
        for path, error in self._failed_files:
            failed_lines.append(f"{path}: {error}")
        self._failed_text.setPlainText("\n".join(failed_lines))

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

    def has_failures(self) -> bool:
        """Check if there were any loading failures.

        Returns:
            True if there are failed files.
        """
        return len(self._failed_files) > 0

    def get_loaded_count(self) -> int:
        """Get the number of successfully loaded files.

        Returns:
            Number of loaded files.
        """
        return len(self._loaded_files)

    def get_failed_count(self) -> int:
        """Get the number of failed files.

        Returns:
            Number of failed files.
        """
        return len(self._failed_files)

    @staticmethod
    def show_status(
        loaded_files: list[Path],
        failed_files: list[tuple[Path, str]],
        parent: QtWidgets.QWidget | None = None,
    ) -> bool:
        """Show the dialog and return whether user wants to continue.

        Args:
            loaded_files: List of successfully loaded file paths.
            failed_files: List of tuples (path, error_message) for failed files.
            parent: Parent widget.

        Returns:
            True if user clicked OK, False if cancelled.
        """
        dialog = LoadFilesStatusDialog(loaded_files, failed_files, parent)
        return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted

    def show(self) -> None:
        """Show the dialog."""
        self.ui.show()
