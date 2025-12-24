"""Remove pages confirmation dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtWidgets import QStyle

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent


class RemovePagesDialog(QtWidgets.QDialog):
    """Dialog to confirm removal of pages from a project.

    Shows a warning message with the number of pages to be removed
    and information about what will happen to the files.
    """

    def __init__(
        self,
        page_count: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the dialog.

        Args:
            page_count: Number of pages to be removed.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._page_count = page_count

        # Load UI
        self.ui = load_ui_widget(_UI_FOLDER / "RemovePagesDialog.ui", self)
        self._setup_widgets()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._icon_label = get_cwidget(self.ui, QtWidgets.QLabel, "icon")
        self._text_label = get_cwidget(self.ui, QtWidgets.QLabel, "text")
        self._warning_label = get_cwidget(
            self.ui, QtWidgets.QLabel, "multiPageWarning"
        )
        self._button_box = get_cwidget(
            self.ui, QtWidgets.QDialogButtonBox, "buttonBox"
        )

        # Set the question icon
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
        self._icon_label.setPixmap(icon.pixmap(48, 48))

        # Set the text
        if self._page_count == 1:
            self._text_label.setText("Remove 1 page from project?")
        else:
            self._text_label.setText(f"Remove {self._page_count} pages from project?")

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

    @staticmethod
    def confirm_removal(
        page_count: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> bool:
        """Show the dialog and return whether removal was confirmed.

        Args:
            page_count: Number of pages to be removed.
            parent: Parent widget.

        Returns:
            True if user confirmed removal, False otherwise.
        """
        dialog = RemovePagesDialog(page_count, parent)
        return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted

    def show(self) -> None:
        """Show the dialog."""
        self.ui.show()
