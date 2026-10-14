"""Remove pages confirmation dialog."""

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtWidgets import QStyle

from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent


class RemovePagesDialog(QtWidgets.QDialog):
    """Dialog to confirm removal of pages from a project.

    Shows the number of pages to be removed and lets the user choose whether
    the output files written for them are deleted too. Input files always remain.
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
        embed_ui(self, self.ui)
        self._setup_widgets()
        self._connect_signals()

    def _setup_widgets(self) -> None:
        """Set up widget references."""
        self._icon_label = get_cwidget(self.ui, QtWidgets.QLabel, "icon")
        self._text_label = get_cwidget(self.ui, QtWidgets.QLabel, "text")
        self._warning_label = get_cwidget(self.ui, QtWidgets.QLabel, "multiPageWarning")
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")
        self._erase_check = get_cwidget(
            self.ui, QtWidgets.QCheckBox, "eraseOutputFiles"
        )
        self._output_note = self._warning_label.text()
        self._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            self.tr("Remove")
        )

        # Set the question icon
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
        self._icon_label.setPixmap(icon.pixmap(48, 48))

        # Set the text
        if self._page_count == 1:
            self._text_label.setText(self.tr("Remove 1 page from project?"))
        else:
            self._text_label.setText(
                self.tr("Remove {count} pages from project?").format(
                    count=self._page_count
                )
            )

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        self._erase_check.toggled.connect(self._show_output_note)
        self._show_output_note(self._erase_check.isChecked())

    def _show_output_note(self, erase: bool) -> None:
        self._warning_label.setText(
            self._output_note
            if erase
            else self.tr("Input and output files will remain.")
        )

    @property
    def erase_output_files(self) -> bool:
        """Whether the output files of the removed pages are to be deleted."""
        return self._erase_check.isChecked()

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
        return RemovePagesDialog.ask_removal(page_count, parent) is not None

    @staticmethod
    def ask_removal(
        page_count: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> bool | None:
        """Show the dialog and report what the user chose.

        Args:
            page_count (int): Number of pages to be removed.
            parent (QtWidgets.QWidget | None): Parent widget.

        Returns:
            bool | None: None if cancelled, otherwise whether output files are
                to be deleted as well.
        """
        dialog = RemovePagesDialog(page_count, parent)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dialog.erase_output_files
