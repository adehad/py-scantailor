"""Asks which pages of a PDF to extract, and what to do with pages already there."""

import enum
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtWidgets

DEFAULT_IMPORT_DPI = 300
_MIN_DPI = 72
_MAX_DPI = 1200


@dataclass(frozen=True)
class ExtractOptions:
    """What to render from a PDF.

    Attributes:
        first_page (int): First page to render, one-based.
        last_page (int | None): Last page to render, one-based and inclusive;
            None for the last page of the PDF.
        dpi (int): Resolution to render at.
    """

    first_page: int
    last_page: int | None
    dpi: int


class ExistingPages(enum.Enum):
    """What to do with page images an earlier extraction left in the folder."""

    USE = "use"
    RENDER_AGAIN = "render_again"
    CANCEL = "cancel"


class PdfExtractDialog(QtWidgets.QDialog):
    """Asks for the page range and resolution of an extraction.

    The destination is shown but not editable: pages always go beside the PDF.
    """

    def __init__(
        self,
        pdf_path: Path,
        page_count: int,
        destination: Path,
        default_dpi: int = DEFAULT_IMPORT_DPI,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Build the dialog.

        Args:
            pdf_path (Path): The PDF whose pages are extracted.
            page_count (int): Number of pages in the PDF; at least 1.
            destination (Path): Folder the rendered pages are written to.
            default_dpi (int): Resolution offered at first.
            parent (QtWidgets.QWidget | None): Owner of the dialog.
        """
        super().__init__(parent)
        self.setWindowTitle(self.tr("Extract pages"))
        self._all_pages = QtWidgets.QRadioButton(f"All {page_count} pages", self)
        self._page_range = QtWidgets.QRadioButton(self.tr("Pages"), self)
        self._first = QtWidgets.QSpinBox(self)
        self._first.setRange(1, page_count)
        self._last = QtWidgets.QSpinBox(self)
        self._last.setRange(1, page_count)
        self._last.setValue(page_count)
        self._dpi = QtWidgets.QSpinBox(self)
        self._dpi.setRange(_MIN_DPI, _MAX_DPI)
        self._dpi.setSuffix(" dpi")
        self._dpi.setValue(min(max(default_dpi, _MIN_DPI), _MAX_DPI))
        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            self.tr("Extract")
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._all_pages.setChecked(True)
        self._all_pages.toggled.connect(self._update_enabled)
        self._first.valueChanged.connect(self._update_enabled)
        self._last.valueChanged.connect(self._update_enabled)

        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(self._page_range)
        range_row.addWidget(self._first)
        range_row.addWidget(QtWidgets.QLabel(self.tr("to"), self))
        range_row.addWidget(self._last)
        range_row.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(pdf_path.name, self))
        layout.addWidget(self._all_pages)
        layout.addLayout(range_row)
        form = QtWidgets.QFormLayout()
        form.addRow("Render at", self._dpi)
        form.addRow("Write to", QtWidgets.QLabel(str(destination), self))
        layout.addLayout(form)
        layout.addWidget(self._buttons)
        self._update_enabled()

    def _update_enabled(self) -> None:
        """Enable the page range boxes only for a range, and Extract only when valid."""
        in_range = self._page_range.isChecked()
        self._first.setEnabled(in_range)
        self._last.setEnabled(in_range)
        valid = not in_range or self._first.value() <= self._last.value()
        self._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            valid
        )

    def options(self) -> ExtractOptions:
        """Return the chosen pages and resolution.

        Returns:
            ExtractOptions: All pages, or the range; the last page is None when
            it is the last page of the PDF.
        """
        if self._all_pages.isChecked():
            return ExtractOptions(1, None, self._dpi.value())
        return ExtractOptions(
            self._first.value(), self._last.value(), self._dpi.value()
        )


def ask_existing_pages(
    folder: Path, count: int, parent: QtWidgets.QWidget
) -> ExistingPages:
    """Ask what to do when the destination already holds rendered pages.

    Args:
        folder (Path): The destination folder.
        count (int): Number of rendered pages in it.
        parent (QtWidgets.QWidget): Owner of the question.

    Returns:
        ExistingPages: The choice; CANCEL when the question is closed.
    """
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle(
        QtCore.QCoreApplication.translate("PdfExtractDialog", "Pages already extracted")
    )
    box.setText(f"{folder} already holds {count} rendered pages.")
    box.setInformativeText(
        "Use them, or render the pages again? Rendering again replaces "
        "the page images in that folder."
    )
    use = box.addButton("Use them", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    again = box.addButton(
        "Render again", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
    )
    box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
    box.exec()
    clicked = box.clickedButton()
    box.deleteLater()
    if clicked is use:
        return ExistingPages.USE
    if clicked is again:
        return ExistingPages.RENDER_AGAIN
    return ExistingPages.CANCEL
