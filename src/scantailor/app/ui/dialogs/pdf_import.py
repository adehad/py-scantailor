"""Asks for the resolution a PDF is rendered at."""

from pathlib import Path

from PySide6 import QtWidgets

DEFAULT_IMPORT_DPI = 300
_MIN_DPI = 72
_MAX_DPI = 1200


class PdfImportDialog(QtWidgets.QDialog):
    """Small dialog that asks for the import resolution of a PDF."""

    def __init__(
        self,
        pdf_path: Path,
        default_dpi: int = DEFAULT_IMPORT_DPI,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Build the dialog.

        Args:
            pdf_path (Path): The PDF about to be imported.
            default_dpi (int): Resolution offered at first.
            parent (QtWidgets.QWidget | None): Owner of the dialog.
        """
        super().__init__(parent)
        self.setWindowTitle(self.tr("Open PDF"))
        self._dpi = QtWidgets.QSpinBox(self)
        self._dpi.setRange(_MIN_DPI, _MAX_DPI)
        self._dpi.setSuffix(" dpi")
        self._dpi.setValue(min(max(default_dpi, _MIN_DPI), _MAX_DPI))
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QtWidgets.QFormLayout(self)
        form.addRow(QtWidgets.QLabel(pdf_path.name, self))
        form.addRow(self.tr("Render pages at"), self._dpi)
        form.addRow(buttons)

    def dpi(self) -> int:
        """Return the chosen resolution."""
        return self._dpi.value()
