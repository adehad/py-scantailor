"""Options of the Export PDF command."""

from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtWidgets

from scantailor.core.pdf_export import BilevelCodec

_CODECS: tuple[tuple[str, BilevelCodec], ...] = (
    ("JBIG2", "jbig2"),
    ("CCITT G4", "g4"),
)
_DEFAULT_JPEG_QUALITY = 85


@dataclass(frozen=True)
class PdfExportOptions:
    """What the user chose in ``PdfExportDialog``.

    Attributes:
        out_path: Destination PDF.
        bilevel_codec: Codec of black and white pages.
        jpeg_quality: JPEG quality of grey and colour pages.
        carry_text: Carry the source PDF's text layer onto the output.
    """

    out_path: Path
    bilevel_codec: BilevelCodec
    jpeg_quality: int
    carry_text: bool


class PdfExportDialog(QtWidgets.QDialog):
    """Asks for the codec, JPEG quality, text layer and destination of the PDF."""

    def __init__(
        self,
        default_path: Path,
        *,
        can_carry_text: bool,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Build the dialog.

        Args:
            default_path (Path): Destination offered at first.
            can_carry_text (bool): Whether the project has a source PDF whose text
                can be carried; the option is disabled otherwise.
            parent (QtWidgets.QWidget | None): Owner of the dialog.
        """
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export PDF"))
        self._codec = QtWidgets.QComboBox(self)
        for label, codec in _CODECS:
            self._codec.addItem(label, codec)
        self._quality = QtWidgets.QSpinBox(self)
        self._quality.setRange(1, 100)
        self._quality.setValue(_DEFAULT_JPEG_QUALITY)
        self._carry_text = QtWidgets.QCheckBox(self.tr("Carry text layer"), self)
        self._carry_text.setEnabled(can_carry_text)
        self._carry_text.setChecked(can_carry_text)
        self._path = QtWidgets.QLineEdit(str(default_path), self)
        browse = QtWidgets.QPushButton(self.tr("Browse..."), self)
        browse.clicked.connect(self._browse)
        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._path.textChanged.connect(self._update_ok)

        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self._path, 1)
        path_row.addWidget(browse)
        form = QtWidgets.QFormLayout(self)
        form.addRow(self.tr("Black and white pages"), self._codec)
        form.addRow(self.tr("JPEG quality"), self._quality)
        form.addRow(self._carry_text)
        form.addRow(self.tr("Output file"), path_row)
        form.addRow(self._buttons)
        self._update_ok()

    def options(self) -> PdfExportOptions:
        """Return what is currently chosen."""
        path = Path(self._path.text().strip())
        if path.suffix.lower() != ".pdf":
            path = path.with_name(path.name + ".pdf")
        return PdfExportOptions(
            out_path=path,
            bilevel_codec=self._codec.currentData(),
            jpeg_quality=self._quality.value(),
            carry_text=self._carry_text.isEnabled() and self._carry_text.isChecked(),
        )

    def _browse(self) -> None:
        chosen, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, self.tr("Export PDF"), self._path.text(), "PDF (*.pdf)"
        )
        if chosen:
            self._path.setText(chosen)

    def _update_ok(self) -> None:
        ok = self._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(bool(self._path.text().strip()))
