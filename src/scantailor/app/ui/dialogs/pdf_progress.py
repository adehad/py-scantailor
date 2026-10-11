"""Progress window of a PDF import or export."""

from PySide6 import QtCore, QtWidgets


class PdfProgressDialog(QtWidgets.QDialog):
    """Shows how far a PDF job is and lets the user cancel it.

    The dialog does not stop the work itself; it asks for it through
    ``cancel_requested``. Closing it counts as a cancel request.

    Signals:
        cancel_requested: The user asked to stop the job.
    """

    cancel_requested = QtCore.Signal()

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        """Build the dialog.

        Args:
            title (str): Window title naming the job.
            parent (QtWidgets.QWidget | None): Owner of the dialog.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._cancelled = False
        self._status = QtWidgets.QLabel(self)
        self._bar = QtWidgets.QProgressBar(self)
        self._bar.setRange(0, 0)
        self._button = QtWidgets.QPushButton(self.tr("Cancel"), self)
        self._button.clicked.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        for widget in (self._status, self._bar, self._button):
            layout.addWidget(widget)

    @property
    def status_text(self) -> str:
        """The line shown above the bar."""
        return self._status.text()

    @property
    def bar_value(self) -> int:
        """Current value of the bar."""
        return self._bar.value()

    def set_progress(self, done: int, total: int, text: str) -> None:
        """Show the state of the job.

        Args:
            done (int): Units finished.
            total (int): Units in all; 0 when unknown.
            text (str): What the job is doing.
        """
        if self._cancelled:
            return
        self._bar.setRange(0, total)
        self._bar.setValue(done)
        self._status.setText(f"{text} {done}/{total}" if total else text)

    def reject(self) -> None:
        """Ask for a cancel; the owner closes the dialog once the job ended."""
        if self._cancelled:
            return
        self._cancelled = True
        self._button.setEnabled(False)
        self._status.setText(self.tr("Cancelling..."))
        self.cancel_requested.emit()

    def close_finished(self) -> None:
        """Close the dialog after the job ended."""
        self._cancelled = True
        super().reject()
