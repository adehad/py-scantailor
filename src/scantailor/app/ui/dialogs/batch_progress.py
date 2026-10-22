"""Progress window of a running batch."""

from PySide6 import QtCore, QtWidgets

from scantailor.app.ui.panels.batch_processing import BatchProcessingLowerPanel
from scantailor.app.ui.thumbnails.model import page_label
from scantailor.core.batch import BatchResult, PageDetected, PageDone, ProgressEvent


class BatchProgressDialog(QtWidgets.QDialog):
    """Shows how far a batch is and how each page ended.

    The dialog does not stop the work itself; it asks for it through
    ``stop_requested``. Closing it while the batch runs counts as a stop request.
    Its system load slider stores the chosen page count in the settings, where
    the next batch finds it.

    Signals:
        stop_requested: The user asked to stop the batch.
        load_changed: The user chose how many pages run at once. Args: count (int).
    """

    stop_requested = QtCore.Signal()
    load_changed = QtCore.Signal(int)

    def __init__(
        self,
        total: int,
        parent: QtWidgets.QWidget | None = None,
        *,
        renders: bool = True,
    ) -> None:
        """Build the dialog.

        Args:
            total (int): Number of pages in the batch.
            parent (QtWidgets.QWidget | None): Owner of the dialog.
            renders (bool): False for a batch that stops before Output, which
                only has the detecting phase.
        """
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch processing"))
        self.setModal(False)
        self._total = total
        self._renders = renders
        self._running = True
        self._detected = 0
        self._rendered = 0
        self._rendering = False

        layout = QtWidgets.QVBoxLayout(self)
        self._status = QtWidgets.QLabel(self)
        self._bar = QtWidgets.QProgressBar(self)
        self._bar.setRange(0, max(total, 1))
        self._results = QtWidgets.QListWidget(self)
        self._panel = BatchProcessingLowerPanel(self)
        self._panel.load_changed.connect(self.load_changed)
        self._button = QtWidgets.QPushButton(self.tr("Stop"), self)
        self._button.clicked.connect(self._on_button)
        for widget in (
            self._status,
            self._bar,
            self._results,
            self._panel,
            self._button,
        ):
            layout.addWidget(widget)
        self._update_status()

    @property
    def is_running(self) -> bool:
        """Whether the batch has not finished yet."""
        return self._running

    @property
    def load(self) -> int:
        """Number of pages the system load slider lets run at once."""
        return self._panel.get_load()

    @property
    def beep_when_finished(self) -> bool:
        """Whether the user asked for a beep when the batch completes."""
        return self._panel.is_beep_enabled()

    @beep_when_finished.setter
    def beep_when_finished(self, enabled: bool) -> None:
        self._panel.set_beep_enabled(enabled)

    @property
    def reported(self) -> int:
        """Number of pages reported so far."""
        return self._results.count()

    @property
    def phase(self) -> str:
        """Either ``"detecting"`` or ``"rendering"``."""
        return "rendering" if self._rendering else "detecting"

    def result_lines(self) -> list[str]:
        """Return the per-page result lines in report order."""
        return [self._results.item(row).text() for row in range(self._results.count())]

    def add_detected(self, event: PageDetected) -> None:
        """Count a page whose geometry was measured.

        Args:
            event (PageDetected): The measured page.
        """
        self._detected += 1
        if not self._renders:
            self._bar.setValue(self._detected + self._results.count())
        elif self._detected >= self._total:
            self._rendering = True
        self._update_status()

    def add_result(self, event: ProgressEvent) -> None:
        """Record how a page ended.

        Args:
            event (ProgressEvent): The page's outcome.
        """
        name = page_label(event.page)
        if isinstance(event, PageDone):
            self._rendered += 1
            self._rendering = True
            line = f"{name}: {event.path.name}"
        else:
            line = f"{name}: {self.tr('failed')} ({event.error})"
        self._results.addItem(line)
        self._results.scrollToBottom()
        self._bar.setValue(
            self._results.count() + (0 if self._renders else self._detected)
        )
        self._update_status()

    def finish(self, result: BatchResult) -> None:
        """Show the outcome and turn the stop button into a close button.

        Args:
            result (BatchResult): The finished batch.
        """
        self._running = False
        if self._renders and result.reused:
            summary = self.tr("Done: {written} written, {kept} kept, {failed} failed")
            summary = summary.format(
                written=len(result.written) - len(result.reused),
                kept=len(result.reused),
                failed=len(result.failed),
            )
        elif self._renders:
            summary = self.tr("Done: {written} written, {failed} failed").format(
                written=len(result.written), failed=len(result.failed)
            )
        else:
            summary = self.tr("Done: {processed} processed, {failed} failed").format(
                processed=len(result.detected), failed=len(result.failed)
            )
        if result.cancelled:
            summary += self.tr(", {count} not started").format(
                count=len(result.cancelled)
            )
        self._status.setText(summary)
        self._button.setText(self.tr("Close"))
        self._button.setEnabled(True)

    def reject(self) -> None:
        """Ask for a stop while running; otherwise close."""
        if self._running:
            self._request_stop()
            return
        super().reject()

    def _on_button(self) -> None:
        if self._running:
            self._request_stop()
        else:
            self.accept()

    def _request_stop(self) -> None:
        self._button.setEnabled(False)
        self._status.setText(self.tr("Stopping..."))
        self.stop_requested.emit()

    def _update_status(self) -> None:
        if self._rendering:
            text = self.tr("Rendering {done}/{total}").format(
                done=self._rendered, total=self._total
            )
        else:
            text = self.tr("Detecting {done}/{total}").format(
                done=self._detected, total=self._total
            )
        self._status.setText(text)
