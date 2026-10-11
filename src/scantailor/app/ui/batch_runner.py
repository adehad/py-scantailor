"""Runs a batch on a worker thread and reports progress as signals."""

import threading
from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from PySide6 import QtCore

from scantailor.core.batch import BatchEvent, BatchResult, PageDetected, run_batch
from scantailor.core.models import PageId
from scantailor.core.page_source import DpiOverrides
from scantailor.core.project import LayoutDirection
from scantailor.core.stage_sequence import StageSequence


class _BatchWorker(QtCore.QObject):
    detected = QtCore.Signal(object)
    progress = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        stages: StageSequence,
        pages: Sequence[PageId],
        output_dir: Path,
        layout_direction: LayoutDirection,
        max_workers: int | None,
        cancel: threading.Event,
        dpi_overrides: DpiOverrides,
    ) -> None:
        super().__init__()
        self._arguments = (stages, pages, output_dir)
        self._layout_direction = layout_direction
        self._max_workers = max_workers
        self._cancel = cancel
        self._dpi_overrides = dpi_overrides

    def _on_event(self, event: BatchEvent) -> None:
        if isinstance(event, PageDetected):
            self.detected.emit(event)
        else:
            self.progress.emit(event)

    @QtCore.Slot()
    def run(self) -> None:
        """Run the batch; called on the worker thread."""
        stages, pages, output_dir = self._arguments
        try:
            result = run_batch(
                stages,
                pages,
                output_dir,
                layout_direction=self._layout_direction,
                max_workers=self._max_workers,
                on_progress=self._on_event,
                cancel=self._cancel,
                dpi_overrides=self._dpi_overrides,
            )
        except Exception as error:
            logger.exception("Batch processing failed")
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.finished.emit(result)


class BatchRunner(QtCore.QObject):
    """Owns the worker thread of the current batch.

    While a batch runs, the stage settings belong to the worker: callers must
    not edit them until ``finished`` or ``failed`` arrives.

    Signals:
        detected: A page's geometry was measured. Args: event (PageDetected).
        progress: A page ended. Args: event (ProgressEvent).
        finished: The batch ended, possibly cancelled. Args: result (BatchResult).
        failed: The batch could not run at all. Args: message (str).
    """

    detected = QtCore.Signal(object)
    progress = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Create an idle runner.

        Args:
            parent (QtCore.QObject | None): Owner of the runner.
        """
        super().__init__(parent)
        self._thread: QtCore.QThread | None = None
        self._worker: _BatchWorker | None = None
        self._cancel = threading.Event()
        self._outcome: BatchResult | str | None = None

    @property
    def is_running(self) -> bool:
        """Whether a batch has started and its thread has not ended."""
        return self._thread is not None

    def start(
        self,
        stages: StageSequence,
        pages: Sequence[PageId],
        output_dir: Path,
        layout_direction: LayoutDirection = "LTR",
        max_workers: int | None = None,
        dpi_overrides: DpiOverrides | None = None,
    ) -> None:
        """Start a batch on a new thread.

        Args:
            stages (StageSequence): Settings of the project; updated by detection.
            pages (Sequence[PageId]): Pages to process, in order.
            output_dir (Path): Directory for the output files.
            layout_direction (LayoutDirection): Reading direction, for file names.
            max_workers (int | None): Worker processes; None for the CPU count.
            dpi_overrides (DpiOverrides | None): Snapshot of the project's
                resolutions, taken on the calling thread.

        Raises:
            RuntimeError: If a batch is already running.
        """
        if self.is_running:
            msg = "A batch is already running"
            raise RuntimeError(msg)
        self._cancel = threading.Event()
        self._outcome = None
        worker = _BatchWorker(
            stages,
            pages,
            output_dir,
            layout_direction,
            max_workers,
            self._cancel,
            dpi_overrides or {},
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.detected.connect(self.detected)
        worker.progress.connect(self.progress)
        worker.finished.connect(self._on_outcome)
        worker.failed.connect(self._on_outcome)
        # Direct, so the thread ends even while the GUI thread blocks in wait().
        direct = QtCore.Qt.ConnectionType.DirectConnection
        worker.finished.connect(thread.quit, direct)
        worker.failed.connect(thread.quit, direct)
        thread.finished.connect(self._on_thread_finished)
        self._worker, self._thread = worker, thread
        thread.start()

    def cancel(self) -> None:
        """Stop starting further pages; pages already running finish."""
        self._cancel.set()

    def wait(self, timeout_ms: int = -1) -> bool:
        """Block until the batch thread has ended and its signals were delivered.

        Args:
            timeout_ms (int): Longest wait in milliseconds; negative waits forever.

        Returns:
            bool: True if no batch is running afterwards.
        """
        thread = self._thread
        if thread is None:
            return True
        if not thread.wait(QtCore.QDeadlineTimer(timeout_ms)):
            return False
        QtCore.QCoreApplication.sendPostedEvents(self)
        return not self.is_running

    def _on_outcome(self, outcome: BatchResult | str) -> None:
        self._outcome = outcome

    def _on_thread_finished(self) -> None:
        thread, worker = self._thread, self._worker
        self._thread = self._worker = None
        outcome, self._outcome = self._outcome, None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()
        if isinstance(outcome, BatchResult):
            self.finished.emit(outcome)
        else:
            self.failed.emit(outcome or "Batch processing stopped unexpectedly")
