"""Runs a batch on a worker thread and reports progress as signals."""

import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import shiboken6
from loguru import logger
from PySide6 import QtCore

from scantailor.core.batch import (
    BatchEvent,
    BatchResult,
    PageDetected,
    PageLister,
    PagesListed,
    run_batch,
)
from scantailor.core.models import PageId
from scantailor.core.page_source import DpiOverrides
from scantailor.core.project import LayoutDirection, PageOutputRecord
from scantailor.core.stage_sequence import FilterStage, StageSequence


class _BatchWorker(QtCore.QObject):
    detected = QtCore.Signal(object)
    pages_listed = QtCore.Signal(object)
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
        file_labels: Mapping[Path, int] | None,
        end_stage: FilterStage,
        worker_limit: Callable[[], int | None],
        output_records: Mapping[PageId, PageOutputRecord] | None,
        page_lister: PageLister | None,
    ) -> None:
        super().__init__()
        self._arguments = (stages, pages, output_dir)
        self._layout_direction = layout_direction
        self._max_workers = max_workers
        self._cancel = cancel
        self._dpi_overrides = dpi_overrides
        self._file_labels = file_labels
        self._end_stage = end_stage
        self._worker_limit = worker_limit
        self._output_records = output_records
        self._page_lister = page_lister

    def _on_event(self, event: BatchEvent) -> None:
        if isinstance(event, PageDetected):
            self.detected.emit(event)
        elif isinstance(event, PagesListed):
            self.pages_listed.emit(event)
        else:
            self.progress.emit(event)

    @QtCore.Slot()
    def run(self) -> None:
        """Run the batch; called on the worker thread."""
        with logger.contextualize(stage="batch"):
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
                    file_labels=self._file_labels,
                    end_stage=self._end_stage,
                    worker_limit=self._worker_limit,
                    output_records=self._output_records,
                    page_lister=self._page_lister,
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
        pages_listed: Page Split changed the batch's pages. Args: event
            (PagesListed).
        progress: A page ended. Args: event (ProgressEvent).
        finished: The batch ended, possibly cancelled. Args: result (BatchResult).
        failed: The batch could not run at all. Args: message (str).
    """

    detected = QtCore.Signal(object)
    pages_listed = QtCore.Signal(object)
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
        self._worker_limit: int | None = None

    @property
    def worker_limit(self) -> int | None:
        """Pages processed at once, or None for one per worker process.

        A running batch picks up a new value the next time it starts a page.
        """
        return self._worker_limit

    @worker_limit.setter
    def worker_limit(self, count: int | None) -> None:
        self._worker_limit = count

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
        file_labels: Mapping[Path, int] | None = None,
        end_stage: FilterStage = FilterStage.OUTPUT,
        output_records: Mapping[PageId, PageOutputRecord] | None = None,
        page_lister: PageLister | None = None,
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
            file_labels (Mapping[Path, int] | None): Stored output file labels of
                the project's source files.
            end_stage (FilterStage): Last stage to run; only Output writes files.
            output_records (Mapping[PageId, PageOutputRecord] | None): Stored
                records of earlier output; pages that still match them are
                not rendered again.
            page_lister (PageLister | None): Answers the batch's Page Split
                questions, as ``run_batch`` takes it. It runs on the batch
                thread, so it must only read.

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
            file_labels,
            end_stage,
            lambda: self._worker_limit,
            output_records,
            page_lister,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.detected.connect(self.detected)
        worker.pages_listed.connect(self.pages_listed)
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
        if thread is not None:
            # finished is emitted before the thread runs its last deferred deletes
            # and exits, so a deleteLater() on the worker would race this thread
            # for the object. Once the thread has ended, delete it here instead.
            thread.wait()
            thread.deleteLater()
        if worker is not None:
            shiboken6.delete(worker)
        if isinstance(outcome, BatchResult):
            self.finished.emit(outcome)
        else:
            self.failed.emit(
                outcome or self.tr("Batch processing stopped unexpectedly")
            )
