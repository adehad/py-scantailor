"""PDF import and export work that runs on a worker thread."""

import os
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium
from loguru import logger
from pydantic import BaseModel, Field
from PySide6 import QtCore

from scantailor.core.batch import pool_size, spawn_pool
from scantailor.core.models import Dpi
from scantailor.core.pdf_export import BilevelCodec, export_pdf
from scantailor.core.pdf_import import import_pdf
from scantailor.core.text_layer import TextLayer, TextLayerJob, build_text_layers
from scantailor.core.transformation import ImageTransformation

SOURCE_FILE_NAME = "pdf_source.json"
_PAGE_POLL_SECONDS = 0.1
# Spawning workers costs seconds, which a short document never earns back.
_POOL_MIN_PAGES = 8

Report = Callable[[int, int, str], None]
Work = Callable[[Report, threading.Event], object]


class JobCancelled(Exception):
    """Raised by work that noticed its cancel event and stopped."""


class PdfSource(BaseModel):
    """Where a project's page images came from, stored beside the images.

    Attributes:
        pdf_path: The PDF the pages were rendered from.
        dpi: Resolution the pages were rendered at.
        pages: Zero-based PDF page index per image file name.
    """

    pdf_path: Path
    dpi: int = Field(gt=0)
    pages: dict[str, int] = Field(default_factory=dict)

    def save(self, directory: Path) -> None:
        """Write the mapping into ``directory``."""
        (directory / SOURCE_FILE_NAME).write_text(
            self.model_dump_json(indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "PdfSource | None":
        """Read the mapping stored in ``directory``; None when absent or unreadable."""
        path = directory / SOURCE_FILE_NAME
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @classmethod
    def find(cls, image_files: Iterable[Path]) -> "PdfSource | None":
        """Return the first mapping stored beside any of ``image_files``."""
        seen: set[Path] = set()
        for file in image_files:
            if file.parent in seen:
                continue
            seen.add(file.parent)
            if (source := cls.load(file.parent)) is not None:
                return source
        return None

    def page_index(self, image_file: Path) -> int | None:
        """Return the PDF page an image file was rendered from, if it was."""
        return self.pages.get(image_file.name)

    @property
    def is_available(self) -> bool:
        """Whether the PDF still exists, so its text can be read."""
        return self.pdf_path.is_file()


@dataclass(frozen=True)
class ImportedPdf:
    """Result of ``import_pdf_pages``.

    Attributes:
        source: Mapping of the rendered images to PDF pages.
        images: The rendered image files in page order.
    """

    source: PdfSource
    images: list[Path]


def import_pdf_pages(
    pdf_path: Path, out_dir: Path, dpi: int, report: Report, cancel: threading.Event
) -> ImportedPdf:
    """Render a PDF into ``out_dir`` and record where the images came from.

    ``import_pdf`` cannot be interrupted, so progress is read from the files it
    leaves behind and a cancel takes effect once the rendering call returns, when
    its files are removed again.

    Args:
        pdf_path (Path): Source PDF.
        out_dir (Path): Directory for the page images.
        dpi (int): Rendering resolution.
        report (Report): Receives ``(done, total, text)``.
        cancel (threading.Event): Set to discard the result.

    Returns:
        ImportedPdf: The images and their mapping, which is also saved in ``out_dir``.

    Raises:
        JobCancelled: If ``cancel`` was set.
        ValueError: If the PDF is unreadable or has no pages.
    """
    total = _page_count(pdf_path)
    finished = threading.Event()

    def watch_files() -> None:
        while not finished.wait(_PAGE_POLL_SECONDS):
            report(sum(1 for _ in out_dir.glob("page_*.png")), total, "Rendering")

    watcher = threading.Thread(target=watch_files, daemon=True)
    watcher.start()
    try:
        with ExitStack() as stack:
            pool = _pool_if_worth_it(stack, total)
            images = import_pdf(pdf_path, out_dir, dpi, executor=pool)
    finally:
        finished.set()
        watcher.join()
    if cancel.is_set():
        for image in images:
            image.unlink(missing_ok=True)
        raise JobCancelled
    source = PdfSource(
        pdf_path=pdf_path,
        dpi=dpi,
        pages={image.name: index for index, image in enumerate(images)},
    )
    source.save(out_dir)
    report(len(images), len(images), "Rendering")
    return ImportedPdf(source, images)


def _page_count(pdf_path: Path) -> int:
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except pdfium.PdfiumError:
        return 0
    try:
        return len(document)
    finally:
        document.close()


@dataclass(frozen=True)
class TextRequest:
    """What is needed to carry one page's source text over.

    Attributes:
        page_index: Zero-based page of the source PDF.
        transformation: Source pixels to output pixels.
        output_dpi: Resolution of the output image.
        dewarped: True when the output was dewarped.
    """

    page_index: int
    transformation: ImageTransformation
    output_dpi: Dpi
    dewarped: bool


@dataclass(frozen=True)
class PagePlan:
    """One output page of the PDF.

    Attributes:
        image: The processed output image.
        dpi: Its resolution, horizontal and vertical.
        text: Source text to carry over, or None.
    """

    image: Path
    dpi: tuple[float, float]
    text: TextRequest | None = None


@dataclass(frozen=True)
class PdfExportRequest:
    """Everything the export needs, as plain data captured on the GUI thread.

    Attributes:
        pages: Output pages in order.
        out_path: Destination PDF.
        bilevel_codec: Codec of black and white pages.
        jpeg_quality: JPEG quality of grey and colour pages.
        source: Mapping to the source PDF; needed only for text.
    """

    pages: list[PagePlan]
    out_path: Path
    bilevel_codec: BilevelCodec = "jbig2"
    jpeg_quality: int = 85
    source: PdfSource | None = None


@dataclass
class PdfExportResult:
    """Outcome of ``export_pdf_pages``.

    Attributes:
        out_path: The written PDF.
        pages: Number of pages written.
        text_pages: Number of pages that carry a text layer.
    """

    out_path: Path
    pages: int = 0
    text_pages: int = field(default=0)


def export_pdf_pages(
    request: PdfExportRequest, report: Report, cancel: threading.Event
) -> PdfExportResult:
    """Build the text layers and write the PDF.

    The PDF is written beside its destination and moved into place only when
    the export was not cancelled, so a cancel never leaves a partial file or
    replaces an earlier one.

    Args:
        request (PdfExportRequest): The pages and options.
        report (Report): Receives ``(done, total, text)``.
        cancel (threading.Event): Set to stop; checked between pages and before
            the file is moved into place.

    Returns:
        PdfExportResult: Page and text layer counts.

    Raises:
        JobCancelled: If ``cancel`` was set.
    """
    if request.source is not None and _same_file(
        request.out_path, request.source.pdf_path
    ):
        msg = "The output would replace the source PDF; choose another file name"
        raise ValueError(msg)
    total = len(request.pages)
    report(0, total, "Reading text")
    with ExitStack() as stack:
        pool = _pool_if_worth_it(stack, total)
        layers = _text_layers_of(request, pool)
        if cancel.is_set():
            raise JobCancelled
        carries_text = any(layer is not None for layer in layers)
        report(total, total, "Writing PDF")
        _write_pdf(request, layers if carries_text else None, pool, cancel)
    logger.info("Exported {} pages to {}", total, request.out_path)
    return PdfExportResult(
        request.out_path, total, sum(layer is not None for layer in layers)
    )


def _pool_if_worth_it(stack: ExitStack, pages: int) -> ProcessPoolExecutor | None:
    if pages < _POOL_MIN_PAGES:
        return None
    return stack.enter_context(spawn_pool(pool_size(None)))


def _write_pdf(
    request: PdfExportRequest,
    layers: list[TextLayer | None] | None,
    pool: ProcessPoolExecutor | None,
    cancel: threading.Event,
) -> None:
    partial = request.out_path.with_name(request.out_path.name + ".part")
    try:
        export_pdf(
            [(page.image, page.dpi) for page in request.pages],
            partial,
            bilevel_codec=request.bilevel_codec,
            colour_quality=request.jpeg_quality,
            text_layers=layers,
            executor=pool,
        )
        if cancel.is_set():
            raise JobCancelled
        os.replace(partial, request.out_path)
    finally:
        partial.unlink(missing_ok=True)


def _text_layers_of(
    request: PdfExportRequest, pool: ProcessPoolExecutor | None
) -> list[TextLayer | None]:
    source = request.source
    jobs: list[TextLayerJob] = []
    positions: list[int] = []
    for position, page in enumerate(request.pages):
        if source is None or page.text is None:
            continue
        jobs.append(
            TextLayerJob.of(
                source.pdf_path,
                page.text.page_index,
                page.text.transformation,
                Dpi.uniform(source.dpi),
                page.text.output_dpi,
                dewarped=page.text.dewarped,
            )
        )
        positions.append(position)
    built = build_text_layers(jobs, executor=pool)
    layers: list[TextLayer | None] = [None] * len(request.pages)
    for position, layer in zip(positions, built, strict=True):
        layers[position] = layer
    return layers


def _same_file(first: Path, second: Path) -> bool:
    return os.path.normcase(first.resolve()) == os.path.normcase(second.resolve())


class _Worker(QtCore.QObject):
    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, work: Work, cancel: threading.Event) -> None:
        super().__init__()
        self._work = work
        self._cancel = cancel

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._work(self.progress.emit, self._cancel)
        except JobCancelled:
            self.cancelled.emit()
        except Exception as error:
            logger.exception("PDF job failed")
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.done.emit(result)


class PdfJob(QtCore.QObject):
    """Runs one piece of work on its own thread and reports it as signals.

    Signals:
        progress: Args: done (int), total (int), text (str).
        finished: The work returned. Args: result (object).
        cancelled: The work stopped after ``cancel``.
        failed: The work raised. Args: message (str).
    """

    progress = QtCore.Signal(int, int, str)
    finished = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Create an idle job.

        Args:
            parent (QtCore.QObject | None): Owner of the job.
        """
        super().__init__(parent)
        self._thread: QtCore.QThread | None = None
        self._worker: _Worker | None = None
        self._cancel = threading.Event()
        self._outcome: tuple[str, object] | None = None

    @property
    def is_running(self) -> bool:
        """Whether the work has started and its thread has not ended."""
        return self._thread is not None

    def start(self, work: Work) -> None:
        """Run ``work(report, cancel)`` on a new thread.

        Args:
            work (Work): Called on the worker thread; it must not touch widgets.

        Raises:
            RuntimeError: If the job is already running.
        """
        if self.is_running:
            msg = "The job is already running"
            raise RuntimeError(msg)
        self._cancel = threading.Event()
        self._outcome = None
        worker = _Worker(work, self._cancel)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress)
        worker.done.connect(lambda result: self._record("done", result))
        worker.cancelled.connect(lambda: self._record("cancelled", None))
        worker.failed.connect(lambda message: self._record("failed", message))
        # Direct, so the thread ends even while the GUI thread blocks in wait().
        direct = QtCore.Qt.ConnectionType.DirectConnection
        for signal in (worker.done, worker.cancelled, worker.failed):
            signal.connect(thread.quit, direct)
        thread.finished.connect(self._on_thread_finished)
        self._worker, self._thread = worker, thread
        thread.start()

    def cancel(self) -> None:
        """Ask the work to stop; it ends at its next check."""
        self._cancel.set()

    def wait(self, timeout_ms: int = -1) -> bool:
        """Block until the thread has ended and its outcome was delivered.

        Args:
            timeout_ms (int): Longest wait in milliseconds; negative waits forever.

        Returns:
            bool: True if the job is not running afterwards.
        """
        thread = self._thread
        if thread is None:
            return True
        if not thread.wait(QtCore.QDeadlineTimer(timeout_ms)):
            return False
        QtCore.QCoreApplication.sendPostedEvents(self)
        return not self.is_running

    def _record(self, kind: str, payload: object) -> None:
        self._outcome = (kind, payload)

    def _on_thread_finished(self) -> None:
        thread, worker = self._thread, self._worker
        self._thread = self._worker = None
        outcome, self._outcome = self._outcome, None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()
        kind, payload = outcome or ("failed", "The job stopped unexpectedly")
        if kind == "done":
            self.finished.emit(payload)
        elif kind == "cancelled":
            self.cancelled.emit()
        else:
            self.failed.emit(str(payload))
