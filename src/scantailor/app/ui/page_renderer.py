"""Background rendering of a page up to a filter stage."""

import threading

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui

from scantailor.core import pipeline
from scantailor.core.image_io import ImageLoadError
from scantailor.core.models import PageId
from scantailor.core.page_source import FALLBACK_DPI, DpiOverrides, load_page
from scantailor.core.stage_sequence import FilterStage, StageSequence

__all__ = ["FALLBACK_DPI", "PageRenderer", "render_page", "render_thumbnail"]

_MAX_THREADS = 2


def render_page(
    page_id: PageId,
    sequence: StageSequence,
    end_stage: FilterStage,
    dpi_overrides: DpiOverrides | None = None,
) -> NDArray[np.uint8]:
    """Load a page and run the pipeline up to ``end_stage``.

    Args:
        page_id (PageId): Page to render.
        sequence (StageSequence): Stages holding the page's parameters; only read.
        end_stage (FilterStage): Last stage to run.
        dpi_overrides (DpiOverrides | None): Project resolutions that win over
            the image file's.

    Returns:
        NDArray[np.uint8]: Display image, ``(H, W)`` gray or ``(H, W, 3)`` RGB.

    Raises:
        ImageLoadError: If the source image cannot be decoded.
    """
    page = load_page(page_id.image_id, dpi_overrides)
    options = pipeline.PipelineOptions(end_stage=end_stage, dpi=page.horizontal_dpi())
    image = pipeline.process_page(page.pixels, page_id, sequence, options).image
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image[:, :, ::-1])
    return image


def render_thumbnail(
    page_id: PageId,
    sequence: StageSequence,
    end_stage: FilterStage,
    max_size: QtCore.QSize,
    dpi_overrides: DpiOverrides | None = None,
) -> QtGui.QImage:
    """Render a page up to ``end_stage`` and shrink it to fit ``max_size``.

    Safe to call from worker threads; ``sequence`` is only read.

    Args:
        page_id (PageId): Page to render.
        sequence (StageSequence): Stages holding the page's parameters.
        end_stage (FilterStage): Last stage to run.
        max_size (QtCore.QSize): Bounding size of the result.
        dpi_overrides (DpiOverrides | None): Project resolutions that win over
            the image file's.

    Returns:
        QtGui.QImage: Detached image, 8-bit gray or RGB888.

    Raises:
        ImageLoadError: If the source image cannot be decoded.
    """
    image = _render(page_id, sequence, end_stage, dpi_overrides)
    height, width = image.shape[:2]
    scale = min(max_size.width() / width, max_size.height() / height, 1.0)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = np.ascontiguousarray(image)
    pixel_format = (
        QtGui.QImage.Format.Format_Grayscale8
        if image.ndim == 2
        else QtGui.QImage.Format.Format_RGB888
    )
    height, width = image.shape[:2]
    return QtGui.QImage(
        image.data, width, height, image.strides[0], pixel_format
    ).copy()


def _render(
    page_id: PageId,
    sequence: StageSequence,
    end_stage: FilterStage,
    dpi_overrides: DpiOverrides | None,
) -> NDArray[np.uint8]:
    return render_page(page_id, sequence, end_stage, dpi_overrides)


class _RenderSignals(QtCore.QObject):
    done = QtCore.Signal(int, object, object)
    failed = QtCore.Signal(int, object, str)


class _RenderTask(QtCore.QRunnable):
    """One render; a cancelled task never reports."""

    def __init__(
        self,
        generation: int,
        page_id: PageId,
        sequence: StageSequence,
        stage: FilterStage,
        dpi_overrides: DpiOverrides | None,
        signals: _RenderSignals,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.cancelled = threading.Event()
        self._generation = generation
        self._page_id = page_id
        self._sequence = sequence
        self._stage = stage
        self._dpi_overrides = dpi_overrides
        self._signals = signals

    def run(self) -> None:
        """Render on a pool thread and report unless cancelled."""
        if self.cancelled.is_set():
            return
        try:
            image = _render(
                self._page_id, self._sequence, self._stage, self._dpi_overrides
            )
        except Exception as error:
            logger.opt(exception=not isinstance(error, ImageLoadError)).warning(
                f"Cannot render {self._page_id}: {error}"
            )
            if not self.cancelled.is_set():
                self._signals.failed.emit(self._generation, self._page_id, str(error))
            return
        if not self.cancelled.is_set():
            self._signals.done.emit(self._generation, self._page_id, image)


class PageRenderer(QtCore.QObject):
    """Renders pages off the GUI thread and delivers results as signals.

    Only the latest request is reported: requesting again cancels the previous
    one, whose result is dropped even if it is already running.

    Signals:
        render_done: A render finished. Args: page_id (PageId), image (ndarray).
        render_failed: A render failed. Args: page_id (PageId), message (str).
    """

    render_done = QtCore.Signal(object, object)
    render_failed = QtCore.Signal(object, str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Create the renderer.

        Args:
            parent (QtCore.QObject | None): Owner of the renderer.
        """
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(_MAX_THREADS)
        self._signals = _RenderSignals(self)
        self._signals.done.connect(self._on_done)
        self._signals.failed.connect(self._on_failed)
        self._generation = 0
        self._task: _RenderTask | None = None

    def request(
        self,
        page_id: PageId,
        sequence: StageSequence,
        stage: FilterStage,
        dpi_overrides: DpiOverrides | None = None,
    ) -> None:
        """Start rendering a page, cancelling any render still pending.

        Args:
            page_id (PageId): Page to render.
            sequence (StageSequence): Stages holding the parameters to use.
            stage (FilterStage): Last stage to run.
            dpi_overrides (DpiOverrides | None): Snapshot of the project's
                resolutions, read on the worker thread; do not mutate it.
        """
        self.cancel()
        task = _RenderTask(
            self._generation, page_id, sequence, stage, dpi_overrides, self._signals
        )
        self._task = task
        self._pool.start(task)

    def cancel(self) -> None:
        """Drop the pending render, if any."""
        if self._task is not None:
            self._task.cancelled.set()
            self._task = None
        self._generation += 1

    def wait_for_done(self, timeout_ms: int = -1) -> bool:
        """Block until every started render has returned.

        Args:
            timeout_ms (int): Longest wait in milliseconds; negative waits forever.

        Returns:
            bool: False if the timeout expired first.
        """
        return self._pool.waitForDone(timeout_ms)

    @QtCore.Slot(int, object, object)
    def _on_done(self, generation: int, page_id: PageId, image: object) -> None:
        if generation == self._generation:
            self.render_done.emit(page_id, image)

    @QtCore.Slot(int, object, str)
    def _on_failed(self, generation: int, page_id: PageId, message: str) -> None:
        if generation == self._generation:
            self.render_failed.emit(page_id, message)
