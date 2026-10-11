"""Batch processing of pages through the pipeline on a process pool.

A batch first runs the geometry pass in the calling process, so every page's
detected parameters and content size are known, then renders and writes the
pages in worker processes. Workers receive the stage parameters as plain JSON
data and rebuild their own settings, so no lock-holding object is pickled.
"""

import multiprocessing
import os
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, PageId, SubPage
from scantailor.core.page_source import DpiOverrides, load_page
from scantailor.core.pipeline import (
    PipelineOptions,
    fix_aggregate_page_size,
    process_page,
)
from scantailor.core.project import LayoutDirection, StageParamsEntry
from scantailor.core.stage_params import apply_stage_params, collect_stage_params
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.output.split_output import write_split_images

DEFAULT_DPI = 300.0

StageParamsData = dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class PageDone:
    """A page was rendered and written.

    Attributes:
        page: The finished page.
        path: Where its output was written.
    """

    page: PageId
    path: Path


@dataclass(frozen=True)
class PageFailed:
    """A page could not be processed; the rest of the batch continues.

    Attributes:
        page: The page that failed.
        error: Description of the failure.
    """

    page: PageId
    error: str


ProgressEvent = PageDone | PageFailed


@dataclass
class BatchResult:
    """Outcome of a batch run.

    Attributes:
        written: Output path of every page that finished.
        failed: Error text of every page that failed.
        cancelled: Pages that were never started because of cancellation.
    """

    written: dict[PageId, Path] = field(default_factory=dict)
    failed: dict[PageId, str] = field(default_factory=dict)
    cancelled: list[PageId] = field(default_factory=list)


def output_file_names(
    pages: Iterable[PageId], layout_direction: LayoutDirection = "LTR"
) -> dict[PageId, str]:
    """Name the output file of each page the way the C++ generator does.

    Source files that share a file name but live in different directories get
    ``(1)``, ``(2)``, ... in the order they first appear in ``pages``. Pages of
    a multi-page file get ``_pageNNNN`` and the halves of a split image get
    ``_1L``/``_2R``, numbered in reading order of the layout direction.

    Args:
        pages (Iterable[PageId]): Pages in project order; the order decides
            which colliding source file gets which label.
        layout_direction (LayoutDirection): Reading direction of the project.

    Returns:
        dict[PageId, str]: File name (with ``.tif`` suffix) per page.
    """
    pages = list(pages)
    labels: dict[Path, int] = {}
    files_per_name: dict[str, int] = {}
    for page in pages:
        path = page.image_id.file_path
        if path not in labels:
            labels[path] = files_per_name.get(path.name, 0)
            files_per_name[path.name] = labels[path] + 1

    left_to_right = layout_direction == "LTR"
    names: dict[PageId, str] = {}
    for page in pages:
        image_id = page.image_id
        name = image_id.file_path.stem
        if label := labels[image_id.file_path]:
            name += f"({label})"
        if image_id.is_multi_page_file():
            name += f"_page{image_id.page:04d}"
        if page.sub_page != SubPage.SINGLE_PAGE:
            is_left = page.sub_page == SubPage.LEFT_PAGE
            name += f"_{1 if left_to_right == is_left else 2}{'L' if is_left else 'R'}"
        names[page] = name + ".tif"
    return names


def configure_worker_threads() -> None:
    """Limit OpenCV and Numba to one thread in this process.

    The pool already runs one page per core, so per-process threading only
    oversubscribes the CPU. Numba reads its variable on first import, so this
    must run before anything in the process imports Numba.
    """
    os.environ["NUMBA_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)


def _init_pool_worker(
    extra: Callable[..., None] | None, extra_args: tuple[Any, ...]
) -> None:
    configure_worker_threads()
    if extra is not None:
        extra(*extra_args)


def run_pool[Job, Result](
    function: Callable[[Job], Result],
    jobs: Sequence[Job],
    on_outcome: Callable[[Job, Result | None, BaseException | None], None],
    *,
    max_workers: int | None = None,
    cancel: Event | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[Job]:
    """Run ``function`` over ``jobs`` in spawned worker processes.

    At most one job per worker is in flight, so setting ``cancel`` stops new
    jobs from being scheduled within one job's duration while running jobs
    finish and are reported. Outcomes are reported from the calling thread.

    Args:
        function (Callable[[Job], Result]): Picklable module-level callable.
        jobs (Sequence[Job]): Picklable job arguments, run in order.
        on_outcome (Callable[[Job, Result | None, BaseException | None], None]):
            Called with ``(job, result, None)`` on success and
            ``(job, None, error)`` on failure.
        max_workers (int | None): Pool size; defaults to the CPU count.
        cancel (Event | None): When set, no further jobs are scheduled.
        initializer (Callable[..., None] | None): Picklable callable run once in
            each worker after the thread limits are applied.
        initargs (tuple[Any, ...]): Arguments for ``initializer``.

    Returns:
        list[Job]: Jobs that were never scheduled because of cancellation.
    """
    workers = max_workers or os.cpu_count() or 1
    queue = deque(jobs)
    in_flight: dict[Future[Result], Job] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_pool_worker,
        initargs=(initializer, initargs),
    ) as executor:
        while True:
            while queue and len(in_flight) < workers and not _is_set(cancel):
                job = queue.popleft()
                try:
                    in_flight[executor.submit(function, job)] = job
                except BrokenProcessPool as error:
                    on_outcome(job, None, error)
            if not in_flight:
                break
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                job = in_flight.pop(future)
                error = future.exception()
                if error is None:
                    on_outcome(job, future.result(), None)
                else:
                    on_outcome(job, None, error)
    return list(queue)


def _is_set(event: Event | None) -> bool:
    return event is not None and event.is_set()


def _geometry_pass(
    pages: Sequence[PageId],
    stages: StageSequence,
    default_dpi: float,
    dpi_overrides: DpiOverrides | None,
    cancel: Event | None,
    result: BatchResult,
) -> list[PageId]:
    """Detect and store every page's geometry, one page in memory at a time.

    Pages that cannot be measured are recorded in ``result.failed`` and
    dropped; pages left over after cancellation go to ``result.cancelled``.

    Returns:
        list[PageId]: Pages that are ready to render.
    """
    ready: list[PageId] = []
    for index, page in enumerate(pages):
        if _is_set(cancel):
            result.cancelled.extend(pages[index:])
            break
        try:
            loaded = load_page(page.image_id, dpi_overrides)
            options = PipelineOptions(dpi=loaded.horizontal_dpi(default_dpi))
            fix_aggregate_page_size([(loaded.pixels, page)], stages, options)
        except Exception as error:
            logger.warning("Geometry pass failed for {}: {}", page, error)
            result.failed[page] = _describe(error)
        else:
            ready.append(page)
    return ready


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


@dataclass(frozen=True)
class _RenderJob:
    page: PageId
    output_path: Path
    dpi_override: Dpi | None = None


_worker_stages: StageSequence | None = None
_worker_default_dpi = DEFAULT_DPI


def _init_render_worker(stage_params: StageParamsData, default_dpi: float) -> None:
    global _worker_stages, _worker_default_dpi
    stages = StageSequence()
    apply_stage_params(
        stages,
        {
            key: [StageParamsEntry.model_validate(entry) for entry in entries]
            for key, entries in stage_params.items()
        },
    )
    _worker_stages = stages
    _worker_default_dpi = default_dpi


def _as_writable(image: NDArray[np.uint8], is_binary: bool) -> NDArray:
    """Convert pipeline output to what ``write_image`` accepts.

    Binary output is uint8 0 (ink) or 255 and becomes a ``True``-is-black mask.
    """
    if is_binary:
        return image == 0
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., :3]
    return image


def _render_page(job: _RenderJob) -> Path:
    """Render one page in a worker and write it; returns the written path."""
    if _worker_stages is None:
        raise RuntimeError("Render worker was not initialised")
    overrides = {job.page.image_id: job.dpi_override} if job.dpi_override else None
    loaded = load_page(job.page.image_id, overrides)
    options = PipelineOptions(dpi=loaded.horizontal_dpi(_worker_default_dpi))
    rendered = process_page(loaded.pixels, job.page, _worker_stages, options)
    is_binary = bool(rendered.metadata.get("OUTPUT", {}).get("is_binary"))
    dpi = rendered.dpi or (None if loaded.dpi.is_null() else loaded.dpi)
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(job.output_path, _as_writable(rendered.image, is_binary), dpi)
    if rendered.split is not None:
        write_split_images(
            job.output_path.parent, job.output_path.name, rendered.split, dpi
        )
    return job.output_path


def run_batch(
    stages: StageSequence,
    pages: Sequence[PageId],
    output_dir: Path,
    *,
    layout_direction: LayoutDirection = "LTR",
    default_dpi: float = DEFAULT_DPI,
    max_workers: int | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    cancel: Event | None = None,
    dpi_overrides: DpiOverrides | None = None,
) -> BatchResult:
    """Process pages through the full pipeline and write them as TIFFs.

    The geometry pass runs here and fills ``stages`` with each page's detected
    parameters, which are then handed to the workers as JSON-compatible data.

    Args:
        stages (StageSequence): Settings of the project; updated by detection.
        pages (Sequence[PageId]): Pages to process, in project order.
        output_dir (Path): Directory for the output files; created if missing.
        layout_direction (LayoutDirection): Reading direction, for file names.
        default_dpi (float): Resolution for sources that do not define one.
        max_workers (int | None): Worker processes; defaults to the CPU count.
        on_progress (Callable[[ProgressEvent], None] | None): Receives a
            ``PageDone`` or ``PageFailed`` for every page that is attempted, from
            the calling thread.
        cancel (Event | None): Setting it stops further pages from starting;
            pages already running finish and are reported.
        dpi_overrides (DpiOverrides | None): Project resolutions that win over
            each image file's; handed to the workers as plain data.

    Returns:
        BatchResult: Written, failed and never-started pages.
    """
    result = BatchResult()

    def report(event: ProgressEvent) -> None:
        if isinstance(event, PageDone):
            result.written[event.page] = event.path
        else:
            result.failed[event.page] = event.error
        if on_progress is not None:
            on_progress(event)

    ready = _geometry_pass(pages, stages, default_dpi, dpi_overrides, cancel, result)
    for page in pages:
        if page in result.failed:
            report(PageFailed(page, result.failed[page]))

    names = output_file_names(pages, layout_direction)
    overrides = dpi_overrides or {}
    jobs = [
        _RenderJob(page, Path(output_dir) / names[page], overrides.get(page.image_id))
        for page in ready
    ]
    stage_params: StageParamsData = {
        key: [entry.model_dump(mode="json") for entry in entries]
        for key, entries in collect_stage_params(stages, list(pages)).items()
    }

    def on_outcome(
        job: _RenderJob, path: Path | None, error: BaseException | None
    ) -> None:
        if path is not None:
            report(PageDone(job.page, path))
        else:
            report(PageFailed(job.page, _describe(error or RuntimeError("no result"))))

    unscheduled = run_pool(
        _render_page,
        jobs,
        on_outcome,
        max_workers=max_workers,
        cancel=cancel,
        initializer=_init_render_worker,
        initargs=(stage_params, default_dpi),
    )
    result.cancelled.extend(job.page for job in unscheduled)
    return result
