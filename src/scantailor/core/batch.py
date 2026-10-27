"""Batch processing of pages through the pipeline on a process pool.

A batch first detects every page's geometry on the worker pool and stores the
results in the calling process's settings, so every page's parameters and
content size are known; it then renders and writes the pages on the same pool.
Workers receive the stage defaults and parameters as plain JSON data and rebuild
their own settings, so no lock-holding object is pickled.
"""

import functools
import multiprocessing
import os
import statistics
import sys
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.dewarp_spec import DewarpSpec
from scantailor.core.image_io import tiff_compressions_in_use, write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage, page_name
from scantailor.core.output_reuse import build_record, is_reusable, output_params_hash
from scantailor.core.page_source import DpiOverrides, load_page
from scantailor.core.pipeline import (
    PipelineOptions,
    PipelineResult,
    fix_aggregate_page_size,
    process_page,
)
from scantailor.core.project import (
    LayoutDirection,
    PageOutputRecord,
    StageParamsEntry,
    assign_output_file_labels,
)
from scantailor.core.settings import (
    ApplicationSettings,
    BlackOnWhiteDetection,
    TiffCompressions,
)
from scantailor.core.stage_params import (
    apply_stage_defaults,
    apply_stage_params,
    collect_effective_stage_params,
    collect_stage_defaults,
    collect_stage_params,
)
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.core.worker_log import (
    WorkerLogging,
    start_worker_logging,
    worker_logging,
)
from scantailor.filters.output.generator import applied_post_deskew_angle
from scantailor.filters.output.params import AutoPictureZones, DistortionModelSpec
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.output.split_output import (
    BACKGROUND_DIR,
    FOREGROUND_DIR,
    ORIGINAL_BACKGROUND_DIR,
    write_split_images,
)
from scantailor.filters.page_layout.params import ContentSize

DEFAULT_DPI = 300.0

StageParamsData = dict[str, list[dict[str, Any]]]
StageDefaultsData = dict[str, Any]


@dataclass(frozen=True)
class PageDetected:
    """A page's geometry was detected and stored; it is not rendered yet.

    Attributes:
        page: The measured page.
        source_size: Width and height of the source image the worker decoded,
            before Fix Orientation; None when not reported.
    """

    page: PageId
    source_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class RenderInfo:
    """How a written page relates to its source, for carrying text over.

    Attributes:
        transformation: Source pixels to output pixels.
        output_dpi: Resolution of the written image.
        was_dewarped: True when the output was dewarped, which no affine
            transformation describes.
        distortion_model: Page edges that automatic or marginal dewarping
            found, to be stored in the page's parameters; None if none was built.
        dewarp: How the page was dewarped, for placing its text; None when it
            was not dewarped or no model is known.
        post_deskew_angle: The rotation the worker measured after dewarping, to
            be stored in the page's parameters; None when the page was not
            dewarped.
        auto_picture_zones: The auto picture zones the worker settled, to be
            stored in the page's parameters; None when they stayed.
        black_on_white: The polarity Output worked in, which the worker may
            have measured; None when the pipeline did not report it.
        seconds: Wall time the worker spent loading, rendering and writing
            the page.
        source_size: Width and height of the source image the worker decoded,
            before Fix Orientation; None when not reported.
    """

    transformation: ImageTransformation
    output_dpi: Dpi
    was_dewarped: bool
    distortion_model: DistortionModelSpec | None = None
    dewarp: DewarpSpec | None = None
    post_deskew_angle: float | None = None
    auto_picture_zones: AutoPictureZones | None = None
    black_on_white: bool | None = None
    seconds: float = 0.0
    source_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class PageDone:
    """A page was rendered and written, or its unchanged output was kept.

    Attributes:
        page: The finished page.
        path: Where its output is.
        info: Geometry of the written page; None when not recorded, and for a
            page whose output was kept.
        files: Every file written for the page, the output file first.
        reused: True when the page's existing output files matched its
            settings, so nothing was rendered.
    """

    page: PageId
    path: Path
    info: RenderInfo | None = None
    files: tuple[Path, ...] = ()
    reused: bool = False


@dataclass(frozen=True)
class PageFailed:
    """A page could not be processed; the rest of the batch continues.

    Attributes:
        page: The page that failed.
        error: Description of the failure.
    """

    page: PageId
    error: str


@dataclass(frozen=True)
class PagesListed:
    """Page Split changed which pages the batch carries through later stages.

    Attributes:
        pages: Every page the batch now processes past Page Split, in order.
    """

    pages: tuple[PageId, ...]


class PageLister(Protocol):
    """What a batch asks about Page Split, answered from the project."""

    def is_settled(self, image_id: ImageId) -> bool:
        """Whether Page Split would keep the image's stored layout as it is."""
        ...

    def pages_after_split(self, pages: Sequence[PageId]) -> list[PageId]:
        """List the pages the stored layouts give, in the order of ``pages``."""
        ...


ProgressEvent = PageDone | PageFailed
BatchEvent = PageDetected | PagesListed | PageDone | PageFailed


@dataclass
class BatchResult:
    """Outcome of a batch run.

    Attributes:
        detected: Pages whose geometry was measured, in the order they finished.
        written: Output path of every page that finished, including kept ones.
        reused: Pages whose existing output was kept instead of rendered.
        render_info: Geometry of every page rendered in this batch.
        output_records: What the output of every finished page was rendered
            from, to store in the project for the next batch.
        failed: Error text of every page that failed.
        cancelled: Pages that were never started because of cancellation.
        replaced: Pages that Page Split replaced with a different number of
            pages of the same image; their output files were deleted, and
            their records are for ``Project.forget_output_records``.
        source_sizes: Width and height each worker decoded an image's source
            to, before Fix Orientation, for images whose file may have been
            replaced since the project recorded its size.
    """

    detected: list[PageId] = field(default_factory=list)
    written: dict[PageId, Path] = field(default_factory=dict)
    reused: list[PageId] = field(default_factory=list)
    render_info: dict[PageId, RenderInfo] = field(default_factory=dict)
    output_records: dict[PageId, PageOutputRecord] = field(default_factory=dict)
    failed: dict[PageId, str] = field(default_factory=dict)
    cancelled: list[PageId] = field(default_factory=list)
    replaced: list[PageId] = field(default_factory=list)
    source_sizes: dict[ImageId, tuple[int, int]] = field(default_factory=dict)

    def reached_images(self) -> list[ImageId]:
        """List each image with a page the batch measured, wrote or failed.

        These are the images whose stored Page Split layout the batch may have
        changed, for ``propagation.adopt_stored_page_counts``.

        Returns:
            list[ImageId]: Each image once, measured ones first.
        """
        pages = (*self.detected, *self.written, *self.failed)
        return list(dict.fromkeys(page.image_id for page in pages))


def output_file_names(
    pages: Iterable[PageId],
    layout_direction: LayoutDirection = "LTR",
    file_labels: Mapping[Path, int] | None = None,
) -> dict[PageId, str]:
    """Name the output file of each page the way the C++ generator does.

    Source files that share a file name but live in different directories get
    ``(1)``, ``(2)``, ... in the order they first appear in ``pages``. Pages of
    a multi-page file get ``_pageNNNN`` and the halves of a split image get
    ``_1L``/``_2R``, numbered in reading order of the layout direction.

    Args:
        pages (Iterable[PageId]): Pages in project order; the order decides
            which colliding source file gets which label.
        layout_direction (str): Reading direction of the project.
        file_labels (Mapping[pathlib.Path, int] | None): Labels already given to source
            files, as stored in ``Project.output_file_labels``. They win over the
            order of ``pages``; a file without one is labelled after the others
            of its name, as the C++ FileNameDisambiguator does when it registers
            a file.

    Returns:
        dict[PageId, str]: File name (with ``.tif`` suffix) per page.
    """
    pages = list(pages)
    labels = assign_output_file_labels(
        (page.image_id.file_path for page in pages), file_labels
    )

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


_ERASE_VARIANTS = {
    SubPage.SINGLE_PAGE: (SubPage.SINGLE_PAGE, SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE),
    SubPage.LEFT_PAGE: (SubPage.SINGLE_PAGE, SubPage.LEFT_PAGE),
    SubPage.RIGHT_PAGE: (SubPage.SINGLE_PAGE, SubPage.RIGHT_PAGE),
}
_SPLIT_LAYER_DIRS = (FOREGROUND_DIR, BACKGROUND_DIR, ORIGINAL_BACKGROUND_DIR)


def erase_output_files(
    removed: Iterable[PageId],
    output_dir: Path,
    project_pages: Sequence[PageId],
    layout_direction: LayoutDirection = "LTR",
    keep: Iterable[Path] = (),
    file_labels: Mapping[Path, int] | None = None,
) -> list[Path]:
    """Delete the files the batch wrote for pages that leave the project.

    A removed page may have been written as a single page or as either half of
    a split, so each name it could have had is tried, as in C++. Split layer
    copies in the sub-folders are deleted with them. Only files named by
    ``output_file_names`` directly in ``output_dir`` (or in the layer
    sub-folders) are touched, never folders, and never a source image.

    Args:
        removed (Iterable[PageId]): Pages being removed.
        output_dir (pathlib.Path): The project's output folder.
        project_pages (Sequence[PageId]): The project's pages as they were
            before the removal; labels that tell same-named source files apart
            depend on them.
        layout_direction (str): Reading direction, for file names.
        keep (Iterable[pathlib.Path]): Files that must survive, such as the source
            images when the output folder is also the input folder.
        file_labels (Mapping[pathlib.Path, int] | None): Stored labels of source files,
            as passed to ``output_file_names``.

    Returns:
        list[pathlib.Path]: The files that were deleted.
    """
    removed = list(removed)
    variants = [
        PageId(image_id=page.image_id, sub_page=sub_page)
        for page in removed
        for sub_page in _ERASE_VARIANTS[page.sub_page]
    ]
    names = output_file_names(
        [*project_pages, *variants], layout_direction, file_labels
    )
    root = Path(os.path.abspath(output_dir))
    protected = {os.path.normcase(os.path.abspath(path)) for path in keep}
    candidates = {names[variant] for variant in variants}

    deleted: list[Path] = []
    for name in sorted(candidates):
        for folder in (root, *(root / layer for layer in _SPLIT_LAYER_DIRS)):
            path = folder / name
            if os.path.normcase(os.path.abspath(path)) in protected:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            try:
                path.unlink()
            except OSError as error:
                logger.warning("Could not delete output file {}: {}", path, error)
                continue
            deleted.append(path)
    return deleted


def rename_split_outputs(
    output_dir: Path,
    pages: Sequence[PageId],
    old_direction: LayoutDirection,
    new_direction: LayoutDirection,
    file_labels: Mapping[Path, int] | None = None,
) -> list[tuple[Path, Path]]:
    """Rename the outputs of split pages to the names a new reading order gives.

    Each half keeps its content and only changes its ``_1L``/``_2L`` or
    ``_1R``/``_2R`` number, in the output folder and in the layer sub-folders.
    A half's old and new name never coincide with another half's, so no file
    overwrites a file that is itself being renamed.

    Args:
        output_dir (pathlib.Path): The project's output folder.
        pages (Sequence[PageId]): All pages of the project.
        old_direction (str): Reading direction the files were named
            under.
        new_direction (str): Reading direction to name them for.
        file_labels (Mapping[pathlib.Path, int] | None): Stored labels of source files,
            as passed to ``output_file_names``.

    Returns:
        list[tuple[pathlib.Path, pathlib.Path]]:
            Old and new path of every file that was renamed.
            A file that cannot be renamed is logged and left where it is.
    """
    old_names = output_file_names(pages, old_direction, file_labels)
    new_names = output_file_names(pages, new_direction, file_labels)
    root = Path(os.path.abspath(output_dir))
    renamed: list[tuple[Path, Path]] = []
    for page in pages:
        if page.sub_page == SubPage.SINGLE_PAGE or old_names[page] == new_names[page]:
            continue
        for folder in (root, *(root / layer for layer in _SPLIT_LAYER_DIRS)):
            source, target = folder / old_names[page], folder / new_names[page]
            if source.is_symlink() or not source.is_file():
                continue
            try:
                os.replace(source, target)
            except OSError as error:
                logger.warning("Could not rename {} to {}: {}", source, target, error)
                continue
            renamed.append((source, target))
    return renamed


# ProcessPoolExecutor refuses more workers than WaitForMultipleObjects can wait on.
_MAX_WINDOWS_WORKERS = 61


def pool_size(max_workers: int | None) -> int:
    """Worker count to use: the request or the CPU count, within platform limits."""
    workers = max_workers or os.cpu_count() or 1
    if sys.platform == "win32":
        workers = min(workers, _MAX_WINDOWS_WORKERS)
    return workers


def configure_worker_threads() -> None:
    """Limit OpenCV to one thread in this process.

    The pool already runs one page per core, so per-process threading only
    oversubscribes the CPU.
    """
    cv2.setNumThreads(1)


def _init_pool_worker(
    worker_log: WorkerLogging | None,
    extra: Callable[..., None] | None,
    extra_args: tuple[Any, ...],
) -> None:
    start_worker_logging(worker_log)
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
    executor: ProcessPoolExecutor | None = None,
    name_job: Callable[[Job], str] | None = None,
    worker_limit: Callable[[], int | None] | None = None,
) -> list[Job]:
    """Run ``function`` over ``jobs`` in spawned worker processes.

    At most one job per worker is in flight, so setting ``cancel`` stops new
    jobs from being scheduled within one job's duration while running jobs
    finish and are reported. Outcomes are reported from the calling thread.

    Args:
        function (Callable[[Any], Any]): Picklable module-level callable.
        jobs (Sequence[Any]): Picklable job arguments, run in order.
        on_outcome (Callable[[Any, Any | None, BaseException | None], None]):
            Called with ``(job, result, None)`` on success and
            ``(job, None, error)`` on failure.
        max_workers (int | None): Pool size; defaults to the CPU count.
        cancel (threading.Event | None): When set, no further jobs are scheduled.
        initializer (Callable[..., None] | None): Picklable callable run once in
            each worker after the thread limits are applied.
        initargs (tuple[Any, ...]): Arguments for ``initializer``.
        executor (concurrent.futures.ProcessPoolExecutor | None):
            A pool from ``spawn_pool`` to
            reuse, so consecutive phases pay the worker start-up cost once;
            ``max_workers`` must then be its size. It is left running.
        name_job (Callable[[Any], str] | None): Names a job in the log line
            written when a worker process dies, which lists the jobs that were
            running; no such line is written when None.
        worker_limit (Callable[[], int | None] | None): Asked before each job is
            scheduled for how many jobs may run at once, so the limit can change
            while the pool runs. Its answer is clamped to between one and the
            pool size, and None means the pool size; so does passing None.

    Returns:
        list[Any]: Jobs that were never scheduled because of cancellation.
    """
    workers = pool_size(max_workers)

    def running_limit() -> int:
        limit = None if worker_limit is None else worker_limit()
        return workers if limit is None else min(workers, max(1, limit))

    if executor is None:
        with spawn_pool(workers, initializer, initargs) as own_pool:
            return _drain(
                own_pool, running_limit, function, jobs, on_outcome, cancel, name_job
            )
    return _drain(executor, running_limit, function, jobs, on_outcome, cancel, name_job)


def spawn_pool(
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> ProcessPoolExecutor:
    """Create a pool of spawned workers with the thread limits applied.

    The workers log into the current diagnostics session, if there is one.

    Args:
        workers (int): Pool size.
        initializer (Callable[..., None] | None): Picklable callable run once in
            each worker after the thread limits are applied.
        initargs (tuple[Any, ...]): Arguments for ``initializer``.

    Returns:
        concurrent.futures.ProcessPoolExecutor: A pool to use as a context manager.
    """
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_pool_worker,
        initargs=(worker_logging(), initializer, initargs),
    )


def _drain[Job, Result](
    executor: ProcessPoolExecutor,
    running_limit: Callable[[], int],
    function: Callable[[Job], Result],
    jobs: Sequence[Job],
    on_outcome: Callable[[Job, Result | None, BaseException | None], None],
    cancel: Event | None,
    name_job: Callable[[Job], str] | None = None,
) -> list[Job]:
    queue = deque(jobs)
    in_flight: dict[Future[Result], Job] = {}
    broken_pool_logged = False
    while True:
        while queue and len(in_flight) < running_limit() and not _is_set(cancel):
            job = queue.popleft()
            try:
                in_flight[executor.submit(function, job)] = job
            except BrokenProcessPool as error:
                on_outcome(job, None, error)
        if not in_flight:
            break
        finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        if (
            name_job is not None
            and not broken_pool_logged
            and any(_broke_the_pool(future) for future in finished)
        ):
            running = [
                name_job(job)
                for future, job in in_flight.items()
                if not future.done() or _broke_the_pool(future)
            ]
            _log_broken_pool(running, len(queue))
            broken_pool_logged = True
        for future in finished:
            job = in_flight.pop(future)
            error = future.exception()
            if error is None:
                on_outcome(job, future.result(), None)
            else:
                on_outcome(job, None, error)
    return list(queue)


def _broke_the_pool(future: Future[Any]) -> bool:
    return future.done() and isinstance(future.exception(), BrokenProcessPool)


def _log_broken_pool(running: Sequence[str], not_started: int) -> None:
    """Name the jobs a dead worker process took down with the pool.

    Every running job fails with the same error, so only the crash file and the
    last line a worker forwarded tell which job caused it.
    """
    text = (
        f"worker process died; running: {', '.join(running)}; "
        f"{not_started} more not started"
    )
    forwarded = worker_logging()
    if forwarded is not None and forwarded.folder is not None:
        text += "; see crash-worker-*.log"
    with logger.contextualize(stage="batch"):
        logger.error(text)


def _is_set(event: Event | None) -> bool:
    return event is not None and event.is_set()


def _log_page_failure(page: PageId, phase: str, error: BaseException) -> None:
    """Log a page's failure as an error, with a traceback unless the pool broke.

    A broken pool fails every running and queued page with the same
    ``BrokenProcessPool``, and the pool's own line already names the pages that
    were running, so a traceback per page would repeat one stack many times.
    """
    with logger.contextualize(page=page_name(page), stage="batch"):
        if isinstance(error, BrokenProcessPool):
            logger.error("{} failed: worker process died ({})", phase, _describe(error))
        else:
            logger.opt(exception=error).error("{} failed: {}", phase, _describe(error))


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _dump_entries(entries: dict[str, list[StageParamsEntry]]) -> StageParamsData:
    return {
        key: [entry.model_dump(mode="json") for entry in stage_entries]
        for key, stage_entries in entries.items()
    }


def _load_entries(data: StageParamsData) -> dict[str, list[StageParamsEntry]]:
    return {
        key: [StageParamsEntry.model_validate(entry) for entry in entries]
        for key, entries in data.items()
    }


@dataclass(frozen=True)
class _PhaseConfig:
    """Plain-data settings of one batch phase, shared by all of its jobs.

    Workers rebuild their stages when ``token`` differs from the one they hold,
    so a pool can serve consecutive phases with different settings.
    """

    token: str
    stage_params: StageParamsData
    defaults: StageDefaultsData
    default_dpi: float
    tiff_compressions: TiffCompressions = field(default_factory=TiffCompressions)
    end_stage: FilterStage = FilterStage.OUTPUT
    black_on_white_detection: BlackOnWhiteDetection = field(
        default_factory=BlackOnWhiteDetection
    )


@dataclass(frozen=True)
class _PageJob:
    page: PageId
    config: _PhaseConfig
    output_path: Path | None = None
    dpi_override: Dpi | None = None


def _job_page_name(job: _PageJob) -> str:
    return page_name(job.page)


_worker_token: str | None = None
_worker_stages = StageSequence()


def _worker_stages_for(config: _PhaseConfig) -> StageSequence:
    global _worker_token, _worker_stages
    if config.token != _worker_token:
        stages = StageSequence()
        apply_stage_defaults(stages, config.defaults)
        apply_stage_params(stages, _load_entries(config.stage_params))
        _worker_stages, _worker_token = stages, config.token
    return _worker_stages


def _source_size(pixels: NDArray) -> tuple[int, int]:
    height, width = pixels.shape[:2]
    return width, height


def _detect_page(job: _PageJob) -> tuple[StageParamsData, tuple[int, int]]:
    """Detect one page's geometry in a worker.

    Returns:
        tuple[StageParamsData, tuple[int, int]]: The page's parameters as data,
            and the width and height of its decoded source.
    """
    with logger.contextualize(page=page_name(job.page)):
        stages = _worker_stages_for(job.config)
        overrides = {job.page.image_id: job.dpi_override} if job.dpi_override else None
        loaded = load_page(job.page.image_id, overrides)
        options = PipelineOptions(
            dpi=loaded.resolution(job.config.default_dpi),
            end_stage=job.config.end_stage,
            black_on_white_detection=job.config.black_on_white_detection,
        )
        fix_aggregate_page_size([(loaded.pixels, job.page)], stages, options)
        params = _dump_entries(collect_stage_params(stages, [job.page]))
        return params, _source_size(loaded.pixels)


def _detect_geometry(
    stages: StageSequence,
    pages: Sequence[PageId],
    default_dpi: float,
    dpi_overrides: DpiOverrides | None,
    pool: ProcessPoolExecutor,
    workers: int,
    worker_limit: Callable[[], int | None] | None,
    cancel: Event | None,
    report: Callable[[BatchEvent], None],
    result: BatchResult,
    end_stage: FilterStage,
    black_on_white_detection: BlackOnWhiteDetection,
) -> list[PageId]:
    """Detect every page's geometry up to ``end_stage`` and store it in ``stages``.

    Pages that cannot be measured are reported as failed and dropped; pages
    never started because of cancellation go to ``result.cancelled``. The
    aggregate page size follows from the stored layout parameters of all pages,
    so it needs no separate step.

    Returns:
        list[PageId]: Pages that are ready to render, in project order.
    """
    overrides = dpi_overrides or {}
    config = _PhaseConfig(
        token=uuid.uuid4().hex,
        stage_params=_dump_entries(collect_stage_params(stages, list(pages))),
        defaults=collect_stage_defaults(stages),
        default_dpi=default_dpi,
        end_stage=end_stage,
        black_on_white_detection=black_on_white_detection,
    )
    jobs = [
        _PageJob(page, config, dpi_override=overrides.get(page.image_id))
        for page in pages
    ]
    detected: set[PageId] = set()

    def on_outcome(
        job: _PageJob,
        outcome: tuple[StageParamsData, tuple[int, int]] | None,
        error: BaseException | None,
    ) -> None:
        if outcome is None:
            error = error or RuntimeError("no result")
            _log_page_failure(job.page, "geometry pass", error)
            report(PageFailed(job.page, _describe(error)))
            return
        params, source_size = outcome
        apply_stage_params(stages, _load_entries(params))
        detected.add(job.page)
        report(PageDetected(job.page, source_size))

    unscheduled = run_pool(
        _detect_page,
        jobs,
        on_outcome,
        max_workers=workers,
        cancel=cancel,
        executor=pool,
        name_job=_job_page_name,
        worker_limit=worker_limit,
    )
    result.cancelled.extend(job.page for job in unscheduled)
    return [page for page in pages if page in detected]


def _as_writable(image: NDArray[np.uint8], is_binary: bool) -> NDArray:
    """Convert pipeline output to what ``write_image`` accepts.

    Binary output is uint8 0 (ink) or 255 and becomes a ``True``-is-black mask.
    """
    if is_binary:
        return image == 0
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., :3]
    return image


def _render_page(job: _PageJob) -> tuple[Path, RenderInfo, tuple[Path, ...]]:
    """Render one page in a worker and write it.

    Returns:
        tuple[pathlib.Path, RenderInfo, tuple[pathlib.Path, ...]]: The output
            path, the page's geometry and every file written, output first.
    """
    started = time.perf_counter()
    with logger.contextualize(page=page_name(job.page)):
        if job.output_path is None:
            raise ValueError("A render job needs an output path")
        stages = _worker_stages_for(job.config)
        overrides = {job.page.image_id: job.dpi_override} if job.dpi_override else None
        loaded = load_page(job.page.image_id, overrides)
        options = PipelineOptions(
            dpi=loaded.resolution(job.config.default_dpi),
            black_on_white_detection=job.config.black_on_white_detection,
        )
        rendered = process_page(loaded.pixels, job.page, stages, options)
        is_binary = bool(rendered.metadata.get("OUTPUT", {}).get("is_binary"))
        dpi = rendered.dpi or (None if loaded.dpi.is_null() else loaded.dpi)
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        compressions = job.config.tiff_compressions
        write_image(
            job.output_path, _as_writable(rendered.image, is_binary), dpi, compressions
        )
        written = [job.output_path]
        if rendered.split is not None:
            with tiff_compressions_in_use(compressions):
                written += write_split_images(
                    job.output_path.parent, job.output_path.name, rendered.split, dpi
                )
        if rendered.transformation is None or dpi is None:
            raise RuntimeError("The pipeline returned no output geometry")
        was_dewarped = bool(rendered.metadata.get("OUTPUT", {}).get("was_dewarped"))
        info = RenderInfo(
            rendered.transformation,
            dpi,
            was_dewarped,
            rendered.distortion_model,
            _dewarp_spec(rendered, stages.output_filter.get_params(job.page))
            if was_dewarped
            else None,
            post_deskew_angle=_measured_post_deskew_angle(rendered)
            if was_dewarped
            else None,
            auto_picture_zones=rendered.auto_picture_zones,
            black_on_white=rendered.metadata.get("OUTPUT", {}).get("black_on_white"),
            seconds=time.perf_counter() - started,
            source_size=_source_size(loaded.pixels),
        )
        return job.output_path, info, tuple(written)


def _measured_post_deskew_angle(rendered: PipelineResult) -> float | None:
    """Return the post-dewarp rotation the pipeline measured, if it reported one."""
    angle = rendered.metadata.get("OUTPUT", {}).get("post_deskew_angle")
    return None if angle is None else float(angle)


def _dewarp_spec(rendered: PipelineResult, params: OutputParams) -> DewarpSpec | None:
    """Describe the dewarp that produced a rendered page, or None if unknown."""
    model = rendered.distortion_model
    if model is None and params.distortion_model.is_defined():
        model = params.distortion_model
    if model is None:
        return None
    return DewarpSpec(
        top=model.top,
        bottom=model.bottom,
        depth_perception=params.depth_perception,
        post_deskew_angle=applied_post_deskew_angle(params),
    )


def run_batch(
    stages: StageSequence,
    pages: Sequence[PageId],
    output_dir: Path,
    *,
    layout_direction: LayoutDirection = "LTR",
    default_dpi: float = DEFAULT_DPI,
    max_workers: int | None = None,
    on_progress: Callable[[BatchEvent], None] | None = None,
    cancel: Event | None = None,
    dpi_overrides: DpiOverrides | None = None,
    executor: ProcessPoolExecutor | None = None,
    tiff_compressions: TiffCompressions | None = None,
    file_labels: Mapping[Path, int] | None = None,
    end_stage: FilterStage = FilterStage.OUTPUT,
    worker_limit: Callable[[], int | None] | None = None,
    output_records: Mapping[PageId, PageOutputRecord] | None = None,
    black_on_white_detection: BlackOnWhiteDetection | None = None,
    page_lister: PageLister | None = None,
) -> BatchResult:
    """Process pages through the pipeline up to ``end_stage``.

    With ``page_lister``, a batch that goes past Page Split first runs Fix
    Orientation and Page Split once on each image whose split is not settled,
    then asks it for the pages the stored layouts give, so a detected spread
    goes on as its two halves.
    A ``PagesListed`` event reports a list that differs from ``pages``, and
    the output files of the pages that left it are deleted.

    Geometry detection runs per page on the worker pool and fills ``stages``
    with each page's detected parameters. When ``end_stage`` is Output,
    rendering then reuses the same pool, whose workers receive the effective
    parameters of every page, defaults included, as JSON-compatible data, and
    write each page as a TIFF. A batch that ends earlier writes no files.

    A page whose ``output_records`` entry still matches its settings, its
    source file and its output files is not rendered again; it is reported as
    a ``PageDone`` with ``reused`` set.

    Args:
        stages (StageSequence): Settings of the project; updated by detection.
        pages (Sequence[PageId]): Pages to process, in project order.
        output_dir (pathlib.Path): Directory for the output files; created if missing.
        layout_direction (str): Reading direction, for file names.
        default_dpi (float): Resolution for sources that do not define one.
        max_workers (int | None): Worker processes; defaults to the CPU count.
        on_progress (Callable[[PageDetected | PageDone | PageFailed], None] | None):
            Receives a
            ``PageDetected`` per measured page, then a ``PageDone`` or
            ``PageFailed`` for every page that is attempted, from the calling
            thread.
        cancel (threading.Event | None): Setting it stops further pages from starting;
            pages already running finish and are reported.
        dpi_overrides (Mapping[ImageId, Dpi] | None): Project resolutions that win over
            each image file's; handed to the workers as plain data.
        executor (concurrent.futures.ProcessPoolExecutor | None):
            A pool from ``spawn_pool`` to
            reuse across phases; ``max_workers`` must then be its size. It is
            left running. A private pool is used and closed when None.
        tiff_compressions (TiffCompressions | None): Methods for the written
            files, handed to the workers as plain data; the application settings
            as they are when the batch starts when None.
        file_labels (Mapping[pathlib.Path, int] | None):
            Stored output file labels of the
            sources, as in ``output_file_names``.
        end_stage (FilterStage): Last stage to run. Only Output renders.
        worker_limit (Callable[[], int | None] | None): Asked before each page is
            started for how many pages may run at once, at most ``max_workers``.
            None, as the argument or the answer, means ``max_workers``.
        output_records (Mapping[PageId, PageOutputRecord] | None): Records of
            an earlier batch, as stored in ``Project.output_records``; every
            page is rendered when None.
        black_on_white_detection (BlackOnWhiteDetection | None): Whether the
            workers measure each page's polarity, handed to them as plain
            data; the application settings as they are when the batch starts
            when None.
        page_lister (PageLister | None): Tells which images need the Page
            Split pass and which pages follow it, as
            ``propagation.ProjectPageLister`` does. Called on the calling
            thread. The pages stay as given when None.

    Returns:
        BatchResult: Written, failed and never-started pages.
    """
    result = BatchResult()
    compressions = (
        tiff_compressions or ApplicationSettings.get_instance().tiff_compressions
    )
    polarity = (
        black_on_white_detection
        or ApplicationSettings.get_instance().polarity_detection
    )

    # Every hash comes after geometry detection, and nothing later changes the
    # layout, so one snapshot of the shared page size serves the whole batch.
    @functools.cache
    def aggregate_mm() -> ContentSize:
        return stages.page_layout_filter.settings.get_aggregate_hard_size_mm()

    def params_hash(page: PageId) -> str:
        return output_params_hash(
            stages,
            page,
            dpi_override=(dpi_overrides or {}).get(page.image_id),
            default_dpi=default_dpi,
            tiff_compressions=compressions,
            aggregate_mm=aggregate_mm(),
        )

    out_dir = Path(output_dir)

    def report(event: BatchEvent) -> None:
        if isinstance(event, PageDone):
            result.written[event.page] = event.path
            if event.reused:
                result.reused.append(event.page)
            if event.info is not None:
                result.render_info[event.page] = event.info
                if event.info.source_size is not None:
                    result.source_sizes[event.page.image_id] = event.info.source_size
                if event.info.distortion_model is not None:
                    stages.output_filter.store_distortion_model(
                        event.page, event.info.distortion_model
                    )
                if event.info.post_deskew_angle is not None:
                    stages.output_filter.settings.set_post_deskew_angle(
                        event.page, event.info.post_deskew_angle
                    )
                if event.info.auto_picture_zones is not None:
                    stages.output_filter.settings.replace_auto_picture_zones(
                        event.page, event.info.auto_picture_zones
                    )
                if event.info.black_on_white is not None:
                    stages.output_filter.settings.set_detected_black_on_white(
                        event.page, event.info.black_on_white
                    )
            if not event.reused:
                _record_output(result, event, params_hash(event.page), out_dir)
        elif isinstance(event, PageDetected):
            result.detected.append(event.page)
            if event.source_size is not None:
                result.source_sizes[event.page.image_id] = event.source_size
        elif isinstance(event, PageFailed):
            result.failed[event.page] = event.error
        if on_progress is not None:
            on_progress(event)

    workers = pool_size(max_workers)
    started = time.perf_counter()
    with logger.contextualize(stage="batch"):
        logger.info(
            "batch start: {} pages to {}, {} workers",
            len(pages),
            end_stage.name.lower(),
            workers,
        )
    with ExitStack() as stack:
        pool = executor or stack.enter_context(spawn_pool(workers))

        def detect(
            batch_pages: Sequence[PageId],
            last_stage: FilterStage,
            sink: Callable[[BatchEvent], None],
        ) -> list[PageId]:
            return _detect_geometry(
                stages,
                batch_pages,
                default_dpi,
                dpi_overrides,
                pool,
                workers,
                worker_limit,
                cancel,
                sink,
                result,
                last_stage,
                polarity,
            )

        # Labels follow the pages as given, so a same-named source that the
        # Page Split pass drops does not move the labels of the others.
        labels = assign_output_file_labels(
            (page.image_id.file_path for page in pages), file_labels
        )
        if page_lister is not None and end_stage > FilterStage.PAGE_SPLIT:
            given = pages
            pages = _split_images(pages, page_lister, detect, report, result)
            if result.replaced:
                _erase_replaced_outputs(
                    result.replaced, given, pages, out_dir, layout_direction, labels
                )
        ready = detect(pages, end_stage, report)
        if end_stage == FilterStage.OUTPUT:
            names = output_file_names(pages, layout_direction, labels)
            ready = _keep_unchanged_outputs(
                ready, output_records or {}, names, out_dir, params_hash, report, result
            )
            # The workers hold only the pages they render and would size them
            # against each other alone; held at the batch's page size, a page
            # rendered again matches the pages that were kept.
            defaults = collect_stage_defaults(stages)
            defaults["page_layout_frozen_aggregate_size"] = aggregate_mm().model_dump(
                mode="json"
            )
            _render_ready_pages(
                ready,
                _PhaseConfig(
                    token=uuid.uuid4().hex,
                    stage_params=collect_effective_stage_params(stages, ready),
                    defaults=defaults,
                    default_dpi=default_dpi,
                    tiff_compressions=compressions,
                    black_on_white_detection=polarity,
                ),
                names,
                out_dir,
                dpi_overrides or {},
                pool,
                workers,
                worker_limit,
                cancel,
                report,
                result,
            )
    with logger.contextualize(stage="batch"):
        logger.info(
            "batch end: {}", _batch_summary(result, time.perf_counter() - started)
        )
    return result


def _split_images(
    pages: Sequence[PageId],
    page_lister: PageLister,
    detect: Callable[
        [Sequence[PageId], FilterStage, Callable[[BatchEvent], None]], list[PageId]
    ],
    report: Callable[[BatchEvent], None],
    result: BatchResult,
) -> list[PageId]:
    """Run Page Split on each unsettled image and list the pages that follow.

    One page of each image whose split is not settled is decoded; settled
    images are not. Only failures are reported from this pass, once for each
    of the failed image's pages; the full detection pass reports every page
    it measures.
    An image never started goes to ``result.cancelled`` with all its pages.

    Returns:
        list[PageId]: The pages the later stages process, in order.
    """
    pages_of: dict[ImageId, list[PageId]] = {}
    for page in pages:
        pages_of.setdefault(page.image_id, []).append(page)

    def report_failures(event: BatchEvent) -> None:
        if isinstance(event, PageFailed):
            for page in pages_of[event.page.image_id]:
                report(PageFailed(page, event.error))

    unsettled = [
        image_id for image_id in pages_of if not page_lister.is_settled(image_id)
    ]
    first_cancelled = len(result.cancelled)
    split = detect(
        [pages_of[image_id][0] for image_id in unsettled],
        FilterStage.PAGE_SPLIT,
        report_failures,
    )
    result.cancelled[first_cancelled:] = [
        page
        for unstarted in result.cancelled[first_cancelled:]
        for page in pages_of[unstarted.image_id]
    ]
    dropped = set(unsettled) - {page.image_id for page in split}
    kept = [page for page in pages if page.image_id not in dropped]
    listed = page_lister.pages_after_split(kept)
    if listed != kept:
        still_listed = set(listed)
        result.replaced = [page for page in kept if page not in still_listed]
        report(PagesListed(tuple(listed)))
    return listed


def _erase_replaced_outputs(
    replaced: Sequence[PageId],
    given: Sequence[PageId],
    pages: Sequence[PageId],
    output_dir: Path,
    layout_direction: LayoutDirection,
    file_labels: Mapping[Path, int] | None,
) -> None:
    """Delete the output files of pages Page Split replaced, sparing the new pages'.

    ``erase_output_files`` tries every name a removed page could have had,
    which includes the names of its image's new pages, so those files and
    the source images are passed as ``keep``. ``given`` is the batch's page
    list before Page Split and ``pages`` the list after it.
    """
    new_names = output_file_names(pages, layout_direction, file_labels).values()
    folders = (output_dir, *(output_dir / layer for layer in _SPLIT_LAYER_DIRS))
    keep = [folder / name for folder in folders for name in new_names]
    keep += [page.image_id.file_path for page in given]
    deleted = erase_output_files(
        replaced,
        output_dir,
        given,
        layout_direction,
        keep=keep,
        file_labels=file_labels,
    )
    for path in deleted:
        logger.info("deleted {}: Page Split replaced its page", path)


def _keep_unchanged_outputs(
    ready: Sequence[PageId],
    records: Mapping[PageId, PageOutputRecord],
    names: Mapping[PageId, str],
    output_dir: Path,
    params_hash: Callable[[PageId], str],
    report: Callable[[BatchEvent], None],
    result: BatchResult,
) -> list[PageId]:
    """Report the pages whose output files can be kept; return the rest.

    Returns:
        list[PageId]: Pages that need rendering, in project order.
    """
    to_render: list[PageId] = []
    for page in ready:
        record = records.get(page)
        if record is None or not is_reusable(
            record, params_hash(page), output_dir, names[page]
        ):
            to_render.append(page)
            continue
        result.output_records[page] = record
        files = tuple(output_dir / key for key in record.files)
        report(PageDone(page, output_dir / names[page], files=files, reused=True))
    return to_render


def _record_output(
    result: BatchResult, done: PageDone, params_hash: str, output_dir: Path
) -> None:
    """Remember what a rendered page's files came from, or forget a stale record."""
    record = build_record(done.page, params_hash, output_dir, done.files)
    if record is None:
        result.output_records.pop(done.page, None)
    else:
        result.output_records[done.page] = record


def _render_ready_pages(
    ready: Sequence[PageId],
    config: _PhaseConfig,
    names: Mapping[PageId, str],
    output_dir: Path,
    dpi_overrides: DpiOverrides,
    pool: ProcessPoolExecutor,
    workers: int,
    worker_limit: Callable[[], int | None] | None,
    cancel: Event | None,
    report: Callable[[BatchEvent], None],
    result: BatchResult,
) -> None:
    """Render the measured pages on the pool and write each one as a TIFF.

    Args:
        ready (Sequence[PageId]): Pages to render, in project order.
        config (_PhaseConfig): Plain-data settings handed to the workers.
        names (Mapping[PageId, str]): Output file name of every page.
        output_dir (pathlib.Path): Directory for the output files.
        dpi_overrides (DpiOverrides): Project resolutions of the images.
        pool (concurrent.futures.ProcessPoolExecutor): Workers to render on.
        workers (int): Size of ``pool``.
        worker_limit (Callable[[], int | None] | None): Live limit on the pages
            rendered at once, as in ``run_pool``.
        cancel (threading.Event | None): Stops further pages from starting.
        report (Callable[[BatchEvent], None]): Receives a ``PageDone`` or
            ``PageFailed`` per attempted page.
        result (BatchResult): Gets the pages never started.
    """
    jobs = [
        _PageJob(
            page, config, output_dir / names[page], dpi_overrides.get(page.image_id)
        )
        for page in ready
    ]

    def on_outcome(
        job: _PageJob,
        rendered: tuple[Path, RenderInfo, tuple[Path, ...]] | None,
        error: BaseException | None,
    ) -> None:
        if rendered is not None:
            path, info, files = rendered
            report(PageDone(job.page, path, info, files))
        else:
            error = error or RuntimeError("no result")
            _log_page_failure(job.page, "render", error)
            report(PageFailed(job.page, _describe(error)))

    unscheduled = run_pool(
        _render_page,
        jobs,
        on_outcome,
        max_workers=workers,
        cancel=cancel,
        executor=pool,
        name_job=_job_page_name,
        worker_limit=worker_limit,
    )
    result.cancelled.extend(job.page for job in unscheduled)


def _batch_summary(result: BatchResult, seconds: float) -> str:
    """Count a batch's outcomes and name its slowest page, for the log."""
    kept = f"{len(result.reused)} kept, " if result.reused else ""
    text = (
        f"{len(result.detected)} detected, "
        f"{len(result.written) - len(result.reused)} written, {kept}"
        f"{len(result.failed)} failed, "
        f"{len(result.cancelled)} cancelled in {seconds:.1f} s"
    )
    timings = {page: info.seconds for page, info in result.render_info.items()}
    if timings:
        slowest = max(timings, key=timings.__getitem__)
        text += (
            f" (median page {statistics.median(timings.values()):.1f} s, "
            f"slowest {page_name(slowest)} {timings[slowest]:.1f} s)"
        )
    return text
