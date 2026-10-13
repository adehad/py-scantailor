"""Batch processing of pages through the pipeline on a process pool.

A batch first detects every page's geometry on the worker pool and stores the
results in the calling process's settings, so every page's parameters and
content size are known; it then renders and writes the pages on the same pool.
Workers receive the stage defaults and parameters as plain JSON data and rebuild
their own settings, so no lock-holding object is pickled.
"""

import multiprocessing
import os
import sys
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.dewarp_spec import DewarpSpec
from scantailor.core.image_io import tiff_compressions_in_use, write_image
from scantailor.core.models import Dpi, PageId, SubPage
from scantailor.core.page_source import DpiOverrides, load_page
from scantailor.core.pipeline import (
    PipelineOptions,
    PipelineResult,
    fix_aggregate_page_size,
    process_page,
)
from scantailor.core.project import (
    LayoutDirection,
    StageParamsEntry,
    assign_output_file_labels,
)
from scantailor.core.settings import ApplicationSettings, TiffCompressions
from scantailor.core.stage_params import (
    STAGE_KEYS,
    apply_stage_defaults,
    apply_stage_params,
    collect_stage_defaults,
    collect_stage_params,
)
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.output.generator import applied_post_deskew_angle
from scantailor.filters.output.params import DistortionModelSpec
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.output.split_output import (
    BACKGROUND_DIR,
    FOREGROUND_DIR,
    ORIGINAL_BACKGROUND_DIR,
    write_split_images,
)

DEFAULT_DPI = 300.0

StageParamsData = dict[str, list[dict[str, Any]]]
StageDefaultsData = dict[str, Any]


@dataclass(frozen=True)
class PageDetected:
    """A page's geometry was detected and stored; it is not rendered yet.

    Attributes:
        page: The measured page.
    """

    page: PageId


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
    """

    transformation: ImageTransformation
    output_dpi: Dpi
    was_dewarped: bool
    distortion_model: DistortionModelSpec | None = None
    dewarp: DewarpSpec | None = None
    post_deskew_angle: float | None = None


@dataclass(frozen=True)
class PageDone:
    """A page was rendered and written.

    Attributes:
        page: The finished page.
        path: Where its output was written.
        info: Geometry of the written page; None when not recorded.
    """

    page: PageId
    path: Path
    info: RenderInfo | None = None


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
BatchEvent = PageDetected | PageDone | PageFailed


@dataclass
class BatchResult:
    """Outcome of a batch run.

    Attributes:
        written: Output path of every page that finished.
        render_info: Geometry of every written page.
        failed: Error text of every page that failed.
        cancelled: Pages that were never started because of cancellation.
    """

    written: dict[PageId, Path] = field(default_factory=dict)
    render_info: dict[PageId, RenderInfo] = field(default_factory=dict)
    failed: dict[PageId, str] = field(default_factory=dict)
    cancelled: list[PageId] = field(default_factory=list)


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
        layout_direction (LayoutDirection): Reading direction of the project.
        file_labels (Mapping[Path, int] | None): Labels already given to source
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
        output_dir (Path): The project's output folder.
        project_pages (Sequence[PageId]): The project's pages as they were
            before the removal; labels that tell same-named source files apart
            depend on them.
        layout_direction (LayoutDirection): Reading direction, for file names.
        keep (Iterable[Path]): Files that must survive, such as the source
            images when the output folder is also the input folder.
        file_labels (Mapping[Path, int] | None): Stored labels of source files,
            as passed to ``output_file_names``.

    Returns:
        list[Path]: The files that were deleted.
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
        output_dir (Path): The project's output folder.
        pages (Sequence[PageId]): All pages of the project.
        old_direction (LayoutDirection): Reading direction the files were named
            under.
        new_direction (LayoutDirection): Reading direction to name them for.
        file_labels (Mapping[Path, int] | None): Stored labels of source files,
            as passed to ``output_file_names``.

    Returns:
        list[tuple[Path, Path]]: Old and new path of every file that was renamed.
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
    executor: ProcessPoolExecutor | None = None,
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
        executor (ProcessPoolExecutor | None): A pool from ``spawn_pool`` to
            reuse, so consecutive phases pay the worker start-up cost once;
            ``max_workers`` must then be its size. It is left running.

    Returns:
        list[Job]: Jobs that were never scheduled because of cancellation.
    """
    workers = pool_size(max_workers)
    if executor is None:
        with spawn_pool(workers, initializer, initargs) as own_pool:
            return _drain(own_pool, workers, function, jobs, on_outcome, cancel)
    return _drain(executor, workers, function, jobs, on_outcome, cancel)


def spawn_pool(
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> ProcessPoolExecutor:
    """Create a pool of spawned workers with the thread limits applied.

    Args:
        workers (int): Pool size.
        initializer (Callable[..., None] | None): Picklable callable run once in
            each worker after the thread limits are applied.
        initargs (tuple[Any, ...]): Arguments for ``initializer``.

    Returns:
        ProcessPoolExecutor: A pool to use as a context manager.
    """
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_pool_worker,
        initargs=(initializer, initargs),
    )


def _drain[Job, Result](
    executor: ProcessPoolExecutor,
    workers: int,
    function: Callable[[Job], Result],
    jobs: Sequence[Job],
    on_outcome: Callable[[Job, Result | None, BaseException | None], None],
    cancel: Event | None,
) -> list[Job]:
    queue = deque(jobs)
    in_flight: dict[Future[Result], Job] = {}
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


def collect_effective_stage_params(
    stages: StageSequence, pages: Sequence[PageId]
) -> StageParamsData:
    """Gather the parameters every page and stage would use, defaults included.

    Unlike ``collect_stage_params``, pages without explicit parameters are
    included with the values ``Settings.get_params`` falls back to, so a worker
    renders them the way the calling process would.

    Args:
        stages (StageSequence): The stages whose settings are read.
        pages (Sequence[PageId]): Pages to export.

    Returns:
        StageParamsData: Entries per stage key, JSON-compatible.
    """
    entries = collect_stage_params(stages, list(pages))
    for stage in FilterStage:
        if stage is FilterStage.FIX_ORIENTATION:
            continue
        settings: Any = stages.filter_at(stage).settings
        entries[STAGE_KEYS[stage]] = [
            StageParamsEntry(
                page=page, params=settings.get_params(page).model_dump(mode="json")
            )
            for page in pages
        ]
    return _dump_entries(entries)


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


@dataclass(frozen=True)
class _PageJob:
    page: PageId
    config: _PhaseConfig
    output_path: Path | None = None
    dpi_override: Dpi | None = None


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


def _detect_page(job: _PageJob) -> StageParamsData:
    """Detect one page's geometry in a worker; returns its parameters as data."""
    stages = _worker_stages_for(job.config)
    overrides = {job.page.image_id: job.dpi_override} if job.dpi_override else None
    loaded = load_page(job.page.image_id, overrides)
    options = PipelineOptions(dpi=loaded.horizontal_dpi(job.config.default_dpi))
    fix_aggregate_page_size([(loaded.pixels, job.page)], stages, options)
    return _dump_entries(collect_stage_params(stages, [job.page]))


def _detect_geometry(
    stages: StageSequence,
    pages: Sequence[PageId],
    default_dpi: float,
    dpi_overrides: DpiOverrides | None,
    pool: ProcessPoolExecutor,
    workers: int,
    cancel: Event | None,
    report: Callable[[BatchEvent], None],
    result: BatchResult,
) -> list[PageId]:
    """Detect every page's geometry on the pool and store it in ``stages``.

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
    )
    jobs = [
        _PageJob(page, config, dpi_override=overrides.get(page.image_id))
        for page in pages
    ]
    detected: set[PageId] = set()

    def on_outcome(
        job: _PageJob, params: StageParamsData | None, error: BaseException | None
    ) -> None:
        if params is None:
            error = error or RuntimeError("no result")
            logger.warning("Geometry pass failed for {}: {}", job.page, error)
            report(PageFailed(job.page, _describe(error)))
            return
        apply_stage_params(stages, _load_entries(params))
        detected.add(job.page)
        report(PageDetected(job.page))

    unscheduled = run_pool(
        _detect_page,
        jobs,
        on_outcome,
        max_workers=workers,
        cancel=cancel,
        executor=pool,
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


def _render_page(job: _PageJob) -> tuple[Path, RenderInfo]:
    """Render one page in a worker and write it; returns its path and geometry."""
    if job.output_path is None:
        raise ValueError("A render job needs an output path")
    stages = _worker_stages_for(job.config)
    overrides = {job.page.image_id: job.dpi_override} if job.dpi_override else None
    loaded = load_page(job.page.image_id, overrides)
    options = PipelineOptions(dpi=loaded.horizontal_dpi(job.config.default_dpi))
    rendered = process_page(loaded.pixels, job.page, stages, options)
    is_binary = bool(rendered.metadata.get("OUTPUT", {}).get("is_binary"))
    dpi = rendered.dpi or (None if loaded.dpi.is_null() else loaded.dpi)
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    compressions = job.config.tiff_compressions
    write_image(
        job.output_path, _as_writable(rendered.image, is_binary), dpi, compressions
    )
    if rendered.split is not None:
        with tiff_compressions_in_use(compressions):
            write_split_images(
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
    )
    return job.output_path, info


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
) -> BatchResult:
    """Process pages through the full pipeline and write them as TIFFs.

    Geometry detection runs per page on the worker pool and fills ``stages``
    with each page's detected parameters. Rendering then reuses the same pool,
    whose workers receive the effective parameters of every page, defaults
    included, as JSON-compatible data.

    Args:
        stages (StageSequence): Settings of the project; updated by detection.
        pages (Sequence[PageId]): Pages to process, in project order.
        output_dir (Path): Directory for the output files; created if missing.
        layout_direction (LayoutDirection): Reading direction, for file names.
        default_dpi (float): Resolution for sources that do not define one.
        max_workers (int | None): Worker processes; defaults to the CPU count.
        on_progress (Callable[[BatchEvent], None] | None): Receives a
            ``PageDetected`` per measured page, then a ``PageDone`` or
            ``PageFailed`` for every page that is attempted, from the calling
            thread.
        cancel (Event | None): Setting it stops further pages from starting;
            pages already running finish and are reported.
        dpi_overrides (DpiOverrides | None): Project resolutions that win over
            each image file's; handed to the workers as plain data.
        executor (ProcessPoolExecutor | None): A pool from ``spawn_pool`` to
            reuse across phases; ``max_workers`` must then be its size. It is
            left running. A private pool is used and closed when None.
        tiff_compressions (TiffCompressions | None): Methods for the written
            files, handed to the workers as plain data; the application settings
            as they are when the batch starts when None.
        file_labels (Mapping[Path, int] | None): Stored output file labels of the
            sources, as in ``output_file_names``.

    Returns:
        BatchResult: Written, failed and never-started pages.
    """
    result = BatchResult()
    compressions = (
        tiff_compressions or ApplicationSettings.get_instance().tiff_compressions
    )

    def report(event: BatchEvent) -> None:
        if isinstance(event, PageDone):
            result.written[event.page] = event.path
            if event.info is not None:
                result.render_info[event.page] = event.info
                if event.info.distortion_model is not None:
                    stages.output_filter.store_distortion_model(
                        event.page, event.info.distortion_model
                    )
                if event.info.post_deskew_angle is not None:
                    stages.output_filter.settings.set_post_deskew_angle(
                        event.page, event.info.post_deskew_angle
                    )
        elif isinstance(event, PageFailed):
            result.failed[event.page] = event.error
        if on_progress is not None:
            on_progress(event)

    workers = pool_size(max_workers)
    with ExitStack() as stack:
        pool = executor or stack.enter_context(spawn_pool(workers))
        ready = _detect_geometry(
            stages,
            pages,
            default_dpi,
            dpi_overrides,
            pool,
            workers,
            cancel,
            report,
            result,
        )
        names = output_file_names(pages, layout_direction, file_labels)
        overrides = dpi_overrides or {}
        config = _PhaseConfig(
            token=uuid.uuid4().hex,
            stage_params=collect_effective_stage_params(stages, ready),
            defaults=collect_stage_defaults(stages),
            default_dpi=default_dpi,
            tiff_compressions=compressions,
        )
        jobs = [
            _PageJob(
                page,
                config,
                Path(output_dir) / names[page],
                overrides.get(page.image_id),
            )
            for page in ready
        ]

        def on_outcome(
            job: _PageJob,
            rendered: tuple[Path, RenderInfo] | None,
            error: BaseException | None,
        ) -> None:
            if rendered is not None:
                report(PageDone(job.page, *rendered))
            else:
                error = error or RuntimeError("no result")
                report(PageFailed(job.page, _describe(error)))

        unscheduled = run_pool(
            _render_page,
            jobs,
            on_outcome,
            max_workers=workers,
            cancel=cancel,
            executor=pool,
        )
    result.cancelled.extend(job.page for job in unscheduled)
    return result
