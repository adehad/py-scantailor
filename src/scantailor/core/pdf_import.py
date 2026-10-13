"""Build a project's page images from a PDF.

Pages are rasterised with pypdfium2 at a fixed resolution and stored as PNG
files carrying that resolution, so the rest of the pipeline treats them as
ordinary scans. Pages are always rendered rather than extracted at native
resolution: a scan PDF stores its image in one of several codecs (DCT, JBIG2,
CCITT, JPX) with per-page masks, and re-rendering gives one uniform path.
"""

import os
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Final

import cv2
import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.batch import pool_size
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi
from scantailor.core.pdf_pool import PdfCancelled, map_pool_progress

_POINTS_PER_INCH: Final = 72.0
# Largest per-pixel channel spread, in grey levels, still treated as neutral;
# anti-aliasing and JPEG noise on a grayscale page stay below this.
_COLOUR_SPREAD_THRESHOLD: Final = 12
_MAX_PAGES_PER_JOB: Final = 16

PageProgress = Callable[[int, int], None]


def _has_colour(bgr: NDArray[np.uint8]) -> bool:
    """Return True when any pixel's channels differ by more than the noise floor."""
    channels = bgr.astype(np.int16)
    spread = channels.max(axis=2) - channels.min(axis=2)
    return bool((spread > _COLOUR_SPREAD_THRESHOLD).any())


@dataclass(frozen=True)
class _RenderJob:
    pdf_path: str
    indices: tuple[int, ...]
    out_dir: str
    dpi: int


def _render_pages(
    job: _RenderJob, after_page: Callable[[], None] | None = None
) -> list[str]:
    """Render the pages of one job to PNG files; opens the PDF in this process.

    ``after_page`` runs after each page is written and may raise to stop.
    """
    document = pdfium.PdfDocument(job.pdf_path)
    resolution = Dpi.uniform(job.dpi)
    scale = job.dpi / _POINTS_PER_INCH
    written: list[str] = []
    try:
        for index in job.indices:
            page = document[index]
            try:
                bitmap = page.render(scale=scale, may_draw_forms=False)
                bgr = np.ascontiguousarray(bitmap.to_numpy()[..., :3])
            finally:
                page.close()
            pixels = bgr if _has_colour(bgr) else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            path = Path(job.out_dir) / f"page_{index + 1:04d}.png"
            write_image(path, pixels, resolution)
            written.append(str(path))
            if after_page is not None:
                after_page()
    finally:
        document.close()
    return written


def _page_count(pdf_path: Path) -> int:
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except pdfium.PdfiumError as error:
        raise ValueError(f"{pdf_path} is not a readable PDF with pages") from error
    try:
        return len(document)
    finally:
        document.close()


def _split_into_jobs(
    pdf_path: Path, out_dir: Path, dpi: int, page_count: int, workers: int
) -> list[_RenderJob]:
    """Cut the pages into runs small enough to balance across the workers."""
    run = max(1, min(_MAX_PAGES_PER_JOB, -(-page_count // (workers * 4))))
    return [
        _RenderJob(
            str(pdf_path),
            tuple(range(start, min(start + run, page_count))),
            str(out_dir),
            dpi,
        )
        for start in range(0, page_count, run)
    ]


def _render_all(
    pdf_path: Path,
    staging: Path,
    dpi: int,
    page_count: int,
    executor: ProcessPoolExecutor | None,
    max_workers: int | None,
    cancel: Event | None,
    on_page: PageProgress | None,
) -> list[str]:
    done = 0

    def finish_pages(count: int) -> None:
        nonlocal done
        done += count
        if on_page is not None:
            on_page(done, page_count)

    if executor is None:
        job = _RenderJob(str(pdf_path), tuple(range(page_count)), str(staging), dpi)

        def after_page() -> None:
            if cancel is not None and cancel.is_set():
                raise PdfCancelled
            finish_pages(1)

        return _render_pages(job, after_page)

    jobs = _split_into_jobs(pdf_path, staging, dpi, page_count, pool_size(max_workers))
    runs = map_pool_progress(
        _render_pages,
        jobs,
        executor,
        max_workers,
        cancel=cancel,
        on_done=lambda _index, _job, written: finish_pages(len(written)),
    )
    return [name for run in runs for name in run]


def import_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 300,
    *,
    executor: ProcessPoolExecutor | None = None,
    max_workers: int | None = None,
    cancel: Event | None = None,
    on_page: PageProgress | None = None,
) -> list[Path]:
    """Render every page of a PDF to a PNG image.

    The pages are rendered into a scratch folder beside ``out_dir`` and moved
    into it only when all of them are done, so a cancel or a failure leaves
    ``out_dir`` and any images from an earlier import exactly as they were.

    Args:
        pdf_path (Path): Source PDF.
        out_dir (Path): Directory for the images; created when missing.
        dpi (int): Rendering resolution, also stored as the PNG's DPI.
        executor (ProcessPoolExecutor | None): A pool from ``spawn_pool`` that
            renders page runs in its workers, each opening the PDF itself
            because pypdfium2 documents cannot be shared across processes.
            Pages render in this process when None.
        max_workers (int | None): Size of ``executor``.
        cancel (Event | None): Setting it stops the import. In this process it
            is checked after every page; on a pool, between page runs.
        on_page (PageProgress | None): Called as ``(pages_done, page_count)``
            from the calling thread after each page, or after each run of pages
            on a pool.

    Returns:
        list[Path]: ``page_0001.png``, ``page_0002.png``, ... in page order.
        Pages with colour content are written as BGR, all others as grayscale.

    Raises:
        ValueError: If ``dpi`` is not positive, or the file is unreadable or
            has no pages.
        PdfCancelled: If ``cancel`` was set before every page was rendered.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    if cancel is not None and cancel.is_set():
        raise PdfCancelled
    out_dir = Path(out_dir)
    page_count = _page_count(pdf_path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.importing-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        names = _render_all(
            pdf_path, staging, dpi, page_count, executor, max_workers, cancel, on_page
        )
        if cancel is not None and cancel.is_set():
            raise PdfCancelled
        out_dir.mkdir(exist_ok=True)
        paths = []
        for name in names:
            target = out_dir / Path(name).name
            os.replace(name, target)
            paths.append(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("Imported {} pages from {} at {} dpi", len(paths), pdf_path, dpi)
    return paths
