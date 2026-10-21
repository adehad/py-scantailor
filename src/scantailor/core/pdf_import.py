"""Build a project's page images from a PDF.

Pages are rasterised with pypdfium2 at a fixed resolution and stored as PNG
files carrying that resolution, so the rest of the pipeline treats them as
ordinary scans. Pages are always rendered rather than extracted at native
resolution: a scan PDF stores its image in one of several codecs (DCT, JBIG2,
CCITT, JPX) with per-page masks, and re-rendering gives one uniform path.
"""

import os
import re
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
_RENDERED_NAME: Final = re.compile(r"page_\d{4,}\.png")

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


def page_count(pdf_path: Path) -> int:
    """Return the number of pages of a PDF.

    Args:
        pdf_path (pathlib.Path): The PDF to open.

    Returns:
        int: Number of pages.

    Raises:
        ValueError: If the file is not a readable PDF with pages.
    """
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except pdfium.PdfiumError as error:
        raise ValueError(f"{pdf_path} is not a readable PDF with pages") from error
    try:
        return len(document)
    finally:
        document.close()


def _page_indices(total: int, first_page: int, last_page: int | None) -> range:
    """Turn a one-based inclusive page range into zero-based indices."""
    last = total if last_page is None else last_page
    if not 1 <= first_page <= last <= total:
        msg = f"pages {first_page} to {last} are outside the {total} pages of the PDF"
        raise ValueError(msg)
    return range(first_page - 1, last)


def _split_into_jobs(
    pdf_path: Path, out_dir: Path, dpi: int, indices: range, workers: int
) -> list[_RenderJob]:
    """Cut the pages into runs small enough to balance across the workers."""
    count = len(indices)
    run = max(1, min(_MAX_PAGES_PER_JOB, -(-count // (workers * 4))))
    return [
        _RenderJob(
            str(pdf_path),
            tuple(indices[start : start + run]),
            str(out_dir),
            dpi,
        )
        for start in range(0, count, run)
    ]


def _name_pages(indices: tuple[int, ...]) -> str:
    """Name a run of zero-based page indices as one-based PDF pages."""
    if len(indices) == 1:
        return f"PDF page {indices[0] + 1}"
    return f"PDF pages {indices[0] + 1}-{indices[-1] + 1}"


def _render_all(
    pdf_path: Path,
    staging: Path,
    dpi: int,
    indices: range,
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
            on_page(done, len(indices))

    if executor is None:
        job = _RenderJob(str(pdf_path), tuple(indices), str(staging), dpi)

        def after_page() -> None:
            if cancel is not None and cancel.is_set():
                raise PdfCancelled
            finish_pages(1)

        return _render_pages(job, after_page)

    jobs = _split_into_jobs(pdf_path, staging, dpi, indices, pool_size(max_workers))
    runs = map_pool_progress(
        _render_pages,
        jobs,
        executor,
        max_workers,
        cancel=cancel,
        on_done=lambda _index, _job, written: finish_pages(len(written)),
        name_job=lambda _index, job: _name_pages(job.indices),
    )
    return [name for run in runs for name in run]


def rendered_pages(out_dir: Path) -> list[Path]:
    """List the ``page_NNNN.png`` images an earlier import left in a folder.

    Args:
        out_dir (pathlib.Path): Folder that may hold rendered pages.

    Returns:
        list[pathlib.Path]: The rendered pages in name order; empty when the
            folder is missing.
    """
    if not out_dir.is_dir():
        return []
    return sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file() and _RENDERED_NAME.fullmatch(path.name)
    )


def clear_rendered_pages(out_dir: Path) -> None:
    """Delete the rendered page images in a folder, leaving other files alone.

    Args:
        out_dir (pathlib.Path): Folder holding earlier rendered pages.
    """
    for path in rendered_pages(out_dir):
        path.unlink()


def import_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 300,
    *,
    first_page: int = 1,
    last_page: int | None = None,
    clear_existing: bool = False,
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
        pdf_path (pathlib.Path): Source PDF.
        out_dir (pathlib.Path): Directory for the images; created when missing.
        dpi (int): Rendering resolution, also stored as the PNG's DPI.
        first_page (int): First page to render, one-based.
        last_page (int | None): Last page to render, one-based and inclusive;
            None renders to the end. Each image keeps its PDF page number in its name.
        clear_existing (bool): Delete the ``page_NNNN.png`` images already in
            ``out_dir`` just before the new ones are moved in, so a shorter range
            leaves no stale pages. Other files are kept.
        executor (concurrent.futures.ProcessPoolExecutor | None):
            A pool from ``spawn_pool`` that
            renders page runs in its workers, each opening the PDF itself
            because pypdfium2 documents cannot be shared across processes.
            Pages render in this process when None.
        max_workers (int | None): Size of ``executor``.
        cancel (threading.Event | None): Setting it stops the import. In this process it
            is checked after every page; on a pool, between page runs.
        on_page (Callable[[int, int], None] | None):
            Called as ``(pages_done, page_count)``
            from the calling thread after each page, or after each run of pages
            on a pool.

    Returns:
        list[pathlib.Path]: ``page_0001.png``, ``page_0002.png``, ... in page order.
        Pages with colour content are written as BGR, all others as grayscale.

    Raises:
        ValueError: If ``dpi`` is not positive, the file is unreadable or
            has no pages, or the page range is outside the document.
        PdfCancelled: If ``cancel`` was set before every page was rendered.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    if cancel is not None and cancel.is_set():
        raise PdfCancelled
    out_dir = Path(out_dir)
    indices = _page_indices(page_count(pdf_path), first_page, last_page)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.importing-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        names = _render_all(
            pdf_path, staging, dpi, indices, executor, max_workers, cancel, on_page
        )
        if cancel is not None and cancel.is_set():
            raise PdfCancelled
        if clear_existing:
            clear_rendered_pages(out_dir)
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
