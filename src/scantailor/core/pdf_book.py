"""Rebuild a scanned PDF: import its pages, process them, export a new PDF.

The source PDF's text layer, when it has one, is carried onto the processed
pages as invisible text, and its bookmarks and page labels when every page
was exported.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from loguru import logger

from scantailor.core.batch import (
    BatchEvent,
    BatchResult,
    PageDetected,
    PageDone,
    pool_size,
    run_batch,
    spawn_pool,
)
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.pdf_export import BilevelCodec, export_pdf
from scantailor.core.pdf_import import import_pdf
from scantailor.core.stage_sequence import StageSequence
from scantailor.core.text_layer import TextLayer, TextLayerJob, build_text_layers


@dataclass
class BookResult:
    """Outcome of ``rebuild_pdf``.

    Attributes:
        out_path: The written PDF.
        pages_exported: Number of pages in it.
        failed: Error text per source page index that could not be processed
            and is missing from the output.
        text_pages: Number of output pages that carry a text layer.
        seconds: Wall time per phase: ``import``, ``detect`` (includes worker
            start-up), ``render``, ``text`` and ``export``.
    """

    out_path: Path
    pages_exported: int = 0
    failed: dict[int, str] = field(default_factory=dict)
    text_pages: int = 0
    seconds: dict[str, float] = field(default_factory=dict)


def rebuild_pdf(
    pdf_path: Path,
    out_path: Path,
    work_dir: Path,
    *,
    dpi: int = 300,
    output_dpi: int | None = None,
    bilevel_codec: BilevelCodec = "jbig2",
    carry_text: bool = True,
    max_workers: int | None = None,
    stages: StageSequence | None = None,
    on_progress: Callable[[BatchEvent], None] | None = None,
    cancel: Event | None = None,
) -> BookResult:
    """Import ``pdf_path``, run every page through the pipeline and export a PDF.

    Args:
        pdf_path (pathlib.Path): Source PDF.
        out_path (pathlib.Path): Destination PDF.
        work_dir (pathlib.Path): Scratch directory for the imported and processed pages;
            kept so the intermediate files can be inspected.
        dpi (int): Resolution the source pages are rendered at.
        output_dpi (int | None): Output resolution; ``dpi`` when None. Ignored
            when ``stages`` is given.
        bilevel_codec (str): Codec of black and white pages.
        carry_text (bool): Carry the source text layer onto the output.
        max_workers (int | None): Worker processes of the batch.
        stages (StageSequence | None): Project settings; defaults otherwise.
        on_progress (Callable[[PageDetected | PageDone | PageFailed], None] | None):
            Batch events.
        cancel (threading.Event | None): Stops the batch from starting further pages.

    Returns:
        BookResult: Timings, failures and the output location. Pages that
        failed are absent from the PDF. No PDF is written, and an existing one
        is left alone, when the batch was cancelled or no page was written.
    """
    result = BookResult(out_path)
    clock = _PhaseClock(result.seconds)
    workers = pool_size(max_workers)

    with spawn_pool(workers) as pool:
        clock.start("import")
        images = import_pdf(
            pdf_path, work_dir / "source", dpi, executor=pool, max_workers=workers
        )
        pages = [PageId(image_id=ImageId(file_path=path)) for path in images]
        index_of = {page: index for index, page in enumerate(pages)}

        if stages is None:
            stages = StageSequence()
            stages.output_filter.settings.set_default_dpi(
                Dpi.uniform(output_dpi or dpi)
            )

        def forward(event: BatchEvent) -> None:
            if isinstance(event, PageDetected):
                clock.start("detect")
            elif isinstance(event, PageDone):
                clock.start("render")
            if on_progress is not None:
                on_progress(event)

        clock.start("detect")
        batch = run_batch(
            stages,
            pages,
            work_dir / "output",
            max_workers=workers,
            on_progress=forward,
            cancel=cancel,
            executor=pool,
        )
        for page, error in batch.failed.items():
            result.failed[index_of[page]] = error

        exported = [page for page in pages if page in batch.written]
        if batch.cancelled or not exported:
            clock.stop()
            logger.info(
                "pdf export: not written, {} pages cancelled, {} failed",
                len(batch.cancelled),
                len(batch.failed),
            )
            return result

        clock.start("text")
        layers: list[TextLayer | None] = []
        if carry_text:
            jobs = [
                TextLayerJob.of(
                    pdf_path,
                    index_of[page],
                    info.transformation,
                    Dpi.uniform(dpi),
                    info.output_dpi,
                    dewarped=info.was_dewarped,
                    dewarp=info.dewarp,
                )
                for page in exported
                if (info := batch.render_info.get(page)) is not None
            ]
            built = iter(build_text_layers(jobs, executor=pool, max_workers=workers))
            layers = [
                next(built) if page in batch.render_info else None for page in exported
            ]
        result.text_pages = sum(layer is not None for layer in layers)

        clock.start("export")
        export_pdf(
            [
                (batch.written[page], _output_dpi(batch, page, output_dpi or dpi))
                for page in exported
            ],
            out_path,
            bilevel_codec=bilevel_codec,
            text_layers=layers if carry_text else None,
            executor=pool,
            max_workers=workers,
            outline_source=pdf_path,
        )
    clock.stop()
    result.pages_exported = len(exported)
    logger.info("pdf export: {}", _export_summary(result, bilevel_codec, carry_text))
    return result


def _export_summary(result: BookResult, codec: BilevelCodec, carry_text: bool) -> str:
    """Say what a rebuilt PDF holds and where the time went, for the log."""
    size_mb = result.out_path.stat().st_size / 1e6 if result.out_path.exists() else 0
    text_layer = f"text layer on {result.text_pages}" if carry_text else "no text"
    phases = ", ".join(
        f"{name} {seconds:.1f}" for name, seconds in result.seconds.items()
    )
    return (
        f"{result.pages_exported} pages, {len(result.failed)} failed, bilevel {codec}, "
        f"{text_layer}, {size_mb:.1f} MB in {sum(result.seconds.values()):.1f} s "
        f"({phases}) to {result.out_path}"
    )


def _output_dpi(batch: BatchResult, page: PageId, fallback: int) -> tuple[float, float]:
    info = batch.render_info.get(page)
    if info is None:
        return float(fallback), float(fallback)
    return float(info.output_dpi.horizontal), float(info.output_dpi.vertical)


class _PhaseClock:
    """Accumulates wall time into named phases; starting a phase ends the last."""

    def __init__(self, seconds: dict[str, float]) -> None:
        self._seconds = seconds
        self._phase: str | None = None
        self._started = 0.0

    def start(self, phase: str) -> None:
        if phase == self._phase:
            return
        self.stop()
        self._phase, self._started = phase, time.perf_counter()

    def stop(self) -> None:
        if self._phase is not None:
            elapsed = time.perf_counter() - self._started
            self._seconds[self._phase] = self._seconds.get(self._phase, 0.0) + elapsed
            self._phase = None
