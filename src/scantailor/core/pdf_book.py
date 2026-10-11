"""Rebuild a scanned PDF: import its pages, process them, export a new PDF.

The source PDF's text layer, when it has one, is carried onto the processed
pages as invisible text.
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
    run_batch,
)
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.pdf_export import BilevelCodec, export_pdf
from scantailor.core.pdf_import import import_pdf
from scantailor.core.stage_sequence import StageSequence
from scantailor.core.text_layer import TextLayer, build_text_layer


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
        pdf_path (Path): Source PDF.
        out_path (Path): Destination PDF.
        work_dir (Path): Scratch directory for the imported and processed pages;
            kept so the intermediate files can be inspected.
        dpi (int): Resolution the source pages are rendered at.
        output_dpi (int | None): Output resolution; ``dpi`` when None. Ignored
            when ``stages`` is given.
        bilevel_codec (BilevelCodec): Codec of black and white pages.
        carry_text (bool): Carry the source text layer onto the output.
        max_workers (int | None): Worker processes of the batch.
        stages (StageSequence | None): Project settings; defaults otherwise.
        on_progress (Callable[[BatchEvent], None] | None): Batch events.
        cancel (Event | None): Stops the batch from starting further pages.

    Returns:
        BookResult: Timings, failures and the output location. Pages that
        failed or were cancelled are absent from the PDF.
    """
    result = BookResult(out_path)
    clock = _PhaseClock(result.seconds)

    clock.start("import")
    images = import_pdf(pdf_path, work_dir / "source", dpi)
    pages = [PageId(image_id=ImageId(file_path=path)) for path in images]
    index_of = {page: index for index, page in enumerate(pages)}

    if stages is None:
        stages = StageSequence()
        stages.output_filter.settings.set_default_dpi(Dpi.uniform(output_dpi or dpi))

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
        max_workers=max_workers,
        on_progress=forward,
        cancel=cancel,
    )
    for page, error in batch.failed.items():
        result.failed[index_of[page]] = error

    clock.start("text")
    exported = [page for page in pages if page in batch.written]
    layers: list[TextLayer | None] = []
    for page in exported:
        info = batch.render_info.get(page)
        layer = None
        if carry_text and info is not None:
            layer = build_text_layer(
                pdf_path,
                index_of[page],
                info.transformation,
                Dpi.uniform(dpi),
                info.output_dpi,
                dewarped=info.was_dewarped,
            )
        layers.append(layer)
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
    )
    clock.stop()
    result.pages_exported = len(exported)
    logger.info("Rebuilt {} pages into {}", result.pages_exported, out_path)
    return result


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
