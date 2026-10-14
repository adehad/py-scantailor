"""Text of a dewarped page lands on the dewarped words of the exported PDF."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from numpy.typing import NDArray

from scantailor.core.batch import BatchResult, run_batch
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.pdf_export import export_pdf
from scantailor.core.pdf_import import import_pdf
from scantailor.core.stage_sequence import StageSequence
from scantailor.core.text_layer import TextLayerJob, build_text_layers
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.params import DistortionModelSpec

PAGE_WIDTH_PT, PAGE_HEIGHT_PT = 300, 400
DPI = 144
PX_PER_PT = DPI / 72
FONT_SIZE = 14
# Each line is a row of words whose baselines rise towards the middle of the
# page, as the lines of an open book do. Words are short and spaced apart so
# the ink around one text box is that word's alone. Bold strokes survive
# binarisation at this resolution whole, so the ink spans each word.
LINES = [
    ("ALPHA", "BRAVO", "DELTA", "ECHO"),
    ("HOTEL", "INDIA", "KILO", "LIMA"),
    ("MIKE", "OSCAR", "PAPA", "TANGO"),
]
WORD_COUNT = sum(len(line) for line in LINES)
WORD_PITCH_PT = 64
FIRST_BASELINE_PT, LINE_PITCH_PT = 290, 70
LEFT_PT, RIGHT_PT = 30, 270
SAG_PT = 18
TOLERANCE_PX = 4

Processed = tuple[Path, list[PageId], BatchResult]


def _lift_pt(x_pt: float) -> float:
    """How far a line is raised at ``x_pt``: zero at the edges, ``SAG_PT`` midway."""
    middle, half = (LEFT_PT + RIGHT_PT) / 2, (RIGHT_PT - LEFT_PT) / 2
    return SAG_PT * (1 - ((x_pt - middle) / half) ** 2)


def _page_content(lift: Callable[[float], float]) -> str:
    words = []
    for row, line in enumerate(LINES):
        baseline = FIRST_BASELINE_PT - row * LINE_PITCH_PT
        for column, word in enumerate(line):
            x = LEFT_PT + 8 + column * WORD_PITCH_PT
            y = baseline + lift(x + WORD_PITCH_PT / 3)
            words.append(f"BT /F1 {FONT_SIZE} Tf {x:.2f} {y:.2f} Td ({word}) Tj ET")
    return "\n".join(words)


def _write_source(path: Path) -> Path:
    """A curved page, then a flat one, each with visible text as its text layer."""
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name("/Helvetica-Bold"),
        )
    )
    for content in (_page_content(_lift_pt), _page_content(lambda _x: 0.0)):
        page = pdf.add_blank_page(page_size=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
        page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
        page.Contents = pdf.make_stream(content.encode())
    pdf.save(path)
    return path


def _edge_px(y_pt: float) -> tuple[tuple[float, float], ...]:
    """A model edge following the curve of the lines, in source pixels."""
    return tuple(
        (float(x * PX_PER_PT), float((PAGE_HEIGHT_PT - y_pt - _lift_pt(x)) * PX_PER_PT))
        for x in np.linspace(LEFT_PT - 10, RIGHT_PT + 10, 9)
    )


def _dewarp_manually(stages: StageSequence, page: PageId) -> None:
    settings = stages.output_filter.settings
    last_baseline = FIRST_BASELINE_PT - (len(LINES) - 1) * LINE_PITCH_PT
    model = DistortionModelSpec(
        top=_edge_px(FIRST_BASELINE_PT + 40), bottom=_edge_px(last_baseline - 40)
    )
    params = settings.get_params(page)
    settings.set_params(
        page,
        params.with_dewarping(
            DewarpingOptions(mode=DewarpingMode.MANUAL)
        ).with_distortion_model(model),
    )


def _process(source: Path, work: Path) -> tuple[list[PageId], BatchResult]:
    images = import_pdf(source, work / "source", DPI)
    pages = [PageId(image_id=ImageId(file_path=path)) for path in images]
    stages = StageSequence()
    stages.output_filter.settings.set_default_dpi(Dpi.uniform(DPI))
    _dewarp_manually(stages, pages[0])
    batch = run_batch(stages, pages, work / "output", max_workers=1)
    assert batch.failed == {}
    return pages, batch


def _jobs(processed: Processed, *, through_dewarp: bool) -> list[TextLayerJob]:
    """Text jobs as an export makes them.

    Without ``through_dewarp`` the page transformation alone places the text,
    as if the page had not been dewarped.
    """
    source, pages, batch = processed
    jobs = []
    for index, page in enumerate(pages):
        info = batch.render_info[page]
        jobs.append(
            TextLayerJob.of(
                source,
                index,
                info.transformation,
                Dpi.uniform(DPI),
                info.output_dpi,
                dewarped=info.was_dewarped and through_dewarp,
                dewarp=info.dewarp if through_dewarp else None,
            )
        )
    return jobs


def _export(out: Path, processed: Processed, jobs: list[TextLayerJob]) -> Path:
    _source, pages, batch = processed
    export_pdf(
        [(batch.written[page], DPI) for page in pages],
        out,
        bilevel_codec="g4",
        text_layers=build_text_layers(jobs),
    )
    return out


def _word_boxes_px(document: pdfium.PdfDocument, index: int) -> list[NDArray]:
    """Left, top, right and bottom of each word's text, in output pixels."""
    page = document[index]
    textpage = page.get_textpage()
    text = textpage.get_text_range()
    visible = [position for position, char in enumerate(text) if not char.isspace()]
    letters = "".join(text[position] for position in visible)
    height_pt = page.get_height()
    boxes = []
    for word in (word for line in LINES for word in line):
        start = letters.find(word)
        if start < 0:
            continue
        chars = np.array(
            [
                textpage.get_charbox(visible[start + n], loose=True)
                for n in range(len(word))
            ]
        )
        left, right = chars[:, 0].min(), chars[:, 2].max()
        top, bottom = height_pt - chars[:, 3].max(), height_pt - chars[:, 1].min()
        boxes.append(np.array([left, top, right, bottom]) * PX_PER_PT)
    return boxes


def _ink(document: pdfium.PdfDocument, index: int) -> NDArray[np.bool_]:
    bitmap = document[index].render(scale=PX_PER_PT, grayscale=True)
    return np.asarray(bitmap.to_numpy()).reshape(bitmap.height, bitmap.width) < 128


def _ink_extent_near(ink: NDArray[np.bool_], box: NDArray) -> NDArray | None:
    """Left, top, right and bottom of the ink around a text box."""
    margin = 12
    left, top = (max(int(v) - margin, 0) for v in box[:2])
    right, bottom = (int(v) + margin for v in box[2:])
    rows, cols = np.nonzero(ink[top:bottom, left:right])
    if rows.size == 0:
        return None
    return np.array(
        [cols.min() + left, rows.min() + top, cols.max() + left, rows.max() + top],
        dtype=float,
    )


def _placement_errors(out: Path, index: int) -> NDArray:
    """Per word, in pixels: left edge, right edge and vertical centre errors.

    The ink of capitals sits at a fixed share of a loose text box's height, so
    the vertical error is how far the ink centre is from that share, taken as
    the page's median.
    """
    document = pdfium.PdfDocument(str(out))
    try:
        boxes = _word_boxes_px(document, index)
        ink = _ink(document, index)
    finally:
        document.close()
    rows = []
    for box in boxes:
        extent = _ink_extent_near(ink, box)
        if extent is not None:
            height = box[3] - box[1]
            share = ((extent[1] + extent[3]) / 2 - box[1]) / height
            rows.append((extent[0] - box[0], extent[2] - box[2], share, height))
    found = np.array(rows, dtype=float).reshape(-1, 4)
    if len(found):
        found[:, 2] = (found[:, 2] - np.median(found[:, 2])) * found[:, 3]
    return found[:, :3]


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory) -> Processed:
    work = tmp_path_factory.mktemp("dewarped_text")
    source = _write_source(work / "source.pdf")
    pages, batch = _process(source, work)
    return source, pages, batch


@pytest.mark.slow
def test_the_curved_page_is_dewarped_and_the_flat_one_is_not(
    processed: Processed,
) -> None:
    _source, pages, batch = processed

    curved, flat = (batch.render_info[page] for page in pages)

    assert curved.was_dewarped
    assert curved.dewarp is not None
    assert not flat.was_dewarped


@pytest.mark.slow
def test_text_of_the_dewarped_page_lies_on_the_dewarped_words(
    processed: Processed, tmp_path: Path
) -> None:
    jobs = _jobs(processed, through_dewarp=True)

    out = _export(tmp_path / "out.pdf", processed, jobs)

    errors = _placement_errors(out, 0)
    assert len(errors) == WORD_COUNT
    assert np.abs(errors).max() <= TOLERANCE_PX


@pytest.mark.slow
def test_the_page_transformation_alone_misses_the_dewarped_words(
    processed: Processed, tmp_path: Path
) -> None:
    jobs = _jobs(processed, through_dewarp=False)

    out = _export(tmp_path / "out.pdf", processed, jobs)

    errors = _placement_errors(out, 0)
    assert len(errors) < WORD_COUNT or np.abs(errors).max() > 2 * TOLERANCE_PX


@pytest.mark.slow
def test_text_of_the_page_without_dewarping_lies_on_its_words(
    processed: Processed, tmp_path: Path
) -> None:
    jobs = _jobs(processed, through_dewarp=True)

    out = _export(tmp_path / "out.pdf", processed, jobs)

    errors = _placement_errors(out, 1)
    assert len(errors) == WORD_COUNT
    assert np.abs(errors).max() <= TOLERANCE_PX
