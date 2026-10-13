"""PDF import and export jobs: guards, progress, stored geometry and cancel."""

import threading
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from pikepdf import Dictionary, Name, OutlineItem

from scantailor.app.ui.pdf_worker import (
    PagePlan,
    PdfExportRequest,
    PdfSource,
    StoredTextRequest,
    TextRequest,
    export_pdf_pages,
    import_pdf_pages,
)
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi
from scantailor.core.pdf_pool import PdfCancelled
from scantailor.core.text_layer import TextGeometry
from scantailor.core.transformation import ImageTransformation

WORDS = ["alpha", "bravo"]
DPI = Dpi.uniform(144)
SIZE = (400, 600)


def _source(path: Path) -> Path:
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica)
    )
    for word in WORDS:
        page = pdf.add_blank_page(page_size=(200, 300))
        page.Resources = Dictionary(Font=Dictionary(F1=font))
        page.Contents = pdf.make_stream(
            f"BT /F1 24 Tf 40 200 Td ({word}) Tj ET".encode()
        )
    with pdf.open_outline() as outline:
        outline.root.extend([OutlineItem("One", 0), OutlineItem("Two", 1)])
    pdf.save(path)
    return path


def _images(folder: Path) -> list[Path]:
    paths = []
    for index in range(len(WORDS)):
        path = folder / f"out_{index}.tif"
        write_image(path, np.zeros(SIZE[::-1], dtype=bool), DPI)
        paths.append(path)
    return paths


def _transformation() -> ImageTransformation:
    return ImageTransformation.from_size(*SIZE, DPI)


def _plans(images: list[Path], text: object = None) -> list[PagePlan]:
    plans = []
    for index, image in enumerate(images):
        request = None
        if text == "live":
            request = TextRequest(index, _transformation(), DPI, False)
        elif text == "stored":
            request = StoredTextRequest(index, TextGeometry.of(_transformation(), DPI))
        plans.append(PagePlan(image, (144.0, 144.0), request))
    return plans


def _page_text(path: Path, index: int) -> str:
    document = pdfium.PdfDocument(str(path))
    try:
        return document[index].get_textpage().get_text_range()
    finally:
        document.close()


def _export(
    tmp_path: Path,
    *,
    out: Path | None = None,
    carry_text: bool = True,
    text: object = "live",
    source: PdfSource | None = None,
    cancel: threading.Event | None = None,
    report: list | None = None,
):
    pdf = _source(tmp_path / "book.pdf")
    images = _images(tmp_path)
    request = PdfExportRequest(
        _plans(images, text),
        out or tmp_path / "out.pdf",
        source=source or PdfSource(pdf_path=pdf, dpi=144),
        carry_text=carry_text,
    )
    sink = [] if report is None else report
    return export_pdf_pages(
        request, lambda *a: sink.append(a), cancel or threading.Event()
    )


def test_the_source_pdf_is_never_overwritten_when_text_is_carried(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="replace the source PDF"):
        _export(tmp_path, out=tmp_path / "book.pdf")

    assert "alpha" in _page_text(tmp_path / "book.pdf", 0)


def test_the_source_pdf_is_never_overwritten_when_text_is_not_carried(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="replace the source PDF"):
        _export(tmp_path, out=tmp_path / "book.pdf", carry_text=False)

    assert "alpha" in _page_text(tmp_path / "book.pdf", 0)


def test_without_carry_text_the_pages_have_no_text_but_keep_the_bookmarks(
    tmp_path: Path,
) -> None:
    result = _export(tmp_path, carry_text=False)

    assert result.text_pages == 0
    assert _page_text(result.out_path, 0).strip() == ""
    with pikepdf.open(result.out_path) as pdf, pdf.open_outline() as outline:
        assert [str(item.title) for item in outline.root] == ["One", "Two"]


def test_text_is_carried_by_default(tmp_path: Path) -> None:
    result = _export(tmp_path)

    assert result.text_pages == len(WORDS)
    assert "alpha" in _page_text(result.out_path, 0)


def test_geometry_stored_with_the_project_carries_the_same_text(
    tmp_path: Path,
) -> None:
    result = _export(tmp_path, text="stored")

    assert result.text_pages == len(WORDS)
    for index, word in enumerate(WORDS):
        assert word in _page_text(result.out_path, index)


def test_the_export_reports_each_page_and_the_save(tmp_path: Path) -> None:
    report: list[tuple[int, int, str]] = []

    _export(tmp_path, report=report)

    assert report[0] == (0, 2, "Reading text")
    assert (1, 2, "Writing PDF") in report
    assert (2, 2, "Writing PDF") in report
    assert report[-1] == (2, 2, "Saving PDF")


def test_a_cancelled_export_leaves_neither_the_file_nor_a_partial_one(
    tmp_path: Path,
) -> None:
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(PdfCancelled):
        _export(tmp_path, cancel=cancel)

    assert not (tmp_path / "out.pdf").exists()
    assert not (tmp_path / "out.pdf.part").exists()


def test_a_missing_source_pdf_still_exports_without_text(tmp_path: Path) -> None:
    pdf = _source(tmp_path / "book.pdf")
    source = PdfSource(pdf_path=pdf, dpi=144)
    images = _images(tmp_path)
    pdf.unlink()
    request = PdfExportRequest(_plans(images), tmp_path / "out.pdf", source=source)

    result = export_pdf_pages(request, lambda *_: None, threading.Event())

    assert result.pages == len(WORDS)
    assert result.text_pages == 0


def test_a_cancelled_import_leaves_an_earlier_import_in_the_folder(
    tmp_path: Path,
) -> None:
    pdf = _source(tmp_path / "book.pdf")
    pages = tmp_path / "book_pages"
    first = import_pdf_pages(pdf, pages, 72, lambda *_: None, threading.Event())
    before = {p.name: p.read_bytes() for p in first.images}
    cancel = threading.Event()

    with pytest.raises(PdfCancelled):
        import_pdf_pages(pdf, pages, 100, lambda *_: cancel.set(), cancel)

    assert {p.name: p.read_bytes() for p in pages.glob("page_*.png")} == before
    assert PdfSource.load(pages) == first.source


def test_import_reports_the_pages_as_they_are_rendered(tmp_path: Path) -> None:
    pdf = _source(tmp_path / "book.pdf")
    report: list[tuple[int, int, str]] = []

    import_pdf_pages(
        pdf, tmp_path / "pages", 72, lambda *a: report.append(a), threading.Event()
    )

    assert report == [
        (0, 2, "Rendering"),
        (1, 2, "Rendering"),
        (2, 2, "Rendering"),
    ]
