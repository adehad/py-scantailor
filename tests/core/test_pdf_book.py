"""A PDF goes through import, the pipeline and export, keeping its text."""

from pathlib import Path

import pikepdf
import pypdfium2 as pdfium
import pytest

from scantailor.core.pdf_book import BookResult, _export_summary, rebuild_pdf

PAGE_WORDS = ["alpha", "bravo", "charlie"]


def _write_text_pdf(path: Path) -> None:
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
        )
    )
    for word in PAGE_WORDS:
        page = pdf.add_blank_page(page_size=(300, 400))
        page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
        page.Contents = pdf.make_stream(
            f"BT /F1 32 Tf 60 200 Td ({word}) Tj ET".encode()
        )
    pdf.save(path)


def _page_text(path: Path, index: int) -> str:
    document = pdfium.PdfDocument(str(path))
    try:
        return document[index].get_textpage().get_text_range()
    finally:
        document.close()


@pytest.mark.slow
def test_rebuilt_pdf_keeps_every_page_and_its_text(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _write_text_pdf(source)
    out = tmp_path / "out.pdf"

    result = rebuild_pdf(source, out, tmp_path / "work", dpi=150, max_workers=2)

    assert result.failed == {}
    assert result.pages_exported == len(PAGE_WORDS)
    assert result.text_pages == len(PAGE_WORDS)
    assert set(result.seconds) == {"import", "detect", "render", "text", "export"}
    for index, word in enumerate(PAGE_WORDS):
        assert word in _page_text(out, index)


@pytest.mark.slow
def test_text_can_be_left_out(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _write_text_pdf(source)
    out = tmp_path / "out.pdf"

    result = rebuild_pdf(
        source, out, tmp_path / "work", dpi=150, carry_text=False, max_workers=2
    )

    assert result.text_pages == 0
    assert _page_text(out, 0).strip() == ""


def test_the_export_summary_gives_counts_size_and_phase_times(tmp_path: Path):
    out = tmp_path / "book.pdf"
    out.write_bytes(b"x" * 2_500_000)
    result = BookResult(
        out,
        pages_exported=410,
        failed={3: "ValueError: bad", 9: "ValueError: bad"},
        text_pages=395,
        seconds={"import": 1.0, "detect": 2.0, "render": 3.0, "text": 0.5},
    )

    summary = _export_summary(result, "jbig2", carry_text=True)

    assert summary == (
        "410 pages, 2 failed, bilevel jbig2, text layer on 395, 2.5 MB in 6.5 s "
        f"(import 1.0, detect 2.0, render 3.0, text 0.5) to {out}"
    )
