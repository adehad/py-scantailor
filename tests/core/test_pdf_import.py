"""Importing a PDF yields one correctly sized, DPI-tagged image per page."""

import threading
from pathlib import Path

import numpy as np
import pikepdf
import pytest

from scantailor.core.image_io import load_image
from scantailor.core.models import ImageId
from scantailor.core.pdf_import import (
    clear_rendered_pages,
    import_pdf,
    page_count,
    rendered_pages,
)
from scantailor.core.pdf_pool import PdfCancelled


def _write_pdf(path: Path, pages: list[tuple[float, float, str]]) -> None:
    pdf = pikepdf.Pdf.new()
    for width, height, content in pages:
        page = pdf.add_blank_page(page_size=(width, height))
        page.Contents = pdf.make_stream(content.encode())
    pdf.save(path)


def _load(path: Path) -> tuple[np.ndarray, int]:
    pixels, metadata = load_image(ImageId(file_path=path, page=0))
    return pixels, metadata.dpi.horizontal


def test_each_page_becomes_an_image_of_the_requested_resolution(tmp_path):
    from scantailor.core.pdf_import import import_pdf

    source = tmp_path / "book.pdf"
    _write_pdf(
        source, [(72, 144, "0 g 10 10 20 20 re f"), (144, 72, "0 g 10 10 20 20 re f")]
    )

    paths = import_pdf(source, tmp_path / "pages", dpi=150)

    assert [p.name for p in paths] == ["page_0001.png", "page_0002.png"]
    first, first_dpi = _load(paths[0])
    second, _ = _load(paths[1])
    assert first.shape[:2] == (300, 150)
    assert second.shape[:2] == (150, 300)
    assert first_dpi == 150


def test_a_black_and_white_page_is_stored_as_grayscale(tmp_path):
    from scantailor.core.pdf_import import import_pdf

    source = tmp_path / "bw.pdf"
    _write_pdf(source, [(72, 72, "0 g 0 0 36 72 re f")])

    pixels, _ = _load(import_pdf(source, tmp_path / "pages", dpi=72)[0])

    assert pixels.ndim == 2
    assert pixels[36, 10] < 30
    assert pixels[36, 60] > 225


def test_a_page_with_colour_is_stored_in_colour(tmp_path):
    from scantailor.core.pdf_import import import_pdf

    source = tmp_path / "colour.pdf"
    _write_pdf(source, [(72, 72, "1 0 0 rg 0 0 72 72 re f")])

    pixels, _ = _load(import_pdf(source, tmp_path / "pages", dpi=72)[0])

    assert pixels.ndim == 3
    blue, green, red = pixels[36, 36]
    assert red > 200
    assert blue < 50
    assert green < 50


def test_non_positive_dpi_is_rejected(tmp_path):
    from scantailor.core.pdf_import import import_pdf

    source = tmp_path / "x.pdf"
    _write_pdf(source, [(72, 72, "")])

    with pytest.raises(ValueError, match="dpi"):
        import_pdf(source, tmp_path / "pages", dpi=0)


def test_a_rotated_page_is_rendered_in_its_displayed_orientation(tmp_path):
    from scantailor.core.pdf_import import import_pdf

    source = tmp_path / "rotated.pdf"
    _write_pdf(source, [(72, 144, "0 g 0 0 10 10 re f")])
    with pikepdf.open(source, allow_overwriting_input=True) as pdf:
        pdf.pages[0].Rotate = 90
        pdf.save(source)

    pixels, _ = _load(import_pdf(source, tmp_path / "pages", dpi=72)[0])

    assert pixels.shape[:2] == (72, 144)


def test_a_pdf_without_pages_is_rejected(tmp_path):
    from scantailor.core.pdf_import import import_pdf

    source = tmp_path / "empty.pdf"
    pikepdf.Pdf.new().save(source)

    with pytest.raises(ValueError, match="pages"):
        import_pdf(source, tmp_path / "pages")


def _three_page_pdf(path: Path) -> Path:
    _write_pdf(path, [(72, 72, "0 g 0 0 10 10 re f")] * 3)
    return path


def test_a_page_range_renders_only_those_pages_under_their_pdf_numbers(tmp_path):
    source = _three_page_pdf(tmp_path / "book.pdf")

    paths = import_pdf(source, tmp_path / "pages", dpi=72, first_page=2, last_page=3)

    assert [p.name for p in paths] == ["page_0002.png", "page_0003.png"]


def test_a_range_without_an_end_renders_to_the_last_page(tmp_path):
    source = _three_page_pdf(tmp_path / "book.pdf")

    paths = import_pdf(source, tmp_path / "pages", dpi=72, first_page=3)

    assert [p.name for p in paths] == ["page_0003.png"]


@pytest.mark.parametrize(("first", "last"), [(0, 2), (2, 1), (1, 4)])
def test_a_range_outside_the_document_is_refused(tmp_path, first, last):
    source = _three_page_pdf(tmp_path / "book.pdf")

    with pytest.raises(ValueError, match="outside the 3 pages"):
        import_pdf(source, tmp_path / "pages", dpi=72, first_page=first, last_page=last)


def test_page_count_reads_the_pages_and_refuses_a_file_that_is_not_a_pdf(tmp_path):
    source = _three_page_pdf(tmp_path / "book.pdf")
    not_a_pdf = tmp_path / "notes.pdf"
    not_a_pdf.write_text("plain text", encoding="utf-8")

    assert page_count(source) == 3
    with pytest.raises(ValueError, match="not a readable PDF"):
        page_count(not_a_pdf)


def test_rendered_pages_lists_only_page_images_in_name_order(tmp_path):
    folder = tmp_path / "pages"
    import_pdf(_three_page_pdf(tmp_path / "book.pdf"), folder, dpi=72)
    (folder / "notes.txt").write_text("keep", encoding="utf-8")
    (folder / "page_draft.png").write_text("keep", encoding="utf-8")

    assert [p.name for p in rendered_pages(folder)] == [
        "page_0001.png",
        "page_0002.png",
        "page_0003.png",
    ]
    assert rendered_pages(tmp_path / "missing") == []


def test_clearing_rendered_pages_leaves_other_files_alone(tmp_path):
    folder = tmp_path / "pages"
    import_pdf(_three_page_pdf(tmp_path / "book.pdf"), folder, dpi=72)
    (folder / "notes.txt").write_text("keep", encoding="utf-8")

    clear_rendered_pages(folder)

    assert rendered_pages(folder) == []
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_a_shorter_range_that_replaces_leaves_no_stale_pages(tmp_path):
    source = _three_page_pdf(tmp_path / "book.pdf")
    folder = tmp_path / "pages"
    import_pdf(source, folder, dpi=72)

    import_pdf(source, folder, dpi=72, first_page=1, last_page=1, clear_existing=True)

    assert [p.name for p in rendered_pages(folder)] == ["page_0001.png"]


def test_a_cancelled_range_keeps_the_pages_already_there(tmp_path):
    source = _three_page_pdf(tmp_path / "book.pdf")
    folder = tmp_path / "pages"
    import_pdf(source, folder, dpi=72)
    before = {p.name: p.read_bytes() for p in rendered_pages(folder)}
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(PdfCancelled):
        import_pdf(
            source,
            folder,
            dpi=100,
            first_page=1,
            last_page=1,
            clear_existing=True,
            cancel=cancel,
        )

    assert {p.name: p.read_bytes() for p in rendered_pages(folder)} == before
