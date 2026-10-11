"""Importing a PDF yields one correctly sized, DPI-tagged image per page."""

from pathlib import Path

import numpy as np
import pikepdf
import pytest

from scantailor.core.image_io import load_image
from scantailor.core.models import ImageId


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
