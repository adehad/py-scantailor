"""Fast checks that the sample book has the sizes, metadata and text layer promised."""

import importlib.util
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from scantailor.filters.deskew.oblique import find_oblique_degrees
from scantailor.filters.deskew.params import MAX_OBLIQUE_DEG
from scantailor.imageproc.binarize import binarize_otsu

_TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load_sample_book() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sample_book", _TOOLS / "sample_book.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def book(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, object]:
    """Write the sample book once for every test in this module."""
    module = _load_sample_book()
    folder = tmp_path_factory.mktemp("sample_book")
    return module, module.write_sample_book(folder)


def test_every_page_has_its_expected_size_and_resolution(
    book: tuple[ModuleType, object],
) -> None:
    """Each page is the documented size, 8-bit RGB, and carries 150 dpi metadata."""
    module, sample = book
    assert len(sample.pages) == len(module.SAMPLE_PAGE_NAMES)
    for path, name in zip(sample.pages, module.SAMPLE_PAGE_NAMES, strict=True):
        assert path.name == f"{name}.png"
        with Image.open(path) as image:
            assert image.size == module.expected_size(name), name
            assert image.mode == "RGB", name
            assert tuple(round(v) for v in image.info["dpi"]) == (150, 150), name


def test_pages_are_identical_between_runs(tmp_path: Path) -> None:
    """Two renders of the book write the same bytes for every page and the PDF."""
    module = _load_sample_book()
    first = module.write_sample_book(tmp_path / "a")
    second = module.write_sample_book(tmp_path / "b")
    for left, right in zip(
        (*first.pages, first.pdf), (*second.pages, second.pdf), strict=True
    ):
        assert left.read_bytes() == right.read_bytes(), left.name


def test_pages_have_content_and_no_flat_regions(
    book: tuple[ModuleType, object],
) -> None:
    """Every page has text or pictures; none is a blank sheet."""
    _, sample = book
    for path in sample.pages:
        with Image.open(path) as image:
            grey = np.asarray(image.convert("L"))
        assert grey.std() > 5, f"{path.name} is blank"


def test_sample_pdf_has_three_pages_with_extractable_text(
    book: tuple[ModuleType, object],
) -> None:
    """The PDF opens with three pages, and its text layer returns the headings."""
    module, sample = book
    document = pdfium.PdfDocument(str(sample.pdf))
    try:
        assert len(document) == len(module.PDF_PAGE_NAMES) == 3
        first_text = document[0].get_textpage().get_text_range()
        assert "Chapter One" in first_text
        assert "ledger bound in" in first_text
    finally:
        document.close()


def test_oblique_sample_page_is_detected_as_leaning_within_the_correction_limit() -> (
    None
):
    """The detector finds most of the shear, short of the clamp at the limit.

    The manual's oblique shot relies on automatic oblique correction picking up
    the shear of ``06_oblique``.
    """
    sample = _load_sample_book()

    page = sample.render_sample_page("06_oblique")
    found = find_oblique_degrees(
        binarize_otsu(cv2.cvtColor(page.image, cv2.COLOR_RGB2GRAY))
    )

    assert found is not None
    assert found == pytest.approx(-sample._OBLIQUE_DEGREES, abs=1.5)
    assert abs(found) < MAX_OBLIQUE_DEG
