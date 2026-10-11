"""The pool-backed PDF import, export and text layers match the serial paths."""

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pikepdf
import pytest

from scantailor.core.batch import spawn_pool
from scantailor.core.image_io import load_image
from scantailor.core.models import Dpi, ImageId
from scantailor.core.pdf_book import rebuild_pdf
from scantailor.core.pdf_export import export_pdf
from scantailor.core.pdf_import import import_pdf
from scantailor.core.text_layer import (
    TextLayerJob,
    build_text_layer,
    build_text_layers,
)
from scantailor.core.transformation import ImageTransformation

WORKERS = 2
PAGE_WORDS = ["alpha", "bravo", "charlie", "delta", "echo"]


@pytest.fixture(scope="module")
def pool() -> Iterator[ProcessPoolExecutor]:
    with spawn_pool(WORKERS) as executor:
        yield executor


def _write_book(path: Path) -> None:
    """Five pages: text on every page, a colour fill on the odd ones."""
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
        )
    )
    for index, word in enumerate(PAGE_WORDS):
        page = pdf.add_blank_page(page_size=(200 + index * 10, 300))
        page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
        fill = "1 0 0 rg 10 10 40 40 re f 0 g " if index % 2 else "0 g "
        page.Contents = pdf.make_stream(
            f"{fill}BT /F1 24 Tf 30 150 Td ({word}) Tj ET".encode()
        )
    pdf.save(path)


def _pixels(path: Path) -> np.ndarray:
    return load_image(ImageId(file_path=path, page=0))[0]


def test_imported_pages_match_the_serial_render_pixel_for_pixel(tmp_path, pool):
    source = tmp_path / "book.pdf"
    _write_book(source)

    serial = import_pdf(source, tmp_path / "serial", dpi=100)
    parallel = import_pdf(
        source, tmp_path / "parallel", dpi=100, executor=pool, max_workers=WORKERS
    )

    assert [p.name for p in parallel] == [p.name for p in serial]
    assert len(parallel) == len(PAGE_WORDS)
    for serial_path, parallel_path in zip(serial, parallel, strict=True):
        assert np.array_equal(_pixels(serial_path), _pixels(parallel_path))


def test_a_pool_import_rejects_a_pdf_without_pages(tmp_path, pool):
    source = tmp_path / "empty.pdf"
    pikepdf.Pdf.new().save(source)

    with pytest.raises(ValueError, match="pages"):
        import_pdf(source, tmp_path / "pages", executor=pool, max_workers=WORKERS)


def _page_streams(path: Path) -> list[tuple[bytes, bytes, list[float]]]:
    """Raw image data, content and size per page; the trailer ID differs per save."""
    with pikepdf.open(path) as pdf:
        return [
            (
                page.Resources.XObject.Im0.read_raw_bytes(),
                page.Contents.read_bytes(),
                [float(v) for v in page.MediaBox],
            )
            for page in pdf.pages
        ]


def _mixed_pages() -> list[tuple[np.ndarray, float]]:
    rng = np.random.default_rng(7)
    bilevel = rng.random((300, 240)) > 0.9
    grey = np.tile(np.linspace(0, 255, 120, dtype=np.uint8), (80, 1))
    colour = np.zeros((90, 110, 3), dtype=np.uint8)
    colour[:, :, 2] = 200
    return [(bilevel, 300), (grey, 150), (colour, 150), (bilevel[:100], 200)]


@pytest.mark.parametrize("codec", ["jbig2", "g4"])
def test_pool_export_writes_the_same_bytes_as_the_serial_export(tmp_path, pool, codec):
    pages = _mixed_pages()
    serial, parallel = tmp_path / "serial.pdf", tmp_path / "parallel.pdf"

    export_pdf(pages, serial, bilevel_codec=codec)
    export_pdf(
        pages,
        parallel,
        bilevel_codec=codec,
        executor=pool,
        max_workers=WORKERS,
    )

    assert _page_streams(serial) == _page_streams(parallel)


def test_pool_export_keeps_page_order_when_paths_are_given(tmp_path, pool):
    paths = []
    for index, (pixels, _) in enumerate(_mixed_pages()):
        path = tmp_path / f"page{index}.png"
        image = pixels.astype(np.uint8) * 255 if pixels.dtype == bool else pixels
        cv2.imwrite(str(path), image)
        paths.append(path)
    pages = [(path, 150) for path in paths]
    serial, parallel = tmp_path / "serial.pdf", tmp_path / "parallel.pdf"

    export_pdf(pages, serial)
    export_pdf(pages, parallel, executor=pool, max_workers=WORKERS)

    assert _page_streams(serial) == _page_streams(parallel)


def test_pool_export_reports_an_unreadable_image(tmp_path, pool):
    pages = [(_mixed_pages()[0][0], 300), (tmp_path / "missing.png", 300)]

    with pytest.raises(ValueError, match="Cannot read image"):
        export_pdf(pages, tmp_path / "out.pdf", executor=pool, max_workers=WORKERS)


def _job(source: Path, index: int, *, dewarped: bool = False) -> TextLayerJob:
    transformation = ImageTransformation.from_size(300, 400, Dpi.uniform(150))
    transformation.set_post_rotation(3.0)
    return TextLayerJob.of(
        source,
        index,
        transformation,
        Dpi.uniform(150),
        Dpi.uniform(300),
        dewarped=dewarped,
    )


def test_worker_built_layers_match_the_serial_layers(tmp_path, pool):
    source = tmp_path / "book.pdf"
    _write_book(source)
    jobs = [_job(source, index) for index in range(len(PAGE_WORDS))]

    serial = build_text_layers(jobs)
    parallel = build_text_layers(jobs, executor=pool, max_workers=WORKERS)

    assert all(layer is not None for layer in parallel)
    for expected, actual in zip(serial, parallel, strict=True):
        assert expected is not None and actual is not None
        assert actual.content == expected.content
        assert list(actual.resources["/Font"].keys()) == list(
            expected.resources["/Font"].keys()
        )
        assert (
            actual.resources["/Font"]["/F1"].BaseFont
            == expected.resources["/Font"]["/F1"].BaseFont
        )


def test_single_page_helper_agrees_with_the_batch_builder(tmp_path):
    source = tmp_path / "book.pdf"
    _write_book(source)
    job = _job(source, 2)
    transformation = ImageTransformation.from_size(300, 400, Dpi.uniform(150))
    transformation.set_post_rotation(3.0)

    single = build_text_layer(
        source, 2, transformation, Dpi.uniform(150), Dpi.uniform(300)
    )
    batched = build_text_layers([job])[0]

    assert single is not None and batched is not None
    assert single.content == batched.content


def test_pages_without_a_layer_stay_none_in_position(tmp_path, pool):
    source = tmp_path / "book.pdf"
    _write_book(source)
    jobs = [_job(source, 0), _job(source, 1, dewarped=True), _job(source, 2)]

    layers = build_text_layers(jobs, executor=pool, max_workers=WORKERS)

    assert [layer is None for layer in layers] == [False, True, False]


def test_a_font_shared_by_pages_is_one_object_in_the_exported_pdf(tmp_path, pool):
    source = tmp_path / "book.pdf"
    _write_book(source)
    jobs = [_job(source, index) for index in range(len(PAGE_WORDS))]
    layers = build_text_layers(jobs, executor=pool, max_workers=WORKERS)
    pages = [(np.ones((400, 300), dtype=bool), 150)] * len(PAGE_WORDS)
    out = tmp_path / "out.pdf"

    export_pdf(pages, out, text_layers=layers, executor=pool, max_workers=WORKERS)

    with pikepdf.open(out) as pdf:
        font_ids = {page.Resources.Font.F1.objgen for page in pdf.pages}
    assert len(font_ids) == 1


def test_a_cancelled_rebuild_writes_no_pdf(tmp_path):
    source = tmp_path / "book.pdf"
    _write_book(source)
    out = tmp_path / "out.pdf"
    cancel = Event()
    cancel.set()

    result = rebuild_pdf(
        source, out, tmp_path / "work", dpi=72, max_workers=WORKERS, cancel=cancel
    )

    assert result.pages_exported == 0
    assert not out.exists()


def _write_embedded_font_book(path: Path) -> bytes:
    """One page whose font carries an embedded font program; returns its bytes."""
    program = bytes(range(256)) * 40
    pdf = pikepdf.Pdf.new()
    descriptor = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.FontDescriptor,
            FontName=pikepdf.Name.Embedded,
            FontFile=pdf.make_stream(program),
        )
    )
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Embedded,
            FontDescriptor=descriptor,
        )
    )
    page = pdf.add_blank_page(page_size=(200, 300))
    page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    page.Contents = pdf.make_stream(b"BT /F1 24 Tf 30 150 Td (word) Tj ET")
    pdf.save(path)
    return program


@pytest.mark.parametrize("use_pool", [False, True])
def test_an_embedded_font_program_survives_the_source_being_closed(
    tmp_path, pool, use_pool
):
    source = tmp_path / "embedded.pdf"
    program = _write_embedded_font_book(source)
    options = {"executor": pool, "max_workers": WORKERS} if use_pool else {}
    layers = build_text_layers([_job(source, 0)], **options)
    out = tmp_path / "out.pdf"

    export_pdf([(np.ones((400, 300), dtype=bool), 150)], out, text_layers=layers)

    with pikepdf.open(out) as pdf:
        font = pdf.pages[0].Resources.Font.F1
        assert font.FontDescriptor.FontFile.read_bytes() == program


def test_a_single_page_pdf_imports_on_the_pool(tmp_path, pool):
    source = tmp_path / "one.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(100, 100))
    pdf.save(source)

    paths = import_pdf(
        source, tmp_path / "pages", dpi=72, executor=pool, max_workers=WORKERS
    )

    assert [p.name for p in paths] == ["page_0001.png"]


def test_an_encrypted_pdf_is_rejected_on_the_pool(tmp_path, pool):
    source = tmp_path / "locked.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(100, 100))
    pdf.save(source, encryption=pikepdf.Encryption(owner="o", user="u"))

    with pytest.raises(ValueError, match="readable"):
        import_pdf(source, tmp_path / "pages", executor=pool, max_workers=WORKERS)


def test_a_page_without_text_has_no_layer_on_the_pool(tmp_path, pool):
    source = tmp_path / "blank.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(200, 300))
    pdf.save(source)

    layers = build_text_layers([_job(source, 0)], executor=pool, max_workers=WORKERS)

    assert layers == [None]
