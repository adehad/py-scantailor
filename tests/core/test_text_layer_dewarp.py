"""A dewarped page's text lands where the dewarp moved the picture."""

from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from pikepdf import Dictionary, Name

from scantailor.core import text_layer
from scantailor.core.models import Dpi
from scantailor.core.text_layer import (
    DewarpSpec,
    TextGeometry,
    TextLayer,
    build_text_layer,
)
from scantailor.core.transformation import ImageTransformation
from scantailor.dewarping.page_dewarper import dewarper_from_polylines
from scantailor.dewarping.raster_dewarper import (
    compute_dewarp_maps,
    output_model_domain,
    remap_with_maps,
)

PAGE_WIDTH_PT, PAGE_HEIGHT_PT = 200, 300
DPI = Dpi.uniform(144)
WIDTH_PX, HEIGHT_PX = 400, 600
DEPTH = 2.0
WORD = "HELLO"
# Page edges that sag in the middle, in source pixels.
SPEC = DewarpSpec(
    top=((20.0, 60.0), (200.0, 20.0), (380.0, 60.0)),
    bottom=((20.0, 580.0), (200.0, 560.0), (380.0, 580.0)),
    depth_perception=DEPTH,
)


def _write_source(path: Path) -> Path:
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica)
    )
    page = pdf.add_blank_page(page_size=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
    page.Resources = Dictionary(Font=Dictionary(F1=font))
    page.Contents = pdf.make_stream(f"BT /F1 30 Tf 30 230 Td ({WORD}) Tj ET".encode())
    pdf.save(path)
    return path


def _identity() -> ImageTransformation:
    return ImageTransformation.from_size(WIDTH_PX, HEIGHT_PX, DPI)


def _output_pdf(path: Path, layer: TextLayer) -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
    fonts = Dictionary()
    for name, font in layer.resources["/Font"].items():
        fonts[name] = pdf.copy_foreign(font)
    page.Resources = Dictionary(Font=fonts)
    page.Contents = pdf.make_stream(layer.content)
    pdf.save(path)


def _word_extent_px(path: Path) -> tuple[float, float, float, float]:
    """Left, top, right and bottom, in output pixels, of the word's characters."""
    document = pdfium.PdfDocument(str(path))
    try:
        textpage = document[0].get_textpage()
        text = textpage.get_text_range()
        start = text.index(WORD)
        boxes = np.array(
            [textpage.get_charbox(start + i, loose=True) for i in range(len(WORD))]
        )
    finally:
        document.close()
    scale = DPI.horizontal / 72
    return (
        float(boxes[:, 0].min() * scale),
        float((PAGE_HEIGHT_PT - boxes[:, 3].max()) * scale),
        float(boxes[:, 2].max() * scale),
        float((PAGE_HEIGHT_PT - boxes[:, 1].min()) * scale),
    )


def _render_gray(source: Path) -> np.ndarray:
    document = pdfium.PdfDocument(str(source))
    try:
        bitmap = document[0].render(scale=DPI.horizontal / 72, grayscale=True)
        return np.array(bitmap.to_numpy(), dtype=np.uint8).reshape(HEIGHT_PX, WIDTH_PX)
    finally:
        document.close()


def _ink_extent(gray: np.ndarray) -> tuple[float, float, float, float]:
    rows, cols = np.nonzero(gray < 128)
    return float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())


def _dewarped_ink_extent(source: Path) -> tuple[float, float, float, float]:
    """Dewarp a rendering of the source page and return its ink's bounding box."""
    dewarper = dewarper_from_polylines(SPEC.top, SPEC.bottom, DEPTH)
    assert dewarper is not None
    domain = output_model_domain(dewarper, (WIDTH_PX, HEIGHT_PX))
    maps = compute_dewarp_maps(dewarper, (WIDTH_PX, HEIGHT_PX), domain)
    return _ink_extent(remap_with_maps(_render_gray(source), maps, 255))


def _undewarped_ink_extent(source: Path) -> tuple[float, float, float, float]:
    return _ink_extent(_render_gray(source))


def test_dewarped_text_moves_by_as_much_as_the_dewarped_picture(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source.pdf")
    dewarped = build_text_layer(
        source, 0, _identity(), DPI, DPI, dewarped=True, dewarp=SPEC
    )
    flat = build_text_layer(source, 0, _identity(), DPI, DPI)
    assert dewarped is not None
    assert flat is not None
    _output_pdf(tmp_path / "dewarped.pdf", dewarped)
    _output_pdf(tmp_path / "flat.pdf", flat)

    ink_flat = _undewarped_ink_extent(source)
    ink_dewarped = _dewarped_ink_extent(source)
    text_flat = _word_extent_px(tmp_path / "flat.pdf")
    text_dewarped = _word_extent_px(tmp_path / "dewarped.pdf")

    # How far the left, top and right edges of the word were moved. The bottom
    # is left out: the text box reaches below the baseline, where the
    # non-uniform vertical stretch of the dewarp moves it more than the ink.
    ink_shift = np.subtract(ink_dewarped, ink_flat)[:3]
    text_shift = np.subtract(text_dewarped, text_flat)[:3]
    assert abs(ink_shift[1]) > 15
    assert text_shift == pytest.approx(ink_shift, abs=6)


def test_the_dewarp_actually_moves_the_text(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.pdf")

    warped = _dewarped_ink_extent(source)
    flat = _undewarped_ink_extent(source)

    assert abs(warped[1] - flat[1]) > 10 or abs(warped[0] - flat[0]) > 10


def test_dewarped_text_is_invisible_and_extracts_as_the_source_text(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source.pdf")
    layer = build_text_layer(
        source, 0, _identity(), DPI, DPI, dewarped=True, dewarp=SPEC
    )
    assert layer is not None
    output = tmp_path / "output.pdf"
    _output_pdf(output, layer)

    document = pdfium.PdfDocument(str(output))
    try:
        page = document[0]
        assert WORD in page.get_textpage().get_text_range()
        ink = (np.asarray(page.render(scale=1).to_numpy())[..., :3] < 250).any()
    finally:
        document.close()
    assert not ink


def test_a_dewarped_page_without_a_dewarp_model_has_no_text_layer(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source.pdf")

    assert build_text_layer(source, 0, _identity(), DPI, DPI, dewarped=True) is None


def test_an_unusable_dewarp_model_gives_no_text_layer(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.pdf")
    flat = DewarpSpec(top=((0.0, 10.0),), bottom=((0.0, 20.0),), depth_perception=2.0)

    assert (
        build_text_layer(source, 0, _identity(), DPI, DPI, dewarped=True, dewarp=flat)
        is None
    )


def test_post_deskew_turns_the_text_with_the_picture(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.pdf")
    turned = SPEC.model_copy(update={"post_deskew_angle": 5.0})
    plain = build_text_layer(
        source, 0, _identity(), DPI, DPI, dewarped=True, dewarp=SPEC
    )
    rotated = build_text_layer(
        source, 0, _identity(), DPI, DPI, dewarped=True, dewarp=turned
    )
    assert plain is not None
    assert rotated is not None
    _output_pdf(tmp_path / "plain.pdf", plain)
    _output_pdf(tmp_path / "rotated.pdf", rotated)

    plain_box = _word_extent_px(tmp_path / "plain.pdf")
    rotated_box = _word_extent_px(tmp_path / "rotated.pdf")

    # Clockwise about the centre moves a word left of centre up.
    assert rotated_box[1] < plain_box[1] - 3


def test_short_text_uses_one_generated_font(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.pdf")

    layer = build_text_layer(
        source, 0, _identity(), DPI, DPI, dewarped=True, dewarp=SPEC
    )

    assert layer is not None
    assert set(layer.resources["/Font"].keys()) == {"/STU0"}


def test_more_distinct_characters_than_one_font_holds_use_a_second_font(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path / "source.pdf")
    count = 300
    characters = [
        text_layer._SourceChar(
            0x4E00 + n, 30.0 + (n % 10), 100.0, 36.0 + (n % 10), 112.0
        )
        for n in range(count)
    ]
    monkeypatch.setattr(text_layer, "_source_characters", lambda *_args: characters)

    layer = build_text_layer(
        source, 0, _identity(), DPI, DPI, dewarped=True, dewarp=SPEC
    )

    assert layer is not None
    assert set(layer.resources["/Font"].keys()) == {"/STU0", "/STU1"}
    output = tmp_path / "output.pdf"
    _output_pdf(output, layer)
    document = pdfium.PdfDocument(str(output))
    try:
        text = document[0].get_textpage().get_text_range()
    finally:
        document.close()
    assert all(chr(0x4E00 + n) in text for n in (0, 254, 255, count - 1))


def test_geometry_survives_a_json_round_trip_and_builds_the_same_job(
    tmp_path: Path,
) -> None:
    xform = _identity()
    geometry = TextGeometry.of(xform, DPI, dewarped=True, dewarp=SPEC)

    restored = TextGeometry.model_validate_json(geometry.model_dump_json())

    assert restored == geometry
    source = tmp_path / "source.pdf"
    assert restored.job(source, 0, DPI) == geometry.job(source, 0, DPI)
    assert geometry.job(source, 0, DPI).rect_width == WIDTH_PX


BOOK_LINES = [
    "The quick brown fox jumps over",
    "a lazy dog while the afternoon",
    "light slides across the pages of",
    "an old and much loved volume",
    "that nobody has opened for years",
    "Pack my box with five dozen jugs",
    "of liquor before the long winter",
    "sets in and the roads are closed",
]


def _write_book_page(path: Path) -> Path:
    """A page of text lines that spans the width between the curved edges."""
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica)
    )
    page = pdf.add_blank_page(page_size=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
    page.Resources = Dictionary(Font=Dictionary(F1=font))
    first_baseline, pitch = 250, 26
    content = "".join(
        f"BT /F1 11 Tf 22 {first_baseline - n * pitch} Td ({line}) Tj ET\n"
        for n, line in enumerate(BOOK_LINES)
    )
    page.Contents = pdf.make_stream(content.encode())
    pdf.save(path)
    return path


def _character_boxes_px(path: Path) -> np.ndarray:
    """Left, top, right and bottom, in output pixels, of every visible character."""
    document = pdfium.PdfDocument(str(path))
    try:
        textpage = document[0].get_textpage()
        text = textpage.get_text_range()
        boxes = [
            textpage.get_charbox(i, loose=True)
            for i, char in enumerate(text)
            if not char.isspace()
        ]
    finally:
        document.close()
    scale = DPI.horizontal / 72
    left, bottom, right, top = np.array(boxes).T
    return (
        np.column_stack([left, PAGE_HEIGHT_PT - top, right, PAGE_HEIGHT_PT - bottom])
        * scale
    )


def _ink_offsets_from_boxes(ink: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Vertical distance of the ink under each character box from the box centre."""
    offsets = []
    for left, top, right, bottom in boxes:
        centre = (top + bottom) / 2
        rows, _cols = np.nonzero(
            ink[
                max(int(centre) - 22, 0) : int(centre) + 23,
                max(int(left), 0) : int(right) + 1,
            ]
        )
        if rows.size >= 3:
            offsets.append(rows.mean() + max(int(centre) - 22, 0) - centre)
    return np.array(offsets)


def test_text_of_a_curved_book_page_follows_the_dewarped_lines(
    tmp_path: Path,
) -> None:
    source = _write_book_page(tmp_path / "book.pdf")
    dewarper = dewarper_from_polylines(SPEC.top, SPEC.bottom, DEPTH)
    assert dewarper is not None
    maps = compute_dewarp_maps(
        dewarper,
        (WIDTH_PX, HEIGHT_PX),
        output_model_domain(dewarper, (WIDTH_PX, HEIGHT_PX)),
    )
    ink = remap_with_maps(_render_gray(source), maps, 255) < 128
    dewarped = build_text_layer(
        source, 0, _identity(), DPI, DPI, dewarped=True, dewarp=SPEC
    )
    flat = build_text_layer(source, 0, _identity(), DPI, DPI)
    assert dewarped is not None
    assert flat is not None
    _output_pdf(tmp_path / "dewarped.pdf", dewarped)
    _output_pdf(tmp_path / "flat.pdf", flat)

    through_dewarp = _ink_offsets_from_boxes(
        ink, _character_boxes_px(tmp_path / "dewarped.pdf")
    )
    without_dewarp = _ink_offsets_from_boxes(
        ink, _character_boxes_px(tmp_path / "flat.pdf")
    )

    # The ink sits at a fixed height inside a character box, so a layer that
    # follows the lines leaves a constant offset and one that does not, a drifting one.
    assert through_dewarp.size > 150
    assert through_dewarp.std() < 2.5
    assert without_dewarp.std() > 2 * through_dewarp.std()
