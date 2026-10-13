"""Text drawn by form XObjects and Type 3 fonts is carried over, invisibly."""

from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from pikepdf import Array, Dictionary, Name

from scantailor.core.models import Dpi
from scantailor.core.text_layer import TextLayer, build_text_layer
from scantailor.core.transformation import ImageTransformation

PAGE = (300, 400)
DPI = Dpi.uniform(144)
XFORM_SIZE = (600, 800)


def _font(pdf: pikepdf.Pdf, base: str) -> pikepdf.Object:
    return pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name(f"/{base}"))
    )


def _identity() -> ImageTransformation:
    return ImageTransformation.from_size(*XFORM_SIZE, DPI)


def _carry(source: Path, tmp_path: Path) -> tuple[TextLayer, Path]:
    layer = build_text_layer(source, 0, _identity(), DPI, DPI)
    assert layer is not None
    output = tmp_path / "output.pdf"
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=PAGE)
    fonts = Dictionary()
    for name, font in layer.resources["/Font"].items():
        fonts[name] = pdf.copy_foreign(font)
    page.Resources = Dictionary(Font=fonts)
    page.Contents = pdf.make_stream(layer.content)
    pdf.save(output)
    return layer, output


def _text_and_boxes(path: Path) -> tuple[str, list[tuple[float, ...]]]:
    document = pdfium.PdfDocument(str(path))
    try:
        textpage = document[0].get_textpage()
        text = textpage.get_text_range()
        boxes = [textpage.get_charbox(i) for i in range(textpage.count_chars())]
        return text, boxes
    finally:
        document.close()


def _render_ink(path: Path) -> int:
    document = pdfium.PdfDocument(str(path))
    try:
        bitmap = document[0].render(scale=1.0)
        return int((np.asarray(bitmap.to_numpy())[..., :3] < 128).any(axis=2).sum())
    finally:
        document.close()


def _page_with_form(path: Path, *, form_font: str = "Courier") -> Path:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=PAGE)
    page.Resources = Dictionary(
        Font=Dictionary(F1=_font(pdf, "Helvetica")),
        XObject=Dictionary(
            Fm0=pdf.make_stream(
                b"BT /F1 24 Tf 0 200 Td (WORLD) Tj ET",
                Type=Name.XObject,
                Subtype=Name.Form,
                BBox=Array([0, 0, 300, 400]),
                Matrix=Array([1, 0, 0, 1, 100, -50]),
                Resources=Dictionary(Font=Dictionary(F1=_font(pdf, form_font))),
            )
        ),
    )
    page.Contents = pdf.make_stream(
        b"BT /F1 24 Tf 40 200 Td (HELLO) Tj ET /Fm0 Do",
    )
    pdf.save(path)
    return path


def test_text_inside_a_form_xobject_is_carried_with_the_form_matrix(
    tmp_path: Path,
) -> None:
    source = _page_with_form(tmp_path / "form.pdf")
    reference = tmp_path / "reference.pdf"
    ref = pikepdf.Pdf.new()
    ref_page = ref.add_blank_page(page_size=PAGE)
    ref_page.Resources = Dictionary(
        Font=Dictionary(F1=_font(ref, "Helvetica"), F2=_font(ref, "Courier"))
    )
    ref_page.Contents = ref.make_stream(
        b"BT /F1 24 Tf 40 200 Td (HELLO) Tj ET BT /F2 24 Tf 100 150 Td (WORLD) Tj ET"
    )
    ref.save(reference)

    _layer, output = _carry(source, tmp_path)

    text, boxes = _text_and_boxes(output)
    ref_text, ref_boxes = _text_and_boxes(reference)
    assert text.replace("\r\n", "") == ref_text.replace("\r\n", "")
    assert np.allclose(np.array(boxes), np.array(ref_boxes), atol=0.1)


def test_two_resource_dictionaries_using_one_font_name_do_not_clash(
    tmp_path: Path,
) -> None:
    source = _page_with_form(tmp_path / "form.pdf")

    layer, _output = _carry(source, tmp_path)

    fonts = layer.resources["/Font"]
    assert sorted(str(f.BaseFont) for f in fonts.values()) == [
        "/Courier",
        "/Helvetica",
    ]


def test_a_font_shared_by_the_page_and_a_form_is_copied_once(tmp_path: Path) -> None:
    pdf = pikepdf.Pdf.new()
    shared = _font(pdf, "Helvetica")
    page = pdf.add_blank_page(page_size=PAGE)
    page.Resources = Dictionary(
        Font=Dictionary(F1=shared),
        XObject=Dictionary(
            Fm0=pdf.make_stream(
                b"BT /F9 24 Tf 0 0 Td (B) Tj ET",
                Type=Name.XObject,
                Subtype=Name.Form,
                BBox=Array([0, 0, 300, 400]),
                Resources=Dictionary(Font=Dictionary(F9=shared)),
            )
        ),
    )
    page.Contents = pdf.make_stream(b"BT /F1 24 Tf 40 200 Td (A) Tj ET /Fm0 Do")
    source = tmp_path / "shared.pdf"
    pdf.save(source)

    layer, _output = _carry(source, tmp_path)

    assert list(layer.resources["/Font"].keys()) == ["/F1"]


def test_a_form_that_draws_itself_does_not_loop_forever(tmp_path: Path) -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=PAGE)
    form = pdf.make_stream(
        b"BT /F1 12 Tf 10 10 Td (loop) Tj ET /Fm0 Do",
        Type=Name.XObject,
        Subtype=Name.Form,
        BBox=Array([0, 0, 300, 400]),
    )
    form.Resources = Dictionary(
        Font=Dictionary(F1=_font(pdf, "Helvetica")), XObject=Dictionary(Fm0=form)
    )
    page.Resources = Dictionary(XObject=Dictionary(Fm0=form))
    page.Contents = pdf.make_stream(b"/Fm0 Do")
    source = tmp_path / "loop.pdf"
    pdf.save(source)

    layer = build_text_layer(source, 0, _identity(), DPI, DPI)

    assert layer is not None
    assert layer.content.count(b"loop") == 1


def test_a_form_without_text_adds_nothing(tmp_path: Path) -> None:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=PAGE)
    page.Resources = Dictionary(
        XObject=Dictionary(
            Fm0=pdf.make_stream(
                b"0 0 50 50 re f",
                Type=Name.XObject,
                Subtype=Name.Form,
                BBox=Array([0, 0, 300, 400]),
            )
        )
    )
    page.Contents = pdf.make_stream(b"/Fm0 Do")
    source = tmp_path / "shapes.pdf"
    pdf.save(source)

    assert build_text_layer(source, 0, _identity(), DPI, DPI) is None


def _type3_page(path: Path) -> Path:
    pdf = pikepdf.Pdf.new()
    to_unicode = pdf.make_stream(
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
        b"1 begincodespacerange <00> <FF> endcodespacerange\n"
        b"1 beginbfchar\n<61> <0061>\nendbfchar\n"
        b"endcmap CMapName currentdict /CMap defineresource pop end end"
    )
    font = pdf.make_indirect(
        Dictionary(
            Type=Name.Font,
            Subtype=Name.Type3,
            FontBBox=Array([0, 0, 1000, 1000]),
            FontMatrix=Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=Dictionary(
                sq=pdf.make_stream(b"1000 0 0 0 1000 1000 d1 0 0 1000 1000 re f")
            ),
            Encoding=Dictionary(Type=Name.Encoding, Differences=Array([97, Name.sq])),
            FirstChar=97,
            LastChar=97,
            Widths=Array([1000]),
            ToUnicode=to_unicode,
        )
    )
    page = pdf.add_blank_page(page_size=PAGE)
    page.Resources = Dictionary(Font=Dictionary(T3=font))
    page.Contents = pdf.make_stream(b"BT /T3 40 Tf 50 100 Td (aaa) Tj ET")
    pdf.save(path)
    return path


def test_the_source_type_3_font_does_paint_so_it_needs_a_stand_in(
    tmp_path: Path,
) -> None:
    source = _type3_page(tmp_path / "t3.pdf")

    assert _render_ink(source) > 1000


def test_type_3_text_is_carried_but_paints_nothing(tmp_path: Path) -> None:
    source = _type3_page(tmp_path / "t3.pdf")

    layer, output = _carry(source, tmp_path)

    font = layer.resources["/Font"]["/T3"]
    assert font.Subtype == "/Type3"
    for proc in font.CharProcs.values():
        assert proc.read_bytes().split()[-1] == b"d0"
    assert _render_ink(output) == 0
    text, boxes = _text_and_boxes(output)
    assert text.startswith("aaa")
    assert len(boxes) >= 3


def test_type_3_stand_in_keeps_the_advance_of_the_original(tmp_path: Path) -> None:
    source = _type3_page(tmp_path / "t3.pdf")

    _layer, output = _carry(source, tmp_path)

    _text, boxes = _text_and_boxes(output)
    lefts = [box[0] for box in boxes[:3]]
    assert np.diff(lefts) == pytest.approx([40.0, 40.0], abs=0.5)
