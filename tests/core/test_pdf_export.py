"""Tests for PDF export of output pages."""

from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from pikepdf import PdfImage

from scantailor.core.pdf_export import export_pdf
from scantailor.core.text_layer import TextLayer


def _text_page(width: int = 400, height: int = 300) -> np.ndarray:
    """Black text-like bars on white, as a bool array (True is black)."""
    black = np.zeros((height, width), dtype=bool)
    rng = np.random.default_rng(1)
    for row in range(20, height - 20, 18):
        start = 20
        while start < width - 40:
            word = int(rng.integers(10, 40))
            black[row : row + 9, start : start + word] = True
            start += word + 8
    return black


def _grey_page() -> np.ndarray:
    ramp = np.tile(np.linspace(0, 255, 120, dtype=np.uint8), (80, 1))
    return ramp


def _colour_page() -> np.ndarray:
    page = np.zeros((90, 110, 3), dtype=np.uint8)
    page[:, :, 2] = 220  # red in BGR
    page[30:60, 30:80] = (200, 40, 40)  # blue patch
    return page


def _render(pdf_path: Path, index: int, scale: float) -> np.ndarray:
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        bitmap = document[index].render(scale=scale, rev_byteorder=False)
        return bitmap.to_numpy()
    finally:
        document.close()


@pytest.mark.parametrize("codec", ["jbig2", "g4"])
def test_bilevel_page_round_trips_through_rendering(tmp_path: Path, codec: str):
    black = _text_page()
    out = tmp_path / "out.pdf"
    export_pdf([(black, 300)], out, bilevel_codec=codec)

    rendered = _render(out, 0, scale=300 / 72)
    grey = cv2.cvtColor(rendered, cv2.COLOR_BGRA2GRAY)
    assert grey.shape == black.shape
    rendered_black = grey < 128
    mismatch = np.count_nonzero(rendered_black != black) / black.size
    assert mismatch < 0.001


def test_g4_stream_decodes_to_the_input_with_pikepdf(tmp_path: Path):
    black = _text_page()
    out = tmp_path / "g4.pdf"
    export_pdf([(black, 300)], out, bilevel_codec="g4")

    with pikepdf.open(out) as pdf:
        image = PdfImage(pdf.pages[0].Resources.XObject.Im0)
        decoded = np.asarray(image.as_pil_image().convert("L")) < 128
    assert np.array_equal(decoded, black)


def test_uint8_binary_page_is_stored_as_bilevel(tmp_path: Path):
    grey = np.where(_text_page(), 0, 255).astype(np.uint8)
    out = tmp_path / "out.pdf"
    export_pdf([(grey, 300)], out, bilevel_codec="g4")

    with pikepdf.open(out) as pdf:
        xobject = pdf.pages[0].Resources.XObject.Im0
        assert xobject.BitsPerComponent == 1
        assert xobject.Filter == "/CCITTFaxDecode"


def test_ccitt_parameters_declare_group4_with_black_as_zero(tmp_path: Path):
    out = tmp_path / "out.pdf"
    export_pdf([(_text_page(), 300)], out, bilevel_codec="g4")

    with pikepdf.open(out) as pdf:
        decode_params = pdf.pages[0].Resources.XObject.Im0.DecodeParms
        assert decode_params.K == -1
        assert not decode_params.get("/BlackIs1", False)


def test_jbig2_page_uses_jbig2_filter_without_globals(tmp_path: Path):
    out = tmp_path / "out.pdf"
    export_pdf([(_text_page(), 300)], out, bilevel_codec="jbig2")

    with pikepdf.open(out) as pdf:
        xobject = pdf.pages[0].Resources.XObject.Im0
        assert xobject.Filter == "/JBIG2Decode"
        assert "/DecodeParms" not in xobject


def test_jbig2_is_smaller_than_g4_for_a_text_page(tmp_path: Path):
    black = _text_page(1600, 1200)
    jbig2_path, g4_path = tmp_path / "a.pdf", tmp_path / "b.pdf"
    export_pdf([(black, 300)], jbig2_path, bilevel_codec="jbig2")
    export_pdf([(black, 300)], g4_path, bilevel_codec="g4")
    assert jbig2_path.stat().st_size < g4_path.stat().st_size


@pytest.mark.parametrize(
    ("image", "colour_space"),
    [(_grey_page(), "/DeviceGray"), (_colour_page(), "/DeviceRGB")],
    ids=["grey", "colour"],
)
def test_grey_and_colour_pages_are_dct_encoded(
    tmp_path: Path, image: np.ndarray, colour_space: str
):
    out = tmp_path / "out.pdf"
    export_pdf([(image, 150)], out, colour_quality=95)

    with pikepdf.open(out) as pdf:
        xobject = pdf.pages[0].Resources.XObject.Im0
        assert xobject.Filter == "/DCTDecode"
        assert xobject.ColorSpace == colour_space
        decoded = np.asarray(PdfImage(xobject).as_pil_image())
    expected = image if image.ndim == 2 else image[:, :, ::-1]
    assert decoded.shape == expected.shape
    assert np.abs(decoded.astype(int) - expected.astype(int)).mean() < 4


def test_page_size_in_points_follows_pixels_and_dpi(tmp_path: Path):
    out = tmp_path / "out.pdf"
    export_pdf([(_text_page(600, 300), 300), (_grey_page(), (150, 300))], out)

    with pikepdf.open(out) as pdf:
        first = [float(v) for v in pdf.pages[0].MediaBox]
        second = [float(v) for v in pdf.pages[1].MediaBox]
    assert first == pytest.approx([0, 0, 600 / 300 * 72, 300 / 300 * 72])
    assert second == pytest.approx([0, 0, 120 / 150 * 72, 80 / 300 * 72])


def test_mixed_three_page_pdf_opens_and_checks_clean(tmp_path: Path):
    out = tmp_path / "mixed.pdf"
    export_pdf(
        [(_text_page(), 300), (_grey_page(), 200), (_colour_page(), 100)],
        out,
        bilevel_codec="jbig2",
    )

    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 3
        assert pdf.check_pdf_syntax() == []
        filters = [str(p.Resources.XObject.Im0.Filter) for p in pdf.pages]
    assert filters == ["/JBIG2Decode", "/DCTDecode", "/DCTDecode"]
    for index in range(3):
        assert _render(out, index, scale=1.0).size > 0


def test_pages_can_be_read_from_image_paths(tmp_path: Path):
    source = tmp_path / "page.png"
    assert cv2.imwrite(str(source), _colour_page())
    out = tmp_path / "out.pdf"
    export_pdf([(source, 100)], out)

    with pikepdf.open(out) as pdf:
        assert pdf.pages[0].Resources.XObject.Im0.Width == 110


def test_pdf_uses_object_streams(tmp_path: Path):
    out = tmp_path / "out.pdf"
    export_pdf([(_text_page(), 300)] * 3, out)
    assert b"/ObjStm" in out.read_bytes()


def test_text_layer_is_drawn_before_the_image_with_its_fonts(tmp_path: Path):
    layer = TextLayer(
        content=b"BT 3 Tr /F1 12 Tf 20 100 Td (hello world) Tj ET",
        resources={
            "/Font": {
                "F1": {"Type": "/Font", "Subtype": "/Type1", "BaseFont": "/Helvetica"}
            }
        },
    )
    out = tmp_path / "out.pdf"
    export_pdf([(_text_page(), 300)], out, text_layers=[layer])

    with pikepdf.open(out) as pdf:
        page = pdf.pages[0]
        content = page.Contents.read_bytes()
        assert content.index(b"hello world") < content.index(b"/Im0 Do")
        assert page.Resources.Font.F1.BaseFont == "/Helvetica"
    document = pdfium.PdfDocument(str(out))
    try:
        text = document[0].get_textpage().get_text_range()
    finally:
        document.close()
    assert "hello world" in text


def test_text_layer_font_from_another_pdf_is_copied(tmp_path: Path):
    source = pikepdf.Pdf.new()
    font = source.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Courier,
        )
    )
    layer = TextLayer(b"BT 3 Tr /F9 10 Tf 5 5 Td (x) Tj ET", {"/Font": {"F9": font}})
    out = tmp_path / "out.pdf"
    export_pdf([(_text_page(), 300)], out, text_layers=[layer])

    with pikepdf.open(out) as pdf:
        assert pdf.pages[0].Resources.Font.F9.BaseFont == "/Courier"


def test_text_layers_may_skip_pages(tmp_path: Path):
    out = tmp_path / "out.pdf"
    layer = TextLayer(
        b"BT 3 Tr /F1 10 Tf 5 5 Td (a) Tj ET",
        {"/Font": {"F1": {"BaseFont": "/Courier"}}},
    )
    export_pdf([(_text_page(), 300)] * 2, out, text_layers=[None, layer])

    with pikepdf.open(out) as pdf:
        assert "/Font" not in pdf.pages[0].Resources
        assert "/Font" in pdf.pages[1].Resources


def test_empty_project_writes_a_valid_pdf_with_no_pages(tmp_path: Path):
    out = tmp_path / "empty.pdf"
    export_pdf([], out)
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 0


@pytest.mark.parametrize("codec", ["jbig2", "g4"])
def test_sub_one_dpi_and_non_contiguous_pages_are_accepted(tmp_path: Path, codec: str):
    out = tmp_path / "out.pdf"
    view = _text_page().T[::2]
    export_pdf([(view, 0.4), (view, [300, 200])], out, bilevel_codec=codec)
    with pikepdf.open(out) as pdf:
        width = view.shape[1]
        assert float(pdf.pages[0].MediaBox[2]) == pytest.approx(width / 0.4 * 72)
        assert float(pdf.pages[1].MediaBox[3]) == pytest.approx(
            view.shape[0] / 200 * 72
        )


@pytest.mark.parametrize(
    ("kwargs", "pages"),
    [
        ({}, [(_text_page(), float("nan"))]),
        ({"bilevel_codec": "png"}, [(_text_page(), 300)]),
        ({"text_layers": []}, [(_text_page(), 300)]),
        ({}, [(_text_page(), 0)]),
        ({}, [(np.zeros((4, 4, 2), dtype=np.uint8), 300)]),
        ({}, [(Path("missing-image.png"), 300)]),
    ],
    ids=["nan-dpi", "codec", "layers-length", "dpi", "shape", "missing-file"],
)
def test_invalid_input_raises_value_error(tmp_path: Path, kwargs, pages):
    with pytest.raises(ValueError):
        export_pdf(pages, tmp_path / "out.pdf", **kwargs)


def test_three_channel_bool_image_is_rejected_with_dtype_message(tmp_path: Path):
    with pytest.raises(ValueError, match="uint8"):
        export_pdf([(np.zeros((4, 4, 3), dtype=bool), 300)], tmp_path / "out.pdf")
