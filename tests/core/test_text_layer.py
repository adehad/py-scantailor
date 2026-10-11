"""Text carried over from a source PDF lands where the picture moved it to."""

from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest

from scantailor.core.models import Dpi, OrthogonalRotation
from scantailor.core.text_layer import TextLayer, build_text_layer
from scantailor.core.transformation import ImageTransformation

PAGE_WIDTH_PT = 200
PAGE_HEIGHT_PT = 300
SOURCE_DPI = Dpi.uniform(144)
OUTPUT_DPI = Dpi.uniform(300)
TEXT = "HELLO"


def _write_source_pdf(
    path: Path,
    *,
    rotate: int = 0,
    preamble: str = "",
    cropbox: tuple[float, float, float, float] | None = None,
) -> None:
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
        )
    )
    page = pdf.add_blank_page(page_size=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
    page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    stream = (
        f"{preamble} BT /F1 24 Tf 1 Tr 40 200 Td ({TEXT}) Tj ET "
        "0 0 1 rg 10 10 30 30 re f"
    )
    page.Contents = pdf.make_stream(stream.encode())
    if rotate:
        page.Rotate = rotate
    if cropbox is not None:
        page.CropBox = list(cropbox)
    pdf.save(path)


def _first_char_box(pdf_path: Path) -> tuple[float, float, float, float]:
    """Return (left, bottom, right, top) in points of the first glyph on page 0."""
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        textpage = document[0].get_textpage()
        assert textpage.get_text_range().startswith(TEXT)
        return textpage.get_charbox(0)
    finally:
        document.close()


def _output_pdf(path: Path, layer: TextLayer, width_pt: float, height_pt: float):
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(width_pt, height_pt))
    fonts = pikepdf.Dictionary()
    for name, font in layer.resources["/Font"].items():
        fonts[name] = pdf.copy_foreign(font)
    page.Resources = pikepdf.Dictionary(Font=fonts)
    page.Contents = pdf.make_stream(layer.content)
    pdf.save(path)


def _transformation(width_px: int = 400, height_px: int = 600) -> ImageTransformation:
    """Rotate 90 degrees, crop, deskew by 7 degrees and scale to 300 dpi."""
    xform = ImageTransformation.from_size(width_px, height_px, SOURCE_DPI)
    xform.set_pre_rotation(OrthogonalRotation(degrees=90))
    xform.set_pre_crop_area(
        np.array([[20, 10], [380, 10], [380, 380], [20, 380]], dtype=np.float64)
    )
    xform.set_post_rotation(7.0)
    xform.post_scale_to_dpi(OUTPUT_DPI)
    return xform


def _box_centre(box: tuple[float, float, float, float]) -> np.ndarray:
    left, bottom, right, top = box
    return np.array([(left + right) / 2, (bottom + top) / 2])


@pytest.mark.parametrize("rotate", [0, 90])
def test_text_lands_where_the_transformation_moves_the_picture(tmp_path, rotate):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source, rotate=rotate)
    xform = _transformation(*((600, 400) if rotate == 90 else (400, 600)))

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, OUTPUT_DPI)
    assert layer is not None

    rect = xform.resulting_rect
    width_pt = rect.width * 72 / OUTPUT_DPI.horizontal
    height_pt = rect.height * 72 / OUTPUT_DPI.vertical
    output = tmp_path / "output.pdf"
    _output_pdf(output, layer, width_pt, height_pt)

    source_centre = _box_centre(_first_char_box(source))
    if rotate == 90:
        # pdfium reports boxes in unrotated user space; a clockwise page turn
        # puts user y on the horizontal axis of the rendered pixels.
        px_x, px_y = source_centre[1] * 2, source_centre[0] * 2
    else:
        px_x, px_y = source_centre[0] * 2, (PAGE_HEIGHT_PT - source_centre[1]) * 2
    out_x, out_y = xform.transform_point(px_x, px_y)
    expected_px = np.array([out_x - rect.x, out_y - rect.y])

    centre_pt = _box_centre(_first_char_box(output))
    actual_px = np.array(
        [
            centre_pt[0] * OUTPUT_DPI.horizontal / 72,
            (height_pt - centre_pt[1]) * OUTPUT_DPI.vertical / 72,
        ]
    )
    # The glyph box of a rotated string is its axis-aligned bound, so allow a
    # few output pixels (about 0.3 mm at 300 dpi) of slack.
    assert actual_px == pytest.approx(expected_px, abs=8)


def test_identity_mapping_at_equal_dpi_keeps_the_source_glyph_box(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source)
    xform = ImageTransformation.from_size(400, 600, SOURCE_DPI)

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, SOURCE_DPI)
    assert layer is not None
    output = tmp_path / "output.pdf"
    _output_pdf(output, layer, PAGE_WIDTH_PT, PAGE_HEIGHT_PT)

    assert _first_char_box(output) == pytest.approx(_first_char_box(source), abs=0.05)


def test_text_inside_a_scaled_graphics_state_is_mapped_through_it(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source, preamble="q 0.5 0 0 0.5 20 30 cm")
    xform = ImageTransformation.from_size(400, 600, SOURCE_DPI)

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, SOURCE_DPI)
    assert layer is not None
    output = tmp_path / "output.pdf"
    _output_pdf(output, layer, PAGE_WIDTH_PT, PAGE_HEIGHT_PT)

    assert _first_char_box(output) == pytest.approx(_first_char_box(source), abs=0.05)


def test_a_tiny_graphics_state_scale_keeps_its_precision(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source)
    scaled = pikepdf.open(source, allow_overwriting_input=True)
    scaled.pages[0].Contents = scaled.make_stream(
        b"q 0.0001 0 0 0.0001 0 0 cm BT /F1 240000 Tf 400000 2000000 Td (HELLO) Tj ET Q"
    )
    scaled.save(source)
    scaled.close()
    xform = ImageTransformation.from_size(400, 600, SOURCE_DPI)

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, SOURCE_DPI)
    assert layer is not None
    output = tmp_path / "output.pdf"
    _output_pdf(output, layer, PAGE_WIDTH_PT, PAGE_HEIGHT_PT)

    assert _first_char_box(output) == pytest.approx(_first_char_box(source), abs=0.05)


def test_a_cropbox_offsets_the_text_like_the_rendered_page(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source, cropbox=(20, 30, 180, 270))
    xform = ImageTransformation.from_size(320, 480, SOURCE_DPI)

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, SOURCE_DPI)
    assert layer is not None
    output = tmp_path / "output.pdf"
    _output_pdf(output, layer, 160, 240)

    left, bottom, right, top = _first_char_box(source)
    assert _first_char_box(output) == pytest.approx(
        (left - 20, bottom - 30, right - 20, top - 30), abs=0.05
    )


def test_every_text_block_is_forced_to_the_invisible_render_mode(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source)
    xform = ImageTransformation.from_size(400, 600, SOURCE_DPI)

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, SOURCE_DPI)

    assert layer is not None
    modes = [
        str(item.operands[0])
        for item in _instructions(layer)
        if str(item.operator) == "Tr"
    ]
    assert modes == ["3"]


def _instructions(layer: TextLayer) -> list:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
    page.Contents = pdf.make_stream(layer.content)
    return [
        item
        for item in pikepdf.parse_content_stream(page)
        if not isinstance(item, pikepdf.ContentStreamInlineImage)
    ]


def test_the_non_text_drawing_of_the_source_is_not_carried_over(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source)
    xform = ImageTransformation.from_size(400, 600, SOURCE_DPI)

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, SOURCE_DPI)

    assert layer is not None
    operators = {str(item.operator) for item in _instructions(layer)}
    assert {"re", "f", "rg"}.isdisjoint(operators)


def test_the_referenced_fonts_are_copied(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source)
    xform = ImageTransformation.from_size(400, 600, SOURCE_DPI)

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, SOURCE_DPI)

    assert layer is not None
    fonts = layer.resources["/Font"]
    assert list(fonts.keys()) == ["/F1"]
    assert str(fonts["/F1"].BaseFont) == "/Helvetica"


def test_a_dewarped_page_has_no_text_layer(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source_pdf(source)
    xform = ImageTransformation.from_size(400, 600, SOURCE_DPI)

    assert (
        build_text_layer(source, 0, xform, SOURCE_DPI, OUTPUT_DPI, dewarped=True)
        is None
    )


def test_a_page_without_text_has_no_text_layer(tmp_path):
    source = tmp_path / "blank.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(PAGE_WIDTH_PT, PAGE_HEIGHT_PT))
    pdf.save(source)
    xform = ImageTransformation.from_size(400, 600, SOURCE_DPI)

    assert build_text_layer(source, 0, xform, SOURCE_DPI, OUTPUT_DPI) is None


def test_a_cropped_source_page_maps_text_relative_to_its_crop_box(tmp_path):
    source = tmp_path / "cropped.pdf"
    _write_source_pdf(source)
    with pikepdf.open(source, allow_overwriting_input=True) as pdf:
        pdf.pages[0].CropBox = [30, 100, 190, 290]
        pdf.save(source)
    document = pdfium.PdfDocument(str(source))
    try:
        crop_width_pt, crop_height_pt = document[0].get_size()
    finally:
        document.close()
    xform = ImageTransformation.from_size(
        round(crop_width_pt * 2), round(crop_height_pt * 2), SOURCE_DPI
    )

    layer = build_text_layer(source, 0, xform, SOURCE_DPI, SOURCE_DPI)

    assert layer is not None
    output = tmp_path / "output.pdf"
    _output_pdf(output, layer, crop_width_pt, crop_height_pt)
    left, bottom, _, _ = _first_char_box(source)
    out_left, out_bottom, _, _ = _first_char_box(output)
    assert (out_left, out_bottom) == pytest.approx((left - 30, bottom - 100), abs=0.5)
