"""Export output pages to a single PDF.

Each page is placed at its physical size (pixels / dpi * 72 pt). Bilevel pages are
stored losslessly as JBIG2 or CCITT G4, grey and colour pages as JPEG.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pikepdf
from jbig2_numba import encode_pdf_stream
from numpy.typing import NDArray
from PIL import Image

from scantailor.core.text_layer import TextLayer

PDF_POINTS_PER_INCH = 72.0
BilevelCodec = Literal["jbig2", "g4"]
PageSource = NDArray[Any] | str | Path
PageInput = tuple[PageSource, "float | tuple[float, float]"]


def export_pdf(
    pages: Sequence[PageInput],
    out_path: str | Path,
    *,
    bilevel_codec: BilevelCodec = "jbig2",
    colour_quality: int = 85,
    text_layers: Sequence[TextLayer | None] | None = None,
) -> None:
    """Write pages to one PDF.

    Args:
        pages (Sequence[PageInput]): ``(image, dpi)`` per page. ``image`` is a path or
            an array: 2-D ``bool`` (``True`` is black), 2-D ``uint8`` or 3-channel
            BGR ``uint8``. A single-channel ``uint8`` image containing only 0 and 255
            is stored as bilevel. ``dpi`` is one number or ``(x, y)``.
        out_path (str | Path): Destination PDF.
        bilevel_codec (BilevelCodec): ``"jbig2"`` or ``"g4"`` for bilevel pages.
        colour_quality (int): JPEG quality, 1 to 100, for grey and colour pages.
        text_layers (Sequence[TextLayer | None] | None): One entry per page, or
            ``None`` for no text.

    Raises:
        ValueError: On an unknown codec, a mismatched ``text_layers`` length, a
            non-positive dpi or an image that cannot be read or has an unsupported
            shape.
    """
    if bilevel_codec not in ("jbig2", "g4"):
        raise ValueError(f"Unknown bilevel codec: {bilevel_codec!r}")
    if text_layers is not None and len(text_layers) != len(pages):
        raise ValueError("text_layers must have one entry per page")

    pdf = pikepdf.Pdf.new()
    for index, (source, dpi) in enumerate(pages):
        image = _load(source)
        dpi_x, dpi_y = _dpi_pair(dpi)
        text = text_layers[index] if text_layers is not None else None
        _add_page(pdf, image, dpi_x, dpi_y, bilevel_codec, colour_quality, text)

    pdf.save(
        str(out_path),
        object_stream_mode=pikepdf.ObjectStreamMode.generate,
        compress_streams=True,
    )


def _dpi_pair(dpi: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(dpi, tuple | list):
        dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
    else:
        dpi_x = dpi_y = float(dpi)
    if not (np.isfinite(dpi_x) and np.isfinite(dpi_y) and min(dpi_x, dpi_y) > 0):
        raise ValueError(f"dpi must be positive, got {dpi!r}")
    return dpi_x, dpi_y


def _load(source: PageSource) -> NDArray[Any]:
    if isinstance(source, np.ndarray):
        image = source
    else:
        try:
            data = np.fromfile(str(source), dtype=np.uint8)
        except OSError as error:
            raise ValueError(f"Cannot read image: {source}") from error
        decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
        if decoded is None:
            raise ValueError(f"Cannot read image: {source}")
        image = decoded
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.size == 0:
        raise ValueError("Image is empty")
    return image


def _is_bilevel(image: NDArray[Any]) -> bool:
    if image.ndim != 2:
        return False
    if image.dtype == np.bool_:
        return True
    if image.dtype != np.uint8:
        return False
    return bool(np.all((image == 0) | (image == 255)))


def _black_mask(image: NDArray[Any]) -> NDArray[np.bool_]:
    if image.dtype == np.bool_:
        return image
    return image == 0


def _encode_g4(black: NDArray[np.bool_]) -> bytes:
    """Return raw CCITT Group 4 data where black pixels are coded as black runs."""
    pil = Image.fromarray(black).convert("1")
    buffer = io.BytesIO()
    pil.save(buffer, format="TIFF", compression="group4")
    buffer.seek(0)
    with Image.open(buffer) as tiff:
        tags = getattr(tiff, "tag_v2")
        offsets, counts = tags[273], tags[279]
    raw = buffer.getvalue()
    return b"".join(raw[o : o + c] for o, c in zip(offsets, counts, strict=True))


def _add_page(
    pdf: pikepdf.Pdf,
    image: NDArray[Any],
    dpi_x: float,
    dpi_y: float,
    bilevel_codec: BilevelCodec,
    colour_quality: int,
    text: TextLayer | None,
) -> None:
    height, width = image.shape[:2]
    page_w = width / dpi_x * PDF_POINTS_PER_INCH
    page_h = height / dpi_y * PDF_POINTS_PER_INCH

    if _is_bilevel(image):
        xobject = _bilevel_xobject(pdf, image, bilevel_codec, dpi_x, dpi_y)
    else:
        xobject = _jpeg_xobject(pdf, image, colour_quality)

    resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=xobject))
    parts: list[bytes] = []
    if text is not None and text.content:
        fonts = pikepdf.Dictionary()
        for name, font in text.resources.get("/Font", {}).items():
            fonts[f"/{name.lstrip('/')}"] = _import_font(pdf, font)
        if len(fonts):
            resources["/Font"] = fonts
        parts.append(b"q\n" + text.content + b"\nQ\n")
    parts.append(f"q {page_w:.6f} 0 0 {page_h:.6f} 0 0 cm /Im0 Do Q\n".encode("ascii"))
    contents = pdf.make_stream(b"".join(parts))

    pdf.pages.append(
        pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[0, 0, page_w, page_h],
                Resources=resources,
                Contents=contents,
            )
        )
    )


def _import_font(pdf: pikepdf.Pdf, font: Any) -> pikepdf.Object:
    if isinstance(font, pikepdf.Object):
        try:
            return pdf.copy_foreign(font)
        except ValueError:
            return pdf.make_indirect(font)
    return pdf.make_indirect(pikepdf.Object.parse(_serialise(font)))


def _serialise(value: Any) -> bytes:
    if isinstance(value, Mapping):
        body = b" ".join(
            b"/" + str(k).lstrip("/").encode("ascii") + b" " + _serialise(v)
            for k, v in value.items()
        )
        return b"<< " + body + b" >>"
    if isinstance(value, list | tuple):
        return b"[ " + b" ".join(_serialise(v) for v in value) + b" ]"
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, int | float):
        return repr(value).encode("ascii")
    text = str(value)
    if text.startswith("/"):
        return text.encode("ascii")
    return (
        b"("
        + text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .encode("latin-1")
        + b")"
    )


def _bilevel_xobject(
    pdf: pikepdf.Pdf,
    image: NDArray[Any],
    codec: BilevelCodec,
    dpi_x: float,
    dpi_y: float,
) -> pikepdf.Stream:
    black = _black_mask(image)
    height, width = black.shape
    if codec == "jbig2":
        metadata_dpi = (max(1, round(dpi_x)), max(1, round(dpi_y)))
        stream = pdf.make_stream(encode_pdf_stream(black, dpi=metadata_dpi))
        stream["/Filter"] = pikepdf.Name.JBIG2Decode
    else:
        stream = pdf.make_stream(_encode_g4(black))
        stream["/Filter"] = pikepdf.Name.CCITTFaxDecode
        stream["/DecodeParms"] = pikepdf.Dictionary(
            K=-1, Columns=width, Rows=height, BlackIs1=False
        )
    stream["/Type"] = pikepdf.Name.XObject
    stream["/Subtype"] = pikepdf.Name.Image
    stream["/Width"] = width
    stream["/Height"] = height
    stream["/ColorSpace"] = pikepdf.Name.DeviceGray
    stream["/BitsPerComponent"] = 1
    return stream


def _jpeg_xobject(
    pdf: pikepdf.Pdf, image: NDArray[Any], quality: int
) -> pikepdf.Stream:
    if image.dtype != np.uint8:
        raise ValueError(f"Grey and colour pages must be uint8, got {image.dtype}")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    height, width = image.shape[:2]
    stream = pdf.make_stream(encoded.tobytes())
    stream["/Type"] = pikepdf.Name.XObject
    stream["/Subtype"] = pikepdf.Name.Image
    stream["/Width"] = width
    stream["/Height"] = height
    stream["/ColorSpace"] = (
        pikepdf.Name.DeviceGray if image.ndim == 2 else pikepdf.Name.DeviceRGB
    )
    stream["/BitsPerComponent"] = 8
    stream["/Filter"] = pikepdf.Name.DCTDecode
    return stream
