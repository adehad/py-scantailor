"""Export output pages to a single PDF.

Each page is placed at its physical size (pixels / dpi * 72 pt). Bilevel pages are
stored losslessly as JBIG2 or CCITT G4, grey and colour pages as JPEG.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Literal

import cv2
import numpy as np
import pikepdf
from jbig2 import encode_pdf_stream
from numpy.typing import NDArray
from PIL import Image

from scantailor.core.pdf_outline import carry_outline_and_labels
from scantailor.core.pdf_pool import PdfCancelled, map_pool_progress
from scantailor.core.text_layer import TextLayer

PDF_POINTS_PER_INCH = 72.0
BilevelCodec = Literal["jbig2", "g4"]
PageSource = NDArray[Any] | str | Path
PageInput = tuple[PageSource, "float | tuple[float, float]"]
ExportProgress = Callable[[int, int, str], None]


def export_pdf(
    pages: Sequence[PageInput],
    out_path: str | Path,
    *,
    bilevel_codec: BilevelCodec = "jbig2",
    colour_quality: int = 85,
    text_layers: Sequence[TextLayer | None] | None = None,
    executor: ProcessPoolExecutor | None = None,
    max_workers: int | None = None,
    cancel: Event | None = None,
    on_progress: ExportProgress | None = None,
    outline_source: str | Path | None = None,
) -> None:
    """Write pages to one PDF.

    Args:
        pages (Sequence[tuple[Any, Any]]): ``(image, dpi)`` per page. ``image`` is a
            path, or an array: 2-D ``bool`` (``True`` is black), 2-D ``uint8`` or
            3-channel BGR ``uint8``. A single-channel ``uint8`` image containing only
            0 and 255 is stored as bilevel. ``dpi`` is one number or ``(x, y)``.
        out_path (str | pathlib.Path): Destination PDF.
        bilevel_codec (str): ``"jbig2"`` or ``"g4"`` for bilevel pages.
        colour_quality (int): JPEG quality, 1 to 100, for grey and colour pages.
        text_layers (Sequence[scantailor.core.text_layer.TextLayer | None] | None):
            One entry per page, or
            ``None`` for no text.
        executor (concurrent.futures.ProcessPoolExecutor | None):
            A pool from ``spawn_pool`` whose
            workers load and encode the page images; this process assembles the
            PDF in page order. Pages are encoded in this process when None.
        max_workers (int | None): Size of ``executor``.
        cancel (threading.Event | None):
            Setting it stops the export before the next page
            is encoded or added, and before the file is written.
        on_progress (Callable[[int, int, str], None] | None):
            Called as ``(done, total, phase)``
            from the calling thread: per encoded page on a pool ("Encoding
            pages"), per page added ("Writing PDF"), then once more, with
            ``done == total``, before the file is saved ("Saving PDF").
        outline_source (str | pathlib.Path | None): A PDF whose bookmarks and page
            labels are copied when it has as many pages as this export.

    Raises:
        PdfCancelled: If ``cancel`` was set.
        ValueError: On an unknown codec, a mismatched ``text_layers`` length, a
            non-positive dpi or an image that cannot be read or has an unsupported
            shape.
    """
    if bilevel_codec not in ("jbig2", "g4"):
        raise ValueError(f"Unknown bilevel codec: {bilevel_codec!r}")
    if text_layers is not None and len(text_layers) != len(pages):
        raise ValueError("text_layers must have one entry per page")

    jobs = [
        _EncodeJob(source, *_dpi_pair(dpi), bilevel_codec, colour_quality)
        for source, dpi in pages
    ]
    total = len(jobs)

    def report(done: int, phase: str) -> None:
        if on_progress is not None:
            on_progress(done, total, phase)

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise PdfCancelled

    encoded_count = 0

    def page_encoded(_index: int, _job: _EncodeJob, _page: EncodedPage) -> None:
        nonlocal encoded_count
        encoded_count += 1
        report(encoded_count, "Encoding pages")

    if executor is None:
        encoded: Iterable[EncodedPage] = (_encode_page(job) for job in jobs)
    else:
        encoded = map_pool_progress(
            _encode_page,
            jobs,
            executor,
            max_workers,
            cancel=cancel,
            on_done=page_encoded,
        )

    pdf = pikepdf.Pdf.new()
    for index, page in enumerate(encoded):
        check_cancel()
        text = text_layers[index] if text_layers is not None else None
        _add_page(pdf, page, text)
        report(index + 1, "Writing PDF")
    check_cancel()
    if outline_source is not None:
        carry_outline_and_labels(Path(outline_source), pdf)

    report(total, "Saving PDF")
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


@dataclass(frozen=True)
class _EncodeJob:
    source: PageSource
    dpi_x: float
    dpi_y: float
    codec: BilevelCodec
    quality: int


@dataclass(frozen=True)
class EncodedPage:
    """A page image as stream data plus the dictionary entries that describe it.

    Attributes:
        data (bytes): Image stream data in ``codec_filter``'s encoding.
        codec_filter (str): ``JBIG2Decode``, ``CCITTFaxDecode`` or ``DCTDecode``.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        colour_space (str): ``DeviceGray`` or ``DeviceRGB``.
        bits_per_component (int): 1 for bilevel images, otherwise 8.
        page_width (float): Page width in points.
        page_height (float): Page height in points.
    """

    data: bytes
    codec_filter: str
    width: int
    height: int
    colour_space: str
    bits_per_component: int
    page_width: float
    page_height: float


def _encode_page(job: _EncodeJob) -> EncodedPage:
    """Load one page image and encode it; runs in a worker or in this process."""
    image = _load(job.source)
    height, width = image.shape[:2]
    page_width = width / job.dpi_x * PDF_POINTS_PER_INCH
    page_height = height / job.dpi_y * PDF_POINTS_PER_INCH
    if _is_bilevel(image):
        black = _black_mask(image)
        if job.codec == "jbig2":
            metadata_dpi = (max(1, round(job.dpi_x)), max(1, round(job.dpi_y)))
            data = encode_pdf_stream(black, dpi=metadata_dpi)
            codec_filter = "JBIG2Decode"
        else:
            data = _encode_g4(black)
            codec_filter = "CCITTFaxDecode"
        return EncodedPage(
            data, codec_filter, width, height, "DeviceGray", 1, page_width, page_height
        )
    if image.dtype != np.uint8:
        raise ValueError(f"Grey and colour pages must be uint8, got {image.dtype}")
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(job.quality)]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return EncodedPage(
        encoded.tobytes(),
        "DCTDecode",
        width,
        height,
        "DeviceGray" if image.ndim == 2 else "DeviceRGB",
        8,
        page_width,
        page_height,
    )


def _add_page(pdf: pikepdf.Pdf, page: EncodedPage, text: TextLayer | None) -> None:
    xobject = _image_xobject(pdf, page)
    page_w, page_h = page.page_width, page.page_height

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


def _image_xobject(pdf: pikepdf.Pdf, page: EncodedPage) -> pikepdf.Stream:
    stream = pdf.make_stream(page.data)
    stream["/Type"] = pikepdf.Name.XObject
    stream["/Subtype"] = pikepdf.Name.Image
    stream["/Width"] = page.width
    stream["/Height"] = page.height
    stream["/ColorSpace"] = pikepdf.Name(f"/{page.colour_space}")
    stream["/BitsPerComponent"] = page.bits_per_component
    stream["/Filter"] = pikepdf.Name(f"/{page.codec_filter}")
    if page.codec_filter == "CCITTFaxDecode":
        stream["/DecodeParms"] = pikepdf.Dictionary(
            K=-1, Columns=page.width, Rows=page.height, BlackIs1=False
        )
    return stream
