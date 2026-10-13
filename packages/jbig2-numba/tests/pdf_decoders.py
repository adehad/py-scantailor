"""Wrap JBIG2 streams in a minimal PDF and decode them with MuPDF and poppler.

Both decoders read the image XObject itself (MuPDF via ``Pixmap(doc, xref)``,
poppler via ``pdfimages -png``) so no page rendering or resampling is
involved. With ``/DeviceGray`` a decoded sample of 0 is black.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image

POPPLER_MISSING_REASON = "poppler's pdfimages is not on PATH; install poppler-utils"


@dataclass(frozen=True)
class EmbeddedImage:
    """One ``/JBIG2Decode`` stream with its declared size."""

    stream: bytes
    width: int
    height: int


@dataclass(frozen=True)
class TestPdf:
    """A built PDF and the object numbers of its image XObjects, one per page."""

    __test__ = False

    data: bytes
    image_xrefs: list[int]


def build_pdf(images: list[EmbeddedImage]) -> TestPdf:
    """Build a PDF with one page per image, each painting its image full-page.

    Args:
        images (list[EmbeddedImage]): Streams to embed, in page order.

    Returns:
        TestPdf: PDF bytes and image object numbers.
    """
    objects: dict[int, bytes] = {}
    page_numbers = []
    image_xrefs = []
    next_number = 3
    for image in images:
        page, content, xobject = next_number, next_number + 1, next_number + 2
        next_number += 3
        page_numbers.append(page)
        image_xrefs.append(xobject)
        objects[page] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {image.width} {image.height}]"
            f" /Resources << /XObject << /Im0 {xobject} 0 R >> >> /Contents {content} 0 R >>"
        ).encode()
        paint = f"q {image.width} 0 0 {image.height} 0 0 cm /Im0 Do Q".encode()
        objects[content] = _stream(b"", paint)
        objects[xobject] = _stream(
            (
                f"/Type /XObject /Subtype /Image /Width {image.width} /Height {image.height}"
                " /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /JBIG2Decode"
            ).encode(),
            image.stream,
        )
    kids = " ".join(f"{n} 0 R" for n in page_numbers)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode()

    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref_offset = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for number in range(1, size):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return TestPdf(bytes(out), image_xrefs)


def _stream(extra_dict: bytes, data: bytes) -> bytes:
    return (
        b"<< " + extra_dict + f" /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"
    )


def decode_with_pymupdf(pdf: TestPdf) -> list[np.ndarray]:
    """Decode every image with MuPDF.

    Args:
        pdf (TestPdf): PDF from ``build_pdf``.

    Returns:
        list[np.ndarray]: ``bool`` images, ``True`` for black, in page order.
    """
    images = []
    with pymupdf.open(stream=pdf.data, filetype="pdf") as doc:
        for xref in pdf.image_xrefs:
            pix = pymupdf.Pixmap(doc, xref)
            rows = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
            gray = rows[:, : pix.width * pix.n].reshape(pix.height, pix.width, pix.n)[..., 0]
            images.append(gray < 128)
    return images


def decode_with_poppler(pdf: TestPdf, workdir: Path) -> list[np.ndarray]:
    """Decode every image with poppler's ``pdfimages -png``.

    Args:
        pdf (TestPdf): PDF from ``build_pdf``.
        workdir (Path): Empty scratch directory.

    Returns:
        list[np.ndarray]: ``bool`` images, ``True`` for black, in page order.

    Raises:
        AssertionError: If poppler reports an error or emits a different number of images.
        pytest.skip.Exception: If ``pdfimages`` is not installed.
    """
    if shutil.which("pdfimages") is None:
        pytest.skip(POPPLER_MISSING_REASON)
    pdf_path = workdir / "test.pdf"
    pdf_path.write_bytes(pdf.data)
    result = subprocess.run(
        ["pdfimages", "-png", str(pdf_path), str(workdir / "img")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr.strip(), f"poppler warnings: {result.stderr}"
    pngs = sorted(workdir.glob("img-*.png"))
    assert len(pngs) == len(pdf.image_xrefs)
    images = []
    for png in pngs:
        with Image.open(png) as decoded:
            images.append(np.asarray(decoded.convert("L")) < 128)
    return images


def decode_with_both(images: list[EmbeddedImage], workdir: Path) -> dict[str, list[np.ndarray]]:
    """Embed images in one PDF and decode them with both decoders.

    Args:
        images (list[EmbeddedImage]): Streams to embed.
        workdir (Path): Empty scratch directory for poppler.

    Returns:
        dict[str, list[np.ndarray]]: Decoded images keyed by decoder name; poppler
        is left out when ``pdfimages`` is missing, so the MuPDF check still runs.
        ``test_poppler_available`` reports the gap as a skip.
    """
    pdf = build_pdf(images)
    decoded = {"mupdf": decode_with_pymupdf(pdf)}
    if shutil.which("pdfimages") is not None:
        decoded["poppler"] = decode_with_poppler(pdf, workdir)
    return decoded
