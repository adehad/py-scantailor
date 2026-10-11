"""Compression and speed of the encoder against CCITT G4 from Pillow (libtiff)."""

import io
import time
from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image
from sample_images import scanned_page, text_page

from jbig2_numba import encode_pdf_stream

# libtiff's G4 coder is a tight C loop over run lengths while MQ coding
# visits every pixel of every non-typical row, so parity is not expected. The
# factor guards against regressions such as losing numba compilation, which
# costs two orders of magnitude.
_MAX_SLOWDOWN_VS_G4 = 40.0


def _g4_strip(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(~image).convert("1").save(buffer, format="TIFF", compression="group4")
    buffer.seek(0)
    with Image.open(buffer) as tiff:
        offsets, counts = tiff.tag_v2[273], tiff.tag_v2[279]
    data = buffer.getvalue()
    return b"".join(data[o : o + c] for o, c in zip(offsets, counts, strict=True))


def _best_time(function: Callable[[], object], repeats: int = 5) -> float:
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


@pytest.fixture(scope="module")
def text_page_2100x3000() -> np.ndarray:
    return text_page(2100, 3000)


@pytest.mark.parametrize("make_page", [text_page, scanned_page], ids=["text_page", "scanned_page"])
def test_stream_is_smaller_than_ccitt_g4(make_page, report_measurement) -> None:
    image = make_page()

    jbig2 = encode_pdf_stream(image)
    g4 = _g4_strip(image)

    report_measurement(
        f"{make_page.__name__} {image.shape[1]}x{image.shape[0]}: JBIG2 {len(jbig2)} B, "
        f"G4 {len(g4)} B, ratio {len(jbig2) / len(g4):.3f}"
    )
    assert len(jbig2) < len(g4)


def test_encode_time_is_within_a_factor_of_g4(text_page_2100x3000, report_measurement) -> None:
    image = text_page_2100x3000

    jbig2_seconds = _best_time(lambda: encode_pdf_stream(image))
    g4_seconds = _best_time(lambda: _g4_strip(image))

    report_measurement(
        "encode 2100x3000 text page (best of 5 after warm-up): "
        f"JBIG2 {jbig2_seconds * 1000:.1f} ms, "
        f"Pillow G4 {g4_seconds * 1000:.1f} ms, ratio {jbig2_seconds / g4_seconds:.1f}"
    )
    assert jbig2_seconds < _MAX_SLOWDOWN_VS_G4 * g4_seconds
