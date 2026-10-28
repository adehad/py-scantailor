"""Compression and speed of the encoder against CCITT G4 from Pillow (libtiff)."""

import io
import sys
import time
from collections.abc import Callable
from types import CodeType, FrameType

import numpy as np
import pytest
from PIL import Image
from sample_images import scanned_page, text_page

from jbig2 import encode_pdf_stream, mq_coder

# Inside a run of equal decisions the coder codes every decision up to the
# next renormalisation with one subtraction, so its loop runs about a dozen
# lines per run on a text page; coding each decision on its own takes twice as
# many.
_MAX_ENCODE_LINES_PER_RUN = 16


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


def _lines_run_in(code: CodeType, function: Callable[[], object]) -> int:
    """Count the source lines ``function`` executes inside ``code``."""
    lines = 0

    def count_lines(_frame: FrameType, event: str, _arg: object) -> object:
        nonlocal lines
        if event == "line":
            lines += 1
        return count_lines

    def enter(frame: FrameType, _event: str, _arg: object) -> object:
        return count_lines if frame.f_code is code else None

    sys.settrace(enter)
    try:
        function()
    finally:
        sys.settrace(None)
    return lines


def test_the_mq_coder_runs_a_few_lines_per_run_of_equal_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = text_page(700, 1000)
    runs = 0
    split = mq_coder._runs

    def counting(contexts: np.ndarray, bits: np.ndarray):
        nonlocal runs
        found = split(contexts, bits)
        runs += len(found[0])
        return found

    monkeypatch.setattr(mq_coder, "_runs", counting)

    lines = _lines_run_in(mq_coder.MQEncoder.encode.__code__, lambda: encode_pdf_stream(image))

    assert runs > 10_000
    assert lines <= _MAX_ENCODE_LINES_PER_RUN * runs


@pytest.mark.benchmark
def test_encode_time_against_g4(text_page_2100x3000, report_measurement) -> None:
    image = text_page_2100x3000

    jbig2_seconds = _best_time(lambda: encode_pdf_stream(image))
    g4_seconds = _best_time(lambda: _g4_strip(image))

    report_measurement(
        "encode 2100x3000 text page (best of 5 after warm-up): "
        f"JBIG2 {jbig2_seconds * 1000:.1f} ms, "
        f"Pillow G4 {g4_seconds * 1000:.1f} ms, ratio {jbig2_seconds / g4_seconds:.1f}"
    )
