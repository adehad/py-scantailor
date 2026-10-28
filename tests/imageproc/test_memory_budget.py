"""Pixel kernels stay within a fixed number of bytes per pixel.

Each budget is ``K * pixels`` plus a constant: K counts the page-sized arrays
the kernel's contract needs (its output and the masks or labels it must keep),
and the constant covers one row band. A full-page float or int64 temporary
would add 4 to 8 bytes a pixel and break the budget.
"""

import tracemalloc
from collections.abc import Callable

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.output.generator import _replace_brightness
from scantailor.filters.output.mixed import compose_mixed
from scantailor.imageproc import row_bands
from scantailor.imageproc.binary_threshold import to_gray
from scantailor.imageproc.despeckle import despeckle
from scantailor.imageproc.filters import bgr_to_gray, savgol_filter_2d, wiener_filter

_SLACK = 64 << 10
_BAND_BYTES = 1 << 20
# One band of temporaries; the bytes-per-row estimates the kernels give the
# band helper are within a factor of four of what they allocate.
_BAND_ALLOWANCE = 8 * _BAND_BYTES
_HEIGHT, _WIDTH = 2400, 1800


def _peak_bytes(kernel: Callable[[], object]) -> int:
    """Traced peak of one call, above what was allocated before it."""
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        kernel()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak - before


@pytest.fixture
def small_bands(monkeypatch: pytest.MonkeyPatch, one_opencv_thread: None) -> None:
    """Bands of about a megabyte, so a band is small against the page."""
    monkeypatch.setattr(row_bands, "_BAND_BYTES", _BAND_BYTES)
    monkeypatch.setattr(row_bands, "_MIN_HALOS_PER_BAND", 1)


def _colour_page() -> NDArray[np.uint8]:
    return np.random.default_rng(0).integers(0, 256, (_HEIGHT, _WIDTH, 3), np.uint8)


def _gray_page() -> NDArray[np.uint8]:
    return cv2.GaussianBlur(bgr_to_gray(_colour_page()), (7, 7), 0)


def _text_page(height: int, width: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(1)
    page = np.full((height, width), 255, dtype=np.uint8)
    for top in range(30, height - 10, 40):
        cv2.putText(
            page, "jiggly words.", (5, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2
        )
    specks = height * width // 2000
    page[rng.integers(0, height, specks), rng.integers(0, width, specks)] = 0
    return page


def _bgr_to_gray() -> Callable[[], object]:
    page = _colour_page()
    return lambda: bgr_to_gray(page)


def _to_gray() -> Callable[[], object]:
    page = _colour_page()
    return lambda: to_gray(page)


def _brightness() -> Callable[[], object]:
    page = _colour_page()
    brightness = bgr_to_gray(page)
    return lambda: _replace_brightness(page, brightness)


def _compose_mixed() -> Callable[[], object]:
    page = _colour_page()
    gray = _gray_page()
    text_region = gray > 30
    ink = gray < 60
    return lambda: compose_mixed(page, text_region, ink)


def _savgol() -> Callable[[], object]:
    gray = _gray_page()
    return lambda: savgol_filter_2d(gray, 11, 4, 4)


def _wiener() -> Callable[[], object]:
    gray = _gray_page()
    return lambda: wiener_filter(gray, 5, 30.0)


# Bytes per pixel each kernel's contract needs: the uint8 output, plus for
# brightness the int16 weighted sum and delta beside the BGR copy, and for
# compose_mixed the BGR copy and its masks.
KERNELS: dict[str, tuple[int, Callable[[], Callable[[], object]]]] = {
    "bgr_to_gray": (1, _bgr_to_gray),
    "to_gray": (1, _to_gray),
    "_replace_brightness": (7, _brightness),
    "compose_mixed": (9, _compose_mixed),
    "savgol_filter_2d": (1, _savgol),
    "wiener_filter": (1, _wiener),
}


@pytest.mark.parametrize("name", list(KERNELS))
def test_kernel_peak_is_within_its_bytes_per_pixel(
    small_bands: None, name: str
) -> None:
    bytes_per_pixel, setup = KERNELS[name]
    kernel = setup()

    peak = _peak_bytes(kernel)

    assert peak <= bytes_per_pixel * _HEIGHT * _WIDTH + _BAND_ALLOWANCE + _SLACK


@pytest.mark.parametrize(("dpi", "level"), [(150, 1.0), (300, 3.0)])
def test_despeckle_keeps_eleven_bytes_a_pixel_plus_one_band(
    small_bands: None, dpi: int, level: float
) -> None:
    # Labels and padded node ids are int32, 8 bytes; the ink mask, the result
    # and its copy into the page 3 more.
    height, width = 4800, 900
    page = _text_page(height, width)

    peak = _peak_bytes(lambda: despeckle(page, dpi, level))

    assert peak <= 11 * height * width + _BAND_ALLOWANCE + _SLACK
