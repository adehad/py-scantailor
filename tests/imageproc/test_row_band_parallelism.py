"""Filters that split their work between threads give the values of one thread."""

import time
from collections.abc import Callable

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc import binarize
from scantailor.imageproc.binarize import (
    _ink_below_level,
    binarize_otsu,
    binarize_wolf,
    run_in_row_bands,
)
from scantailor.imageproc.filters import savgol_filter_2d

_SEED = 11


def _gray_page(height: int, width: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(_SEED)
    return rng.normal(150, 70, (height, width)).clip(0, 255).astype(np.uint8)


class TestSavitzkyGolay:
    """Both passes work on independent lines, so any split gives the same pixels."""

    @pytest.mark.parametrize(
        ("height", "width"), [(900, 700), (1031, 513), (300, 1200)]
    )
    @pytest.mark.parametrize(("window", "degree"), [(7, 2), (5, 2), (9, 3)])
    def test_bands_match_a_single_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        height: int,
        width: int,
        window: int,
        degree: int,
    ) -> None:
        image = _gray_page(height, width)
        with monkeypatch.context() as single:
            single.setattr(cv2, "getNumThreads", lambda: 1)
            expected = savgol_filter_2d(image, window, degree, degree)
        monkeypatch.setattr(cv2, "getNumThreads", lambda: 8)

        banded = savgol_filter_2d(image, window, degree, degree)

        assert np.array_equal(banded, expected)


class TestWolf:
    """Windows read rows beyond a band, so bands must still match the whole image."""

    @pytest.mark.parametrize(
        ("height", "width"), [(1001, 333), (700, 500), (130, 90), (5, 7)]
    )
    @pytest.mark.parametrize("window", [1, 2, 25, 50, 301])
    def test_bands_match_a_single_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        height: int,
        width: int,
        window: int,
    ) -> None:
        image = _gray_page(height, width)
        with monkeypatch.context() as single:
            single.setattr(cv2, "getNumThreads", lambda: 1)
            expected = binarize_wolf(image, window, k=0.3)
        monkeypatch.setattr(cv2, "getNumThreads", lambda: 8)

        banded = binarize_wolf(image, window, k=0.3)

        assert np.array_equal(banded, expected)


class TestInkBelowLevel:
    """A gray level is ink when strictly below the threshold, at any threshold."""

    @pytest.mark.parametrize("level", [-5, 0, 1, 30, 128, 255, 256, 300])
    def test_matches_a_comparison_for_every_level(self, level: int) -> None:
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)

        result = _ink_below_level(image, level)

        assert np.array_equal(result, np.where(image < level, 0, 255))
        assert result.dtype == np.uint8

    def test_otsu_threshold_keeps_pixels_at_the_threshold_as_background(self) -> None:
        image = np.array([[10, 10, 200, 200]], dtype=np.uint8)
        threshold = 100

        result = binarize_otsu(image, adjustment=0, bounds=(threshold, threshold))

        assert result.tolist() == [[0, 0, 255, 255]]


def test_wolf_on_a_full_page_runs_several_bands_and_gives_the_one_thread_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _gray_page(2400, 1700)
    with monkeypatch.context() as single:
        single.setattr(cv2, "getNumThreads", lambda: 1)
        expected = binarize_wolf(image, 25)
    bands: list[tuple[int, int]] = []

    def recording(rows: int, band: Callable[[int, int], None]) -> None:
        def recorded(first: int, stop: int) -> None:
            bands.append((int(first), int(stop)))
            band(first, stop)

        run_in_row_bands(rows, recorded)

    monkeypatch.setattr(binarize, "run_in_row_bands", recording)
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 8)

    threaded = binarize_wolf(image, 25)

    assert len(bands) > 1
    assert sorted(bands)[0][0] == 0 and sorted(bands)[-1][1] == image.shape[0]
    assert np.array_equal(threaded, expected)


@pytest.mark.benchmark
def test_wolf_full_page_time_on_threads_and_on_one(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    image = _gray_page(5983, 4280)

    def best_of_three() -> float:
        times = []
        for _ in range(3):
            start = time.perf_counter()
            binarize_wolf(image, 25)
            times.append(time.perf_counter() - start)
        return min(times)

    record_property("benchmark_seconds_threaded", best_of_three())
    with monkeypatch.context() as single:
        single.setattr(cv2, "getNumThreads", lambda: 1)
        record_property("benchmark_seconds_one_thread", best_of_three())
