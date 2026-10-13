"""Filters that split their work between threads give the values of one thread."""

import time

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc.binarize import (
    _ink_below_level,
    binarize_otsu,
    binarize_wolf,
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


@pytest.mark.slow
@pytest.mark.skipif(cv2.getNumThreads() < 4, reason="needs several cores")
def test_wolf_is_faster_on_threads_than_on_one_for_a_full_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured against the same call restricted to one thread in this test."""
    image = _gray_page(5983, 4280)

    def best_of_three() -> float:
        times = []
        for _ in range(3):
            start = time.perf_counter()
            binarize_wolf(image, 25)
            times.append(time.perf_counter() - start)
        return min(times)

    threaded = best_of_three()
    with monkeypatch.context() as single:
        single.setattr(cv2, "getNumThreads", lambda: 1)
        serial = best_of_three()

    assert threaded < serial
