"""Kernels cut into row bands give the whole image's pixels on any thread count."""

import itertools
import time
from collections.abc import Callable

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc import row_bands
from scantailor.imageproc.binarize import (
    binarize_sauvola,
    binarize_window,
    binarize_wolf,
)
from scantailor.imageproc.despeckle import despeckle
from scantailor.imageproc.filters import savgol_filter_2d, wiener_filter
from scantailor.imageproc.row_bands import RowBand

Kernel = Callable[[NDArray[np.uint8]], NDArray[np.uint8]]


def _gray_page(height: int, width: int, seed: int = 11) -> NDArray[np.uint8]:
    """Noise over a smooth gradient, so every window sees a different mix."""
    rng = np.random.default_rng(seed)
    gradient = (
        np.linspace(60, 220, width)[None, :] + np.linspace(-40, 30, height)[:, None]
    )
    return (gradient + rng.normal(0, 35, (height, width))).clip(0, 255).astype(np.uint8)


def _speckled_page(height: int, width: int, seed: int = 5) -> NDArray[np.uint8]:
    """Words, dots beside them and loose specks, some on the page edges."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width), 255, dtype=np.uint8)
    for top in range(20, height - 20, 45):
        cv2.putText(page, "jiggly.", (10, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    specks = 300
    page[rng.integers(0, height, specks), rng.integers(0, width, specks)] = 0
    page[0, 40:60] = 0
    page[-1, 100:104] = 0
    return page


KERNELS: dict[str, tuple[Kernel, Callable[[int, int], NDArray[np.uint8]]]] = {
    "despeckle cautious": (lambda page: despeckle(page, 150, 1.0), _speckled_page),
    "despeckle aggressive": (lambda page: despeckle(page, 150, 3.0), _speckled_page),
    "savgol 7": (lambda page: savgol_filter_2d(page, 7, 2, 2), _gray_page),
    "savgol 5x11": (lambda page: savgol_filter_2d(page, (5, 11), 3, 4), _gray_page),
    "wiener 5": (lambda page: wiener_filter(page, 5, 30.0), _gray_page),
    "wiener 3x9": (lambda page: wiener_filter(page, (3, 9), 10.0), _gray_page),
    "sauvola 25": (lambda page: binarize_sauvola(page, 25, 0.34), _gray_page),
    "sauvola 9x41": (
        lambda page: binarize_sauvola(page, (9, 41), 0.2, 4.0),
        _gray_page,
    ),
    "wolf 25": (lambda page: binarize_wolf(page, 25, 0.3), _gray_page),
    "wolf 301": (lambda page: binarize_wolf(page, 301, 0.5, -3.0), _gray_page),
    "window 25": (lambda page: binarize_window(page, 25, 0.33), _gray_page),
    "window 2": (lambda page: binarize_window(page, 2, 0.2, 5.0), _gray_page),
}


@pytest.fixture
def band_counts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many bands each banded call cuts."""
    counts: list[int] = []
    cut = row_bands.row_bands

    def counting(rows: int, halo: int, bytes_per_row: int) -> list[RowBand]:
        bands = cut(rows, halo, bytes_per_row)
        counts.append(len(bands))
        return bands

    monkeypatch.setattr(row_bands, "row_bands", counting)
    return counts


@pytest.mark.parametrize("name", list(KERNELS))
def test_one_thread_and_eight_threads_cut_the_same_bands_and_give_the_same_pixels(
    monkeypatch: pytest.MonkeyPatch, band_counts: list[int], name: str
) -> None:
    kernel, make_page = KERNELS[name]
    page = make_page(1400, 420)
    monkeypatch.setattr(row_bands, "_BAND_BYTES", 256 << 10)
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 1)
    one_thread = kernel(page)
    counts_on_one = list(band_counts)
    band_counts.clear()
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 8)

    eight_threads = kernel(page)

    assert max(counts_on_one) > 1
    assert band_counts == counts_on_one
    assert np.array_equal(eight_threads, one_thread)


@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize(
    "height",
    [1, 3, 64],
    ids=["one-row bands", "bands under the window", "64-row bands"],
)
def test_bands_of_any_height_give_the_pixels_of_one_band_over_the_whole_image(
    band_height: Callable[[int], None], name: str, height: int
) -> None:
    kernel, make_page = KERNELS[name]
    # 203 rows is a multiple of none of the band heights.
    page = make_page(203, 150)
    band_height(0)
    whole = kernel(page)
    band_height(height)

    banded = kernel(page)

    assert np.array_equal(banded, whole)


def test_a_non_contiguous_view_is_banded_like_a_copy(
    band_height: Callable[[int], None],
) -> None:
    band_height(16)
    page = _gray_page(400, 600)[::2, 1::3]

    assert np.array_equal(
        savgol_filter_2d(page, 7, 2, 2), savgol_filter_2d(page.copy(), 7, 2, 2)
    )
    assert np.array_equal(
        wiener_filter(page, 5, 9.0), wiener_filter(page.copy(), 5, 9.0)
    )


class TestRowBands:
    def test_bands_cover_every_row_once_in_order(self) -> None:
        bands = row_bands.row_bands(1031, 4, 1 << 20)

        assert bands[0].first == 0
        assert bands[-1].stop == 1031
        assert all(a.stop == b.first for a, b in itertools.pairwise(bands))

    def test_a_band_reads_its_halo_clipped_at_the_image_edges(self) -> None:
        bands = row_bands.row_bands(1000, 10, 1 << 20)

        assert (bands[0].top, bands[0].bottom) == (0, bands[0].stop + 10)
        assert (bands[1].top, bands[1].bottom) == (
            bands[1].first - 10,
            bands[1].stop + 10,
        )
        assert bands[-1].bottom == 1000
        assert bands[1].inside == slice(10, 10 + bands[1].stop - bands[1].first)

    def test_bands_are_sized_by_bytes_per_row(self) -> None:
        narrow = row_bands.row_bands(10_000, 1, 1_000)
        wide = row_bands.row_bands(10_000, 1, 10_000)

        assert len(narrow) == 1
        assert len(wide) == 2
        assert all(
            (band.stop - band.first) * 10_000 <= 2 * row_bands._BAND_BYTES
            for band in wide
        )

    def test_a_band_is_never_shorter_than_eight_halos_unless_the_image_is(self) -> None:
        bands = row_bands.row_bands(10_000, 100, 1 << 30)

        assert min(band.stop - band.first for band in bands) >= 800
        assert row_bands.row_bands(300, 100, 1 << 30) == [RowBand(0, 300, 0, 300)]

    def test_an_image_without_rows_is_one_empty_band(self) -> None:
        assert row_bands.row_bands(0, 5, 100) == [RowBand(0, 0, 0, 0)]

    @pytest.mark.parametrize("threads", [1, 3, 8, 64])
    def test_the_thread_count_does_not_change_the_bands_or_their_order(
        self, monkeypatch: pytest.MonkeyPatch, threads: int
    ) -> None:
        monkeypatch.setattr(cv2, "getNumThreads", lambda: threads)

        firsts = row_bands.map_row_bands(5000, 2, 64 << 10, lambda band: band.first)

        assert firsts == [band.first for band in row_bands.row_bands(5000, 2, 64 << 10)]


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
