"""Tests for proximity-aware despeckling of bilevel images."""

import time

import numpy as np
import pytest

from scantailor.imageproc.despeckle import despeckle

CAUTIOUS = 1.0
NORMAL = 2.0
AGGRESSIVE = 3.0


def _page(height: int = 300, width: int = 300) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _ink(page: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    page[top : top + h, left : left + w] = 0


def _has_ink(image: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    return bool((image[top : top + h, left : left + w] == 0).any())


class TestProximityRules:
    """Small marks survive when they sit next to something substantial."""

    @pytest.mark.parametrize("level", [CAUTIOUS, NORMAL])
    def test_i_dot_above_stem_is_kept(self, level: float) -> None:
        page = _page()
        _ink(page, 150, 100, 40, 4)  # stem
        _ink(page, 136, 100, 4, 4)  # dot, 10 px above the stem

        result = despeckle(page, 300, level)

        assert _has_ink(result, 136, 100, 4, 4)
        assert _has_ink(result, 150, 100, 40, 4)

    @pytest.mark.parametrize("level", [CAUTIOUS, NORMAL, AGGRESSIVE])
    def test_isolated_speck_far_from_text_is_removed(self, level: float) -> None:
        page = _page()
        _ink(page, 150, 100, 40, 4)
        _ink(page, 20, 250, 4, 4)  # same size as the i-dot, far away

        result = despeckle(page, 300, level)

        assert not _has_ink(result, 20, 250, 4, 4)
        assert _has_ink(result, 150, 100, 40, 4)

    @pytest.mark.parametrize("level", [CAUTIOUS, NORMAL])
    def test_full_stop_after_word_is_kept(self, level: float) -> None:
        page = _page()
        for column in (50, 62, 74):
            _ink(page, 100, column, 30, 8)  # glyphs
        _ink(page, 125, 88, 5, 5)  # full stop 6 px after the last glyph

        result = despeckle(page, 300, level)

        assert _has_ink(result, 125, 88, 5, 5)

    @pytest.mark.parametrize("level", [CAUTIOUS, NORMAL])
    def test_arabic_dot_cluster_above_stroke_is_kept(self, level: float) -> None:
        page = _page()
        _ink(page, 150, 100, 6, 40)  # horizontal stroke
        for top, left in ((132, 110), (132, 118), (126, 114)):
            _ink(page, top, left, 3, 3)

        result = despeckle(page, 300, level)

        for top, left in ((132, 110), (132, 118), (126, 114)):
            assert _has_ink(result, top, left, 3, 3)

    def test_speck_chained_through_another_small_mark_is_kept(self) -> None:
        page = _page()
        _ink(page, 150, 100, 40, 4)
        _ink(page, 140, 100, 4, 4)  # attaches to the stem
        _ink(page, 130, 100, 3, 3)  # attaches only to the dot above

        result = despeckle(page, 300, CAUTIOUS)

        assert _has_ink(result, 130, 100, 3, 3)

    def test_blob_at_big_object_size_is_kept_and_distant_smaller_blob_is_removed(
        self,
    ) -> None:
        page = _page()
        _ink(page, 150, 100, 7, 7)  # exactly big-object size at CAUTIOUS
        _ink(page, 150, 200, 5, 5)  # smaller blob, far from it

        result = despeckle(page, 300, CAUTIOUS)

        assert _has_ink(result, 150, 100, 7, 7)
        assert not _has_ink(result, 150, 200, 5, 5)

    def test_image_without_any_big_component_is_cleared(self) -> None:
        page = _page(50, 50)
        _ink(page, 20, 20, 3, 3)

        result = despeckle(page, 300, CAUTIOUS)

        assert (result == 255).all()

    def test_blank_image_is_unchanged(self) -> None:
        page = _page()

        assert np.array_equal(despeckle(page, 300, NORMAL), page)


class TestLevels:
    """Higher levels remove more."""

    def test_aggressive_removes_more_than_cautious_on_noise_field(self) -> None:
        rng = np.random.default_rng(7)
        page = _page(600, 600)
        _ink(page, 300, 50, 12, 500)  # a rule to be near
        for _ in range(150):
            top = int(rng.integers(0, 590))
            left = int(rng.integers(0, 590))
            _ink(page, top, left, 3, 3)

        cautious = despeckle(page, 300, CAUTIOUS)
        aggressive = despeckle(page, 300, AGGRESSIVE)

        assert (aggressive == 0).sum() < (cautious == 0).sum()
        assert (cautious == 0).sum() < (page == 0).sum()

    def test_off_returns_identical_copy(self) -> None:
        page = _page()
        _ink(page, 20, 20, 2, 2)

        result = despeckle(page, 300, 0.0)

        assert np.array_equal(result, page)
        assert result is not page

    def test_input_is_not_modified(self) -> None:
        page = _page()
        _ink(page, 150, 100, 40, 4)
        _ink(page, 20, 250, 4, 4)
        before = page.copy()

        despeckle(page, 300, AGGRESSIVE)

        assert np.array_equal(page, before)

    def test_output_values_stay_bilevel_and_ink_only_shrinks(self) -> None:
        rng = np.random.default_rng(3)
        page = np.where(rng.random((200, 200)) < 0.05, 0, 255).astype(np.uint8)
        _ink(page, 80, 20, 30, 150)

        result = despeckle(page, 300, NORMAL)

        assert set(np.unique(result)) <= {0, 255}
        assert not ((result == 0) & (page != 0)).any()


class TestDpiScaling:
    """Thresholds follow the resolution so a physical page decides alike."""

    @staticmethod
    def _scaled_page(dpi: int) -> tuple[np.ndarray, dict[str, tuple[int, ...]]]:
        s = dpi / 300
        page = _page(round(300 * s), round(300 * s))
        marks = {
            "stem": (150, 100, 40, 4),
            "dot": (136, 100, 4, 4),
            "far_speck": (20, 250, 4, 4),
        }
        scaled = {k: tuple(round(v * s) for v in box) for k, box in marks.items()}
        for box in scaled.values():
            _ink(page, *box)
        return page, scaled

    @pytest.mark.parametrize("level", [CAUTIOUS, NORMAL])
    def test_same_physical_page_gives_same_decisions_at_150_and_300_dpi(
        self, level: float
    ) -> None:
        decisions = {}
        for dpi in (150, 300):
            page, boxes = self._scaled_page(dpi)
            result = despeckle(page, dpi, level)
            decisions[dpi] = {
                name: _has_ink(result, *box) for name, box in boxes.items()
            }

        assert decisions[150] == decisions[300]
        assert decisions[300] == {"stem": True, "dot": True, "far_speck": False}

    def test_non_positive_dpi_falls_back_to_300(self) -> None:
        page, _ = self._scaled_page(300)

        assert np.array_equal(despeckle(page, 0, NORMAL), despeckle(page, 300, NORMAL))


@pytest.mark.slow
class TestPerformance:
    def test_full_page_with_twenty_thousand_specks_is_fast(self) -> None:
        rng = np.random.default_rng(0)
        page = _page(3000, 2100)
        for top in range(200, 2800, 80):
            for left in range(150, 1950, 40):
                _ink(page, top, left, 24, 12)
        page[rng.integers(0, 3000, 20000), rng.integers(0, 2100, 20000)] = 0

        elapsed = float("inf")
        for _ in range(2):
            start = time.perf_counter()
            result = despeckle(page, 300, NORMAL)
            elapsed = min(elapsed, time.perf_counter() - start)

        assert elapsed < 4.0  # loose: ~1.3 s idle, must not flake under CPU load
        assert (result == 0).sum() < (page == 0).sum()
