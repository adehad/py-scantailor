"""The despeckling preview marks the ink that a level removes."""

import numpy as np
import pytest
from PySide6.QtCore import QThreadPool
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.despeckle_preview import (
    SPECKLE_RGB,
    DespecklePreviewer,
    despeckle_preview,
)
from scantailor.filters.output.despeckle import DespeckleLevel, DespeckleStrength
from scantailor.imageproc.despeckle import despeckle

DPI = 300
SPECK = (slice(20, 22), slice(20, 22))
BAR = (slice(100, 160), slice(40, 240))


def _page() -> np.ndarray:
    page = np.full((300, 300), 255, dtype=np.uint8)
    page[BAR] = 0
    page[SPECK] = 0
    return page


class TestPreview:
    def test_a_speck_that_goes_is_red_and_the_text_stays_black(self):
        preview = despeckle_preview(_page(), DPI, DespeckleLevel.AGGRESSIVE)

        assert preview.shape == (300, 300, 3)
        assert np.all(preview[SPECK] == SPECKLE_RGB)
        assert np.all(preview[BAR] == 0)

    def test_background_stays_white(self):
        preview = despeckle_preview(_page(), DPI, DespeckleLevel.AGGRESSIVE)

        assert np.all(preview[0, 150] == 255)
        assert np.all(preview[299, 299] == 255)

    def test_the_off_level_marks_nothing(self):
        preview = despeckle_preview(_page(), DPI, DespeckleLevel.OFF)

        assert not np.any(np.all(preview == SPECKLE_RGB, axis=-1))
        assert np.all(preview[SPECK] == 0)

    def test_a_stronger_level_never_marks_less_than_a_weaker_one(self):
        rng = np.random.default_rng(4)
        page = _page()
        for _ in range(60):
            y, x = rng.integers(180, 290), rng.integers(10, 290)
            page[y : y + rng.integers(1, 6), x : x + rng.integers(1, 6)] = 0

        counts = [
            int(
                np.all(
                    despeckle_preview(page, DPI, level) == SPECKLE_RGB, axis=-1
                ).sum()
            )
            for level in (
                DespeckleLevel.OFF,
                DespeckleLevel.CAUTIOUS,
                DespeckleLevel.NORMAL,
                DespeckleLevel.AGGRESSIVE,
            )
        ]

        assert counts == sorted(counts)
        assert counts[0] == 0
        assert counts[-1] > 0

    @pytest.mark.parametrize("factor", [0.5, 1.7, 2.3, 3.5])
    def test_an_in_between_strength_marks_what_the_generator_removes(self, factor):
        page = _page()
        page[200:204, 100:104] = 0

        preview = despeckle_preview(page, DPI, DespeckleStrength(factor))

        removed = (page == 0) & (despeckle(page, DPI, factor) != 0)
        marked = np.all(preview == SPECKLE_RGB, axis=-1)
        assert np.array_equal(marked, removed)

    def test_a_zero_strength_marks_nothing(self):
        preview = despeckle_preview(_page(), DPI, DespeckleStrength(0.0))

        assert not np.any(np.all(preview == SPECKLE_RGB, axis=-1))

    def test_the_page_is_not_modified(self):
        page = _page()
        before = page.copy()

        despeckle_preview(page, DPI, DespeckleLevel.AGGRESSIVE)

        assert np.array_equal(page, before)


class TestPreviewer:
    def test_delivers_the_preview_on_the_calling_thread(self, qtbot: QtBot):
        previewer = DespecklePreviewer()

        with qtbot.waitSignal(previewer.preview_ready, timeout=10000) as signal:
            previewer.request(_page(), DPI, DespeckleLevel.AGGRESSIVE)

        (preview,) = signal.args
        assert np.all(preview[SPECK] == SPECKLE_RGB)

    def test_only_the_newest_request_is_delivered(self, qtbot: QtBot):
        previewer = DespecklePreviewer()
        delivered: list[np.ndarray] = []
        previewer.preview_ready.connect(delivered.append)

        previewer.request(_page(), DPI, DespeckleLevel.OFF)
        previewer.request(_page(), DPI, DespeckleLevel.AGGRESSIVE)
        QThreadPool.globalInstance().waitForDone(10000)
        qtbot.waitUntil(lambda: bool(delivered), timeout=5000)
        qtbot.wait(100)

        assert len(delivered) == 1
        assert np.all(delivered[0][SPECK] == SPECKLE_RGB)

    @pytest.mark.parametrize("level", list(DespeckleLevel))
    def test_every_level_gives_a_full_size_image(
        self, qtbot: QtBot, level: DespeckleLevel
    ):
        previewer = DespecklePreviewer()

        with qtbot.waitSignal(previewer.preview_ready, timeout=10000) as signal:
            previewer.request(_page(), DPI, level)

        assert signal.args[0].shape == (300, 300, 3)
