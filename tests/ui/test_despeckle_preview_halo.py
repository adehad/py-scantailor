"""The despeckling tab tints the area around removed specks red and shows pictures."""

import numpy as np
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.despeckle_preview import (
    SPECKLE_RGB,
    despeckle_preview,
)
from scantailor.app.ui.image_views.output import ImageViewTab, OutputImageView
from scantailor.filters.output.despeckle import DespeckleLevel

DPI = 300
HEIGHT = 300
WIDTH = 300
SPECK = (slice(20, 22), slice(20, 22))
BAR = (slice(150, 200), slice(60, 260))
# At 300 dpi the tint reaches 45 * 300 / 600 pixels from a speck.
RADIUS = 22.5


def _binary() -> np.ndarray:
    page = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    page[BAR] = 0
    page[SPECK] = 0
    return page


def _redness(pixel: np.ndarray) -> float:
    """How far red a pixel is of a white one: 0 for white, 255 for pure red."""
    return float(pixel[0]) - float(pixel[1])


class TestHalo:
    def test_pixels_next_to_a_speck_are_tinted_red(self):
        preview = despeckle_preview(_binary(), DPI, DespeckleLevel.AGGRESSIVE)

        next_to_it = preview[21, 24]
        assert next_to_it[0] == 255
        assert next_to_it[1] < 120

    def test_the_tint_fades_with_the_distance_from_the_speck(self):
        preview = despeckle_preview(_binary(), DPI, DespeckleLevel.AGGRESSIVE)

        redness = [_redness(preview[21, 22 + step]) for step in (1, 6, 12, 18)]

        assert redness == sorted(redness, reverse=True)
        assert redness[0] > redness[-1] > 0

    def test_nothing_is_tinted_beyond_the_radius(self):
        preview = despeckle_preview(_binary(), DPI, DespeckleLevel.AGGRESSIVE)

        assert np.all(preview[21, 22 + int(RADIUS) + 3] == 255)
        assert np.all(preview[HEIGHT - 1, WIDTH - 1] == 255)

    def test_the_radius_grows_with_the_resolution(self):
        low = despeckle_preview(_binary(), 300, DespeckleLevel.AGGRESSIVE)
        high = despeckle_preview(_binary(), 600, DespeckleLevel.AGGRESSIVE)

        assert np.all(low[21, 22 + 30] == 255)
        assert _redness(high[21, 22 + 30]) > 0

    def test_black_ink_is_never_tinted(self):
        near = _binary()
        near[20:22, 62:64] = 0  # A speck right next to the bar of text.
        preview = despeckle_preview(near, DPI, DespeckleLevel.AGGRESSIVE)

        assert np.all(preview[BAR] == 0)

    def test_a_speck_is_opaque_red(self):
        preview = despeckle_preview(_binary(), DPI, DespeckleLevel.AGGRESSIVE)

        assert SPECKLE_RGB == (255, 0, 0)
        assert np.all(preview[SPECK] == SPECKLE_RGB)

    def test_a_page_without_specks_is_tinted_evenly_when_despeckling_is_on(self):
        page = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        page[BAR] = 0

        preview = despeckle_preview(page, DPI, DespeckleLevel.AGGRESSIVE)

        assert tuple(preview[5, 5]) == (255, 178, 178)
        assert np.all(preview[BAR] == 0)

    def test_the_off_level_does_not_tint_anything(self):
        preview = despeckle_preview(_binary(), DPI, DespeckleLevel.OFF)

        assert np.all(preview[5, 5] == 255)
        assert np.all(preview[21, 24] == 255)


class TestPictures:
    @staticmethod
    def _output_page() -> np.ndarray:
        page = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        page[220:280, 20:120] = (30, 120, 200)  # A picture
        return page

    def test_pictures_of_the_output_show_through(self):
        preview = despeckle_preview(
            _binary(), DPI, DespeckleLevel.AGGRESSIVE, self._output_page()
        )

        assert tuple(preview[250, 70]) == (30, 120, 200)

    def test_the_ink_of_the_bilevel_page_is_drawn_black_over_the_output(self):
        output = self._output_page()
        output[BAR] = 255  # The output has already lost this bar.

        preview = despeckle_preview(_binary(), DPI, DespeckleLevel.AGGRESSIVE, output)

        assert np.all(preview[BAR] == 0)

    def test_a_gray_output_is_shown_in_colour_channels(self):
        gray = np.full((HEIGHT, WIDTH), 200, dtype=np.uint8)

        preview = despeckle_preview(_binary(), DPI, DespeckleLevel.OFF, gray)

        assert preview.shape == (HEIGHT, WIDTH, 3)
        assert tuple(preview[5, 5]) == (200, 200, 200)

    def test_the_output_page_is_not_modified(self):
        output = self._output_page()
        before = output.copy()

        despeckle_preview(_binary(), DPI, DespeckleLevel.AGGRESSIVE, output)

        assert np.array_equal(output, before)

    def test_the_halo_reaches_over_a_picture_near_a_speck(self):
        output = self._output_page()
        output[15:30, 24:40] = (30, 120, 200)

        preview = despeckle_preview(_binary(), DPI, DespeckleLevel.AGGRESSIVE, output)

        assert preview[21, 26][0] > 100
        assert tuple(preview[21, 26]) != (30, 120, 200)


class TestInTheTab:
    @staticmethod
    def _view(qtbot: QtBot) -> OutputImageView:
        view = OutputImageView()
        qtbot.addWidget(view)
        view.resize(QtCore.QSize(400, 400))
        view.show()
        qtbot.waitExposed(view)
        view.set_image(np.full((HEIGHT, WIDTH), 255, dtype=np.uint8))
        view.set_tabs(list(ImageViewTab), ImageViewTab.DESPECKLING)
        return view

    def _pixel(self, view: OutputImageView, x: int, y: int) -> tuple[int, int, int]:
        assert view.image is not None
        color = view.image.pixelColor(x, y)
        return color.red(), color.green(), color.blue()

    def test_the_tab_shows_the_pictures_of_the_output(self, qtbot):
        view = self._view(qtbot)
        output = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        output[220:280, 20:120] = (30, 120, 200)

        view.set_despeckle_source(_binary(), DPI, DespeckleLevel.AGGRESSIVE, output)
        qtbot.waitUntil(lambda: self._pixel(view, 20, 20) == SPECKLE_RGB, timeout=10000)

        assert self._pixel(view, 70, 250) == (30, 120, 200)

    def test_an_output_of_another_size_is_ignored(self, qtbot):
        view = self._view(qtbot)
        output = np.full((HEIGHT + 4, WIDTH, 3), 0, dtype=np.uint8)

        view.set_despeckle_source(_binary(), DPI, DespeckleLevel.AGGRESSIVE, output)
        qtbot.waitUntil(lambda: self._pixel(view, 20, 20) == SPECKLE_RGB, timeout=10000)

        assert self._pixel(view, 5, 290) == (255, 255, 255)

    def test_another_level_keeps_the_pictures(self, qtbot):
        view = self._view(qtbot)
        output = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        output[220:280, 20:120] = (30, 120, 200)
        view.set_despeckle_source(_binary(), DPI, DespeckleLevel.AGGRESSIVE, output)
        qtbot.waitUntil(lambda: self._pixel(view, 20, 20) == SPECKLE_RGB, timeout=10000)

        view.set_despeckle_level(DespeckleLevel.OFF)

        qtbot.waitUntil(lambda: self._pixel(view, 20, 20) == (0, 0, 0), timeout=10000)
        assert self._pixel(view, 70, 250) == (30, 120, 200)
