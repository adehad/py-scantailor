"""The millimetre grid over the page in the Margins and Output views."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import ImageViewTab, OutputImageView
from scantailor.app.ui.image_views.page_grid import (
    GRID_SPACING_MM,
    grid_offsets,
    paint_page_grid,
)
from scantailor.app.ui.image_views.page_layout import PageLayoutImageView
from scantailor.core.models import Dpi
from scantailor.core.settings import ApplicationSettings
from scantailor.core.transformation import ImageTransformation

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
VIEW_SIZE = QtCore.QSize(300, 400)
GRID_TOOLTIP = (
    "Draw a line every 10 mm across the page, through its centre, "
    "to compare positions between pages."
)


def _white_page() -> np.ndarray:
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)


def _transformation() -> ImageTransformation:
    return ImageTransformation.from_size(IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300))


def _grab(view: QtWidgets.QAbstractScrollArea) -> QtGui.QImage:
    viewport = view.viewport()
    assert viewport is not None
    return viewport.grab().toImage()


def _settings_as_stored() -> ApplicationSettings:
    """Read the settings file afresh, as the next session would."""
    return ApplicationSettings.load()


def _inked_columns(image: QtGui.QImage, row: int) -> list[int]:
    white = QtGui.QColor("white")
    return [x for x in range(image.width()) if image.pixelColor(x, row) != white]


@pytest.fixture
def margins_view(qtbot: QtBot) -> PageLayoutImageView:
    view = PageLayoutImageView()
    qtbot.addWidget(view)
    view.resize(VIEW_SIZE)
    view.show()
    qtbot.waitExposed(view)
    view.set_image(_white_page(), transformation=_transformation())
    view.set_mm_per_pixel(0.5, 0.5)
    inner = QtCore.QRectF(200, 300, 200, 200)
    middle = inner.adjusted(-60, -60, 60, 60)
    view.set_rectangles(inner, middle, middle.adjusted(-30, -30, 30, 30))
    return view


@pytest.fixture
def output_view(qtbot: QtBot) -> OutputImageView:
    view = OutputImageView()
    qtbot.addWidget(view)
    view.resize(VIEW_SIZE)
    view.show()
    qtbot.waitExposed(view)
    view.set_image(_white_page(), transformation=_transformation())
    view.set_mm_per_pixel(0.5, 0.5)
    return view


class TestGridOffsets:
    def test_lines_run_symmetrically_through_the_centre(self):
        assert grid_offsets(25.0, 10.0) == [-20.0, -10.0, 0.0, 10.0, 20.0]

    def test_a_line_exactly_on_the_edge_is_kept(self):
        assert grid_offsets(20.0, 10.0) == [-20.0, -10.0, 0.0, 10.0, 20.0]

    def test_a_page_narrower_than_the_spacing_still_has_its_centre_line(self):
        assert grid_offsets(3.0, 10.0) == [0.0]

    @pytest.mark.parametrize("spacing", [0.0, -5.0])
    def test_no_spacing_gives_no_lines(self, spacing: float):
        assert grid_offsets(25.0, spacing) == []


class TestPaintPageGrid:
    @staticmethod
    def _paint(scale: float, mm_per_pixel: tuple[float, float]) -> QtGui.QImage:
        image = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        paint_page_grid(
            painter,
            QtCore.QRectF(0, 0, 200 / scale, 200 / scale),
            mm_per_pixel,
            QtGui.QTransform.fromScale(scale, scale),
        )
        painter.end()
        return image

    def test_lines_are_drawn_every_ten_millimetres(self):
        image = self._paint(1.0, (1.0, 1.0))

        assert _inked_columns(image, 5) == list(range(0, 200, int(GRID_SPACING_MM)))

    def test_zoomed_out_every_other_line_is_left_out(self):
        image = self._paint(0.5, (1.0, 1.0))

        assert _inked_columns(image, 5) == list(range(0, 200, 10))

    def test_an_unknown_resolution_draws_nothing(self):
        image = self._paint(1.0, (0.0, 0.0))

        assert _inked_columns(image, 5) == []


class TestMarginsGrid:
    def test_hidden_by_default(self, margins_view: PageLayoutImageView):
        assert margins_view.grid_shown() is False

    def test_showing_it_draws_over_the_page_and_hiding_it_removes_it(
        self, margins_view: PageLayoutImageView
    ):
        before = _grab(margins_view)

        margins_view.set_grid_shown(True)
        shown = _grab(margins_view)
        margins_view.set_grid_shown(False)

        assert shown != before
        assert _grab(margins_view) == before

    def test_choice_is_stored_in_the_application_settings(
        self, margins_view: PageLayoutImageView
    ):
        margins_view.set_grid_shown(True)

        assert _settings_as_stored().margins_grid_shown is True

    def test_next_view_starts_with_the_stored_choice(self, qtbot: QtBot):
        ApplicationSettings.get_instance().margins_grid_shown = True

        later = PageLayoutImageView()
        qtbot.addWidget(later)

        assert later.grid_shown() is True

    def test_menu_item_is_checkable_explains_itself_and_follows_the_state(
        self, margins_view: PageLayoutImageView
    ):
        margins_view.set_grid_shown(True)
        action = margins_view._show_grid_action

        assert action.text() == "Show grid"
        assert action.toolTip() == GRID_TOOLTIP
        assert action.isCheckable()
        assert action.isChecked()

    def test_triggering_the_menu_item_shows_the_grid(
        self, margins_view: PageLayoutImageView
    ):
        margins_view._show_grid_action.trigger()

        assert margins_view.grid_shown() is True

    def test_a_page_that_is_not_aligned_offers_only_the_grid(
        self, margins_view: PageLayoutImageView
    ):
        margins_view.set_aligned(False)
        centre = margins_view.viewport().rect().center()

        margins_view.contextMenuEvent(
            QtGui.QContextMenuEvent(
                QtGui.QContextMenuEvent.Reason.Mouse,
                centre,
                margins_view.mapToGlobal(centre),
            )
        )
        menu = margins_view._context_menu
        visible = [a.text() for a in menu.actions() if a.isVisible() and a.text()]
        menu.close()

        assert visible == ["Show grid"]


class TestOutputGrid:
    def test_hidden_by_default(self, output_view: OutputImageView):
        assert output_view.grid_shown() is False

    def test_showing_it_draws_over_the_page_and_hiding_it_removes_it(
        self, output_view: OutputImageView
    ):
        before = _grab(output_view)

        output_view.set_grid_shown(True)
        shown = _grab(output_view)
        output_view.set_grid_shown(False)

        assert shown != before
        assert _grab(output_view) == before

    def test_nothing_is_drawn_until_the_page_size_is_known(
        self, output_view: OutputImageView
    ):
        output_view.set_mm_per_pixel(0.0, 0.0)
        before = _grab(output_view)

        output_view.set_grid_shown(True)

        assert _grab(output_view) == before

    def test_the_dewarping_tab_does_not_show_it(self, output_view: OutputImageView):
        output_view.set_tabs(list(ImageViewTab))
        output_view.set_current_tab(ImageViewTab.DEWARPING)
        before = _grab(output_view)

        output_view.set_grid_shown(True)

        assert _grab(output_view) == before

    def test_choice_is_stored_in_the_application_settings(
        self, output_view: OutputImageView
    ):
        output_view.set_grid_shown(True)

        assert _settings_as_stored().output_grid_shown is True

    def test_menu_item_is_checkable_explains_itself_and_follows_the_state(
        self, output_view: OutputImageView
    ):
        output_view.set_grid_shown(True)

        menu = output_view.build_context_menu(QtCore.QPointF(10, 10))
        (action,) = [a for a in menu.actions() if a.text() == "Show grid"]

        assert action.toolTip() == GRID_TOOLTIP
        assert action.isCheckable()
        assert action.isChecked()
        assert menu.toolTipsVisible()

    def test_triggering_the_menu_item_shows_the_grid(
        self, output_view: OutputImageView
    ):
        menu = output_view.build_context_menu(QtCore.QPointF(10, 10))
        (action,) = [a for a in menu.actions() if a.text() == "Show grid"]

        action.trigger()

        assert output_view.grid_shown() is True
