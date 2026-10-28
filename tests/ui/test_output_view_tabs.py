"""Upstream's tab set in the Output view: Output, zones, dewarping and despeckling."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.despeckle_preview import SPECKLE_RGB
from scantailor.app.ui.image_views.output import (
    TAB_LABELS,
    ImageViewTab,
    OutputImageView,
    ViewZone,
    ZoneKind,
)
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.params import DistortionModelSpec
from tests.ui.conftest import flush_events

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
LEFT = QtCore.Qt.MouseButton.LeftButton
CTRL = QtCore.Qt.KeyboardModifier.ControlModifier
ALL_TABS = list(ImageViewTab)
CURVED = DistortionModelSpec(
    top=((100.0, 150.0), (300.0, 120.0), (500.0, 150.0)),
    bottom=((100.0, 650.0), (300.0, 680.0), (500.0, 650.0)),
)


def _page(
    value: int = 255, size: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


@pytest.fixture
def view(qtbot: QtBot) -> OutputImageView:
    widget = OutputImageView()
    qtbot.addWidget(widget)
    widget.resize(QtCore.QSize(400, 500))
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        _page(),
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    return widget


def _bar(view: OutputImageView) -> QtWidgets.QTabBar:
    bar = view.findChild(QtWidgets.QTabBar, "outputTabBar")
    assert bar is not None
    return bar


def _pt(point: QtCore.QPointF) -> QtCore.QPoint:
    return QtCore.QPoint(round(point.x()), round(point.y()))


class TestTabBar:
    def test_a_view_has_no_tabs_until_asked(self, view: OutputImageView):
        assert view.tabs() == []
        assert view.current_tab() is None
        assert not _bar(view).isVisible()

    def test_the_tabs_come_in_upstream_order_with_upstream_names(
        self, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS)

        bar = _bar(view)
        assert [bar.tabText(i) for i in range(bar.count())] == [
            "Output",
            "Picture Zones",
            "Fill Zones",
            "Dewarping",
            "Despeckling",
        ]
        assert view.tabs() == ALL_TABS
        assert bar.isVisible()

    def test_the_first_tab_is_chosen_unless_another_is_given(
        self, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS)
        assert view.current_tab() is ImageViewTab.OUTPUT

        view.set_tabs(ALL_TABS, ImageViewTab.DEWARPING)
        assert view.current_tab() is ImageViewTab.DEWARPING

    def test_every_tab_has_an_icon(self, view: OutputImageView):
        view.set_tabs(ALL_TABS)

        bar = _bar(view)
        assert all(not bar.tabIcon(i).isNull() for i in range(bar.count()))

    def test_the_bar_sits_above_the_image(self, view: OutputImageView):
        view.set_tabs(ALL_TABS)

        bar = _bar(view)
        assert bar.geometry().top() == 0
        assert view.viewport().geometry().top() == bar.height() > 0

    def test_removing_the_tabs_restores_the_full_view(self, view: OutputImageView):
        view.set_tabs(ALL_TABS)
        view.set_tabs([])

        assert not _bar(view).isVisible()
        assert view.viewport().geometry().top() == 0
        assert view.current_tab() is None

    def test_choosing_a_tab_reports_it(self, qtbot: QtBot, view: OutputImageView):
        view.set_tabs(ALL_TABS)

        with qtbot.waitSignal(view.tab_changed) as signal:
            _bar(view).setCurrentIndex(2)

        assert signal.args == [ImageViewTab.FILL_ZONES]
        assert view.current_tab() is ImageViewTab.FILL_ZONES

    def test_set_current_tab_reports_it_and_moves_the_bar(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS)

        with qtbot.waitSignal(view.tab_changed) as signal:
            view.set_current_tab(ImageViewTab.DESPECKLING)

        assert signal.args == [ImageViewTab.DESPECKLING]
        assert _bar(view).currentIndex() == 4

    def test_choosing_the_same_tab_again_reports_nothing(self, view: OutputImageView):
        view.set_tabs(ALL_TABS)
        seen: list[ImageViewTab] = []
        view.tab_changed.connect(seen.append)

        view.set_current_tab(ImageViewTab.OUTPUT)

        assert seen == []

    def test_a_tab_the_view_does_not_have_is_ignored(self, view: OutputImageView):
        view.set_tabs([ImageViewTab.OUTPUT, ImageViewTab.FILL_ZONES])

        view.set_current_tab(ImageViewTab.DEWARPING)

        assert view.current_tab() is ImageViewTab.OUTPUT

    @pytest.mark.parametrize("number", [1, 2, 3, 4, 5])
    def test_ctrl_and_a_digit_choose_the_tab(
        self, qtbot: QtBot, view: OutputImageView, number: int
    ):
        view.set_tabs(ALL_TABS)
        key = getattr(QtCore.Qt.Key, f"Key_{number}")

        qtbot.keyClick(view, key, CTRL)

        assert view.current_tab() is ALL_TABS[number - 1]

    def test_a_shortcut_past_the_last_tab_does_nothing(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.set_tabs([ImageViewTab.OUTPUT, ImageViewTab.FILL_ZONES])

        qtbot.keyClick(view, QtCore.Qt.Key.Key_4, CTRL)

        assert view.current_tab() is ImageViewTab.OUTPUT

    def test_the_status_tip_names_the_shortcuts(self, view: OutputImageView):
        view.set_tabs(ALL_TABS)

        assert view.statusTip() == "Use Ctrl+1..5 to switch the tabs."

    def test_every_tab_has_a_label(self):
        assert set(TAB_LABELS) == set(ImageViewTab)


class TestTabImages:
    def test_a_tab_without_an_image_of_its_own_shows_the_output(
        self, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS)
        output = view.image

        view.set_current_tab(ImageViewTab.FILL_ZONES)

        assert view.image is output

    def test_a_tab_with_its_own_image_shows_it_and_gives_the_output_back(
        self, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS)
        output = view.image
        view.set_tab_image(ImageViewTab.PICTURE_ZONES, _page(100))

        view.set_current_tab(ImageViewTab.PICTURE_ZONES)
        shown = view.image
        view.set_current_tab(ImageViewTab.OUTPUT)

        assert shown is not output
        assert shown is not None and shown.pixelColor(5, 5).red() == 100
        assert view.image is output

    def test_a_new_output_does_not_replace_the_image_of_the_current_tab(
        self, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS)
        view.set_tab_image(ImageViewTab.DEWARPING, _page(100))
        view.set_current_tab(ImageViewTab.DEWARPING)

        view.set_image(_page(10))
        assert view.image is not None and view.image.pixelColor(5, 5).red() == 100

        view.set_current_tab(ImageViewTab.OUTPUT)
        assert view.image is not None and view.image.pixelColor(5, 5).red() == 10

    def test_setting_the_image_of_the_current_tab_shows_it_at_once(
        self, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS, ImageViewTab.DESPECKLING)

        view.set_tab_image(ImageViewTab.DESPECKLING, _page(77))

        assert view.image is not None and view.image.pixelColor(5, 5).red() == 77

    def test_taking_the_image_away_goes_back_to_the_output(self, view: OutputImageView):
        view.set_tabs(ALL_TABS, ImageViewTab.DESPECKLING)
        view.set_tab_image(ImageViewTab.DESPECKLING, _page(77))

        view.set_tab_image(ImageViewTab.DESPECKLING, None)

        assert view.image is not None and view.image.pixelColor(5, 5).red() == 255

    def test_the_zoom_carries_over_between_images_of_one_size(
        self, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS)
        view.set_tab_image(ImageViewTab.FILL_ZONES, _page(100))
        view.set_zoom(1.0)

        view.set_current_tab(ImageViewTab.FILL_ZONES)

        assert view.zoom_level == pytest.approx(1.0)

    def test_an_image_of_another_size_is_fitted(self, view: OutputImageView):
        view.set_tabs(ALL_TABS)
        view.set_tab_image(ImageViewTab.FILL_ZONES, _page(100, (1600, 1200)))
        view.set_zoom(1.0)

        view.set_current_tab(ImageViewTab.FILL_ZONES)

        assert view.zoom_level < 0.5


class TestDewarpingTab:
    @pytest.fixture
    def dewarping(self, view: OutputImageView) -> OutputImageView:
        view.set_tabs(ALL_TABS, ImageViewTab.DEWARPING)
        view.set_distortion_model(CURVED, 2.0)
        return view

    def _widget_pos(self, view: OutputImageView, x: float, y: float) -> QtCore.QPointF:
        return view.image_to_widget().map(QtCore.QPointF(x, y))

    def test_the_control_points_are_the_models_points(self, dewarping: OutputImageView):
        top, bottom = dewarping.control_points()

        assert [(p.x(), p.y()) for p in top] == list(CURVED.top)
        assert [(p.x(), p.y()) for p in bottom] == list(CURVED.bottom)

    def test_dragging_a_control_point_reports_the_new_model(
        self, qtbot: QtBot, dewarping: OutputImageView
    ):
        start = self._widget_pos(dewarping, 300, 120)
        end = start + QtCore.QPointF(0, 20)
        viewport = dewarping.viewport()

        with qtbot.waitSignal(dewarping.distortion_model_changed) as signal:
            qtbot.mousePress(viewport, LEFT, pos=_pt(start))
            for step in range(1, 5):
                qtbot.mouseMove(viewport, _pt(start + (end - start) * (step / 4)))
            qtbot.mouseRelease(viewport, LEFT, pos=_pt(end))

        (model,) = signal.args
        assert isinstance(model, DistortionModelSpec)
        assert model == dewarping.distortion_model()
        top, _ = dewarping.control_points()
        assert top[1].y() > 120

    def test_the_grid_is_drawn_over_the_image(self, dewarping: OutputImageView):
        image = dewarping.viewport().grab().toImage()

        painted = {
            image.pixelColor(x, y).rgb()
            for x in range(0, image.width(), 3)
            for y in range(0, image.height(), 3)
        }

        assert len(painted) > 2

    def test_the_depth_perception_redraws_the_grid(
        self, qtbot: QtBot, dewarping: OutputImageView
    ):
        before = dewarping._dewarp.grid
        assert before is not None

        dewarping.set_depth_perception(1.2)

        qtbot.waitUntil(lambda: dewarping._dewarp.grid is not before)
        after = dewarping._dewarp.grid
        assert after is not None and not np.allclose(before, after)

    def test_d_removes_the_inner_point_under_the_cursor(
        self, qtbot: QtBot, dewarping: OutputImageView
    ):
        spot = self._widget_pos(dewarping, 300, 120)
        QtGui.QGuiApplication.sendEvent(
            dewarping.viewport(),
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                spot,
                QtCore.QPointF(dewarping.viewport().mapToGlobal(_pt(spot))),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            ),
        )

        with qtbot.waitSignal(dewarping.distortion_model_changed):
            qtbot.keyClick(dewarping, QtCore.Qt.Key.Key_D)

        top, _ = dewarping.control_points()
        assert len(top) == 2

    def test_the_other_tabs_do_not_move_the_edges(
        self, qtbot: QtBot, dewarping: OutputImageView
    ):
        dewarping.set_current_tab(ImageViewTab.OUTPUT)
        start = self._widget_pos(dewarping, 300, 120)
        seen: list[object] = []
        dewarping.distortion_model_changed.connect(seen.append)

        qtbot.mousePress(dewarping.viewport(), LEFT, pos=_pt(start))
        qtbot.mouseRelease(dewarping.viewport(), LEFT, pos=_pt(start))

        assert seen == []

    def test_zone_tools_are_off_in_the_dewarping_tab(
        self, qtbot: QtBot, dewarping: OutputImageView
    ):
        dewarping.set_zones(
            [
                ViewZone(
                    ZoneKind.PICTURE,
                    QtGui.QPolygonF(
                        [
                            QtCore.QPointF(50, 400),
                            QtCore.QPointF(150, 400),
                            QtCore.QPointF(100, 500),
                        ]
                    ),
                )
            ]
        )
        seen: list[None] = []
        dewarping.zones_changed.connect(lambda: seen.append(None))

        qtbot.keyClick(dewarping, QtCore.Qt.Key.Key_Delete)

        assert seen == []
        assert len(dewarping.zones()) == 1

    def test_the_menu_is_not_offered_in_the_dewarping_tab(
        self, dewarping: OutputImageView
    ):
        called: list[bool] = []
        original = dewarping.build_context_menu
        dewarping.build_context_menu = lambda pos: called.append(True) or original(pos)  # type: ignore[method-assign]

        dewarping.contextMenuEvent(
            QtGui.QContextMenuEvent(
                QtGui.QContextMenuEvent.Reason.Mouse,
                QtCore.QPoint(5, 5),
                QtCore.QPoint(5, 5),
            )
        )

        assert called == []

    def test_a_press_off_the_edges_pans_instead(
        self, qtbot: QtBot, dewarping: OutputImageView
    ):
        dewarping.set_zoom(2.0)
        far = QtCore.QPoint(200, 300)

        qtbot.mousePress(dewarping.viewport(), LEFT, pos=far)

        assert dewarping._panning
        qtbot.mouseRelease(dewarping.viewport(), LEFT, pos=far)
        assert not dewarping._panning


class TestDespecklingTab:
    SPECK = (slice(20, 22), slice(20, 22))

    @staticmethod
    def _binary() -> np.ndarray:
        page = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)
        page[300:400, 100:500] = 0
        page[TestDespecklingTab.SPECK] = 0
        return page

    def _pixel(self, view: OutputImageView, x: int, y: int) -> tuple[int, int, int]:
        assert view.image is not None
        color = view.image.pixelColor(x, y)
        return color.red(), color.green(), color.blue()

    def test_the_tab_shows_the_page_with_the_removed_specks_marked(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS, ImageViewTab.DESPECKLING)

        view.set_despeckle_source(self._binary(), 300, DespeckleLevel.AGGRESSIVE)
        qtbot.waitUntil(lambda: self._pixel(view, 20, 20) == SPECKLE_RGB, timeout=10000)

        assert self._pixel(view, 200, 350) == (0, 0, 0)
        assert self._pixel(view, 5, 700) == (255, 255, 255)

    def test_another_level_is_previewed_on_the_same_page(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS, ImageViewTab.DESPECKLING)
        view.set_despeckle_source(self._binary(), 300, DespeckleLevel.AGGRESSIVE)
        qtbot.waitUntil(lambda: self._pixel(view, 20, 20) == SPECKLE_RGB, timeout=10000)

        view.set_despeckle_level(DespeckleLevel.OFF)

        qtbot.waitUntil(lambda: self._pixel(view, 20, 20) == (0, 0, 0), timeout=10000)

    def test_the_output_tab_keeps_showing_the_output(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS, ImageViewTab.OUTPUT)

        view.set_despeckle_source(self._binary(), 300, DespeckleLevel.AGGRESSIVE)
        assert QtCore.QThreadPool.globalInstance().waitForDone(10000)
        flush_events()

        assert self._pixel(view, 20, 20) == (255, 255, 255)
        view.set_current_tab(ImageViewTab.DESPECKLING)
        assert self._pixel(view, 20, 20) == SPECKLE_RGB

    def test_removing_the_source_removes_the_preview(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.set_tabs(ALL_TABS, ImageViewTab.DESPECKLING)
        view.set_despeckle_source(self._binary(), 300, DespeckleLevel.AGGRESSIVE)
        qtbot.waitUntil(lambda: self._pixel(view, 20, 20) == SPECKLE_RGB, timeout=10000)

        view.set_despeckle_source(None, 300, DespeckleLevel.AGGRESSIVE)

        assert self._pixel(view, 20, 20) == (255, 255, 255)

    def test_a_level_without_a_source_does_nothing(self, view: OutputImageView):
        view.set_despeckle_level(DespeckleLevel.NORMAL)

        assert view.image is not None
