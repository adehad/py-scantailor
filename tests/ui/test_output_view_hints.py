"""Hover hints of the Output view's zone and dewarping tabs, in the original's words."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.base import DEFAULT_STATUS_TIP
from scantailor.app.ui.image_views.output import (
    ImageViewTab,
    OutputImageView,
    ViewZone,
    ZoneKind,
)
from scantailor.core.models import Dpi
from scantailor.core.settings import ZoneCreationMode
from scantailor.core.transformation import ImageTransformation
from scantailor.core.zones import PictureLayer, PictureZoneProperties
from scantailor.filters.output.params import DistortionModelSpec

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
LEFT = QtCore.Qt.MouseButton.LeftButton
SHIFT = QtCore.Qt.KeyboardModifier.ShiftModifier
CONTROL = QtCore.Qt.KeyboardModifier.ControlModifier
NONE = QtCore.Qt.KeyboardModifier.NoModifier
ALL_TABS = list(ImageViewTab)
ZONE_TIP = (
    "Click to start creating a new zone. Ctrl+Alt+Click to copy the latest created "
    "zone. Use Z, X and C keys to switch zone creation mode."
)
CURVED = DistortionModelSpec(
    top=((100.0, 150.0), (300.0, 120.0), (500.0, 150.0)),
    bottom=((100.0, 650.0), (300.0, 680.0), (500.0, 650.0)),
)


@pytest.fixture
def view(qtbot: QtBot) -> OutputImageView:
    widget = OutputImageView()
    qtbot.addWidget(widget)
    widget.resize(QtCore.QSize(400, 500))
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8),
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    widget.set_creation_mode(ZoneCreationMode.POLYGONAL)
    return widget


def _square() -> QtGui.QPolygonF:
    return QtGui.QPolygonF(
        [
            QtCore.QPointF(200, 300),
            QtCore.QPointF(400, 300),
            QtCore.QPointF(400, 500),
            QtCore.QPointF(200, 500),
        ]
    )


def _hover(
    view: OutputImageView,
    virtual: QtCore.QPointF,
    modifier: QtCore.Qt.KeyboardModifier = NONE,
) -> None:
    viewport = view.viewport()
    assert viewport is not None
    position = view.virtual_to_widget_point(virtual)
    QtWidgets.QApplication.sendEvent(
        viewport,
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            position,
            viewport.mapToGlobal(position),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            modifier,
        ),
    )


def _picture_view(view: OutputImageView) -> OutputImageView:
    view.set_zones(
        [
            ViewZone(
                ZoneKind.PICTURE,
                _square(),
                properties=PictureZoneProperties(layer=PictureLayer.PAINTER2),
            )
        ]
    )
    view.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)
    return view


class TestDefaultHint:
    def test_a_zone_tab_explains_how_to_start_a_zone(self, view):
        view.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)

        assert view.default_status_tip == ZONE_TIP
        assert view.status_tip_text == ZONE_TIP

    def test_the_fill_zone_tab_has_the_same_hint(self, view):
        view.set_tabs(ALL_TABS, ImageViewTab.FILL_ZONES)

        assert view.status_tip_text == ZONE_TIP

    @pytest.mark.parametrize(
        "tab",
        [ImageViewTab.OUTPUT, ImageViewTab.DEWARPING, ImageViewTab.DESPECKLING],
    )
    def test_other_tabs_keep_the_zoom_hint(self, view, tab):
        view.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)

        view.set_current_tab(tab)

        assert view.status_tip_text == DEFAULT_STATUS_TIP

    def test_removing_the_tabs_restores_the_zoom_hint(self, view):
        view.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)

        view.set_tabs([])

        assert view.status_tip_text == DEFAULT_STATUS_TIP

    def test_the_tab_bar_explains_the_tab_shortcuts(self, view):
        view.set_tabs(ALL_TABS)

        assert view.statusTip() == "Use Ctrl+1..5 to switch the tabs."


class TestZoneHints:
    def test_over_a_corner(self, view):
        _picture_view(view)

        _hover(view, QtCore.QPointF(200, 300))

        assert view.status_tip_text == (
            "Drag the vertex. Hold Ctrl to make the vertex angle right. "
            "Press D to delete the vertex."
        )

    def test_over_an_edge(self, view):
        _picture_view(view)

        _hover(view, QtCore.QPointF(300, 300))

        assert view.status_tip_text == "Click to create a new vertex here."

    def test_over_the_zone(self, view):
        _picture_view(view)

        _hover(view, QtCore.QPointF(300, 400))

        assert view.status_tip_text == (
            "Right click to edit zone properties. Hold Shift to drag the zone or "
            "Shift+Ctrl to copy. Press Del to delete this zone."
        )

    def test_over_the_zone_with_shift(self, view):
        _picture_view(view)

        _hover(view, QtCore.QPointF(300, 400), SHIFT)

        assert view.status_tip_text == "Hold left mouse button to drag the zone."

    def test_over_the_zone_with_shift_and_control(self, view):
        _picture_view(view)

        _hover(view, QtCore.QPointF(300, 400), SHIFT | CONTROL)

        assert (
            view.status_tip_text == "Hold left mouse button to copy and drag the zone."
        )

    def test_away_from_every_zone_the_default_returns(self, view):
        _picture_view(view)
        _hover(view, QtCore.QPointF(300, 400))

        _hover(view, QtCore.QPointF(50, 50))

        assert view.status_tip_text == ZONE_TIP

    def test_a_zone_of_the_other_kind_gives_no_hint(self, view):
        _picture_view(view)
        view.set_current_tab(ImageViewTab.FILL_ZONES)

        _hover(view, QtCore.QPointF(300, 400))

        assert view.status_tip_text == ZONE_TIP

    def test_dragging_a_corner_says_it_can_merge_with_a_neighbour(self, qtbot, view):
        _picture_view(view)
        corner = view.virtual_to_widget_point(QtCore.QPointF(200, 300))

        qtbot.mousePress(view.viewport(), LEFT, pos=corner.toPoint())

        assert view.status_tip_text == (
            "Move the vertex to one of its neighbors to merge them."
        )
        qtbot.mouseRelease(view.viewport(), LEFT, pos=corner.toPoint())

    def test_dragging_a_corner_onto_a_neighbour_offers_the_merge(self, qtbot, view):
        _picture_view(view)
        viewport = view.viewport()
        corner = view.virtual_to_widget_point(QtCore.QPointF(200, 300))
        neighbour = view.virtual_to_widget_point(QtCore.QPointF(400, 300))

        qtbot.mousePress(viewport, LEFT, pos=corner.toPoint())
        qtbot.mouseMove(viewport, neighbour.toPoint())

        assert view.status_tip_text == "Merge these two vertices."
        qtbot.mouseRelease(viewport, LEFT, pos=neighbour.toPoint())

    def test_dragging_the_zone_says_how_to_finish(self, qtbot, view):
        _picture_view(view)
        inside = view.virtual_to_widget_point(QtCore.QPointF(300, 400))

        qtbot.mousePress(view.viewport(), LEFT, SHIFT, pos=inside.toPoint())

        assert view.status_tip_text == "Release left mouse button to finish dragging."
        qtbot.mouseRelease(view.viewport(), LEFT, SHIFT, pos=inside.toPoint())

    def test_the_hint_returns_once_the_corner_is_dropped(self, qtbot, view):
        _picture_view(view)
        corner = view.virtual_to_widget_point(QtCore.QPointF(200, 300))
        qtbot.mousePress(view.viewport(), LEFT, pos=corner.toPoint())

        qtbot.mouseRelease(view.viewport(), LEFT, pos=corner.toPoint())

        assert view.status_tip_text != (
            "Move the vertex to one of its neighbors to merge them."
        )

    def test_holding_shift_over_the_zone_changes_the_hint_at_once(self, qtbot, view):
        _picture_view(view)
        _hover(view, QtCore.QPointF(300, 400))

        qtbot.keyPress(view, QtCore.Qt.Key.Key_Shift)
        try:
            assert view.status_tip_text == "Hold left mouse button to drag the zone."
        finally:
            qtbot.keyRelease(view, QtCore.Qt.Key.Key_Shift)


class TestCreationHints:
    def _click(self, qtbot: QtBot, view: OutputImageView, x: float, y: float) -> None:
        at = view.virtual_to_widget_point(QtCore.QPointF(x, y))
        qtbot.mouseClick(view.viewport(), LEFT, pos=at.toPoint())

    def test_the_first_corner_points_to_the_mode_keys(self, qtbot, view):
        view.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)

        self._click(qtbot, view, 100, 100)
        _hover(view, QtCore.QPointF(300, 100))

        assert view.status_tip_text == (
            "Use Z and X keys to switch zone creation mode.  ESC to cancel."
        )

    def test_further_corners_ask_to_connect_the_first_and_last(self, qtbot, view):
        view.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)

        self._click(qtbot, view, 100, 100)
        self._click(qtbot, view, 300, 100)
        _hover(view, QtCore.QPointF(300, 300))

        assert view.status_tip_text == (
            "Connect first and last points to finish this zone.  ESC to cancel."
        )

    def test_the_first_corner_closes_the_zone(self, qtbot, view):
        view.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)

        self._click(qtbot, view, 100, 100)
        self._click(qtbot, view, 300, 100)
        self._click(qtbot, view, 300, 300)
        _hover(view, QtCore.QPointF(100, 100))

        assert view.status_tip_text == "Click to finish this zone.  ESC to cancel."

    def test_a_rectangle_being_dragged_says_how_to_finish(self, qtbot, view):
        view.set_creation_mode(ZoneCreationMode.RECTANGULAR)
        view.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)
        start = view.virtual_to_widget_point(QtCore.QPointF(100, 100))

        qtbot.mousePress(view.viewport(), LEFT, pos=start.toPoint())

        assert view.status_tip_text == (
            "Click to finish this rectangular zone.  ESC to cancel."
        )
        qtbot.mouseRelease(view.viewport(), LEFT, pos=start.toPoint())


class TestColourPickupHint:
    def test_picking_a_colour_says_how_to_pick_or_cancel(self, view):
        view.set_zones([ViewZone(ZoneKind.FILL, _square(), (10, 20, 30), None)])
        view.set_tabs(ALL_TABS, ImageViewTab.FILL_ZONES)

        view.start_color_pickup(0)

        assert view.status_tip_text == (
            "Click on an area to pick up its color, or ESC to cancel."
        )

    def test_cancelling_brings_the_default_back(self, view):
        view.set_zones([ViewZone(ZoneKind.FILL, _square(), (10, 20, 30), None)])
        view.set_tabs(ALL_TABS, ImageViewTab.FILL_ZONES)
        view.start_color_pickup(0)

        view._pickup.cancel()

        assert view.status_tip_text == ZONE_TIP


class TestDewarpingHints:
    @pytest.fixture
    def dewarping(self, view: OutputImageView) -> OutputImageView:
        view.set_tabs(ALL_TABS, ImageViewTab.DEWARPING)
        view.set_distortion_model(CURVED, 2.0)
        return view

    def test_an_end_point_mentions_dragging_along_axes(self, dewarping):
        _hover(dewarping, QtCore.QPointF(100, 150))

        assert dewarping.status_tip_text == (
            "This point can be dragged. Hold Ctrl or Shift to drag along axes."
        )

    def test_a_middle_point_mentions_deleting_it(self, dewarping):
        _hover(dewarping, QtCore.QPointF(300, 120))

        assert dewarping.status_tip_text == (
            "Drag this point or delete it by pressing D key."
        )

    def test_the_curve_offers_a_new_control_point(self, dewarping):
        _hover(dewarping, QtCore.QPointF(200, 129))

        assert dewarping.status_tip_text == "Click to create a new control point."

    def test_away_from_the_grid_the_default_returns(self, dewarping):
        _hover(dewarping, QtCore.QPointF(300, 400))

        assert dewarping.status_tip_text == DEFAULT_STATUS_TIP


class TestEdgeDraggedAlongAnAxis:
    def _drag_end(
        self, qtbot: QtBot, view: OutputImageView, modifier: QtCore.Qt.KeyboardModifier
    ) -> None:
        viewport = view.viewport()
        start = view.virtual_to_widget_point(QtCore.QPointF(100, 150))
        end = start + QtCore.QPointF(20, 30)
        qtbot.mousePress(viewport, LEFT, pos=start.toPoint())
        QtWidgets.QApplication.sendEvent(
            viewport,
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                end,
                viewport.mapToGlobal(end),
                QtCore.Qt.MouseButton.NoButton,
                LEFT,
                modifier,
            ),
        )
        qtbot.mouseRelease(viewport, LEFT, pos=end.toPoint())

    def test_control_moves_the_whole_edge_up_or_down(self, qtbot, view):
        view.set_tabs(ALL_TABS, ImageViewTab.DEWARPING)
        view.set_distortion_model(CURVED, 2.0)
        before, _bottom = view.control_points()

        self._drag_end(qtbot, view, CONTROL)

        after, _bottom = view.control_points()
        shifts = [(a.x() - b.x(), a.y() - b.y()) for a, b in zip(after, before)]
        assert all(dx == pytest.approx(0) for dx, _dy in shifts)
        assert all(dy == pytest.approx(30 / view.zoom_level) for _dx, dy in shifts)

    def test_shift_moves_the_whole_edge_sideways(self, qtbot, view):
        view.set_tabs(ALL_TABS, ImageViewTab.DEWARPING)
        view.set_distortion_model(CURVED, 2.0)
        before, _bottom = view.control_points()

        self._drag_end(qtbot, view, SHIFT)

        after, _bottom = view.control_points()
        shifts = [(a.x() - b.x(), a.y() - b.y()) for a, b in zip(after, before)]
        assert all(dx == pytest.approx(20 / view.zoom_level) for dx, _dy in shifts)
        assert all(dy == pytest.approx(0) for _dx, dy in shifts)

    def test_without_a_key_the_edge_turns_about_its_other_end(self, qtbot, view):
        view.set_tabs(ALL_TABS, ImageViewTab.DEWARPING)
        view.set_distortion_model(CURVED, 2.0)
        before, _bottom = view.control_points()

        self._drag_end(qtbot, view, NONE)

        after, _bottom = view.control_points()
        assert after[-1] == before[-1]
        assert after[0] != before[0]
        assert after[1] != before[1]
