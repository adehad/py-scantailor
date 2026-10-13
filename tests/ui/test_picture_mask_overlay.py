"""The picture mask pulsing over the Picture Zones tab, edited by the zones."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import (
    ImageViewTab,
    OutputImageView,
    ViewZone,
    ZoneKind,
)
from scantailor.app.ui.image_views.picture_mask_overlay import (
    MaskZone,
    PictureMaskOverlay,
    opacity_at,
)
from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation
from scantailor.core.zones import PictureLayer, PictureZoneProperties

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
ALL_TABS = list(ImageViewTab)
FULLEST_PHASE = 90
FAINTEST_PHASE = 270


def _mask() -> np.ndarray:
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    mask[100:300, 100:300] = True
    return mask


def _zone(layer: PictureLayer, left: float, top: float, size: float) -> ViewZone:
    return ViewZone(
        ZoneKind.PICTURE,
        QtGui.QPolygonF(
            [
                QtCore.QPointF(left, top),
                QtCore.QPointF(left + size, top),
                QtCore.QPointF(left + size, top + size),
                QtCore.QPointF(left, top + size),
            ]
        ),
        properties=PictureZoneProperties(layer=layer),
    )


@pytest.fixture
def host() -> QtCore.QObject:
    return QtCore.QObject()


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
    widget.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)
    return widget


def _pixel(view: OutputImageView, x: float, y: float) -> QtGui.QColor:
    spot = view.image_to_widget().map(QtCore.QPointF(x, y)).toPoint()
    return view.viewport().grab().toImage().pixelColor(spot)


def _set_phase(view: OutputImageView, degrees: int) -> None:
    view._picture_mask._phase = degrees
    view.update()


def _tinted(view: OutputImageView, x: float, y: float) -> bool:
    """Whether the overlay changed the pixel from what the page and zones alone show."""
    bare = _bare(view)
    return _pixel(view, x, y) != bare


def _bare(view: OutputImageView) -> QtGui.QColor:
    return QtGui.QColor(
        view.image.pixelColor(0, 0) if view.image is not None else QtGui.QColor()
    )


class TestPulse:
    def test_the_pulse_runs_between_a_faint_and_a_strong_tint(self):
        assert opacity_at(270) == pytest.approx(0.15)
        assert opacity_at(90) == pytest.approx(0.5)
        assert 0.15 < opacity_at(0) < 0.5

    def test_the_pulse_is_periodic(self):
        assert opacity_at(40) == pytest.approx(opacity_at(400))

    def test_a_new_overlay_starts_at_its_faintest(self, host):
        overlay = PictureMaskOverlay(host, lambda: None)

        assert overlay.phase == 270

    def test_the_phase_advances_by_forty_degrees_and_wraps(self, host):
        changes: list[int] = []
        overlay = PictureMaskOverlay(host, lambda: changes.append(1))
        overlay.set_mask(_mask())

        overlay._advance()
        assert overlay.phase == 310
        for _ in range(8):
            overlay._advance()

        assert overlay.phase == (270 + 9 * 40) % 360
        assert len(changes) >= 9

    def test_the_timer_steps_every_hundred_and_twenty_milliseconds(self, host):
        overlay = PictureMaskOverlay(host, lambda: None)

        assert overlay._timer.interval() == 120
        assert overlay._timer.isSingleShot()


class TestShowing:
    def test_the_mask_tints_the_picture_area_of_the_tab(self, view):
        view.set_picture_mask(_mask())
        _set_phase(view, FULLEST_PHASE)

        assert _tinted(view, 200, 200)
        assert not _tinted(view, 450, 600)

    def test_the_tint_pulses(self, view):
        view.set_picture_mask(_mask())

        _set_phase(view, FAINTEST_PHASE)
        faint = _pixel(view, 200, 200)
        _set_phase(view, FULLEST_PHASE)
        strong = _pixel(view, 200, 200)

        assert faint != strong
        assert (
            strong.value() < faint.value() or strong.saturation() > faint.saturation()
        )

    def test_the_mask_is_not_shown_on_the_other_tabs(self, view):
        view.set_picture_mask(_mask())
        _set_phase(view, FULLEST_PHASE)

        for tab in (
            ImageViewTab.OUTPUT,
            ImageViewTab.FILL_ZONES,
            ImageViewTab.DEWARPING,
        ):
            view.set_current_tab(tab)
            assert not _tinted(view, 200, 200)

    def test_it_uses_the_picture_colour_of_the_theme(self, view):
        view.set_picture_mask(_mask())
        _set_phase(view, FULLEST_PHASE)

        pixel = _pixel(view, 200, 200)
        theme = view_color("zone_picture_border")

        # White blended with the theme colour at half opacity leans towards it.
        for channel in ("red", "green", "blue"):
            low, high = sorted((255, getattr(theme, channel)()))
            assert low - 3 <= getattr(pixel, channel)() <= high + 3

    def test_a_mask_without_a_picture_shows_nothing(self, view):
        view.set_picture_mask(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool))
        _set_phase(view, FULLEST_PHASE)

        assert not _tinted(view, 200, 200)
        assert not view._picture_mask.has_mask()

    def test_taking_the_mask_away_removes_the_tint(self, view):
        view.set_picture_mask(_mask())
        view.set_picture_mask(None)
        _set_phase(view, FULLEST_PHASE)

        assert not _tinted(view, 200, 200)

    def test_a_large_mask_is_shown_where_it_belongs(self, qtbot):
        widget = OutputImageView()
        qtbot.addWidget(widget)
        widget.resize(QtCore.QSize(400, 500))
        widget.show()
        qtbot.waitExposed(widget)
        widget.set_image(np.full((3200, 2400), 255, dtype=np.uint8))
        widget.set_tabs(ALL_TABS, ImageViewTab.PICTURE_ZONES)
        mask = np.zeros((3200, 2400), dtype=bool)
        mask[400:1200, 400:1200] = True
        widget.set_picture_mask(mask)
        _set_phase(widget, FULLEST_PHASE)

        assert _tinted(widget, 800, 800)
        assert not _tinted(widget, 1800, 2400)


class TestFollowingTheZones:
    def test_a_layer_that_adds_to_the_mask_tints_its_zone(self, view):
        view.set_picture_mask(_mask())
        view.set_zones([_zone(PictureLayer.NOOP, 400, 500, 100)])
        _set_phase(view, FULLEST_PHASE)
        without = _pixel(view, 450, 550)

        view.set_zones([_zone(PictureLayer.PAINTER2, 400, 500, 100)])
        _set_phase(view, FULLEST_PHASE)

        assert _pixel(view, 450, 550) != without

    def test_a_layer_that_subtracts_cuts_the_mask_away(self, host):
        overlay = PictureMaskOverlay(host, lambda: None)
        overlay.set_mask(_mask())
        shown = overlay._shown
        assert shown is not None and shown.pixelColor(200, 200).alpha() == 255

        overlay.set_zones(
            [
                MaskZone(
                    _zone(PictureLayer.ERASER3, 150, 150, 100).polygon,
                    PictureLayer.ERASER3,
                )
            ]
        )

        shown = overlay._shown
        assert shown is not None
        assert shown.pixelColor(200, 200).alpha() == 0
        assert shown.pixelColor(120, 120).alpha() == 255

    def test_the_layers_apply_in_the_order_of_the_output(self, host):
        overlay = PictureMaskOverlay(host, lambda: None)
        overlay.set_mask(_mask())
        around = _zone(PictureLayer.PAINTER2, 120, 120, 100).polygon
        inside = _zone(PictureLayer.ERASER3, 150, 150, 40).polygon
        # Subtract-from-all runs after add-to-auto, whatever the order of the zones.
        overlay.set_zones(
            [
                MaskZone(inside, PictureLayer.ERASER3),
                MaskZone(around, PictureLayer.PAINTER2),
            ]
        )
        shown = overlay._shown
        assert shown is not None

        assert shown.pixelColor(170, 170).alpha() == 0
        assert shown.pixelColor(125, 125).alpha() == 255

    def test_the_eraser_of_the_auto_layer_runs_before_the_painter(self, host):
        overlay = PictureMaskOverlay(host, lambda: None)
        overlay.set_mask(_mask())
        polygon = _zone(PictureLayer.PAINTER2, 150, 150, 40).polygon

        overlay.set_zones(
            [
                MaskZone(polygon, PictureLayer.PAINTER2),
                MaskZone(polygon, PictureLayer.ERASER1),
            ]
        )

        shown = overlay._shown
        assert shown is not None
        assert shown.pixelColor(170, 170).alpha() == 255

    def test_zones_that_are_not_picture_zones_do_not_change_the_mask(self, view):
        view.set_picture_mask(_mask())
        view.set_zones(
            [
                ViewZone(
                    ZoneKind.FILL,
                    _zone(PictureLayer.NOOP, 150, 150, 100).polygon,
                    (1, 2, 3),
                )
            ]
        )
        _set_phase(view, FULLEST_PHASE)

        assert _tinted(view, 200, 200)

    def test_moving_a_zone_updates_the_mask(self, view):
        view.set_picture_mask(_mask())
        view.set_zones([_zone(PictureLayer.ERASER3, 150, 150, 100)])
        _set_phase(view, FULLEST_PHASE)
        before = _pixel(view, 200, 200)

        view.set_zones([_zone(PictureLayer.ERASER3, 500, 500, 50)])
        _set_phase(view, FULLEST_PHASE)

        assert _pixel(view, 200, 200) != before


class TestRunning:
    def test_the_overlay_pulses_only_on_the_picture_zones_tab(self, view):
        view.set_picture_mask(_mask())
        assert view._picture_mask._timer.isActive()

        view.set_current_tab(ImageViewTab.OUTPUT)
        assert not view._picture_mask._timer.isActive()

        view.set_current_tab(ImageViewTab.PICTURE_ZONES)
        assert view._picture_mask._timer.isActive()

    def test_a_hidden_view_does_not_pulse(self, view):
        view.set_picture_mask(_mask())

        view.hide()

        assert not view._picture_mask._timer.isActive()

    def test_a_shown_view_pulses_again(self, view):
        view.set_picture_mask(_mask())
        view.hide()

        view.show()

        assert view._picture_mask._timer.isActive()

    def test_without_a_mask_nothing_runs(self, view):
        assert not view._picture_mask._timer.isActive()

    def test_the_view_repaints_as_the_phase_advances(self, qtbot, view):
        view.set_picture_mask(_mask())
        start = view.picture_mask_phase()

        qtbot.waitUntil(lambda: view.picture_mask_phase() != start, timeout=3000)
