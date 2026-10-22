"""Editing an auto picture zone in the Output view makes it manual."""

from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import ImageViewTab, OutputImageView, ViewZone
from scantailor.core.zones import PictureLayer, PictureZoneProperties, ZoneCategory
from tests.ui.test_output_zone_tools import (
    CTRL,
    LEFT,
    OTHER,
    SHIFT,
    _drag,
    _hover,
    _picture,
    _pt,
    _tabbed,
    _w,
)

AUTO = PictureZoneProperties(layer=PictureLayer.PAINTER2, category=ZoneCategory.AUTO)


def _auto_view(qtbot: QtBot, *zones: ViewZone) -> OutputImageView:
    return _tabbed(
        qtbot, ImageViewTab.PICTURE_ZONES, *(zones or (_picture(properties=AUTO),))
    )


def _category(zone: ViewZone) -> ZoneCategory:
    assert isinstance(zone.properties, PictureZoneProperties)
    return zone.properties.category


def test_moving_an_auto_zone_makes_it_manual(qtbot: QtBot):
    view = _auto_view(qtbot)
    inside = _w(view, 200, 200)

    with qtbot.waitSignal(view.zones_changed):
        _drag(qtbot, view, inside, inside + QtCore.QPointF(30, 20), SHIFT)

    (zone,) = view.zones()
    assert zone.properties == PictureZoneProperties(layer=PictureLayer.PAINTER2)


def test_dragging_a_corner_of_an_auto_zone_makes_it_manual(qtbot: QtBot):
    view = _auto_view(qtbot)
    corner = _w(view, 300, 300)

    with qtbot.waitSignal(view.zones_changed):
        _drag(qtbot, view, corner, corner + QtCore.QPointF(15, 15))

    assert _category(view.zones()[0]) is ZoneCategory.MANUAL


def test_removing_a_corner_of_an_auto_zone_makes_it_manual(qtbot: QtBot):
    view = _auto_view(qtbot)
    _hover(view, _w(view, 300, 300))

    with qtbot.waitSignal(view.zones_changed):
        qtbot.keyClick(view, QtCore.Qt.Key.Key_D)

    assert _category(view.zones()[0]) is ZoneCategory.MANUAL


def test_a_copy_of_an_auto_zone_is_manual_and_the_original_stays_auto(
    qtbot: QtBot,
):
    view = _auto_view(qtbot)
    inside = _w(view, 200, 200)

    _drag(qtbot, view, inside, inside + QtCore.QPointF(60, 60), SHIFT | CTRL)

    original, copy = view.zones()
    assert _category(original) is ZoneCategory.AUTO
    assert _category(copy) is ZoneCategory.MANUAL


def test_an_untouched_auto_zone_stays_auto_when_another_zone_is_edited(
    qtbot: QtBot,
):
    view = _auto_view(
        qtbot, _picture(properties=AUTO), _picture(OTHER, properties=AUTO)
    )
    inside_other = _w(view, 475, 600)

    _drag(qtbot, view, inside_other, inside_other + QtCore.QPointF(10, 10), SHIFT)

    first, second = view.zones()
    assert _category(first) is ZoneCategory.AUTO
    assert _category(second) is ZoneCategory.MANUAL


def test_a_click_without_movement_leaves_an_auto_zone_auto(qtbot: QtBot):
    view = _auto_view(qtbot)

    qtbot.mouseClick(view.viewport(), LEFT, SHIFT, pos=_pt(_w(view, 200, 200)))

    assert _category(view.zones()[0]) is ZoneCategory.AUTO
