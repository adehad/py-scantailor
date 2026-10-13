"""Zone creation modes of the Output view: polygonal, lasso and rectangular."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import OutputImageView, ZoneKind
from scantailor.core.models import Dpi
from scantailor.core.settings import ApplicationSettings, ZoneCreationMode
from scantailor.core.transformation import ImageTransformation

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800


def _new_view(qtbot: QtBot) -> OutputImageView:
    created = OutputImageView()
    qtbot.addWidget(created)
    created.resize(QtCore.QSize(300, 400))
    created.show()
    qtbot.waitExposed(created)
    created.set_image(
        np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8),
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    return created


@pytest.fixture
def view(qtbot: QtBot) -> OutputImageView:
    return _new_view(qtbot)


def _widget_point(view: OutputImageView, x: float, y: float) -> QtCore.QPoint:
    mapped = view.virtual_to_widget_point(QtCore.QPointF(x, y))
    return QtCore.QPoint(round(mapped.x()), round(mapped.y()))


def _drag(qtbot: QtBot, view: OutputImageView, path: list[tuple[float, float]]) -> None:
    viewport = view.viewport()
    left = QtCore.Qt.MouseButton.LeftButton
    points = [_widget_point(view, x, y) for x, y in path]
    qtbot.mousePress(viewport, left, pos=points[0])
    for point in points[1:]:
        QtGui.QGuiApplication.sendEvent(
            viewport,
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(point),
                QtCore.QPointF(viewport.mapToGlobal(point)),
                QtCore.Qt.MouseButton.NoButton,
                left,
                QtCore.Qt.KeyboardModifier.NoModifier,
            ),
        )
    qtbot.mouseRelease(viewport, left, pos=points[-1])


def _corners(view: OutputImageView) -> list[tuple[float, float]]:
    (zone,) = view.zones()
    return [(p.x(), p.y()) for p in zone.polygon.toList()]


def test_a_new_view_starts_in_the_default_zone_creation_mode(qtbot: QtBot) -> None:
    settings = ApplicationSettings.get_instance()
    settings.default_zone_creation_mode = ZoneCreationMode.LASSO

    assert _new_view(qtbot).creation_mode() is ZoneCreationMode.LASSO


@pytest.mark.parametrize(
    ("key", "mode"),
    [
        (QtCore.Qt.Key.Key_Z, ZoneCreationMode.POLYGONAL),
        (QtCore.Qt.Key.Key_X, ZoneCreationMode.LASSO),
        (QtCore.Qt.Key.Key_C, ZoneCreationMode.RECTANGULAR),
    ],
)
def test_z_x_and_c_choose_the_mode_and_make_it_the_default(
    qtbot: QtBot, view: OutputImageView, key: QtCore.Qt.Key, mode: ZoneCreationMode
) -> None:
    other = ZoneCreationMode.LASSO if mode is ZoneCreationMode.POLYGONAL else None
    view.set_creation_mode(other or ZoneCreationMode.POLYGONAL)

    qtbot.keyClick(view, key)

    assert view.creation_mode() is mode
    assert ApplicationSettings.get_instance().default_zone_creation_mode is mode


def test_the_chosen_mode_is_saved_for_the_next_session(
    qtbot: QtBot, view: OutputImageView
) -> None:
    qtbot.keyClick(view, QtCore.Qt.Key.Key_C)

    assert (
        ApplicationSettings.load().default_zone_creation_mode
        is ZoneCreationMode.RECTANGULAR
    )


def test_dragging_in_rectangular_mode_creates_a_four_cornered_zone(
    qtbot: QtBot, view: OutputImageView
) -> None:
    view.set_creation_mode(ZoneCreationMode.RECTANGULAR)
    view.start_zone_creation(ZoneKind.PICTURE)

    with qtbot.waitSignal(view.zones_changed):
        _drag(qtbot, view, [(100, 100), (200, 200), (300, 260)])

    assert _corners(view) == pytest.approx(
        [(100, 100), (300, 100), (300, 260), (100, 260)], abs=3
    )


def test_a_click_without_a_drag_creates_no_rectangle(
    qtbot: QtBot, view: OutputImageView
) -> None:
    view.set_creation_mode(ZoneCreationMode.RECTANGULAR)
    view.start_zone_creation(ZoneKind.FILL)

    _drag(qtbot, view, [(100, 100)])

    assert view.zones() == []


def test_dragging_in_lasso_mode_follows_the_outline(
    qtbot: QtBot, view: OutputImageView
) -> None:
    view.set_creation_mode(ZoneCreationMode.LASSO)
    view.start_zone_creation(ZoneKind.FILL)
    outline = [(100, 100), (180, 100), (260, 100), (260, 200), (260, 300), (100, 300)]

    with qtbot.waitSignal(view.zones_changed):
        _drag(qtbot, view, outline)

    (zone,) = view.zones()
    assert zone.kind is ZoneKind.FILL
    assert _corners(view) == pytest.approx(outline, abs=3)


def test_a_lasso_of_two_points_creates_no_zone(
    qtbot: QtBot, view: OutputImageView
) -> None:
    view.set_creation_mode(ZoneCreationMode.LASSO)
    view.start_zone_creation(ZoneKind.PICTURE)

    _drag(qtbot, view, [(100, 100), (200, 200)])

    assert view.zones() == []


def test_changing_the_mode_drops_the_zone_being_drawn(
    qtbot: QtBot, view: OutputImageView
) -> None:
    view.start_zone_creation(ZoneKind.PICTURE)
    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=_widget_point(view, 100, 100),
    )

    qtbot.keyClick(view, QtCore.Qt.Key.Key_C)
    _drag(qtbot, view, [(100, 100), (300, 300)])

    assert _corners(view) == pytest.approx(
        [(100, 100), (300, 100), (300, 300), (100, 300)], abs=3
    )
