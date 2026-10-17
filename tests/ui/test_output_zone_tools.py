"""Zone tools of the Output view: whole-zone drag, vertices, menu, layer and colour."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import (
    ImageViewTab,
    OutputImageView,
    ViewZone,
    ZoneKind,
)
from scantailor.app.ui.widgets.picture_zone_dialog import PictureZoneDialog
from scantailor.core.models import Dpi
from scantailor.core.settings import ZoneCreationMode
from scantailor.core.transformation import ImageTransformation
from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
)

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
ZOOM = 0.5
LEFT = QtCore.Qt.MouseButton.LeftButton
SHIFT = QtCore.Qt.KeyboardModifier.ShiftModifier
CTRL = QtCore.Qt.KeyboardModifier.ControlModifier
ALT = QtCore.Qt.KeyboardModifier.AltModifier
NONE = QtCore.Qt.KeyboardModifier.NoModifier
RED = (200, 30, 30)
SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]
OTHER = [(400, 500), (550, 500), (550, 700), (400, 700)]


def _polygon(points: list[tuple[float, float]]) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])


def _picture(points=SQUARE, **kwargs) -> ViewZone:
    return ViewZone(ZoneKind.PICTURE, _polygon(points), **kwargs)


def _fill(points=OTHER, color=(255, 255, 255)) -> ViewZone:
    return ViewZone(
        ZoneKind.FILL, _polygon(points), color, FillZoneProperties(color=color)
    )


def _new_view(qtbot: QtBot, image: np.ndarray | None = None) -> OutputImageView:
    view = OutputImageView()
    qtbot.addWidget(view)
    view.resize(QtCore.QSize(300, 400))
    view.show()
    qtbot.waitExposed(view)
    view.set_image(
        image
        if image is not None
        else np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8),
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    view.set_creation_mode(ZoneCreationMode.POLYGONAL)
    return view


def _tabbed(qtbot: QtBot, tab: ImageViewTab, *zones: ViewZone) -> OutputImageView:
    view = _new_view(qtbot)
    view.set_tabs(list(ImageViewTab), tab)
    view.set_zones(list(zones))
    return view


@pytest.fixture
def pictures(qtbot: QtBot) -> OutputImageView:
    return _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())


def _w(view: OutputImageView, x: float, y: float) -> QtCore.QPointF:
    return view.virtual_to_widget_point(QtCore.QPointF(x, y))


def _pt(point: QtCore.QPointF) -> QtCore.QPoint:
    return QtCore.QPoint(round(point.x()), round(point.y()))


def _moved(
    view: OutputImageView, points: list[tuple[float, float]], dx: float, dy: float
) -> list[tuple[float, float]]:
    """Points moved by a widget distance, in virtual coordinates."""
    zoom = view.zoom_level
    return [(x + dx / zoom, y + dy / zoom) for x, y in points]


def _flat(points: list[tuple[float, float]]) -> list[float]:
    return [value for point in points for value in point]


def _corners(zone: ViewZone) -> list[tuple[float, float]]:
    return [(p.x(), p.y()) for p in zone.polygon.toList()]


def _hover(view: OutputImageView, point: QtCore.QPointF) -> None:
    viewport = view.viewport()
    QtGui.QGuiApplication.sendEvent(
        viewport,
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            point,
            QtCore.QPointF(viewport.mapToGlobal(_pt(point))),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            NONE,
        ),
    )


def _drag(
    qtbot: QtBot,
    view: OutputImageView,
    start: QtCore.QPointF,
    end: QtCore.QPointF,
    modifier: QtCore.Qt.KeyboardModifier = NONE,
    release: bool = True,
) -> None:
    viewport = view.viewport()
    qtbot.mousePress(viewport, LEFT, modifier, pos=_pt(start))
    for step in range(1, 5):
        position = start + (end - start) * (step / 4)
        QtGui.QGuiApplication.sendEvent(
            viewport,
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                position,
                QtCore.QPointF(viewport.mapToGlobal(_pt(position))),
                QtCore.Qt.MouseButton.NoButton,
                LEFT,
                modifier,
            ),
        )
    if release:
        qtbot.mouseRelease(viewport, LEFT, modifier, pos=_pt(end))


class TestWholeZoneDrag:
    def test_shift_drag_moves_the_zone_by_the_mouse_distance(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        inside = _w(pictures, 200, 200)

        with qtbot.waitSignal(pictures.zones_changed):
            _drag(qtbot, pictures, inside, inside + QtCore.QPointF(30, 20), SHIFT)

        (zone,) = pictures.zones()
        assert _flat(_corners(zone)) == pytest.approx(
            _flat(_moved(pictures, SQUARE, 30, 20)), abs=3
        )

    def test_the_drag_reports_one_change_when_it_ends(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        reports: list[None] = []
        pictures.zones_changed.connect(lambda: reports.append(None))
        inside = _w(pictures, 200, 200)

        _drag(qtbot, pictures, inside, inside + QtCore.QPointF(30, 20), SHIFT)

        assert len(reports) == 1

    def test_shift_without_movement_changes_nothing(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        reports: list[None] = []
        pictures.zones_changed.connect(lambda: reports.append(None))
        inside = _w(pictures, 200, 200)

        qtbot.mouseClick(pictures.viewport(), LEFT, SHIFT, pos=_pt(inside))

        assert reports == []
        assert _corners(pictures.zones()[0]) == SQUARE

    def test_shift_ctrl_click_without_movement_reports_the_copy_it_made(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        reports: list[None] = []
        pictures.zones_changed.connect(lambda: reports.append(None))
        inside = _w(pictures, 200, 200)

        qtbot.mouseClick(pictures.viewport(), LEFT, SHIFT | CTRL, pos=_pt(inside))

        assert len(pictures.zones()) == 2
        assert len(reports) == 1

    def test_shift_ctrl_drag_moves_a_copy_and_keeps_the_original(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        inside = _w(pictures, 200, 200)

        with qtbot.waitSignal(pictures.zones_changed):
            _drag(
                qtbot,
                pictures,
                inside,
                inside + QtCore.QPointF(30, 20),
                SHIFT | CTRL,
            )

        original, copy = pictures.zones()
        assert _corners(original) == SQUARE
        assert _flat(_corners(copy)) == pytest.approx(
            _flat(_moved(pictures, SQUARE, 30, 20)), abs=3
        )

    def test_a_copy_keeps_the_layer_of_its_original(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        layer = PictureZoneProperties(layer=PictureLayer.ERASER1)
        pictures.set_zones([_picture(properties=layer)])
        inside = _w(pictures, 200, 200)

        _drag(qtbot, pictures, inside, inside + QtCore.QPointF(30, 20), SHIFT | CTRL)

        assert [zone.properties for zone in pictures.zones()] == [layer, layer]

    def test_only_the_zone_under_the_mouse_moves(self, qtbot: QtBot):
        view = _tabbed(
            qtbot,
            ImageViewTab.PICTURE_ZONES,
            _picture(),
            _picture(OTHER),
        )
        inside = _w(view, 450, 600)

        _drag(qtbot, view, inside, inside + QtCore.QPointF(20, 0), SHIFT)

        first, second = view.zones()
        assert _corners(first) == SQUARE
        assert _corners(second)[0][0] == pytest.approx(
            400 + 20 / view.zoom_level, abs=3
        )

    def test_a_plain_drag_inside_a_zone_starts_a_new_zone_not_a_move(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        inside = _w(pictures, 200, 200)

        qtbot.mouseClick(pictures.viewport(), LEFT, pos=_pt(inside))

        assert _corners(pictures.zones()[0]) == SQUARE
        assert pictures._creating is ZoneKind.PICTURE

    def test_without_tabs_shift_drag_moves_the_zone_too(self, qtbot: QtBot):
        view = _new_view(qtbot)
        view.set_zones([_picture()])
        inside = _w(view, 200, 200)

        _drag(qtbot, view, inside, inside + QtCore.QPointF(30, 20), SHIFT)

        assert _flat(_corners(view.zones()[0])) == pytest.approx(
            _flat(_moved(view, SQUARE, 30, 20)), abs=3
        )


class TestInsertingVertices:
    def test_pressing_on_an_edge_adds_a_vertex_that_follows_the_mouse(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        on_top_edge = _w(pictures, 200, 100)

        with qtbot.waitSignal(pictures.zones_changed):
            _drag(qtbot, pictures, on_top_edge, on_top_edge + QtCore.QPointF(0, -30))

        (zone,) = pictures.zones()
        corners = _corners(zone)
        assert len(corners) == 5
        assert corners[0] == (100, 100)
        assert corners[1] == pytest.approx((200, 100 - 30 / pictures.zoom_level), abs=3)
        assert corners[2:] == [(300, 100), (300, 300), (100, 300)]

    def test_an_edge_click_without_movement_keeps_the_new_vertex(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        on_right_edge = _w(pictures, 300, 200)

        with qtbot.waitSignal(pictures.zones_changed):
            qtbot.mouseClick(pictures.viewport(), LEFT, pos=_pt(on_right_edge))

        (zone,) = pictures.zones()
        assert len(_corners(zone)) == 5
        assert _corners(zone)[2] == pytest.approx((300, 200), abs=2)

    def test_the_closing_edge_takes_a_vertex_too(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        on_left_edge = _w(pictures, 100, 200)

        qtbot.mouseClick(pictures.viewport(), LEFT, pos=_pt(on_left_edge))

        corners = _corners(pictures.zones()[0])
        assert len(corners) == 5
        assert corners[-1] == pytest.approx((100, 200), abs=2)

    def test_a_vertex_wins_over_the_edge_it_sits_on(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        corner = _w(pictures, 300, 100)

        _drag(qtbot, pictures, corner, corner + QtCore.QPointF(10, 10))

        assert len(_corners(pictures.zones()[0])) == 4

    def test_every_zone_of_the_tab_reacts_without_being_selected(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture(), _picture(OTHER))

        qtbot.mouseClick(view.viewport(), LEFT, pos=_pt(_w(view, 475, 500)))

        first, second = view.zones()
        assert len(_corners(first)) == 4
        assert len(_corners(second)) == 5

    def test_a_far_click_does_not_insert_a_vertex(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        pictures.cancel_zone_creation()
        beside = _w(pictures, 200, 100) + QtCore.QPointF(0, -20)

        qtbot.mouseClick(pictures.viewport(), LEFT, pos=_pt(beside))

        assert len(_corners(pictures.zones()[0])) == 4


class TestRemovingVertices:
    def test_d_over_a_vertex_removes_it(self, qtbot: QtBot, pictures: OutputImageView):
        _hover(pictures, _w(pictures, 300, 300))

        with qtbot.waitSignal(pictures.zones_changed):
            qtbot.keyClick(pictures, QtCore.Qt.Key.Key_D)

        assert _corners(pictures.zones()[0]) == [(100, 100), (300, 100), (100, 300)]

    def test_a_triangle_keeps_its_vertices(self, qtbot: QtBot):
        view = _tabbed(
            qtbot,
            ImageViewTab.PICTURE_ZONES,
            _picture([(100, 100), (300, 100), (200, 300)]),
        )
        _hover(view, _w(view, 200, 300))

        qtbot.keyClick(view, QtCore.Qt.Key.Key_D)

        assert len(_corners(view.zones()[0])) == 3

    def test_d_away_from_a_vertex_does_nothing(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        reports: list[None] = []
        pictures.zones_changed.connect(lambda: reports.append(None))
        _hover(pictures, _w(pictures, 200, 200))

        qtbot.keyClick(pictures, QtCore.Qt.Key.Key_D)

        assert reports == []
        assert len(_corners(pictures.zones()[0])) == 4

    def test_d_over_an_edge_does_not_remove_a_vertex(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        _hover(pictures, _w(pictures, 200, 100))

        qtbot.keyClick(pictures, QtCore.Qt.Key.Key_D)

        assert len(_corners(pictures.zones()[0])) == 4

    def test_dropping_a_vertex_on_a_neighbour_merges_them(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        corner = _w(pictures, 300, 100)
        neighbour = _w(pictures, 100, 100)

        with qtbot.waitSignal(pictures.zones_changed):
            _drag(qtbot, pictures, corner, neighbour + QtCore.QPointF(3, 2))

        assert _corners(pictures.zones()[0]) == [(100, 100), (300, 300), (100, 300)]

    def test_a_triangle_vertex_dropped_on_a_neighbour_stays(self, qtbot: QtBot):
        view = _tabbed(
            qtbot,
            ImageViewTab.PICTURE_ZONES,
            _picture([(100, 100), (300, 100), (200, 300)]),
        )
        corner = _w(view, 300, 100)

        _drag(qtbot, view, corner, _w(view, 100, 100))

        assert len(_corners(view.zones()[0])) == 3


class TestRightAngle:
    def test_dragging_with_ctrl_squares_the_corner(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        pictures.set_zones([_picture([(100, 100), (300, 120), (320, 300), (90, 310)])])
        corner = _w(pictures, 300, 120)

        _drag(qtbot, pictures, corner, corner + QtCore.QPointF(10, 10), CTRL)

        points = _corners(pictures.zones()[0])
        here, before, after = points[1], points[0], points[2]
        assert (before[1] == pytest.approx(here[1], abs=0.01)) != (
            before[0] == pytest.approx(here[0], abs=0.01)
        )
        assert (after[1] == pytest.approx(here[1], abs=0.01)) != (
            after[0] == pytest.approx(here[0], abs=0.01)
        )

    def test_without_ctrl_the_neighbours_stay(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        quad = [(100, 100), (300, 120), (320, 300), (90, 310)]
        pictures.set_zones([_picture(quad)])
        corner = _w(pictures, 300, 120)

        _drag(qtbot, pictures, corner, corner + QtCore.QPointF(10, 10))

        points = _corners(pictures.zones()[0])
        assert points[0] == quad[0]
        assert points[2] == quad[2]


class TestDeleting:
    def test_delete_removes_the_zone_under_the_mouse(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture(), _picture(OTHER))
        _hover(view, _w(view, 475, 600))

        with qtbot.waitSignal(view.zones_changed):
            qtbot.keyClick(view, QtCore.Qt.Key.Key_Delete)

        (left,) = view.zones()
        assert _corners(left) == SQUARE

    def test_delete_with_the_mouse_outside_every_zone_removes_the_selected_one(
        self, qtbot: QtBot, pictures: OutputImageView
    ):
        corner = _w(pictures, 100, 100)
        _drag(qtbot, pictures, corner, corner + QtCore.QPointF(5, 5))
        _hover(pictures, _w(pictures, 500, 700))

        qtbot.keyClick(pictures, QtCore.Qt.Key.Key_Delete)

        assert pictures.zones() == []

    def test_delete_does_nothing_without_a_zone(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES)

        qtbot.keyClick(view, QtCore.Qt.Key.Key_Delete)

        assert view.zones() == []

    def test_the_other_kind_of_zone_is_not_deleted_from_this_tab(self, qtbot: QtBot):
        view = _tabbed(
            qtbot,
            ImageViewTab.PICTURE_ZONES,
            _picture(),
            _fill([(110, 110), (290, 110), (290, 290)]),
        )
        _hover(view, _w(view, 150, 200))

        qtbot.keyClick(view, QtCore.Qt.Key.Key_Delete)

        assert [zone.kind for zone in view.zones()] == [ZoneKind.FILL]


class TestCreatingInZoneTabs:
    def _click(self, qtbot: QtBot, view: OutputImageView, x: float, y: float) -> None:
        qtbot.mouseClick(view.viewport(), LEFT, pos=_pt(_w(view, x, y)))

    def test_clicking_corners_and_pressing_enter_creates_a_picture_zone(
        self, qtbot: QtBot
    ):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES)
        for x, y in [(100, 100), (300, 100), (300, 300)]:
            self._click(qtbot, view, x, y)

        with qtbot.waitSignal(view.zones_changed):
            qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        (zone,) = view.zones()
        assert zone.kind is ZoneKind.PICTURE
        assert _flat(_corners(zone)) == pytest.approx(
            _flat([(100, 100), (300, 100), (300, 300)]), abs=3
        )

    def test_a_new_picture_zone_gets_the_default_layer(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES)
        for x, y in [(100, 100), (300, 100), (300, 300)]:
            self._click(qtbot, view, x, y)
        qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        (zone,) = view.zones()

        assert zone.properties == PictureZoneProperties(layer=PictureLayer.PAINTER2)

    def test_the_fill_tab_creates_fill_zones_with_the_default_colour(
        self, qtbot: QtBot
    ):
        view = _tabbed(qtbot, ImageViewTab.FILL_ZONES)
        for x, y in [(100, 100), (300, 100), (300, 300)]:
            self._click(qtbot, view, x, y)
        qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        (zone,) = view.zones()

        assert zone.kind is ZoneKind.FILL
        assert zone.fill_color == (255, 255, 255)
        assert zone.properties == FillZoneProperties(color=(255, 255, 255))

    def test_clicking_the_first_corner_again_finishes_the_zone(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES)
        for x, y in [(100, 100), (300, 100), (300, 300)]:
            self._click(qtbot, view, x, y)

        with qtbot.waitSignal(view.zones_changed):
            self._click(qtbot, view, 100, 100)

        (zone,) = view.zones()
        assert len(_corners(zone)) == 3

    def test_two_corners_are_not_enough_to_close_on_the_first(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES)
        self._click(qtbot, view, 100, 100)
        self._click(qtbot, view, 300, 100)

        self._click(qtbot, view, 100, 100)

        assert view.zones() == []
        assert len(view._new_points) == 3

    def test_escape_drops_the_zone_being_drawn(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES)
        self._click(qtbot, view, 100, 100)

        qtbot.keyClick(view, QtCore.Qt.Key.Key_Escape)

        assert view.zones() == []
        assert view._creating is None

    def test_a_second_zone_can_follow_the_first(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES)
        for corners in (
            [(100, 100), (300, 100), (300, 300)],
            [(400, 500), (550, 500), (550, 700)],
        ):
            for x, y in corners:
                self._click(qtbot, view, x, y)
            qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        assert len(view.zones()) == 2

    def test_ctrl_alt_click_copies_the_latest_zone_of_the_tab_there(self, qtbot: QtBot):
        view = _tabbed(
            qtbot,
            ImageViewTab.PICTURE_ZONES,
            _picture(),
            _fill(OTHER),
        )

        with qtbot.waitSignal(view.zones_changed):
            qtbot.mouseClick(
                view.viewport(), LEFT, CTRL | ALT, pos=_pt(_w(view, 400, 600))
            )

        picture, fill, copy = view.zones()
        assert copy.kind is ZoneKind.PICTURE
        centre = copy.polygon.boundingRect().center()
        assert (centre.x(), centre.y()) == pytest.approx((400, 600), abs=2)
        assert (
            copy.polygon.boundingRect().size() == picture.polygon.boundingRect().size()
        )
        assert _corners(picture) == SQUARE
        assert fill.kind is ZoneKind.FILL

    def test_ctrl_alt_click_with_no_zone_of_the_kind_starts_a_zone_instead(
        self, qtbot: QtBot
    ):
        view = _tabbed(qtbot, ImageViewTab.FILL_ZONES, _picture())

        qtbot.mouseClick(view.viewport(), LEFT, CTRL | ALT, pos=_pt(_w(view, 400, 600)))

        assert [zone.kind for zone in view.zones()] == [ZoneKind.PICTURE]
        assert view._creating is ZoneKind.FILL


class TestTabsSeparateTheKinds:
    def test_a_picture_tab_ignores_fill_zones(self, qtbot: QtBot):
        view = _tabbed(
            qtbot,
            ImageViewTab.PICTURE_ZONES,
            _fill([(100, 100), (300, 100), (300, 300), (100, 300)]),
        )
        corner = _w(view, 300, 100)

        _drag(qtbot, view, corner, corner + QtCore.QPointF(20, 20))

        assert _corners(view.zones()[0]) == [
            (100, 100),
            (300, 100),
            (300, 300),
            (100, 300),
        ]

    def test_the_output_tab_edits_no_zones(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.OUTPUT, _picture())
        corner = _w(view, 300, 100)

        _drag(qtbot, view, corner, corner + QtCore.QPointF(20, 20))
        qtbot.keyClick(view, QtCore.Qt.Key.Key_Delete)

        assert _corners(view.zones()[0]) == SQUARE

    def test_zones_are_painted_in_their_own_tab_only(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.OUTPUT, _picture())
        out = _grab_ink(view)
        view.set_current_tab(ImageViewTab.PICTURE_ZONES)
        shown = _grab_ink(view)
        view.set_current_tab(ImageViewTab.FILL_ZONES)
        other = _grab_ink(view)

        assert out == 0
        assert shown > 0
        assert other == 0


def _grab_ink(view: OutputImageView) -> int:
    """Count viewport pixels that differ from the white page and the background."""
    image = (
        view.viewport()
        .grab()
        .toImage()
        .convertToFormat(QtGui.QImage.Format.Format_RGB888)
    )
    pixels = (
        np.frombuffer(image.constBits(), dtype=np.uint8)
        .reshape(image.height(), image.bytesPerLine())[:, : image.width() * 3]
        .reshape(image.height(), image.width(), 3)
    )
    coloured = pixels.max(axis=2).astype(int) - pixels.min(axis=2).astype(int) > 40
    return int(coloured.sum())


class TestContextMenu:
    @staticmethod
    def _texts(menu: QtWidgets.QMenu) -> list[str]:
        return [a.text() for a in menu.actions() if not a.isSeparator()]

    def test_a_picture_zone_offers_properties_and_delete(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())

        menu = view.build_context_menu(_w(view, 200, 200))

        assert self._texts(menu) == ["Properties", "Delete", "Show guides", "Show grid"]

    def test_a_fill_zone_offers_pick_color_and_delete(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.FILL_ZONES, _fill(SQUARE))

        menu = view.build_context_menu(_w(view, 200, 200))

        assert self._texts(menu) == ["Pick color", "Delete", "Show guides", "Show grid"]

    def test_no_zone_under_the_cursor_gives_only_the_guides_and_grid(
        self, qtbot: QtBot
    ):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())

        menu = view.build_context_menu(_w(view, 500, 700))

        assert self._texts(menu) == ["Show guides", "Show grid"]

    def test_every_zone_under_the_cursor_is_listed_smallest_first(self, qtbot: QtBot):
        big = _picture([(50, 50), (350, 50), (350, 350), (50, 350)])
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, big, _picture())

        menu = view.build_context_menu(_w(view, 200, 200))

        assert self._texts(menu) == [
            "Properties",
            "Delete",
            "Properties",
            "Delete",
            "Show guides",
            "Show grid",
        ]

    def test_several_zones_get_colour_swatches_to_tell_them_apart(self, qtbot: QtBot):
        big = _picture([(50, 50), (350, 50), (350, 350), (50, 350)])
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, big, _picture())

        menu = view.build_context_menu(_w(view, 200, 200))

        items = [a for a in menu.actions() if a.text() in ("Properties", "Delete")]
        assert all(not a.icon().isNull() for a in items)
        assert len({a.icon().pixmap(16, 16).toImage().pixel(2, 2) for a in items}) == 2

    def test_a_single_zone_has_no_swatch(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())

        menu = view.build_context_menu(_w(view, 200, 200))

        assert all(a.icon().isNull() for a in menu.actions() if a.text() == "Delete")

    def test_delete_removes_that_zone(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture(), _picture(OTHER))
        menu = view.build_context_menu(_w(view, 200, 200))
        (delete,) = [a for a in menu.actions() if a.text() == "Delete"]

        with qtbot.waitSignal(view.zones_changed):
            delete.trigger()

        (kept,) = view.zones()
        assert _corners(kept) == OTHER

    def test_hovering_a_zone_item_highlights_that_zone(self, qtbot: QtBot):
        big = _picture([(50, 50), (350, 50), (350, 350), (50, 350)])
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, big, _picture())
        menu = view.build_context_menu(_w(view, 200, 200))
        view._cursor = _w(view, 200, 200)
        deletes = [a for a in menu.actions() if a.text() == "Delete"]

        deletes[1].hovered.emit()

        assert view._menu_highlight is not None
        assert view._menu_highlight[0] == 0  # the big zone

    def test_the_menu_without_tabs_keeps_the_creation_items(self, qtbot: QtBot):
        view = _new_view(qtbot)
        view.set_zones([_picture()])

        menu = view.build_context_menu(_w(view, 200, 200))

        assert {"Create Picture Zone", "Create Fill Zone", "Clear All Zones"} <= set(
            self._texts(menu)
        )
        assert "Properties" in self._texts(menu)

    def test_the_zone_tabs_have_no_creation_items(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())

        menu = view.build_context_menu(_w(view, 200, 200))

        assert "Create Picture Zone" not in self._texts(menu)

    def test_the_output_tab_menu_has_only_the_guides_and_grid(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.OUTPUT, _picture())

        menu = view.build_context_menu(_w(view, 200, 200))

        assert self._texts(menu) == ["Show guides", "Show grid"]


class TestPictureProperties:
    @staticmethod
    def _choose(monkeypatch: pytest.MonkeyPatch, name: str | None, accept: bool):
        """Make the dialog pick a layer, or none, and end as accepted or cancelled."""

        def run(dialog: PictureZoneDialog) -> int:
            if name is not None:
                button = dialog.findChild(QtWidgets.QRadioButton, name)
                assert button is not None
                button.click()
            return int(
                QtWidgets.QDialog.DialogCode.Accepted
                if accept
                else QtWidgets.QDialog.DialogCode.Rejected
            )

        monkeypatch.setattr(PictureZoneDialog, "exec", run)

    def test_accepting_stores_the_layer_and_reports_a_change(
        self, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
    ):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())
        self._choose(monkeypatch, "zoneeraser1", accept=True)

        with qtbot.waitSignal(view.zones_changed):
            assert view.edit_zone_properties(0)

        (zone,) = view.zones()
        assert zone.properties == PictureZoneProperties(layer=PictureLayer.ERASER1)

    def test_cancelling_puts_the_old_layer_back(
        self, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
    ):
        old = PictureZoneProperties(layer=PictureLayer.FOREGROUND)
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture(properties=old))
        reports: list[None] = []
        view.zones_changed.connect(lambda: reports.append(None))
        self._choose(monkeypatch, "zonebackground", accept=False)

        assert not view.edit_zone_properties(0)

        assert view.zones()[0].properties == old
        assert reports == []

    def test_the_zone_shows_the_choice_while_the_dialog_is_open(
        self, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
    ):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())
        seen: list[object] = []

        def run(dialog: PictureZoneDialog) -> int:
            button = dialog.findChild(QtWidgets.QRadioButton, "zonebackground")
            assert button is not None
            button.click()
            seen.append(view._zones[0].properties)
            return int(QtWidgets.QDialog.DialogCode.Rejected)

        monkeypatch.setattr(PictureZoneDialog, "exec", run)

        view.edit_zone_properties(0)

        assert seen == [PictureZoneProperties(layer=PictureLayer.BACKGROUND)]

    def test_keeping_the_layer_reports_no_change(
        self, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
    ):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())
        reports: list[None] = []
        view.zones_changed.connect(lambda: reports.append(None))
        self._choose(monkeypatch, None, accept=True)

        assert not view.edit_zone_properties(0)

        assert reports == []

    def test_the_chosen_layer_becomes_the_default_of_new_zones(
        self, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
    ):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())
        self._choose(monkeypatch, "zoneeraser3", accept=True)
        view.edit_zone_properties(0)

        for x, y in [(400, 500), (550, 500), (550, 700)]:
            qtbot.mouseClick(view.viewport(), LEFT, pos=_pt(_w(view, x, y)))
        qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        assert view.default_picture_layer() is PictureLayer.ERASER3
        assert view.zones()[-1].properties == PictureZoneProperties(
            layer=PictureLayer.ERASER3
        )

    def test_the_menu_item_opens_the_dialog(
        self, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
    ):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())
        self._choose(monkeypatch, "zonebackground", accept=True)
        menu = view.build_context_menu(_w(view, 200, 200))
        (properties,) = [a for a in menu.actions() if a.text() == "Properties"]

        properties.trigger()

        assert view.zones()[0].properties == PictureZoneProperties(
            layer=PictureLayer.BACKGROUND
        )

    def test_a_fill_zone_has_no_layer_dialog(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.FILL_ZONES, _fill())

        assert not view.edit_zone_properties(0)

    def test_an_unknown_index_is_ignored(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES)

        assert not view.edit_zone_properties(3)


class TestColorPickup:
    @pytest.fixture
    def view(self, qtbot: QtBot) -> OutputImageView:
        page = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 255, dtype=np.uint8)
        page[:, 300:] = RED
        view = _new_view(qtbot, page)
        view.set_tabs(list(ImageViewTab), ImageViewTab.FILL_ZONES)
        view.set_zones([_fill([(100, 100), (200, 100), (200, 200)])])
        return view

    def test_picking_recolours_the_zone_and_reports_a_change(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.start_color_pickup(0)

        with qtbot.waitSignal(view.zones_changed):
            view.pick_color_at(_w(view, 450, 500))

        (zone,) = view.zones()
        assert zone.fill_color == RED
        assert zone.properties == FillZoneProperties(color=RED)
        assert not view.color_pickup_active()

    def test_the_picked_colour_is_the_default_of_new_fill_zones(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.start_color_pickup(0)
        view.pick_color_at(_w(view, 450, 500))

        for x, y in [(350, 400), (450, 400), (450, 500)]:
            qtbot.mouseClick(view.viewport(), LEFT, pos=_pt(_w(view, x, y)))
        qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        assert view.default_fill_color() == RED
        assert view.zones()[-1].fill_color == RED

    def test_the_zones_are_hidden_while_picking_so_the_page_shows(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.set_zones(
            [_fill([(300, 300), (500, 300), (500, 500), (300, 500)], (0, 160, 0))]
        )
        centre = _w(view, 400, 400)
        with_zone = _grab_ink(view)

        view.start_color_pickup(0)
        view.pick_color_at(centre)

        assert with_zone > 0
        assert view.zones()[0].fill_color == RED

    def test_a_click_picks_too(self, qtbot: QtBot, view: OutputImageView):
        view.start_color_pickup(0)

        with qtbot.waitSignal(view.zones_changed):
            qtbot.mouseClick(view.viewport(), LEFT, pos=_pt(_w(view, 450, 500)))

        assert view.zones()[0].fill_color == RED

    def test_escape_cancels_the_pickup(self, qtbot: QtBot, view: OutputImageView):
        reports: list[None] = []
        view.zones_changed.connect(lambda: reports.append(None))
        view.start_color_pickup(0)

        qtbot.keyClick(view, QtCore.Qt.Key.Key_Escape)

        assert not view.color_pickup_active()
        assert view.zones()[0].fill_color == (255, 255, 255)
        assert reports == []

    def test_the_menu_item_starts_the_pickup(self, qtbot: QtBot, view: OutputImageView):
        menu = view.build_context_menu(_w(view, 150, 110))
        (pick,) = [a for a in menu.actions() if a.text() == "Pick color"]

        pick.trigger()

        assert view.color_pickup_active()

    def test_only_fill_zones_can_be_recoloured(self, qtbot: QtBot):
        view = _tabbed(qtbot, ImageViewTab.PICTURE_ZONES, _picture())

        view.start_color_pickup(0)

        assert not view.color_pickup_active()

    def test_a_pickup_does_not_start_for_an_unknown_zone(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.start_color_pickup(5)

        assert not view.color_pickup_active()

    def test_a_right_click_while_picking_cancels_instead_of_opening_the_menu(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.start_color_pickup(0)

        view.contextMenuEvent(
            QtGui.QContextMenuEvent(
                QtGui.QContextMenuEvent.Reason.Mouse,
                QtCore.QPoint(150, 110),
                view.mapToGlobal(QtCore.QPoint(150, 110)),
            )
        )

        assert not view.color_pickup_active()

    def test_changing_tab_cancels_the_pickup(self, qtbot: QtBot, view: OutputImageView):
        view.start_color_pickup(0)

        view.set_current_tab(ImageViewTab.OUTPUT)

        assert not view.color_pickup_active()
