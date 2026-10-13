"""Image views driven with mouse, wheel and key events, without a main window."""

import math
from collections.abc import Callable

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.image_views.deskew import DeskewImageView
from scantailor.app.ui.image_views.fix_orientation import FixOrientationImageView
from scantailor.app.ui.image_views.output import OutputImageView, ViewZone, ZoneKind
from scantailor.app.ui.image_views.page_layout import PageLayoutImageView
from scantailor.app.ui.image_views.page_split import PageSplitImageView
from scantailor.app.ui.image_views.select_content import SelectContentImageView
from scantailor.core.models import Dpi, Margins, OrthogonalRotation
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.page_split.page_layout import PageLayout

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
VIEW_SIZE = QtCore.QSize(300, 400)
FIT_ZOOM = 0.5  # VIEW_SIZE shows the image at half size
ZOOMED_IN = 2.0


def _gray_page() -> np.ndarray:
    page = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)
    page[100:200, 100:300] = 40
    return page


def _identity_transformation() -> ImageTransformation:
    return ImageTransformation.from_size(IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300))


@pytest.fixture
def make_view(qtbot: QtBot) -> Callable[..., ImageViewBase]:
    """Create, size and show a view of a synthetic page."""

    def create(view_class: type[ImageViewBase], **set_image_args) -> ImageViewBase:
        view = view_class()
        qtbot.addWidget(view)
        view.resize(VIEW_SIZE)
        view.show()
        qtbot.waitExposed(view)
        set_image_args.setdefault("transformation", _identity_transformation())
        view.set_image(_gray_page(), **set_image_args)
        return view

    return create


def _point(point: QtCore.QPointF) -> QtCore.QPoint:
    return QtCore.QPoint(round(point.x()), round(point.y()))


SHIFT = QtCore.Qt.KeyboardModifier.ShiftModifier


def _drag(
    qtbot: QtBot,
    view: ImageViewBase,
    start: QtCore.QPointF,
    end: QtCore.QPointF,
    steps: int = 4,
    modifier: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> None:
    """Press at ``start``, move to ``end`` in steps and release."""
    viewport = view.viewport()
    assert viewport is not None
    qtbot.mousePress(
        viewport, QtCore.Qt.MouseButton.LeftButton, modifier, pos=_point(start)
    )
    for step in range(1, steps + 1):
        fraction = step / steps
        position = start + (end - start) * fraction
        qtbot.mouseMove(viewport, _point(position))
    qtbot.mouseRelease(
        viewport, QtCore.Qt.MouseButton.LeftButton, modifier, pos=_point(end)
    )


def _wheel(view: ImageViewBase, at: QtCore.QPointF, delta: int, modifiers=None) -> None:
    viewport = view.viewport()
    assert viewport is not None
    event = QtGui.QWheelEvent(
        at,
        viewport.mapToGlobal(at),
        QtCore.QPoint(),
        QtCore.QPoint(0, delta),
        QtCore.Qt.MouseButton.NoButton,
        modifiers or QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(viewport, event)


def _image_pixel_to_widget(view: ImageViewBase, x: float, y: float) -> QtCore.QPointF:
    return view.image_to_widget().map(QtCore.QPointF(x, y))


# Zoom and pan


def test_a_new_image_is_fitted_into_the_window(make_view):
    view = make_view(ImageViewBase)

    assert view.zoom_level == pytest.approx(FIT_ZOOM)
    shown = view.virtual_to_widget_rect(view.virtual_display_area)
    assert shown.width() == pytest.approx(IMAGE_WIDTH * FIT_ZOOM)
    assert shown.height() == pytest.approx(IMAGE_HEIGHT * FIT_ZOOM)
    assert shown.center().x() == pytest.approx(view.viewport().width() / 2)
    assert shown.center().y() == pytest.approx(view.viewport().height() / 2)


def test_a_small_image_is_not_magnified_beyond_full_size(qtbot: QtBot):
    view = ImageViewBase()
    qtbot.addWidget(view)
    view.resize(800, 800)
    view.show()
    qtbot.waitExposed(view)

    view.set_image(np.zeros((100, 100), dtype=np.uint8))

    assert view.zoom_level == pytest.approx(1.0)


def test_wheel_zoom_keeps_the_point_under_the_cursor_fixed(make_view):
    view = make_view(ImageViewBase)
    cursor = QtCore.QPointF(200, 150)
    under_cursor = view.widget_to_image().map(cursor)

    _wheel(view, cursor, delta=240)

    assert view.zoom_level == pytest.approx(FIT_ZOOM * 2**0.5)
    assert view.image_to_widget().map(under_cursor).x() == pytest.approx(200, abs=1e-6)
    assert view.image_to_widget().map(under_cursor).y() == pytest.approx(150, abs=1e-6)


def test_wheel_zoom_out_is_limited_by_the_fit_zoom_floor(make_view):
    view = make_view(ImageViewBase)

    _wheel(view, QtCore.QPointF(150, 200), delta=-1200)

    assert view.zoom_level == pytest.approx(FIT_ZOOM)


def test_deskew_wheel_zoom_keeps_the_view_centre_fixed(make_view):
    view = _deskew_view(make_view)
    centre = QtCore.QPointF(view.viewport().rect().center())
    under_centre = view.widget_to_image().map(centre)

    _wheel(view, QtCore.QPointF(20, 20), delta=240)

    assert view.zoom_level > FIT_ZOOM
    mapped = view.image_to_widget().map(under_centre)
    assert mapped.x() == pytest.approx(centre.x(), abs=1e-6)
    assert mapped.y() == pytest.approx(centre.y(), abs=1e-6)


def test_zoom_changes_the_view_transform_and_emits_zoom_changed(qtbot, make_view):
    view = make_view(ImageViewBase)
    before = view.virtual_to_widget

    with qtbot.waitSignal(view.zoom_changed) as zoomed:
        view.zoom_in()

    assert zoomed.args == [pytest.approx(FIT_ZOOM * ImageViewBase.ZOOM_STEP)]
    assert view.virtual_to_widget != before


def test_dragging_empty_space_pans_a_zoomed_in_image(qtbot, make_view):
    view = make_view(ImageViewBase)
    view.set_zoom(ZOOMED_IN)
    before = view.image_to_widget().map(QtCore.QPointF(300, 400))

    _drag(qtbot, view, QtCore.QPointF(150, 200), QtCore.QPointF(120, 170))

    after = view.image_to_widget().map(QtCore.QPointF(300, 400))
    assert after.x() - before.x() == pytest.approx(-30)
    assert after.y() - before.y() == pytest.approx(-30)


def test_panning_cannot_drag_the_image_away_from_the_window(qtbot, make_view):
    view = make_view(ImageViewBase)
    view.set_zoom(ZOOMED_IN)

    _drag(qtbot, view, QtCore.QPointF(10, 10), QtCore.QPointF(290, 390), steps=8)
    _drag(qtbot, view, QtCore.QPointF(10, 10), QtCore.QPointF(290, 390), steps=8)

    top_left = view.image_to_widget().map(QtCore.QPointF(0, 0))
    assert top_left.x() <= 0 + 1e-6
    assert top_left.y() <= 0 + 1e-6


def test_zoom_to_fit_after_panning_restores_the_fitted_view(qtbot, make_view):
    view = make_view(ImageViewBase)
    fitted = view.virtual_to_widget
    view.set_zoom(ZOOMED_IN)
    _drag(qtbot, view, QtCore.QPointF(150, 200), QtCore.QPointF(100, 120))

    view.zoom_to_fit()

    assert view.virtual_to_widget == fitted


def test_scroll_bars_pan_a_zoomed_in_image(make_view):
    view = make_view(ImageViewBase)
    view.set_zoom(ZOOMED_IN)
    horizontal = view.horizontalScrollBar()
    assert horizontal.maximum() > 0
    before = view.image_to_widget().map(QtCore.QPointF(0, 0)).x()

    horizontal.setValue(horizontal.value() + 25)

    assert view.image_to_widget().map(QtCore.QPointF(0, 0)).x() == pytest.approx(
        before - 25
    )


def test_keyboard_shortcuts_zoom(qtbot, make_view):
    view = make_view(ImageViewBase)

    qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_1)
    assert view.zoom_level == pytest.approx(1.0)
    qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_0)
    assert view.zoom_level == pytest.approx(FIT_ZOOM)


def test_the_view_maps_image_pixels_through_a_rotated_transformation(qtbot):
    view = ImageViewBase()
    qtbot.addWidget(view)
    view.resize(VIEW_SIZE)
    view.show()
    transformation = _identity_transformation()
    transformation.set_post_rotation(10.0)

    view.set_image(_gray_page(), transformation=transformation)

    for x, y in ((0, 0), (600, 0), (600, 800), (123, 456)):
        expected = transformation.transform_point(x, y)
        mapped = view.image_to_virtual.map(QtCore.QPointF(x, y))
        assert (mapped.x(), mapped.y()) == pytest.approx(expected, abs=1e-6)


def test_a_set_transformation_keeps_the_zoom_of_a_zoomed_in_view(make_view):
    view = make_view(ImageViewBase)
    view.set_zoom(ZOOMED_IN)
    transformation = _identity_transformation()
    transformation.set_post_rotation(5.0)

    view.set_transformation(transformation)

    assert view.zoom_level == pytest.approx(ZOOMED_IN)


def test_the_display_extent_enlarges_the_fitted_area(make_view):
    view = make_view(ImageViewBase)

    view.set_display_extent(QtCore.QRectF(-300, -400, 1200, 1600))

    assert view.zoom_level == pytest.approx(FIT_ZOOM / 2)
    assert view.virtual_display_area.contains(QtCore.QPointF(-299, -399))


# Page split


def _two_page_layout(split_x: float = 300.0) -> PageLayout:
    return PageLayout.two_pages(IMAGE_WIDTH, IMAGE_HEIGHT, split_x)


def test_dragging_a_split_line_handle_tilts_the_line_in_image_pixels(qtbot, make_view):
    view = make_view(PageSplitImageView)
    view.set_page_layout(_two_page_layout())
    top_handle, _bottom = view.line_handle_positions()[0]

    with qtbot.waitSignal(view.page_layout_set_locally) as moved:
        _drag(qtbot, view, top_handle, top_handle + QtCore.QPointF(40, 25))

    layout = moved.args[0]
    assert layout.cutter1 is not None
    assert layout.cutter1[0] == pytest.approx((300 + 40 / FIT_ZOOM, 0))
    assert layout.cutter1[1] == pytest.approx((300, IMAGE_HEIGHT))
    assert layout.outline == (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)


def test_dragging_the_split_line_body_moves_it_sideways_only(qtbot, make_view):
    view = make_view(PageSplitImageView)
    view.set_page_layout(_two_page_layout())
    top, bottom = view.line_handle_positions()[0]
    middle = (top + bottom) / 2

    with qtbot.waitSignal(view.page_layout_set_locally) as moved:
        _drag(qtbot, view, middle, middle + QtCore.QPointF(-30, 60))

    layout = moved.args[0]
    assert layout.get_split_line_x() == pytest.approx(300 - 30 / FIT_ZOOM)
    assert layout.cutter1 is not None
    assert layout.cutter1[0][1] == 0
    assert layout.cutter1[1][1] == IMAGE_HEIGHT


def test_a_split_line_cannot_leave_the_page(qtbot, make_view):
    view = make_view(PageSplitImageView)
    view.set_page_layout(_two_page_layout())
    top_handle, _bottom = view.line_handle_positions()[0]

    with qtbot.waitSignal(view.page_layout_set_locally) as moved:
        _drag(qtbot, view, top_handle, top_handle + QtCore.QPointF(2000, 0))

    assert moved.args[0].get_split_line_x() < IMAGE_WIDTH


def test_a_click_on_a_split_line_without_moving_it_reports_nothing(qtbot, make_view):
    view = make_view(PageSplitImageView)
    view.set_page_layout(_two_page_layout())
    top_handle, _bottom = view.line_handle_positions()[0]
    reported: list[object] = []
    view.page_layout_set_locally.connect(reported.append)

    _drag(qtbot, view, top_handle, top_handle)

    assert reported == []


def test_two_cutters_of_a_page_with_offcut_move_independently(qtbot, make_view):
    view = make_view(PageSplitImageView)
    view.set_page_layout(
        PageLayout.single_page_cut(IMAGE_WIDTH, IMAGE_HEIGHT, 100, 500)
    )
    (_l_top, _l_bottom), (right_top, _right_bottom) = view.line_handle_positions()

    with qtbot.waitSignal(view.page_layout_set_locally) as moved:
        _drag(qtbot, view, right_top, right_top + QtCore.QPointF(-20, 0))

    layout = moved.args[0]
    assert np.array(layout.cutter1) == pytest.approx(
        np.array(((100, 0), (100, IMAGE_HEIGHT)))
    )
    assert layout.cutter2 is not None
    assert layout.cutter2[0] == pytest.approx((500 - 20 / FIT_ZOOM, 0))
    assert layout.cutter2[1] == pytest.approx((500, IMAGE_HEIGHT))


def test_cutters_do_not_cross(qtbot, make_view):
    view = make_view(PageSplitImageView)
    view.set_page_layout(
        PageLayout.single_page_cut(IMAGE_WIDTH, IMAGE_HEIGHT, 100, 500)
    )
    (_l_top, _l_bottom), (right_top, _right_bottom) = view.line_handle_positions()

    with qtbot.waitSignal(view.page_layout_set_locally) as moved:
        _drag(qtbot, view, right_top, right_top + QtCore.QPointF(-200, 0))

    layout = moved.args[0]
    assert layout.cutter1[0][0] <= layout.cutter2[0][0]
    assert layout.cutter1[1][0] <= layout.cutter2[1][0]


def test_a_page_without_layout_shows_no_split_handles(make_view):
    view = make_view(PageSplitImageView)

    view.set_page_layout(None)

    assert view.line_handle_positions() == []


def test_an_uncut_page_has_no_split_lines(make_view):
    view = make_view(PageSplitImageView)

    view.set_page_layout(PageLayout.single_page_uncut(IMAGE_WIDTH, IMAGE_HEIGHT))

    assert view.line_handle_positions() == []


def test_dragging_empty_space_in_the_page_split_view_pans_instead(qtbot, make_view):
    view = make_view(PageSplitImageView)
    view.set_page_layout(_two_page_layout())
    view.set_zoom(ZOOMED_IN)
    reported: list[object] = []
    view.page_layout_set_locally.connect(reported.append)

    _drag(qtbot, view, QtCore.QPointF(20, 20), QtCore.QPointF(60, 60))

    assert reported == []


# Deskew


def _deskew_view(make_view) -> DeskewImageView:
    return make_view(DeskewImageView)


def _handle_y_for_angle(view: DeskewImageView, degrees: float) -> float:
    """Widget y of the right handle at an angle."""
    arc = view.rotation_arc_square()
    return arc.center().y() + arc.width() / 2 * math.sin(math.radians(degrees))


def _angle_of_handle_y(view: DeskewImageView, y: float) -> float:
    """Angle the right handle has when it is at a widget y."""
    arc = view.rotation_arc_square()
    return math.degrees(math.asin((y - arc.center().y()) / (arc.width() / 2)))


def test_dragging_the_right_handle_sets_the_angle_in_degrees(qtbot, make_view):
    view = _deskew_view(make_view)
    _left, right = view.rotation_handles()
    target_y = _handle_y_for_angle(view, 2.0)

    with qtbot.waitSignal(view.manual_deskew_angle_set) as rotated:
        _drag(qtbot, view, right, QtCore.QPointF(right.x(), target_y))

    expected = _angle_of_handle_y(view, right.y() + round(target_y) - round(right.y()))
    assert expected == pytest.approx(2.0, abs=0.5)
    assert rotated.args[0] == pytest.approx(expected, abs=1e-6)
    assert view.deskew_angle == pytest.approx(expected, abs=1e-6)


def test_dragging_the_left_handle_up_rotates_clockwise(qtbot, make_view):
    view = _deskew_view(make_view)
    left, _right = view.rotation_handles()
    target_y = _handle_y_for_angle(view, -3.0)

    with qtbot.waitSignal(view.manual_deskew_angle_set) as rotated:
        _drag(qtbot, view, left, QtCore.QPointF(left.x(), target_y))

    expected = -_angle_of_handle_y(view, left.y() + round(target_y) - round(left.y()))
    assert expected == pytest.approx(3.0, abs=0.5)
    assert rotated.args[0] == pytest.approx(expected, abs=1e-6)


def test_the_page_turns_with_the_deskew_angle(make_view):
    view = _deskew_view(make_view)
    before = view.image_to_virtual

    view.set_deskew_angle(4.0)

    assert view.image_to_virtual != before
    expected = _identity_transformation()
    expected.set_post_rotation(4.0)
    mapped = view.image_to_virtual.map(QtCore.QPointF(600, 0))
    assert (mapped.x(), mapped.y()) == pytest.approx(
        expected.transform_point(600, 0), abs=1e-6
    )


def test_setting_the_angle_from_outside_does_not_report_it(make_view):
    view = _deskew_view(make_view)
    reported: list[float] = []
    view.manual_deskew_angle_set.connect(reported.append)

    view.set_deskew_angle(1.5)

    assert reported == []
    assert view.deskew_angle == 1.5


def test_the_angle_is_limited_to_45_degrees(make_view):
    view = _deskew_view(make_view)

    view.set_deskew_angle(80.0)

    assert view.deskew_angle == DeskewImageView.MAX_ROTATION_DEG


def test_ctrl_wheel_rotates_in_tenths_of_a_degree(qtbot, make_view):
    view = _deskew_view(make_view)

    with qtbot.waitSignal(view.manual_deskew_angle_set) as rotated:
        _wheel(
            view,
            QtCore.QPointF(150, 200),
            delta=-120,
            modifiers=QtCore.Qt.KeyboardModifier.ControlModifier,
        )

    assert rotated.args[0] == pytest.approx(0.1)


def test_plain_wheel_in_the_deskew_view_zooms_without_rotating(make_view):
    view = _deskew_view(make_view)
    reported: list[float] = []
    view.manual_deskew_angle_set.connect(reported.append)

    _wheel(view, QtCore.QPointF(150, 200), delta=120)

    assert reported == []
    assert view.zoom_level > FIT_ZOOM


def test_arrow_keys_rotate_in_half_degree_steps(qtbot, make_view):
    view = _deskew_view(make_view)

    with qtbot.waitSignal(view.manual_deskew_angle_set) as rotated:
        qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_Right)

    assert rotated.args[0] == pytest.approx(0.5)


# Select content


def _content_view(make_view) -> SelectContentImageView:
    view = make_view(SelectContentImageView)
    view.set_content_rect(QtCore.QRectF(100, 200, 300, 400))
    return view


def test_dragging_a_content_corner_resizes_the_box_in_virtual_pixels(qtbot, make_view):
    view = _content_view(make_view)
    corner = view.virtual_to_widget_point(QtCore.QPointF(400, 600))

    with qtbot.waitSignal(view.manual_content_rect_set) as resized:
        _drag(qtbot, view, corner, corner + QtCore.QPointF(15, -20))

    rect = resized.args[0]
    assert rect.left() == pytest.approx(100)
    assert rect.top() == pytest.approx(200)
    assert rect.right() == pytest.approx(400 + 15 / FIT_ZOOM)
    assert rect.bottom() == pytest.approx(600 - 20 / FIT_ZOOM)


def test_dragging_a_content_edge_moves_only_that_side(qtbot, make_view):
    view = _content_view(make_view)
    edge_point = view.virtual_to_widget_point(QtCore.QPointF(100, 400))

    with qtbot.waitSignal(view.manual_content_rect_set) as resized:
        _drag(qtbot, view, edge_point, edge_point + QtCore.QPointF(-12, 7))

    rect = resized.args[0]
    assert rect.left() == pytest.approx(100 - 12 / FIT_ZOOM)
    assert rect.top() == pytest.approx(200)
    assert rect.right() == pytest.approx(400)
    assert rect.bottom() == pytest.approx(600)


def test_dragging_the_content_body_moves_the_box_keeping_its_size(qtbot, make_view):
    view = _content_view(make_view)
    inside = view.virtual_to_widget_point(QtCore.QPointF(250, 400))

    with qtbot.waitSignal(view.manual_content_rect_set) as moved:
        _drag(
            qtbot,
            view,
            inside,
            inside + QtCore.QPointF(20, 10),
            modifier=SHIFT,
        )

    rect = moved.args[0]
    assert rect.x() == pytest.approx(100 + 20 / FIT_ZOOM)
    assert rect.y() == pytest.approx(200 + 10 / FIT_ZOOM)
    assert rect.width() == pytest.approx(300)
    assert rect.height() == pytest.approx(400)


def test_a_content_box_cannot_leave_the_page(qtbot, make_view):
    view = _content_view(make_view)
    inside = view.virtual_to_widget_point(QtCore.QPointF(250, 400))

    with qtbot.waitSignal(view.manual_content_rect_set) as moved:
        _drag(qtbot, view, inside, inside + QtCore.QPointF(2000, 2000), modifier=SHIFT)

    rect = moved.args[0]
    assert rect.right() <= IMAGE_WIDTH + 1e-6
    assert rect.bottom() <= IMAGE_HEIGHT + 1e-6


def test_a_content_box_keeps_a_minimum_size(qtbot, make_view):
    view = _content_view(make_view)
    corner = view.virtual_to_widget_point(QtCore.QPointF(400, 600))

    with qtbot.waitSignal(view.manual_content_rect_set) as resized:
        _drag(qtbot, view, corner, corner + QtCore.QPointF(-2000, -2000))

    rect = resized.args[0]
    assert rect.width() >= SelectContentImageView.MIN_BOX_SIZE
    assert rect.height() >= SelectContentImageView.MIN_BOX_SIZE


def test_dragging_the_content_body_without_shift_pans_instead(qtbot, make_view):
    view = _content_view(make_view)
    view.set_zoom(ZOOMED_IN)
    reported: list[QtCore.QRectF] = []
    view.manual_content_rect_set.connect(reported.append)
    inside = view.virtual_to_widget_point(QtCore.QPointF(250, 400))

    _drag(qtbot, view, inside, inside + QtCore.QPointF(20, 10))

    assert reported == []
    assert view.content_rect == QtCore.QRectF(100, 200, 300, 400)


def test_moving_the_content_box_grows_the_page_box_to_contain_it(qtbot, make_view):
    view = _content_view(make_view)
    view.set_page_rect_enabled(True)
    view.set_page_rect(QtCore.QRectF(90, 190, 320, 420))
    inside = view.virtual_to_widget_point(QtCore.QPointF(250, 400))
    pages: list[QtCore.QRectF] = []
    view.manual_page_rect_set.connect(pages.append)

    _drag(qtbot, view, inside, inside + QtCore.QPointF(30, 0), modifier=SHIFT)

    assert len(pages) == 1
    assert pages[0].contains(view.content_rect)
    assert pages[0].left() == pytest.approx(90)
    assert pages[0].right() == pytest.approx(view.content_rect.right())


def test_the_page_box_cannot_shrink_inside_the_content_box(qtbot, make_view):
    view = _content_view(make_view)
    view.set_page_rect_enabled(True)
    view.set_page_rect(QtCore.QRectF(50, 150, 400, 500))
    corner = view.virtual_to_widget_point(QtCore.QPointF(450, 650))

    _drag(qtbot, view, corner, corner + QtCore.QPointF(-1000, -1000))

    assert view.page_rect.contains(view.content_rect)


def test_the_page_box_body_moves_with_shift_but_keeps_the_content(qtbot, make_view):
    view = _content_view(make_view)
    view.set_page_rect_enabled(True)
    view.set_page_rect(QtCore.QRectF(50, 150, 400, 500))
    inside_page_only = view.virtual_to_widget_point(QtCore.QPointF(70, 400))

    with qtbot.waitSignal(view.manual_page_rect_set) as moved:
        _drag(
            qtbot,
            view,
            inside_page_only,
            inside_page_only + QtCore.QPointF(-300, 0),
            modifier=SHIFT,
        )

    assert moved.args[0].contains(view.content_rect)
    assert moved.args[0].width() == pytest.approx(400)


def test_page_box_is_editable_only_when_enabled(qtbot, make_view):
    view = _content_view(make_view)
    view.set_page_rect(QtCore.QRectF(50, 50, 500, 700))
    corner = view.virtual_to_widget_point(QtCore.QPointF(550, 750))
    reported: list[QtCore.QRectF] = []
    view.manual_page_rect_set.connect(reported.append)

    _drag(qtbot, view, corner, corner + QtCore.QPointF(-10, -10))
    assert reported == []

    view.set_page_rect_enabled(True)
    _drag(qtbot, view, corner, corner + QtCore.QPointF(-10, -10))
    assert len(reported) == 1
    assert reported[0].right() == pytest.approx(550 - 10 / FIT_ZOOM)


def test_dragging_outside_the_boxes_pans_and_reports_nothing(qtbot, make_view):
    view = make_view(SelectContentImageView)
    view.set_content_rect(QtCore.QRectF(100, 200, 50, 50))
    view.set_zoom(ZOOMED_IN)
    reported: list[QtCore.QRectF] = []
    view.manual_content_rect_set.connect(reported.append)
    before = view.image_to_widget().map(QtCore.QPointF(300, 400))
    start = QtCore.QPointF(view.viewport().rect().center())

    _drag(qtbot, view, start, start - QtCore.QPointF(20, 20))

    assert reported == []
    after = view.image_to_widget().map(QtCore.QPointF(300, 400))
    assert after.x() - before.x() == pytest.approx(-20)


# Page layout


MARGIN = 60  # Far enough from the content box to be a different drag target


def _layout_view(make_view) -> PageLayoutImageView:
    view = make_view(PageLayoutImageView)
    view.set_margin_links(left_right=False, top_bottom=False)
    view.set_mm_per_pixel(0.1, 0.2)
    inner = QtCore.QRectF(200, 300, 200, 200)
    middle = inner.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
    outer = middle.adjusted(-30, -30, 30, 30)
    view.set_rectangles(inner, middle, outer)
    return view


def test_dragging_a_margin_reports_millimetres_per_side(qtbot, make_view):
    view = _layout_view(make_view)
    left_edge = view.virtual_to_widget_point(QtCore.QPointF(200 - MARGIN, 400))
    zoom = view.zoom_level

    with qtbot.waitSignal(view.margins_set_locally) as dragged:
        _drag(qtbot, view, left_edge, left_edge + QtCore.QPointF(-15, 0))

    margins: Margins = dragged.args[0]
    assert margins.left == pytest.approx((MARGIN + 15 / zoom) * 0.1)
    assert margins.right == pytest.approx(MARGIN * 0.1)
    assert margins.top == pytest.approx(MARGIN * 0.2)
    assert margins.bottom == pytest.approx(MARGIN * 0.2)


def test_dragging_a_margin_corner_moves_both_of_its_sides(qtbot, make_view):
    view = _layout_view(make_view)
    corner = view.virtual_to_widget_point(QtCore.QPointF(200 - MARGIN, 300 - MARGIN))
    zoom = view.zoom_level

    with qtbot.waitSignal(view.margins_set_locally) as dragged:
        _drag(qtbot, view, corner, corner + QtCore.QPointF(-10, -12))

    margins: Margins = dragged.args[0]
    assert margins.left == pytest.approx((MARGIN + 10 / zoom) * 0.1)
    assert margins.top == pytest.approx((MARGIN + 12 / zoom) * 0.2)
    assert margins.right == pytest.approx(MARGIN * 0.1)
    assert margins.bottom == pytest.approx(MARGIN * 0.2)


def test_a_linked_margin_cannot_shrink_its_partner_inside_the_content(qtbot, make_view):
    view = make_view(PageLayoutImageView)
    view.set_margin_links(True, False)
    view.set_mm_per_pixel(0.1, 0.2)
    view.set_rectangles(
        QtCore.QRectF(200, 300, 200, 200),
        QtCore.QRectF(120, 280, 300, 240),
        QtCore.QRectF(120, 280, 300, 240),
    )
    left_edge = view.virtual_to_widget_point(QtCore.QPointF(120, 400))

    with qtbot.waitSignal(view.margins_set_locally) as dragged:
        _drag(qtbot, view, left_edge, left_edge + QtCore.QPointF(500, 0))

    margins: Margins = dragged.args[0]
    assert margins.left == pytest.approx(0)
    assert margins.right == pytest.approx(0)
    assert view._middle_rect.contains(view._inner_rect)


def test_the_vertical_margin_uses_the_vertical_scale(qtbot, make_view):
    view = _layout_view(make_view)
    bottom_edge = view.virtual_to_widget_point(QtCore.QPointF(300, 500 + MARGIN))
    zoom = view.zoom_level

    with qtbot.waitSignal(view.margins_set_locally) as dragged:
        _drag(qtbot, view, bottom_edge, bottom_edge + QtCore.QPointF(0, 10))

    assert dragged.args[0].bottom == pytest.approx((MARGIN + 10 / zoom) * 0.2)


def test_linked_margins_move_together(qtbot, make_view):
    view = _layout_view(make_view)
    view.set_margin_links(left_right=True, top_bottom=False)
    left_edge = view.virtual_to_widget_point(QtCore.QPointF(200 - MARGIN, 400))
    zoom = view.zoom_level

    with qtbot.waitSignal(view.margins_set_locally) as dragged:
        _drag(qtbot, view, left_edge, left_edge + QtCore.QPointF(-15, 0))

    margins = dragged.args[0]
    assert margins.left == pytest.approx(margins.right)
    assert margins.left == pytest.approx((MARGIN + 15 / zoom) * 0.1)


def test_a_margin_cannot_shrink_below_the_content_box(qtbot, make_view):
    view = _layout_view(make_view)
    left_edge = view.virtual_to_widget_point(QtCore.QPointF(200 - MARGIN, 400))

    with qtbot.waitSignal(view.margins_set_locally) as dragged:
        _drag(qtbot, view, left_edge, left_edge + QtCore.QPointF(300, 0))

    assert dragged.args[0].left == pytest.approx(0)


def test_the_view_covers_the_margins_outside_the_page(make_view):
    view = _layout_view(make_view)

    assert view.virtual_display_area.left() <= 200 - MARGIN - 30 - 1e-6
    shown = view.virtual_to_widget_rect(view.virtual_display_area)
    assert shown.left() >= -1e-6


# Fix orientation


def test_pressing_r_asks_for_a_clockwise_quarter_turn(qtbot, make_view):
    view = make_view(FixOrientationImageView)

    with qtbot.waitSignal(view.rotation_requested) as requested:
        qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_R)

    assert requested.args[0] == OrthogonalRotation(degrees=90)


def test_pressing_l_asks_for_a_counter_clockwise_quarter_turn(qtbot, make_view):
    view = make_view(FixOrientationImageView)
    view.set_rotation(OrthogonalRotation(degrees=90))

    with qtbot.waitSignal(view.rotation_requested) as requested:
        qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_L)

    assert requested.args[0] == OrthogonalRotation(degrees=0)


def test_double_click_asks_for_a_clockwise_turn(qtbot, make_view):
    view = make_view(FixOrientationImageView)
    view.set_rotation(OrthogonalRotation(degrees=270))

    with qtbot.waitSignal(view.rotation_requested) as requested:
        qtbot.mouseDClick(
            view.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(100, 100),
        )

    assert requested.args[0] == OrthogonalRotation(degrees=0)


@pytest.mark.parametrize(
    "degrees, expected_size",
    [(0, (600, 800)), (90, (800, 600)), (180, (600, 800)), (270, (800, 600))],
)
def test_the_rotation_turns_the_unrotated_source_in_the_view(
    make_view, degrees, expected_size
):
    view = make_view(FixOrientationImageView)

    view.set_rotation(OrthogonalRotation(degrees=degrees))

    area = view.virtual_display_area
    assert (area.width(), area.height()) == expected_size
    assert view.image.width() == IMAGE_WIDTH
    assert view.image.height() == IMAGE_HEIGHT


def test_a_quarter_turn_moves_the_top_left_pixel_to_the_top_right(make_view):
    view = make_view(FixOrientationImageView)

    view.set_rotation(OrthogonalRotation(degrees=90))

    mapped = view.image_to_virtual.map(QtCore.QPointF(0, 0))
    assert (mapped.x(), mapped.y()) == pytest.approx((IMAGE_HEIGHT, 0))


# Output zones


def _zone_view(make_view) -> OutputImageView:
    return make_view(OutputImageView)


def _create_zone(
    qtbot: QtBot,
    view: OutputImageView,
    kind: ZoneKind,
    corners: list[tuple[float, float]],
) -> None:
    view.start_zone_creation(kind)
    viewport = view.viewport()
    for x, y in corners:
        position = view.virtual_to_widget_point(QtCore.QPointF(x, y))
        qtbot.mouseClick(
            viewport, QtCore.Qt.MouseButton.LeftButton, pos=_point(position)
        )
    last = view.virtual_to_widget_point(QtCore.QPointF(*corners[-1]))
    qtbot.mouseDClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=_point(last))


def test_clicking_corners_and_double_clicking_creates_a_picture_zone(qtbot, make_view):
    view = _zone_view(make_view)

    with qtbot.waitSignal(view.zones_changed):
        _create_zone(
            qtbot, view, ZoneKind.PICTURE, [(100, 100), (300, 100), (300, 300)]
        )

    (zone,) = view.zones()
    assert zone.kind == ZoneKind.PICTURE
    corners = [(p.x(), p.y()) for p in zone.polygon.toList()]
    assert corners[:3] == pytest.approx([(100, 100), (300, 100), (300, 300)], abs=2)


def test_a_zone_needs_three_corners(qtbot, make_view):
    view = _zone_view(make_view)
    reported: list[None] = []
    view.zones_changed.connect(lambda: reported.append(None))

    _create_zone(qtbot, view, ZoneKind.FILL, [(100, 100), (300, 100)])

    assert view.zones() == []
    assert reported == []


def test_dragging_a_corner_of_the_selected_zone_reshapes_it(qtbot, make_view):
    view = _zone_view(make_view)
    view.set_zones(
        [
            ViewZone(
                ZoneKind.PICTURE,
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(100, 100),
                        QtCore.QPointF(300, 100),
                        QtCore.QPointF(300, 300),
                    ]
                ),
            )
        ]
    )
    inside = view.virtual_to_widget_point(QtCore.QPointF(250, 150))
    qtbot.mouseClick(
        view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=_point(inside)
    )
    corner = view.virtual_to_widget_point(QtCore.QPointF(300, 300))

    with qtbot.waitSignal(view.zones_changed):
        _drag(qtbot, view, corner, corner + QtCore.QPointF(10, 20))

    moved = view.zones()[0].polygon.at(2)
    assert moved.x() == pytest.approx(300 + 10 / FIT_ZOOM, abs=1)
    assert moved.y() == pytest.approx(300 + 20 / FIT_ZOOM, abs=1)


def test_delete_removes_the_selected_zone(qtbot, make_view):
    view = _zone_view(make_view)
    view.set_zones(
        [
            ViewZone(
                ZoneKind.FILL,
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(100, 100),
                        QtCore.QPointF(300, 100),
                        QtCore.QPointF(300, 300),
                    ]
                ),
                fill_color=(10, 20, 30),
            )
        ]
    )
    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=_point(view.virtual_to_widget_point(QtCore.QPointF(250, 150))),
    )

    with qtbot.waitSignal(view.zones_changed):
        qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_Delete)

    assert view.zones() == []


def test_set_zones_does_not_report_a_change(make_view):
    view = _zone_view(make_view)
    reported: list[None] = []
    view.zones_changed.connect(lambda: reported.append(None))

    view.set_zones(
        [
            ViewZone(
                ZoneKind.PICTURE,
                QtGui.QPolygonF(
                    [QtCore.QPointF(0, 0), QtCore.QPointF(5, 0), QtCore.QPointF(5, 5)]
                ),
            )
        ]
    )

    assert reported == []
    assert len(view.zones()) == 1


def test_zone_properties_survive_a_round_trip_through_the_view(make_view):
    view = _zone_view(make_view)
    marker = object()

    view.set_zones(
        [
            ViewZone(
                ZoneKind.PICTURE,
                QtGui.QPolygonF(
                    [QtCore.QPointF(0, 0), QtCore.QPointF(5, 0), QtCore.QPointF(5, 5)]
                ),
                properties=marker,
            )
        ]
    )

    assert view.zones()[0].properties is marker


def test_escape_cancels_the_zone_being_created(qtbot, make_view):
    view = _zone_view(make_view)
    view.start_zone_creation(ZoneKind.PICTURE)
    qtbot.mouseClick(
        view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(50, 50)
    )

    qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_Escape)
    qtbot.mouseDClick(
        view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(60, 60)
    )

    assert view.zones() == []
