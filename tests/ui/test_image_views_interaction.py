"""Editing a page's geometry by dragging in the stage image views of the main window."""

import math
from collections.abc import Callable

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.image_views.deskew import DeskewImageView
from scantailor.app.ui.image_views.fix_orientation import FixOrientationImageView
from scantailor.app.ui.image_views.output import OutputImageView, ZoneKind
from scantailor.app.ui.image_views.page_layout import PageLayoutImageView
from scantailor.app.ui.image_views.page_split import PageSplitImageView
from scantailor.app.ui.image_views.select_content import SelectContentImageView
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.image_io import load_image
from scantailor.core.models import OrthogonalRotation, PageId
from scantailor.core.page_geometry import build_transformation
from scantailor.core.stage_sequence import FilterStage
from scantailor.core.zones import PictureLayer
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

RENDER_TIMEOUT_MS = 60_000
PAGE_WIDTH = 600
PAGE_HEIGHT = 800
MM_PER_INCH = 25.4
CONTENT_BOX = ContentBox(x=100, y=200, width=300, height=400)


def _source_dpi(window: MainWindowManager) -> float:
    _pixels, metadata = load_image(window.current_page.image_id)
    return float(metadata.dpi.horizontal)


def _mm_per_pixel(window: MainWindowManager) -> float:
    """Millimetres per pixel of the (square-pixel) transformed page."""
    return MM_PER_INCH / _source_dpi(window)


def _switch_stage(window: MainWindowManager, stage: FilterStage) -> None:
    getattr(window, f"actionSwitchFilter{int(stage) + 1}").trigger()


def _pin_geometry(window: MainWindowManager, page: PageId, angle: float = 0.0) -> None:
    """Fix the earlier stages so that view coordinates equal source coordinates."""
    stages = window.stages
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, angle)
    stages.select_content_filter.set_manual_content(page, CONTENT_BOX)


def _open_stage(
    qtbot: QtBot,
    window: MainWindowManager,
    stage: FilterStage,
    ready: Callable[[ImageViewBase], bool] = lambda _view: True,
) -> ImageViewBase:
    """Switch to a stage and wait until its view shows the page's geometry."""
    with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
        _switch_stage(window, stage)
    view = window._panels.view
    qtbot.waitUntil(
        lambda: view.image is not None and bool(ready(view)),
        timeout=RENDER_TIMEOUT_MS,
    )
    return view


def _point(point: QtCore.QPointF) -> QtCore.QPoint:
    return QtCore.QPoint(round(point.x()), round(point.y()))


def _drag(
    qtbot: QtBot,
    view: ImageViewBase,
    start: QtCore.QPointF,
    end: QtCore.QPointF,
    steps: int = 4,
    modifier: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> QtCore.QPoint:
    """Drag between two widget points and return the whole pixels travelled."""
    viewport = view.viewport()
    qtbot.mousePress(
        viewport, QtCore.Qt.MouseButton.LeftButton, modifier, pos=_point(start)
    )
    for step in range(1, steps + 1):
        qtbot.mouseMove(viewport, _point(start + (end - start) * (step / steps)))
    qtbot.mouseRelease(
        viewport, QtCore.Qt.MouseButton.LeftButton, modifier, pos=_point(end)
    )
    return _point(end) - _point(start)


def _drag_and_rerender(
    qtbot: QtBot,
    window: MainWindowManager,
    view: ImageViewBase,
    start: QtCore.QPointF,
    end: QtCore.QPointF,
    modifier: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> QtCore.QPoint:
    """Drag, then wait for the re-render that the edit asks for."""
    with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
        return _drag(qtbot, view, start, end, modifier=modifier)


def _wheel(view: ImageViewBase, at: QtCore.QPointF, delta: int) -> None:
    viewport = view.viewport()
    event = QtGui.QWheelEvent(
        at,
        viewport.mapToGlobal(at),
        QtCore.QPoint(),
        QtCore.QPoint(0, delta),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(viewport, event)


def _spin_box(
    window: MainWindowManager, name: str | None = None
) -> QtWidgets.QDoubleSpinBox:
    container = window.ui.findChild(QtWidgets.QWidget, "filterOptions")
    assert container is not None
    spin_box = (
        container.findChild(QtWidgets.QDoubleSpinBox, name)
        if name
        else container.findChild(QtWidgets.QDoubleSpinBox)
    )
    assert spin_box is not None
    return spin_box


# Fix Orientation


def test_fix_orientation_view_shows_the_unrotated_source_under_the_stored_rotation(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    project_window.stages.fix_orientation_filter.set_rotation(
        page.image_id, OrthogonalRotation(degrees=90)
    )
    project_window.renderer.wait_for_done()
    project_window._show_current_page()
    view = project_window._panels.view
    assert isinstance(view, FixOrientationImageView)

    qtbot.waitUntil(
        lambda: view.image is not None and view.rotation.degrees == 90,
        timeout=RENDER_TIMEOUT_MS,
    )

    assert (view.image.width(), view.image.height()) == (PAGE_WIDTH, PAGE_HEIGHT)
    area = view.virtual_display_area
    assert (area.width(), area.height()) == (PAGE_HEIGHT, PAGE_WIDTH)
    project_window.renderer.wait_for_done()


def test_fix_orientation_key_stores_the_rotation_and_marks_the_project_modified(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = project_window._panels.view
    qtbot.waitUntil(lambda: view.image is not None, timeout=RENDER_TIMEOUT_MS)
    project_window.modified = False

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_R)

    rotation = project_window.stages.fix_orientation_filter.get_rotation(page.image_id)
    assert rotation.degrees == 90
    assert view.rotation.degrees == 90
    assert project_window.modified


def test_fix_orientation_edit_in_the_view_updates_the_options_widget(
    qtbot: QtBot, project_window: MainWindowManager
):
    view = project_window._panels.view
    qtbot.waitUntil(lambda: view.image is not None, timeout=RENDER_TIMEOUT_MS)
    options = project_window._panels.options

    qtbot.keyClick(view.viewport(), QtCore.Qt.Key.Key_R)

    assert options.get_current_rotation().degrees == 90
    project_window.renderer.wait_for_done()


# Page Split


def _open_split_view(
    qtbot: QtBot, window: MainWindowManager, split_x: float = 300.0
) -> PageSplitImageView:
    window.stages.page_split_filter.set_manual_split(
        window.current_page, split_x, PAGE_WIDTH, PAGE_HEIGHT
    )
    view = _open_stage(
        qtbot,
        window,
        FilterStage.PAGE_SPLIT,
        lambda v: v.page_layout is not None and v.line_handle_positions(),
    )
    assert isinstance(view, PageSplitImageView)
    return view


def test_dragging_a_split_line_stores_a_manual_layout_in_source_pixels(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_split_view(qtbot, project_window)
    top_handle, _bottom = view.line_handle_positions()[0]
    zoom = view.zoom_level

    travel = _drag_and_rerender(
        qtbot, project_window, view, top_handle, top_handle + QtCore.QPointF(-30, 12)
    )

    params = project_window.stages.page_split_filter.get_params(page)
    assert params.is_manual()
    assert params.page_layout.get_split_line_x() == pytest.approx(
        300 + travel.x() / zoom, abs=1e-6
    )
    assert params.page_layout.outline == (0, 0, PAGE_WIDTH, PAGE_HEIGHT)
    assert project_window.modified


def test_a_dragged_split_line_stays_where_it_was_dropped_after_the_rerender(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_split_view(qtbot, project_window)
    top_handle, _bottom = view.line_handle_positions()[0]

    _drag_and_rerender(
        qtbot, project_window, view, top_handle, top_handle + QtCore.QPointF(40, 0)
    )
    project_window.renderer.wait_for_done()
    qtbot.wait(50)

    stored = project_window.stages.page_split_filter.get_params(page).page_layout
    assert view.page_layout.get_split_line_x() == pytest.approx(
        stored.get_split_line_x()
    )


# Deskew


def _handle_target(view: DeskewImageView, degrees: float) -> QtCore.QPointF:
    arc = view.rotation_arc_square()
    right = view.rotation_handles()[1]
    return QtCore.QPointF(
        right.x(), arc.center().y() + arc.width() / 2 * math.sin(math.radians(degrees))
    )


def _angle_at_y(view: DeskewImageView, y: float) -> float:
    arc = view.rotation_arc_square()
    return math.degrees(math.asin((y - arc.center().y()) / (arc.width() / 2)))


def test_dragging_a_deskew_handle_stores_a_manual_angle(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    _pin_geometry(project_window, page, angle=0.0)
    view = _open_stage(qtbot, project_window, FilterStage.DESKEW)
    assert isinstance(view, DeskewImageView)
    right = view.rotation_handles()[1]
    target = _handle_target(view, 2.0)

    travel = _drag_and_rerender(qtbot, project_window, view, right, target)

    expected = _angle_at_y(view, right.y() + travel.y())

    params = project_window.stages.deskew_filter.get_params(page)
    assert params.is_manual()
    assert params.deskew_angle_deg == pytest.approx(expected, abs=1e-6)
    assert expected == pytest.approx(2.0, abs=0.5)
    assert project_window.modified


def test_a_deskew_drag_updates_the_options_widget_spin_box(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    _pin_geometry(project_window, page, angle=0.0)
    view = _open_stage(qtbot, project_window, FilterStage.DESKEW)
    right = view.rotation_handles()[1]

    _drag_and_rerender(qtbot, project_window, view, right, _handle_target(view, 3.0))

    stored = project_window.stages.deskew_filter.get_params(page).deskew_angle_deg
    assert stored != 0
    assert _spin_box(project_window).value() == pytest.approx(stored, abs=0.01)


def test_the_deskew_view_follows_the_options_spin_box(
    qtbot: QtBot, project_window: MainWindowManager
):
    _pin_geometry(project_window, project_window.current_page, angle=0.0)
    view = _open_stage(qtbot, project_window, FilterStage.DESKEW)

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        _spin_box(project_window).setValue(2.5)

    assert view.deskew_angle == pytest.approx(2.5)


# Select Content


def _open_content_view(
    qtbot: QtBot, window: MainWindowManager, angle: float = 0.0
) -> SelectContentImageView:
    _pin_geometry(window, window.current_page, angle=angle)
    view = _open_stage(
        qtbot,
        window,
        FilterStage.SELECT_CONTENT,
        lambda v: not v.content_rect.isEmpty(),
    )
    assert isinstance(view, SelectContentImageView)
    return view


def test_dragging_a_content_corner_stores_a_manual_box_in_source_pixels(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_content_view(qtbot, project_window)
    zoom = view.zoom_level
    corner = view.virtual_to_widget_point(QtCore.QPointF(400, 600))

    travel = _drag_and_rerender(
        qtbot, project_window, view, corner, corner + QtCore.QPointF(18, -24)
    )

    params = project_window.stages.select_content_filter.get_params(page)
    box = params.content_box
    assert params.is_content_manual()
    assert (box.x, box.y) == pytest.approx((100, 200), abs=1e-6)
    assert box.width == pytest.approx(300 + travel.x() / zoom, abs=1e-6)
    assert box.height == pytest.approx(400 + travel.y() / zoom, abs=1e-6)
    assert project_window.modified


def test_the_content_box_of_a_deskewed_page_is_stored_in_deskewed_coordinates(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_content_view(qtbot, project_window, angle=3.0)
    zoom = view.zoom_level
    edge = view.virtual_to_widget_point(QtCore.QPointF(100, 400))

    travel = _drag_and_rerender(
        qtbot, project_window, view, edge, edge + QtCore.QPointF(-20, 0)
    )

    box = project_window.stages.select_content_filter.get_params(page).content_box
    assert box.x == pytest.approx(100 + travel.x() / zoom, abs=1e-6)
    assert box.right == pytest.approx(400, abs=1e-6)


def test_removing_the_content_box_in_the_view_stores_an_empty_manual_box(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_content_view(qtbot, project_window)

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        view._remove_content_box()

    params = project_window.stages.select_content_filter.get_params(page)
    assert params.content_box.is_empty()
    assert params.is_content_manual()


# Page Layout


def _open_layout_view(qtbot: QtBot, window: MainWindowManager) -> PageLayoutImageView:
    _pin_geometry(window, window.current_page)
    view = _open_stage(
        qtbot,
        window,
        FilterStage.PAGE_LAYOUT,
        lambda v: v.margins_mm().left > 0,
    )
    assert isinstance(view, PageLayoutImageView)
    return view


def test_dragging_a_margin_stores_manual_margins_in_millimetres(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_layout_view(qtbot, project_window)
    default = project_window.stages.page_layout_filter.get_margins(page)
    mm_per_px = _mm_per_pixel(project_window)
    left_edge = view.virtual_to_widget_point(
        QtCore.QPointF(CONTENT_BOX.x - default.left / mm_per_px, 400)
    )
    zoom = view.zoom_level

    travel = _drag_and_rerender(
        qtbot, project_window, view, left_edge, left_edge + QtCore.QPointF(-25, 0)
    )

    params = project_window.stages.page_layout_filter.get_params(page)
    assert params.hard_margins_mm.left == pytest.approx(
        default.left - travel.x() / zoom * mm_per_px, abs=1e-6
    )
    assert params.hard_margins_mm.right == pytest.approx(default.right, abs=0.01)
    assert params.hard_margins_mm.top == pytest.approx(default.top, abs=0.01)
    assert not params.auto_margins
    assert project_window.modified


def test_a_margin_drag_updates_the_margin_spin_boxes(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_layout_view(qtbot, project_window)
    default = project_window.stages.page_layout_filter.get_margins(page)
    top_edge = view.virtual_to_widget_point(
        QtCore.QPointF(250, CONTENT_BOX.y - default.top / _mm_per_pixel(project_window))
    )

    _drag_and_rerender(
        qtbot, project_window, view, top_edge, top_edge + QtCore.QPointF(0, -20)
    )

    stored = project_window.stages.page_layout_filter.get_margins(page).top
    assert stored > default.top
    assert _spin_box(project_window, "topMarginSpinBox").value() == pytest.approx(
        stored, abs=0.05
    )


def test_the_page_layout_view_shows_the_stored_margins_around_the_content_box(
    qtbot: QtBot, project_window: MainWindowManager
):
    margins = project_window.stages.page_layout_filter.get_margins(
        project_window.current_page
    )

    view = _open_layout_view(qtbot, project_window)

    shown = view.margins_mm()
    assert shown.left == pytest.approx(margins.left, abs=1e-6)
    assert shown.top == pytest.approx(margins.top, abs=1e-6)
    assert shown.right == pytest.approx(margins.right, abs=1e-6)
    assert shown.bottom == pytest.approx(margins.bottom, abs=1e-6)


# Output zones


def _draw_picture_zone(
    qtbot: QtBot, view: OutputImageView, corners: list[tuple[float, float]]
) -> None:
    view.start_zone_creation(ZoneKind.PICTURE)
    viewport = view.viewport()
    for x, y in corners:
        position = view.virtual_to_widget_point(QtCore.QPointF(x, y))
        qtbot.mouseClick(
            viewport, QtCore.Qt.MouseButton.LeftButton, pos=_point(position)
        )
    last = view.virtual_to_widget_point(QtCore.QPointF(*corners[-1]))
    qtbot.mouseDClick(viewport, QtCore.Qt.MouseButton.LeftButton, pos=_point(last))


def _pin_all_pages(window: MainWindowManager) -> None:
    """Select content on every page, so that the Output stage may render."""
    for page in window.pages():
        _pin_geometry(window, page)


def _open_output_view(qtbot: QtBot, window: MainWindowManager) -> OutputImageView:
    _pin_all_pages(window)
    view = _open_stage(
        qtbot,
        window,
        FilterStage.OUTPUT,
        lambda _view: window._panels._output_transformation is not None,
    )
    assert isinstance(view, OutputImageView)
    return view


def test_a_drawn_zone_is_stored_in_source_pixels(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_output_view(qtbot, project_window)
    xform = build_transformation(
        project_window.stages,
        page,
        (PAGE_WIDTH, PAGE_HEIGHT),
        _source_dpi(project_window),
        FilterStage.PAGE_LAYOUT,
    )
    scale = view.image.width() / xform.resulting_rect.width
    drawn = [(60.0, 80.0), (260.0, 80.0), (260.0, 300.0)]
    # With no rotation or split, the transformation only shifts the page by the
    # position of its top-left corner in the source.
    origin_x, origin_y = xform.transform_point_back(0, 0)

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        _draw_picture_zone(qtbot, view, drawn)

    params = project_window.stages.output_filter.get_params(page)
    (zone,) = params.picture_zones
    np.testing.assert_allclose(
        zone.points,
        [[x / scale + origin_x, y / scale + origin_y] for x, y in drawn],
        atol=2,
    )
    assert zone.properties.layer == PictureLayer.PAINTER2
    assert params.fill_zones == ()
    assert project_window.modified


def test_stored_zones_stay_on_their_source_pixels_when_the_margins_change(
    qtbot: QtBot, project_window: MainWindowManager
):
    page = project_window.current_page
    view = _open_output_view(qtbot, project_window)
    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        _draw_picture_zone(qtbot, view, [(60, 80), (260, 80), (260, 300)])
    project_window.renderer.wait_for_done()
    output_filter = project_window.stages.output_filter
    stored_before = output_filter.get_params(page).picture_zones[0].points
    before = view.zones()[0].polygon.at(0).x()

    layout_filter = project_window.stages.page_layout_filter
    margins = layout_filter.get_margins(page)
    layout_filter.set_margins(
        page, margins.model_copy(update={"left": margins.left + 5})
    )
    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        project_window._show_current_page()
    qtbot.waitUntil(
        lambda: bool(view.zones()) and view.zones()[0].polygon.at(0).x() != before,
        timeout=RENDER_TIMEOUT_MS,
    )

    shift = view.zones()[0].polygon.at(0).x() - before
    scale = (
        view.image.width()
        / build_transformation(
            project_window.stages,
            page,
            (PAGE_WIDTH, PAGE_HEIGHT),
            _source_dpi(project_window),
            FilterStage.PAGE_LAYOUT,
        ).resulting_rect.width
    )
    assert shift == pytest.approx(5 / _mm_per_pixel(project_window) * scale, abs=2)
    assert output_filter.get_params(page).picture_zones[0].points == stored_before


# Zoom and page switches


@pytest.mark.parametrize("stage", list(FilterStage))
def test_zooming_changes_the_view_transform_but_not_the_stored_parameters(
    qtbot: QtBot, project_window: MainWindowManager, stage: FilterStage
):
    page = project_window.current_page
    stages = project_window.stages
    stages.page_split_filter.set_manual_split(page, 300.0, PAGE_WIDTH, PAGE_HEIGHT)
    # A stored two-page layout turns the image into two pages; follow the first.
    project_window._adopt_stored_page_counts([page.image_id])
    page = project_window.current_page
    _pin_all_pages(project_window)
    if stage == FilterStage.FIX_ORIENTATION:
        view = project_window._panels.view
        qtbot.waitUntil(lambda: view.image is not None, timeout=RENDER_TIMEOUT_MS)
    else:
        view = _open_stage(qtbot, project_window, stage)
    project_window.renderer.wait_for_done()
    qtbot.wait(100)

    def snapshot() -> tuple[object, ...]:
        return (
            stages.fix_orientation_filter.get_rotation(page.image_id),
            stages.page_split_filter.get_params(page),
            stages.deskew_filter.get_params(page),
            stages.select_content_filter.get_params(page),
            stages.page_layout_filter.get_params(page),
        )

    params_before = snapshot()
    transform_before = view.virtual_to_widget
    zoom_before = view.zoom_level
    project_window.modified = False

    _wheel(view, QtCore.QPointF(150, 150), delta=240)

    assert view.zoom_level > zoom_before
    assert view.virtual_to_widget != transform_before
    assert snapshot() == params_before
    assert not project_window.modified


def test_zoom_survives_the_rerender_that_follows_an_edit(
    qtbot: QtBot, project_window: MainWindowManager
):
    view = _open_content_view(qtbot, project_window)
    view.set_zoom(1.0)
    zoomed = view.zoom_level
    inside = view.virtual_to_widget_point(QtCore.QPointF(250, 400))

    _drag_and_rerender(
        qtbot,
        project_window,
        view,
        inside,
        inside + QtCore.QPointF(10, 10),
        modifier=QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    project_window.renderer.wait_for_done()
    qtbot.wait(50)

    assert view.zoom_level == pytest.approx(zoomed)


def test_content_box_follows_the_page_after_a_page_switch(
    qtbot: QtBot, project_window: MainWindowManager
):
    first, second = project_window.pages()[:2]
    for page, box in (
        (first, ContentBox(x=100, y=200, width=300, height=400)),
        (second, ContentBox(x=50, y=60, width=200, height=250)),
    ):
        stages = project_window.stages
        stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
        stages.deskew_filter.set_manual_angle(page, 0.0)
        stages.select_content_filter.set_manual_content(page, box)
    view = _open_stage(
        qtbot,
        project_window,
        FilterStage.SELECT_CONTENT,
        lambda v: not v.content_rect.isEmpty(),
    )
    assert view.content_rect == QtCore.QRectF(100, 200, 300, 400)

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        project_window.goNextPage()

    qtbot.waitUntil(
        lambda: view.content_rect == QtCore.QRectF(50, 60, 200, 250),
        timeout=RENDER_TIMEOUT_MS,
    )


def test_the_previous_pages_content_box_is_cleared_while_the_next_page_loads(
    qtbot: QtBot, project_window: MainWindowManager
):
    view = _open_content_view(qtbot, project_window)

    project_window.goNextPage()

    assert view.content_rect.isEmpty()
    project_window.renderer.wait_for_done()


def test_split_lines_follow_the_page_after_a_page_switch(
    qtbot: QtBot, project_window: MainWindowManager
):
    first, second = project_window.pages()[:2]
    split_filter = project_window.stages.page_split_filter
    split_filter.set_manual_split(first, 200.0, PAGE_WIDTH, PAGE_HEIGHT)
    split_filter.set_manual_split(second, 420.0, PAGE_WIDTH, PAGE_HEIGHT)
    project_window._adopt_stored_page_counts([first.image_id, second.image_id])
    other_image_page = next(
        page for page in project_window.pages() if page.image_id == second.image_id
    )
    view = _open_stage(
        qtbot,
        project_window,
        FilterStage.PAGE_SPLIT,
        lambda v: v.page_layout is not None,
    )
    assert view.page_layout.get_split_line_x() == 200

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        project_window.page_selected(other_image_page)

    qtbot.waitUntil(
        lambda: view.page_layout is not None
        and view.page_layout.get_split_line_x() == 420,
        timeout=RENDER_TIMEOUT_MS,
    )


def test_page_layout_rectangles_follow_the_page_after_a_page_switch(
    qtbot: QtBot, project_window: MainWindowManager
):
    first, second = project_window.pages()[:2]
    for page in (first, second):
        _pin_geometry(project_window, page)
    layout_filter = project_window.stages.page_layout_filter
    layout_filter.set_margins(
        second, layout_filter.get_margins(first).model_copy(update={"left": 30.0})
    )
    view = _open_stage(
        qtbot,
        project_window,
        FilterStage.PAGE_LAYOUT,
        lambda v: v.margins_mm().left > 0,
    )
    assert view.margins_mm().left == pytest.approx(10.0)

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        project_window.goNextPage()

    qtbot.waitUntil(
        lambda: view.margins_mm().left == pytest.approx(30.0),
        timeout=RENDER_TIMEOUT_MS,
    )
