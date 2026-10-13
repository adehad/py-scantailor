"""The dewarping tab of the Output view: model grid and manual control points."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import DewarpingImageView
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.output.params import DistortionModelSpec

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
LEFT, RIGHT, TOP, BOTTOM = 100.0, 500.0, 150.0, 650.0

CURVED = DistortionModelSpec(
    top=((LEFT, TOP), (300.0, TOP - 30.0), (RIGHT, TOP)),
    bottom=((LEFT, BOTTOM), (300.0, BOTTOM + 30.0), (RIGHT, BOTTOM)),
)
STRAIGHT = DistortionModelSpec()


def _point(point: QtCore.QPointF) -> QtCore.QPoint:
    return QtCore.QPoint(round(point.x()), round(point.y()))


@pytest.fixture
def view(qtbot: QtBot) -> DewarpingImageView:
    widget = DewarpingImageView()
    qtbot.addWidget(widget)
    widget.resize(QtCore.QSize(600, 800))
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8),
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    return widget


def _widget_pos(view: DewarpingImageView, x: float, y: float) -> QtCore.QPointF:
    return view.image_to_widget().map(QtCore.QPointF(x, y))


def _drag(
    qtbot: QtBot, view: DewarpingImageView, start: QtCore.QPointF, end: QtCore.QPointF
) -> None:
    viewport = view.viewport()
    qtbot.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=_point(start))
    for step in range(1, 5):
        qtbot.mouseMove(viewport, _point(start + (end - start) * (step / 4)))
    qtbot.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=_point(end))


class TestShowingAModel:
    def test_a_defined_model_gives_its_points_as_control_points(
        self, view: DewarpingImageView
    ):
        view.set_distortion_model(CURVED, 2.0)

        top, bottom = view.control_points()
        assert [(p.x(), p.y()) for p in top] == list(CURVED.top)
        assert [(p.x(), p.y()) for p in bottom] == list(CURVED.bottom)

    def test_an_undefined_model_starts_as_straight_edges_along_the_default_area(
        self, view: DewarpingImageView
    ):
        area = QtCore.QRectF(LEFT, TOP, RIGHT - LEFT, BOTTOM - TOP)

        view.set_distortion_model(STRAIGHT, 2.0, area)

        top, bottom = view.control_points()
        assert [(p.x(), p.y()) for p in top] == [(LEFT, TOP), (RIGHT, TOP)]
        assert [(p.x(), p.y()) for p in bottom] == [(LEFT, BOTTOM), (RIGHT, BOTTOM)]

    def test_without_a_default_area_the_edges_follow_the_image(
        self, view: DewarpingImageView
    ):
        view.set_distortion_model(STRAIGHT, 2.0)

        top, bottom = view.control_points()
        assert (top[0].x(), top[0].y()) == (0, 0)
        assert (bottom[1].x(), bottom[1].y()) == (IMAGE_WIDTH, IMAGE_HEIGHT)

    def test_a_long_polyline_is_reduced_to_a_few_control_points(
        self, view: DewarpingImageView
    ):
        xs = np.linspace(LEFT, RIGHT, 40)
        model = DistortionModelSpec(
            top=tuple((float(x), TOP) for x in xs),
            bottom=tuple((float(x), BOTTOM) for x in xs),
        )

        view.set_distortion_model(model, 2.0)

        top, _ = view.control_points()
        assert len(top) == DewarpingImageView.MAX_CONTROL_POINTS
        assert (top[0].x(), top[-1].x()) == (LEFT, RIGHT)

    def test_showing_a_model_does_not_report_a_change(
        self, qtbot: QtBot, view: DewarpingImageView
    ):
        with qtbot.assertNotEmitted(view.distortion_model_changed):
            view.set_distortion_model(CURVED, 2.0)

    def test_sampled_edges_pass_through_the_control_points(
        self, view: DewarpingImageView
    ):
        view.set_distortion_model(CURVED, 2.0)

        edge = np.asarray(view.distortion_model().top)
        distances = np.hypot(edge[:, 0] - 300.0, edge[:, 1] - (TOP - 30.0))
        assert distances.min() < 2.0

    def test_the_grid_is_painted_for_a_valid_model(self, view: DewarpingImageView):
        view.set_distortion_model(CURVED, 2.0)

        assert view.grid() is not None
        assert view.grid().shape == (
            DewarpingImageView.GRID_LINES,
            DewarpingImageView.GRID_LINES,
            2,
        )
        image = view.grab().toImage()
        painted = any(
            QtGui.QColor(image.pixel(x, y)).blue() > 200
            and QtGui.QColor(image.pixel(x, y)).red() < 100
            for x in range(0, image.width(), 3)
            for y in range(0, image.height(), 3)
        )
        assert painted

    def test_changing_depth_perception_changes_the_grid(self, view: DewarpingImageView):
        view.set_distortion_model(CURVED, 1.0)
        before = view.grab().toImage()

        view.set_depth_perception(3.0)

        assert view.grab().toImage() != before


class TestManualEditing:
    def test_dragging_a_control_point_moves_it_and_stores_the_model(
        self, qtbot: QtBot, view: DewarpingImageView
    ):
        view.set_distortion_model(CURVED, 2.0)
        start = _widget_pos(view, 300.0, TOP - 30.0)
        end = _widget_pos(view, 300.0, TOP + 40.0)

        with qtbot.waitSignal(view.distortion_model_changed) as emitted:
            _drag(qtbot, view, start, end)

        top, _ = view.control_points()
        assert top[1].y() == pytest.approx(TOP + 40.0, abs=1.5)
        assert top[1].x() == pytest.approx(300.0, abs=1.5)
        model = emitted.args[0]
        assert isinstance(model, DistortionModelSpec)
        assert model.is_defined()
        assert model != CURVED
        edge = np.asarray(model.top)
        assert np.hypot(edge[:, 0] - top[1].x(), edge[:, 1] - top[1].y()).min() < 2.0

    def test_dragging_an_end_point_moves_the_page_corner(
        self, qtbot: QtBot, view: DewarpingImageView
    ):
        view.set_distortion_model(CURVED, 2.0)

        with qtbot.waitSignal(view.distortion_model_changed) as emitted:
            _drag(
                qtbot,
                view,
                _widget_pos(view, LEFT, TOP),
                _widget_pos(view, LEFT - 20.0, TOP - 10.0),
            )

        first = emitted.args[0].top[0]
        assert first == pytest.approx((LEFT - 20.0, TOP - 10.0), abs=1.5)

    def test_pressing_on_an_edge_adds_a_control_point_there(
        self, qtbot: QtBot, view: DewarpingImageView
    ):
        view.set_distortion_model(STRAIGHT, 2.0, QtCore.QRectF(LEFT, TOP, 400, 500))
        on_edge = _widget_pos(view, 200.0, TOP)

        with qtbot.waitSignal(view.distortion_model_changed):
            _drag(qtbot, view, on_edge, _widget_pos(view, 200.0, TOP + 30.0))

        top, _ = view.control_points()
        assert len(top) == 3
        assert top[1].x() == pytest.approx(200.0, abs=1.5)
        assert top[1].y() == pytest.approx(TOP + 30.0, abs=1.5)

    def test_the_remembered_model_is_kept_when_the_stored_one_comes_back(
        self, qtbot: QtBot, view: DewarpingImageView
    ):
        view.set_distortion_model(CURVED, 2.0)
        _drag(
            qtbot,
            view,
            _widget_pos(view, 300.0, TOP - 30.0),
            _widget_pos(view, 300.0, TOP + 40.0),
        )
        before = view.control_points()
        stored = view.distortion_model()

        view.set_distortion_model(stored, 2.0)

        after = view.control_points()
        assert [(p.x(), p.y()) for p in after[0]] == [(p.x(), p.y()) for p in before[0]]

    def test_pressing_away_from_the_edges_reports_nothing(
        self, qtbot: QtBot, view: DewarpingImageView
    ):
        view.set_distortion_model(CURVED, 2.0)

        with qtbot.assertNotEmitted(view.distortion_model_changed):
            _drag(
                qtbot,
                view,
                _widget_pos(view, 300.0, 400.0),
                _widget_pos(view, 320.0, 420.0),
            )

    def test_pressing_a_control_point_without_moving_reports_nothing(
        self, qtbot: QtBot, view: DewarpingImageView
    ):
        view.set_distortion_model(CURVED, 2.0)
        point = _widget_pos(view, 300.0, TOP - 30.0)

        with qtbot.assertNotEmitted(view.distortion_model_changed):
            viewport = view.viewport()
            qtbot.mousePress(
                viewport, QtCore.Qt.MouseButton.LeftButton, pos=_point(point)
            )
            qtbot.mouseRelease(
                viewport, QtCore.Qt.MouseButton.LeftButton, pos=_point(point)
            )

    def test_d_removes_the_inner_control_point_under_the_cursor(
        self, qtbot: QtBot, view: DewarpingImageView
    ):
        view.set_distortion_model(CURVED, 2.0)
        viewport = view.viewport()
        qtbot.mouseMove(viewport, _point(_widget_pos(view, 300.0, TOP - 30.0)))

        with qtbot.waitSignal(view.distortion_model_changed):
            qtbot.keyClick(view, QtCore.Qt.Key.Key_D)

        top, bottom = view.control_points()
        assert len(top) == 2
        assert len(bottom) == 3

    def test_d_keeps_the_end_points(self, qtbot: QtBot, view: DewarpingImageView):
        view.set_distortion_model(CURVED, 2.0)
        viewport = view.viewport()
        qtbot.mouseMove(viewport, _point(_widget_pos(view, LEFT, TOP)))

        with qtbot.assertNotEmitted(view.distortion_model_changed):
            qtbot.keyClick(view, QtCore.Qt.Key.Key_D)

        assert len(view.control_points()[0]) == 3
