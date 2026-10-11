"""Choosing the page's top and bottom curves from traced text lines."""

import numpy as np
import pytest

from scantailor.dewarping.distortion_model_builder import build_distortion_model
from scantailor.dewarping.text_lines import TextLine, VerticalBounds, trace_text_lines
from tests.dewarping.synthetic import (
    PAGE_DPI,
    PAGE_HEIGHT,
    TEXT_LEFT,
    curved_text_page,
    straight_text_page,
)

LEFT, RIGHT = 100.0, 1100.0


def _bounds() -> VerticalBounds:
    return VerticalBounds(
        left_top=(LEFT, 0.0),
        left_bottom=(LEFT, 1000.0),
        right_top=(RIGHT, 0.0),
        right_bottom=(RIGHT, 1000.0),
    )


def _line(y_of_x, y0: float, xs=None) -> TextLine:
    xs = np.linspace(LEFT + 20, RIGHT - 20, 40) if xs is None else xs
    return TextLine(points=[(float(x), float(y_of_x(x) + y0)) for x in xs])


def _sagging(x: float) -> float:
    return 40.0 * np.sin(np.pi * (x - LEFT) / (RIGHT - LEFT))


class TestBuildDistortionModel:
    """Models built from synthetic lines."""

    def test_curves_follow_the_traced_lines_out_to_the_bounds(self) -> None:
        lines = [_line(_sagging, y0) for y0 in (100.0, 200.0, 300.0, 400.0, 500.0)]

        model = build_distortion_model(lines, _bounds())

        assert model is not None
        top, bottom = model.top_curve.polyline, model.bottom_curve.polyline
        assert top[0, 0] == pytest.approx(LEFT, abs=1.0)
        assert top[-1, 0] == pytest.approx(RIGHT, abs=1.0)
        middle = top[np.argmin(np.abs(top[:, 0] - 600.0))]
        # 40 px of sag at the middle of the page, relative to the ends.
        assert middle[1] - top[0, 1] > 30.0
        assert top[:, 1].mean() < bottom[:, 1].mean()

    def test_top_curve_is_above_the_bottom_curve(self) -> None:
        lines = [_line(_sagging, y0) for y0 in (500.0, 100.0, 300.0)]

        model = build_distortion_model(lines, _bounds())

        assert model is not None
        assert (
            model.top_curve.polyline[:, 1].max()
            < model.bottom_curve.polyline[:, 1].min()
        )

    def test_extension_past_the_data_is_gentle(self) -> None:
        """Lines that stop short of the bounds are extended along their tangent."""
        xs = np.linspace(300.0, 900.0, 30)
        lines = [_line(_sagging, y0, xs) for y0 in (100.0, 300.0, 500.0)]

        model = build_distortion_model(lines, _bounds())

        assert model is not None
        top = model.top_curve.polyline
        assert top[0, 0] == pytest.approx(LEFT, abs=1.0)
        assert np.all(np.abs(top[:, 1] - 100.0) < 100.0)

    def test_fewer_than_two_lines_give_no_model(self) -> None:
        assert build_distortion_model([_line(_sagging, 100.0)], _bounds()) is None
        assert build_distortion_model([], _bounds()) is None

    def test_degenerate_bound_gives_no_model(self) -> None:
        bounds = VerticalBounds(
            left_top=(LEFT, 0.0),
            left_bottom=(LEFT, 0.0),
            right_top=(RIGHT, 0.0),
            right_bottom=(RIGHT, 1000.0),
        )
        lines = [_line(_sagging, y0) for y0 in (100.0, 300.0)]

        assert build_distortion_model(lines, bounds) is None

    def test_lines_with_too_few_points_are_ignored(self) -> None:
        lines = [TextLine(points=[(200.0, 100.0), (300.0, 100.0)])] * 3

        assert build_distortion_model(lines, _bounds()) is None

    def test_result_is_repeatable(self) -> None:
        lines = [_line(_sagging, y0) for y0 in np.arange(100.0, 900.0, 40.0)]

        first = build_distortion_model(lines, _bounds())
        second = build_distortion_model(lines, _bounds())

        assert first is not None and second is not None
        np.testing.assert_array_equal(
            first.top_curve.polyline, second.top_curve.polyline
        )
        np.testing.assert_array_equal(
            first.bottom_curve.polyline, second.bottom_curve.polyline
        )


class TestOnRenderedPages:
    """Text lines traced on rendered pages."""

    def test_curved_page_gives_a_curved_model(self) -> None:
        traced = trace_text_lines(curved_text_page(bow=60.0), float(PAGE_DPI))

        model = build_distortion_model(traced.lines, traced.vertical_bounds)

        assert model is not None
        top = model.top_curve.polyline
        sag = top[len(top) // 2, 1] - 0.5 * (top[0, 1] + top[-1, 1])
        assert sag == pytest.approx(60.0, abs=12.0)

    def test_straight_page_gives_a_flat_model(self) -> None:
        traced = trace_text_lines(straight_text_page(), float(PAGE_DPI))

        model = build_distortion_model(traced.lines, traced.vertical_bounds)

        assert model is not None
        for curve in (model.top_curve, model.bottom_curve):
            assert np.ptp(curve.polyline[:, 1]) < 5.0
        assert model.top_curve.polyline[0, 0] == pytest.approx(TEXT_LEFT, abs=15.0)
        assert model.bottom_curve.polyline[:, 1].max() < PAGE_HEIGHT
