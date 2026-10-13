"""Choosing the page's top and bottom curves from traced curves.

Reference values marked "C++" come from compiling the upstream sources of
``DistortionModelBuilder`` and the cylindrical surface dewarper with g++ 12 and
running them on the same curves.
"""

from typing import ClassVar

import numpy as np
import pytest

from scantailor.dewarping.distortion_model_builder import (
    DistortionModelBuilder,
    _centroid,
    _maybe_trim_polyline,
    _straightness_error,
    build_distortion_model,
)
from scantailor.dewarping.text_line_tracer import TextLineResult, VerticalBounds

LEFT = ((100.0, 0.0), (104.0, 1000.0))
RIGHT = ((900.0, 0.0), (897.0, 1000.0))


def _scene(kind: int) -> list[np.ndarray]:
    curves = []
    for k in range(8):
        y0 = 150 + 110 * k
        xs = np.linspace(110 + 3 * k, 890 - 4 * k, 25 + k)
        wave = np.sin(np.pi * (xs - 100) / 800)
        if kind == 0:
            ys = y0 + 40 * wave * (1 - 0.05 * k)
        elif kind == 1:
            ys = y0 + (-30 + 10 * k) * wave + 0.02 * (xs - 500)
        else:
            ys = y0 + 55 * wave - (y0 - 500) * 0.12 * wave
        points = np.column_stack([xs, ys])
        curves.append(points[::-1] if k % 3 == 1 else points)
    return curves


def _build(curves: list[np.ndarray], bounds=(LEFT, RIGHT)):
    builder = DistortionModelBuilder((0.0, 1.0))
    builder.set_vertical_bounds(*bounds)
    for curve in curves:
        builder.add_horizontal_curve(curve)
    return builder.try_build_model()


class TestAgainstCpp:
    EXPECTED: ClassVar[dict] = {
        0: (
            15,
            15,
            ([100.60019, 150.04719], [500.07179, 189.98817], [899.54993, 150.02314]),
            ([103.68142, 920.35383], [500.44913, 946.009], [897.23921, 920.26479]),
        ),
        1: (
            15,
            15,
            ([100.56792, 141.97895], [500.66482, 120.02231], [899.52609, 157.97147]),
            ([103.20966, 802.41414], [500.26109, 840.01387], [897.54536, 818.2131]),
        ),
        2: (
            29,
            7,
            ([100.60051, 150.12739], [499.88692, 246.9654], [899.54979, 150.06945]),
            ([103.68025, 920.06198], [501.18162, 924.60189], [897.23986, 920.0459]),
        ),
    }

    @pytest.mark.parametrize("kind", [0, 1, 2])
    def test_model_curves_match_the_cpp_builder(self, kind: int) -> None:
        top_len, bottom_len, top_pts, bottom_pts = self.EXPECTED[kind]

        model = _build(_scene(kind))

        assert model.is_valid()
        assert model.top_curve is not None and model.bottom_curve is not None
        top = model.top_curve.polyline
        bottom = model.bottom_curve.polyline
        assert (len(top), len(bottom)) == (top_len, bottom_len)
        for actual, expected in (
            (top, top_pts),
            (bottom, bottom_pts),
        ):
            mid = len(actual) // 2
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-3)
            np.testing.assert_allclose(actual[mid], expected[1], atol=1e-3)
            np.testing.assert_allclose(actual[-1], expected[2], atol=1e-3)


class TestBuilding:
    def test_curves_may_be_added_in_either_direction(self) -> None:
        curves = _scene(0)
        reversed_curves = [curve[::-1] for curve in curves]

        forward = _build(curves)
        backward = _build(reversed_curves)

        assert forward.top_curve is not None and backward.top_curve is not None
        np.testing.assert_allclose(
            forward.top_curve.polyline, backward.top_curve.polyline, atol=1e-9
        )

    def test_the_curves_of_a_model_run_left_to_right_top_above_bottom(self) -> None:
        model = _build(_scene(1))

        assert model.top_curve is not None and model.bottom_curve is not None
        assert model.top_curve.polyline[0, 0] < model.top_curve.polyline[-1, 0]
        assert (
            model.top_curve.polyline[:, 1].mean()
            < model.bottom_curve.polyline[:, 1].mean()
        )

    def test_the_curves_reach_the_bounds(self) -> None:
        model = _build(_scene(0))

        assert model.top_curve is not None
        top = model.top_curve.polyline
        assert top[0, 0] == pytest.approx(100.0 + 0.004 * top[0, 1], abs=1e-3)
        assert top[-1, 0] == pytest.approx(900.0 - 0.003 * top[-1, 1], abs=1e-3)

    def test_two_curves_are_enough_for_a_model(self) -> None:
        model = _build(_scene(0)[:2])

        assert model.is_valid()

    def test_fewer_than_two_curves_give_an_invalid_model(self) -> None:
        assert not _build(_scene(0)[:1]).is_valid()
        assert not _build([]).is_valid()

    def test_a_degenerate_bound_gives_an_invalid_model(self) -> None:
        point = ((100.0, 0.0), (100.0, 0.0))

        assert not _build(_scene(0), (point, RIGHT)).is_valid()
        assert not _build(_scene(0), (LEFT, point)).is_valid()

    def test_curves_with_one_point_are_ignored(self) -> None:
        builder = DistortionModelBuilder((0.0, 1.0))
        builder.set_vertical_bounds(LEFT, RIGHT)
        for point in ([[300.0, 300.0]], [[500.0, 400.0]]):
            builder.add_horizontal_curve(np.array(point))

        assert not builder.try_build_model().is_valid()

    def test_the_bounds_are_returned_as_they_were_set(self) -> None:
        builder = DistortionModelBuilder((0.0, 1.0))
        builder.set_vertical_bounds(RIGHT, LEFT)

        assert builder.vertical_bounds() == (RIGHT, LEFT)

    def test_the_result_is_repeatable(self) -> None:
        first = _build(_scene(2))
        second = _build(_scene(2))

        assert first.top_curve is not None and second.top_curve is not None
        assert first.bottom_curve is not None and second.bottom_curve is not None
        np.testing.assert_array_equal(
            first.top_curve.polyline, second.top_curve.polyline
        )
        np.testing.assert_array_equal(
            first.bottom_curve.polyline, second.bottom_curve.polyline
        )

    def test_a_transformation_moves_curves_bounds_and_directions_together(self) -> None:
        plain = _build(_scene(0))
        builder = DistortionModelBuilder((0.0, 1.0))
        builder.set_vertical_bounds(LEFT, RIGHT)
        for curve in _scene(0):
            builder.add_horizontal_curve(curve)
        scale = np.array([[2.0, 0.0, 30.0], [0.0, 2.0, -40.0], [0.0, 0.0, 1.0]])

        builder.transform(scale)
        moved = builder.try_build_model()

        assert plain.top_curve is not None and moved.top_curve is not None
        before = plain.top_curve.polyline
        after = moved.top_curve.polyline
        for index in (0, -1):
            np.testing.assert_allclose(
                after[index], before[index] * 2.0 + np.array([30.0, -40.0]), atol=0.05
            )
        middle = before[np.argmin(np.abs(before[:, 0] - 500.0))]
        assert np.interp(
            2 * middle[0] + 30.0, after[:, 0], after[:, 1]
        ) == pytest.approx(2 * middle[1] - 40.0, abs=0.1)
        assert builder.vertical_bounds()[0][0] == (230.0, -40.0)

    def test_a_flat_page_gives_straight_curves(self) -> None:
        curves = [
            np.column_stack([np.linspace(110, 890, 30), np.full(30, y)])
            for y in (200.0, 400.0, 600.0, 800.0)
        ]

        model = _build(curves)

        assert model.is_valid()
        assert model.top_curve is not None and model.bottom_curve is not None
        assert np.ptp(model.top_curve.polyline[:, 1]) < 0.01
        assert np.ptp(model.bottom_curve.polyline[:, 1]) < 0.01


class TestTrimming:
    BOUNDS = (((100.0, 0.0), (100.0, 1000.0)), ((900.0, 0.0), (900.0, 1000.0)))

    def test_points_beyond_a_bound_are_cut_and_the_end_put_on_the_bound(self) -> None:
        polyline = np.array(
            [[60.0, 300.0], [90.0, 305.0], [130.0, 310.0], [500.0, 320.0]]
        )

        trimmed = _maybe_trim_polyline(polyline, self.BOUNDS)

        assert len(trimmed) == 3
        assert trimmed[0, 0] == pytest.approx(100.0)
        assert trimmed[0, 1] == pytest.approx(306.25)
        np.testing.assert_array_equal(trimmed[1:], polyline[2:])

    def test_both_ends_are_trimmed(self) -> None:
        polyline = np.array([[60.0, 300.0], [500.0, 300.0], [960.0, 300.0]])

        trimmed = _maybe_trim_polyline(polyline, self.BOUNDS)

        assert trimmed[0, 0] == pytest.approx(100.0)
        assert trimmed[-1, 0] == pytest.approx(900.0)

    def test_a_curve_inside_the_bounds_is_left_alone(self) -> None:
        polyline = np.array([[150.0, 300.0], [500.0, 310.0], [850.0, 300.0]])

        np.testing.assert_array_equal(
            _maybe_trim_polyline(polyline, self.BOUNDS), polyline
        )

    def test_a_curve_keeps_two_points_however_far_out_it_lies(self) -> None:
        polyline = np.array([[10.0, 300.0], [20.0, 300.0], [30.0, 300.0]])

        assert len(_maybe_trim_polyline(polyline, self.BOUNDS)) >= 2


class TestScoring:
    def test_a_straight_dewarped_line_scores_zero(self) -> None:
        xs = np.linspace(0, 1, 20)

        assert _straightness_error(
            np.column_stack([xs, 0.3 + 0.1 * xs])
        ) == pytest.approx(0.0, abs=1e-12)

    def test_a_bowed_line_is_charged_its_mean_distance_from_the_best_fit_line(
        self,
    ) -> None:
        xs = np.linspace(0, 1, 21)
        points = np.column_stack([xs, 0.1 * np.sin(np.pi * xs)])

        assert 0.01 < _straightness_error(points) < 0.1

    def test_a_vertical_line_is_charged_a_fixed_penalty(self) -> None:
        ys = np.linspace(0, 1, 10)

        assert _straightness_error(np.column_stack([np.full(10, 0.5), ys])) == 1000.0

    def test_the_centroid_weights_segments_by_length(self) -> None:
        polyline = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 2.0]])

        centroid = _centroid(polyline)

        assert centroid == pytest.approx((70.0 / 12.0, 2.0 / 12.0))

    def test_the_centroid_of_a_zero_length_curve_is_its_first_point(self) -> None:
        assert _centroid(np.array([[4.0, 5.0], [4.0, 5.0]])) == (4.0, 5.0)


class TestConvenienceFunction:
    def test_it_builds_from_a_tracing_result(self) -> None:
        bounds = VerticalBounds(first=LEFT, second=RIGHT)
        result = TextLineResult(lines=_scene(0), vertical_bounds=bounds)

        model = build_distortion_model(result)

        assert model is not None and model.is_valid()

    def test_it_builds_from_curves_and_bounds(self) -> None:
        model = build_distortion_model(
            _scene(0), VerticalBounds(first=LEFT, second=RIGHT)
        )

        assert model is not None and model.is_valid()

    def test_it_gives_none_when_no_model_can_be_built(self) -> None:
        bounds = VerticalBounds(first=LEFT, second=RIGHT)

        assert build_distortion_model([], bounds) is None
        assert build_distortion_model(_scene(0), None) is None
