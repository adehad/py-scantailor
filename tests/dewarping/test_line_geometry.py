"""Line helpers with the conventions of Qt's QLineF that the tracers rely on."""

import pytest

from scantailor.dewarping.line_geometry import (
    grid_line_cells,
    line_bounded_by_rect,
    line_intersection,
    line_intersection_scalars,
    normal_vector,
    point_at,
    projection_dist,
    projection_point,
    projection_scalar,
    q_round,
    sides_of_line,
)


class TestRounding:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, 1), (1.5, 2), (2.4999, 2), (-0.5, 0), (-1.5, -1), (-1.51, -2), (0.0, 0)],
    )
    def test_rounds_half_up_like_qround(self, value: float, expected: int) -> None:
        assert q_round(value) == expected


class TestNormalAndSides:
    def test_normal_of_a_rightward_line_points_up_the_screen(self) -> None:
        """QLineF turns (dx, dy) into (dy, -dx)."""
        assert normal_vector(((0.0, 0.0), (10.0, 0.0))) == (0.0, -10.0)

    def test_points_on_one_side_give_a_positive_product(self) -> None:
        line = ((0.0, 0.0), (0.0, 10.0))

        assert sides_of_line(line, (1.0, 2.0), (5.0, 9.0)) > 0

    def test_points_on_opposite_sides_give_a_negative_product(self) -> None:
        line = ((0.0, 0.0), (0.0, 10.0))

        assert sides_of_line(line, (-1.0, 2.0), (5.0, 9.0)) < 0

    def test_a_point_on_the_line_gives_zero(self) -> None:
        line = ((0.0, 0.0), (0.0, 10.0))

        assert sides_of_line(line, (0.0, 3.0), (5.0, 9.0)) == 0


class TestProjection:
    def test_scalar_is_zero_at_the_start_and_one_at_the_end(self) -> None:
        line = ((2.0, 2.0), (6.0, 2.0))

        assert projection_scalar(line, (2.0, 9.0)) == 0.0
        assert projection_scalar(line, (6.0, -4.0)) == 1.0

    def test_point_is_the_foot_of_the_perpendicular(self) -> None:
        assert projection_point(((0.0, 0.0), (10.0, 0.0)), (3.0, 4.0)) == pytest.approx(
            (3.0, 0.0)
        )

    def test_distance_is_the_length_of_the_perpendicular(self) -> None:
        assert projection_dist(((0.0, 0.0), (10.0, 0.0)), (3.0, 4.0)) == pytest.approx(
            4.0
        )

    def test_a_degenerate_line_projects_everything_to_its_start(self) -> None:
        line = ((5.0, 5.0), (5.0, 5.0))

        assert projection_scalar(line, (9.0, 9.0)) == 0.0
        assert projection_point(line, (9.0, 9.0)) == (5.0, 5.0)


class TestIntersections:
    def test_crossing_of_two_lines_beyond_their_segments_is_found(self) -> None:
        crossing = line_intersection(
            ((0.0, 0.0), (1.0, 0.0)), ((5.0, -3.0), (5.0, -1.0))
        )

        assert crossing == pytest.approx((5.0, 0.0))

    def test_parallel_lines_do_not_cross(self) -> None:
        assert (
            line_intersection(((0.0, 0.0), (1.0, 0.0)), ((0.0, 1.0), (1.0, 1.0)))
            is None
        )

    def test_scalars_locate_the_crossing_on_both_lines(self) -> None:
        line1 = ((0.0, 0.0), (10.0, 0.0))
        line2 = ((4.0, -2.0), (4.0, 2.0))

        assert line_intersection_scalars(line1, line2) == pytest.approx((0.4, 0.5))

    def test_scalars_of_parallel_lines_are_none(self) -> None:
        assert line_intersection_scalars(((0, 0), (1, 1)), ((0, 1), (1, 2))) is None

    def test_point_at_interpolates_along_the_line(self) -> None:
        assert point_at(((0.0, 0.0), (10.0, 4.0)), 0.25) == (2.5, 1.0)


class TestClippingToARectangle:
    RECT = (0.0, 0.0, 100.0, 50.0)

    def test_a_long_vertical_line_is_cut_to_the_rectangle(self) -> None:
        clipped = line_bounded_by_rect(((40.0, -100.0), (40.0, 300.0)), self.RECT)

        assert clipped is not None
        assert clipped[0] == pytest.approx((40.0, 0.0))
        assert clipped[1] == pytest.approx((40.0, 50.0))

    def test_the_direction_of_the_line_is_kept(self) -> None:
        clipped = line_bounded_by_rect(((40.0, 300.0), (40.0, -100.0)), self.RECT)

        assert clipped is not None
        assert clipped[0][1] > clipped[1][1]

    def test_a_line_pointing_away_from_the_rectangle_is_still_clipped(
        self,
    ) -> None:
        clipped = line_bounded_by_rect(((40.0, 100.0), (40.0, 150.0)), self.RECT)

        assert clipped is not None
        assert clipped[0] == pytest.approx((40.0, 0.0))
        assert clipped[1] == pytest.approx((40.0, 50.0))

    def test_a_line_that_misses_the_rectangle_is_not_clipped(self) -> None:
        assert line_bounded_by_rect(((500.0, 0.0), (500.0, 50.0)), self.RECT) is None

    def test_a_line_parallel_to_two_sides_still_meets_the_others(self) -> None:
        clipped = line_bounded_by_rect(((-10.0, 25.0), (-5.0, 25.0)), self.RECT)

        assert clipped is not None
        assert clipped[0] == pytest.approx((0.0, 25.0))
        assert clipped[1] == pytest.approx((100.0, 25.0))


class TestGridTraversal:
    def test_horizontal_line_visits_every_column_once(self) -> None:
        assert grid_line_cells(((2.0, 5.0), (6.0, 5.0))) == [
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
        ]

    def test_steep_line_steps_along_its_longer_axis(self) -> None:
        cells = grid_line_cells(((0.0, 0.0), (2.0, 8.0)))

        assert [y for _, y in cells] == list(range(9))
        assert cells[0] == (0, 0)
        assert cells[-1] == (2, 8)

    def test_a_single_point_is_one_cell(self) -> None:
        assert grid_line_cells(((3.2, 4.7), (3.2, 4.7))) == [(3, 5)]

    def test_end_points_are_rounded_to_the_nearest_cell(self) -> None:
        cells = grid_line_cells(((0.4, 0.0), (4.6, 0.0)))

        assert cells[0] == (0, 0)
        assert cells[-1] == (5, 0)
