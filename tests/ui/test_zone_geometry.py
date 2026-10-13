"""Hit testing and vertex editing helpers of the zone views."""

import math

import pytest
from PySide6.QtCore import QPointF

from scantailor.app.ui.image_views import zone_geometry as geometry

SQUARE = [QPointF(0, 0), QPointF(100, 0), QPointF(100, 100), QPointF(0, 100)]


class TestNearestOnSegment:
    def test_a_point_beside_the_middle_projects_onto_it(self):
        nearest, gap = geometry.nearest_on_segment(
            QPointF(50, 7), QPointF(0, 0), QPointF(100, 0)
        )

        assert (nearest.x(), nearest.y()) == (50, 0)
        assert gap == pytest.approx(7)

    def test_a_point_past_an_end_gives_that_end(self):
        nearest, gap = geometry.nearest_on_segment(
            QPointF(130, 40), QPointF(0, 0), QPointF(100, 0)
        )

        assert (nearest.x(), nearest.y()) == (100, 0)
        assert gap == pytest.approx(math.hypot(30, 40))

    def test_a_zero_length_segment_gives_its_point(self):
        nearest, gap = geometry.nearest_on_segment(
            QPointF(3, 4), QPointF(0, 0), QPointF(0, 0)
        )

        assert (nearest.x(), nearest.y()) == (0, 0)
        assert gap == pytest.approx(5)


class TestVertexNear:
    def test_picks_the_closest_vertex_within_the_radius(self):
        assert geometry.vertex_near(SQUARE, QPointF(97, 4), 8) == 1

    def test_returns_minus_one_outside_the_radius(self):
        assert geometry.vertex_near(SQUARE, QPointF(50, 50), 8) == -1

    def test_a_vertex_exactly_at_the_radius_still_hits(self):
        assert geometry.vertex_near(SQUARE, QPointF(8, 0), 8) == 0


class TestSegmentNear:
    def test_finds_the_edge_and_the_point_on_it(self):
        index, point = geometry.segment_near(SQUARE, QPointF(50, 4), 8) or (-1, None)

        assert index == 0
        assert point is not None
        assert (point.x(), point.y()) == (50, 0)

    def test_the_closing_edge_runs_from_the_last_vertex_to_the_first(self):
        index, point = geometry.segment_near(SQUARE, QPointF(3, 50), 8) or (-1, None)

        assert index == 3
        assert point is not None
        assert (point.x(), point.y()) == (0, 50)

    def test_none_when_no_edge_is_near(self):
        assert geometry.segment_near(SQUARE, QPointF(50, 50), 8) is None


class TestInsertAndRemove:
    def test_insert_puts_the_vertex_after_the_edge_start(self):
        corners = [QPointF(p) for p in SQUARE]

        index = geometry.insert_vertex(corners, 0, QPointF(50, 0))

        assert index == 1
        assert [(p.x(), p.y()) for p in corners][:3] == [(0, 0), (50, 0), (100, 0)]
        assert len(corners) == 5

    def test_insert_on_the_closing_edge_appends(self):
        corners = [QPointF(p) for p in SQUARE]

        index = geometry.insert_vertex(corners, 3, QPointF(0, 50))

        assert index == 4
        assert (corners[-1].x(), corners[-1].y()) == (0, 50)

    def test_insert_copies_the_point(self):
        corners = [QPointF(p) for p in SQUARE]
        point = QPointF(50, 0)

        geometry.insert_vertex(corners, 0, point)
        point.setX(999)

        assert corners[1].x() == 50

    def test_remove_drops_a_vertex_of_a_quadrilateral(self):
        corners = [QPointF(p) for p in SQUARE]

        assert geometry.remove_vertex(corners, 2)

        assert len(corners) == 3

    def test_remove_keeps_a_triangle(self):
        corners = [QPointF(p) for p in SQUARE[:3]]

        assert not geometry.remove_vertex(corners, 0)

        assert len(corners) == 3

    @pytest.mark.parametrize(("count", "allowed"), [(3, False), (4, True), (9, True)])
    def test_can_remove_needs_more_than_three_vertices(self, count: int, allowed: bool):
        assert geometry.can_remove_vertex(count) is allowed


class TestMergeTarget:
    def test_a_vertex_on_a_neighbour_merges_with_it(self):
        corners = [QPointF(p) for p in SQUARE]
        corners[1] = QPointF(2, 1)

        assert geometry.merge_target(corners, 1, 8) == 0

    def test_the_nearer_neighbour_wins(self):
        corners = [QPointF(0, 0), QPointF(6, 0), QPointF(9, 0), QPointF(50, 50)]

        assert geometry.merge_target(corners, 1, 8) == 2

    def test_a_vertex_far_from_both_neighbours_does_not_merge(self):
        assert geometry.merge_target(SQUARE, 1, 8) is None

    def test_a_triangle_never_merges(self):
        corners = [QPointF(0, 0), QPointF(1, 0), QPointF(50, 50)]

        assert geometry.merge_target(corners, 1, 8) is None

    def test_the_first_vertex_looks_back_to_the_last(self):
        corners = [QPointF(1, 1), QPointF(100, 0), QPointF(100, 100), QPointF(0, 0)]

        assert geometry.merge_target(corners, 0, 8) == 3


class TestRightAngle:
    def test_a_vertex_between_a_flat_and_a_steep_edge_squares_both(self):
        current = QPointF(100, 100)
        previous = QPointF(0, 90)  # nearly horizontal edge
        following = QPointF(95, 200)  # nearly vertical edge

        before, after = geometry.right_angle_neighbours(current, previous, following)

        assert before.y() == current.y()
        assert after.x() == current.x()

    def test_the_other_way_round_squares_the_other_way(self):
        current = QPointF(100, 100)
        previous = QPointF(95, 0)
        following = QPointF(200, 110)

        before, after = geometry.right_angle_neighbours(current, previous, following)

        assert before.x() == current.x()
        assert after.y() == current.y()

    def test_the_corner_ends_up_axis_aligned(self):
        current = QPointF(30, 40)

        before, after = geometry.right_angle_neighbours(
            current, QPointF(10, 5), QPointF(80, 70)
        )

        assert (before.x() == current.x() and after.y() == current.y()) or (
            before.y() == current.y() and after.x() == current.x()
        )

    def test_three_coinciding_points_stay_as_they_are(self):
        point = QPointF(5, 5)

        before, after = geometry.right_angle_neighbours(point, point, point)

        assert before == point
        assert after == point

    def test_the_inputs_are_not_changed(self):
        current, previous, following = QPointF(30, 40), QPointF(10, 5), QPointF(80, 70)

        geometry.right_angle_neighbours(current, previous, following)

        assert (previous.x(), previous.y()) == (10, 5)
        assert (following.x(), following.y()) == (80, 70)
