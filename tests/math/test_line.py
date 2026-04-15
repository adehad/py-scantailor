"""Tests for line and line segment utilities."""

import numpy as np
import pytest

from scantailor.math.line import (
    Line2D,
    LineSegment,
    line_bounded_by_rect,
    line_intersection_scalar,
    point_to_line_distance,
    point_to_segment_distance,
    sides_of_line,
)


class TestLine2D:
    """Tests for Line2D class."""

    def test_from_points_horizontal(self):
        """Horizontal line from two points."""
        line = Line2D.from_points(np.array([0, 5]), np.array([10, 5]))
        # Point on line should have zero distance
        assert pytest.approx(line.distance_to_point(np.array([5, 5])), abs=1e-6) == 0

    def test_from_points_vertical(self):
        """Vertical line from two points."""
        line = Line2D.from_points(np.array([3, 0]), np.array([3, 10]))
        assert pytest.approx(line.distance_to_point(np.array([3, 5])), abs=1e-6) == 0

    def test_from_points_diagonal(self):
        """Diagonal line from two points."""
        line = Line2D.from_points(np.array([0, 0]), np.array([10, 10]))
        # Origin should be on the line
        assert pytest.approx(line.distance_to_point(np.array([5, 5])), abs=1e-6) == 0

    def test_from_points_identical_raises(self):
        """Identical points should raise ValueError."""
        with pytest.raises(ValueError, match="distinct"):
            Line2D.from_points(np.array([5, 5]), np.array([5, 5]))

    def test_from_point_and_normal(self):
        """Create line from point and normal."""
        line = Line2D.from_point_and_normal(np.array([0, 0]), np.array([0, 1]))
        # Line should be horizontal (y = 0)
        assert pytest.approx(line.distance_to_point(np.array([100, 0])), abs=1e-6) == 0

    def test_from_point_and_normal_zero_raises(self):
        """Zero normal should raise ValueError."""
        with pytest.raises(ValueError, match="zero"):
            Line2D.from_point_and_normal(np.array([0, 0]), np.array([0, 0]))

    def test_distance_to_point(self):
        """Distance from point to line."""
        # Horizontal line y = 5
        line = Line2D.from_points(np.array([0, 5]), np.array([10, 5]))
        assert pytest.approx(line.distance_to_point(np.array([0, 10])), abs=1e-6) == 5
        assert pytest.approx(line.distance_to_point(np.array([0, 0])), abs=1e-6) == 5

    def test_signed_distance_to_point(self):
        """Signed distance indicates which side of line."""
        line = Line2D.from_points(np.array([0, 0]), np.array([10, 0]))
        # Points above/below should have opposite signs
        d1 = line.signed_distance_to_point(np.array([5, 5]))
        d2 = line.signed_distance_to_point(np.array([5, -5]))
        assert d1 * d2 < 0  # Opposite signs

    def test_project_point(self):
        """Project point onto line."""
        line = Line2D.from_points(np.array([0, 0]), np.array([10, 0]))
        proj = line.project_point(np.array([5, 10]))
        assert pytest.approx(proj[0], abs=1e-6) == 5
        assert pytest.approx(proj[1], abs=1e-6) == 0

    def test_normalize(self):
        """Normalized line should have unit normal."""
        line = Line2D(a=3, b=4, c=5)
        normalized = line.normalize()
        norm = np.sqrt(normalized.a**2 + normalized.b**2)
        assert pytest.approx(norm, abs=1e-6) == 1


class TestLineSegment:
    """Tests for LineSegment class."""

    def test_create_segment(self):
        """Create a line segment."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        assert seg.length() == 10

    def test_identical_points_raises(self):
        """Identical endpoints should raise ValueError."""
        with pytest.raises(ValueError, match="distinct"):
            LineSegment(p1=np.array([5, 5]), p2=np.array([5, 5]))

    def test_length(self):
        """Segment length calculation."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([3, 4]))
        assert seg.length() == 5

    def test_direction(self):
        """Direction vector should be unit length."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        d = seg.direction()
        assert pytest.approx(np.linalg.norm(d), abs=1e-6) == 1
        assert pytest.approx(d[0], abs=1e-6) == 1
        assert pytest.approx(d[1], abs=1e-6) == 0

    def test_to_line(self):
        """Convert segment to infinite line."""
        seg = LineSegment(p1=np.array([0, 5]), p2=np.array([10, 5]))
        line = seg.to_line()
        # Point on extended line should have zero distance
        assert pytest.approx(line.distance_to_point(np.array([100, 5])), abs=1e-6) == 0

    def test_point_at(self):
        """Get point at parameter t."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 10]))
        p0 = seg.point_at(0)
        p1 = seg.point_at(1)
        p_mid = seg.point_at(0.5)
        assert np.allclose(p0, [0, 0])
        assert np.allclose(p1, [10, 10])
        assert np.allclose(p_mid, [5, 5])

    def test_distance_to_point_on_segment(self):
        """Distance to point perpendicular to segment."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        # Point directly above middle of segment
        assert pytest.approx(seg.distance_to_point(np.array([5, 5])), abs=1e-6) == 5

    def test_distance_to_point_before_segment(self):
        """Distance to point before segment start."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        # Point before p1
        dist = seg.distance_to_point(np.array([-3, 4]))
        assert pytest.approx(dist, abs=1e-6) == 5  # Distance to p1

    def test_distance_to_point_after_segment(self):
        """Distance to point after segment end."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        # Point after p2
        dist = seg.distance_to_point(np.array([13, 4]))
        assert pytest.approx(dist, abs=1e-6) == 5  # Distance to p2

    def test_project_point_on_segment(self):
        """Project point onto segment (within bounds)."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        proj = seg.project_point(np.array([5, 10]))
        assert np.allclose(proj, [5, 0])

    def test_project_point_before_segment(self):
        """Project point before segment returns p1."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        proj = seg.project_point(np.array([-5, 5]))
        assert np.allclose(proj, [0, 0])

    def test_project_point_after_segment(self):
        """Project point after segment returns p2."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        proj = seg.project_point(np.array([15, 5]))
        assert np.allclose(proj, [10, 0])


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_point_to_line_distance(self):
        """point_to_line_distance function."""
        line = Line2D.from_points(np.array([0, 0]), np.array([10, 0]))
        dist = point_to_line_distance(np.array([5, 5]), line)
        assert pytest.approx(dist, abs=1e-6) == 5

    def test_point_to_segment_distance(self):
        """point_to_segment_distance function."""
        seg = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        dist = point_to_segment_distance(np.array([5, 5]), seg)
        assert pytest.approx(dist, abs=1e-6) == 5


class TestSidesOfLine:
    """Tests for sides_of_line function."""

    def test_same_side_above(self):
        """Two points on same side (above) should return positive."""
        line = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        p1 = np.array([2, 5])  # above
        p2 = np.array([8, 3])  # above
        result = sides_of_line(line, p1, p2)
        assert result > 0

    def test_same_side_below(self):
        """Two points on same side (below) should return positive."""
        line = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        p1 = np.array([2, -5])  # below
        p2 = np.array([8, -3])  # below
        result = sides_of_line(line, p1, p2)
        assert result > 0

    def test_different_sides(self):
        """Two points on different sides should return negative."""
        line = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        p1 = np.array([5, 5])  # above
        p2 = np.array([5, -5])  # below
        result = sides_of_line(line, p1, p2)
        assert result < 0

    def test_point_on_line(self):
        """Point on line should return zero."""
        line = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        p1 = np.array([5, 0])  # on line
        p2 = np.array([5, 5])  # above
        result = sides_of_line(line, p1, p2)
        assert result == 0

    def test_both_points_on_line(self):
        """Both points on line should return zero."""
        line = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        p1 = np.array([2, 0])  # on line
        p2 = np.array([8, 0])  # on line
        result = sides_of_line(line, p1, p2)
        assert result == 0

    def test_diagonal_line(self):
        """Test with diagonal line."""
        line = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 10]))
        p1 = np.array([0, 5])  # left of diagonal
        p2 = np.array([5, 0])  # right of diagonal
        result = sides_of_line(line, p1, p2)
        assert result < 0  # different sides


class TestLineIntersectionScalar:
    """Tests for line_intersection_scalar function."""

    def test_perpendicular_lines(self):
        """Two perpendicular lines should intersect."""
        line1 = LineSegment(p1=np.array([0, 5]), p2=np.array([10, 5]))  # horizontal
        line2 = LineSegment(p1=np.array([5, 0]), p2=np.array([5, 10]))  # vertical
        result = line_intersection_scalar(line1, line2)
        assert result is not None
        s1, s2 = result
        assert pytest.approx(s1, abs=1e-6) == 0.5
        assert pytest.approx(s2, abs=1e-6) == 0.5

    def test_diagonal_intersection(self):
        """Two diagonal lines crossing at origin."""
        line1 = LineSegment(p1=np.array([-5, -5]), p2=np.array([5, 5]))
        line2 = LineSegment(p1=np.array([-5, 5]), p2=np.array([5, -5]))
        result = line_intersection_scalar(line1, line2)
        assert result is not None
        s1, s2 = result
        assert pytest.approx(s1, abs=1e-6) == 0.5
        assert pytest.approx(s2, abs=1e-6) == 0.5

    def test_parallel_lines(self):
        """Parallel lines should return None."""
        line1 = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        line2 = LineSegment(p1=np.array([0, 5]), p2=np.array([10, 5]))
        result = line_intersection_scalar(line1, line2)
        assert result is None

    def test_intersection_outside_segments(self):
        """Intersection can be outside segment bounds."""
        line1 = LineSegment(p1=np.array([0, 0]), p2=np.array([1, 0]))
        line2 = LineSegment(p1=np.array([5, -1]), p2=np.array([5, 1]))
        result = line_intersection_scalar(line1, line2)
        assert result is not None
        s1, s2 = result
        # s1 should be > 1 (intersection extends past line1)
        assert s1 > 1

    def test_verify_intersection_point(self):
        """Verify both scalars give the same point."""
        line1 = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 5]))
        line2 = LineSegment(p1=np.array([0, 5]), p2=np.array([10, 0]))
        result = line_intersection_scalar(line1, line2)
        assert result is not None
        s1, s2 = result
        point1 = line1.point_at(s1)
        point2 = line2.point_at(s2)
        assert np.allclose(point1, point2)


class TestLineBoundedByRect:
    """Tests for line_bounded_by_rect function."""

    def test_horizontal_line_through_rect(self):
        """Horizontal line passing through rectangle."""
        line = LineSegment(p1=np.array([0, 5]), p2=np.array([20, 5]))
        rect = (5.0, 0.0, 10.0, 10.0)  # x=5, y=0, w=10, h=10
        result = line_bounded_by_rect(line, rect)
        assert result is not None
        # Line should be clipped to x=[5, 15]
        assert pytest.approx(result.p1[0], abs=1e-6) == 5.0
        assert pytest.approx(result.p2[0], abs=1e-6) == 15.0
        assert pytest.approx(result.p1[1], abs=1e-6) == 5.0
        assert pytest.approx(result.p2[1], abs=1e-6) == 5.0

    def test_vertical_line_through_rect(self):
        """Vertical line passing through rectangle."""
        line = LineSegment(p1=np.array([10, 0]), p2=np.array([10, 20]))
        rect = (5.0, 5.0, 10.0, 10.0)  # x=5, y=5, w=10, h=10
        result = line_bounded_by_rect(line, rect)
        assert result is not None
        # Line should be clipped to y=[5, 15]
        assert pytest.approx(result.p1[1], abs=1e-6) == 5.0
        assert pytest.approx(result.p2[1], abs=1e-6) == 15.0

    def test_diagonal_line_through_rect(self):
        """Diagonal line passing through rectangle."""
        line = LineSegment(p1=np.array([0, 0]), p2=np.array([20, 20]))
        rect = (5.0, 5.0, 10.0, 10.0)  # x=5, y=5, w=10, h=10
        result = line_bounded_by_rect(line, rect)
        assert result is not None
        # Diagonal through [5,15] x [5,15] should enter at (5,5) and exit at (15,15)
        p1, p2 = sorted([tuple(result.p1), tuple(result.p2)])
        assert pytest.approx(p1[0], abs=1e-6) == 5.0
        assert pytest.approx(p1[1], abs=1e-6) == 5.0
        assert pytest.approx(p2[0], abs=1e-6) == 15.0
        assert pytest.approx(p2[1], abs=1e-6) == 15.0

    def test_line_outside_rect(self):
        """Line completely outside rectangle should return None."""
        line = LineSegment(p1=np.array([0, 0]), p2=np.array([10, 0]))
        rect = (0.0, 5.0, 10.0, 10.0)  # y starts at 5, line is at y=0
        result = line_bounded_by_rect(line, rect)
        assert result is None

    def test_line_parallel_to_edge(self):
        """Line parallel to rect edge but inside."""
        line = LineSegment(p1=np.array([0, 7]), p2=np.array([20, 7]))
        rect = (5.0, 5.0, 10.0, 5.0)  # x=5, y=5, w=10, h=5
        result = line_bounded_by_rect(line, rect)
        assert result is not None
        assert pytest.approx(result.p1[0], abs=1e-6) == 5.0
        assert pytest.approx(result.p2[0], abs=1e-6) == 15.0
