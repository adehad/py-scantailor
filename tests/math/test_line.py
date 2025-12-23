"""Tests for line and line segment utilities."""

from __future__ import annotations

import numpy as np
import pytest

from scantailor.math import (
    Line2D,
    LineSegment,
    point_to_line_distance,
    point_to_segment_distance,
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
