"""Tests for polyline intersector."""

import numpy as np
import pytest

from scantailor.dewarping import PolylineIntersector, project_point_to_line


class TestPolylineIntersector:
    """Tests for PolylineIntersector class."""

    def test_simple_horizontal_polyline(self):
        """Intersection with horizontal polyline."""
        polyline = np.array([[0, 10], [100, 10]], dtype=np.float64)
        intersector = PolylineIntersector(polyline)

        # Vertical line through midpoint
        intersection = intersector.intersect(np.array([50, 0]), np.array([50, 100]))
        assert intersection[0] == pytest.approx(50.0)
        assert intersection[1] == pytest.approx(10.0)

    def test_simple_vertical_polyline(self):
        """Intersection with vertical polyline."""
        polyline = np.array([[10, 0], [10, 100]], dtype=np.float64)
        intersector = PolylineIntersector(polyline)

        # Horizontal line through midpoint
        intersection = intersector.intersect(np.array([0, 50]), np.array([100, 50]))
        assert intersection[0] == pytest.approx(10.0)
        assert intersection[1] == pytest.approx(50.0)

    def test_diagonal_polyline(self):
        """Intersection with diagonal polyline."""
        polyline = np.array([[0, 0], [100, 100]], dtype=np.float64)
        intersector = PolylineIntersector(polyline)

        # Horizontal line at y=50
        intersection = intersector.intersect(np.array([0, 50]), np.array([100, 50]))
        assert intersection[0] == pytest.approx(50.0)
        assert intersection[1] == pytest.approx(50.0)

    def test_multi_segment_polyline(self):
        """Intersection with multi-segment polyline."""
        polyline = np.array(
            [
                [0, 10],
                [50, 20],
                [100, 10],
            ],
            dtype=np.float64,
        )
        intersector = PolylineIntersector(polyline)

        # Vertical line at x=25 should intersect first segment
        intersection = intersector.intersect(np.array([25, 0]), np.array([25, 100]))
        assert intersection[0] == pytest.approx(25.0)
        # y should be interpolated between 10 and 20
        assert 10 < intersection[1] < 20

    def test_line_misses_polyline(self):
        """Line that doesn't intersect returns nearest endpoint projection."""
        polyline = np.array([[0, 0], [10, 0]], dtype=np.float64)
        intersector = PolylineIntersector(polyline)

        # Line far above the polyline
        intersection = intersector.intersect(np.array([0, 100]), np.array([10, 100]))
        # Should project onto the line at y=100
        assert intersection[1] == pytest.approx(100.0)

    def test_sequential_queries_optimization(self):
        """Sequential queries use optimization hint."""
        polyline = np.array(
            [
                [0, 0],
                [10, 0],
                [20, 0],
                [30, 0],
                [40, 0],
            ],
            dtype=np.float64,
        )
        intersector = PolylineIntersector(polyline)

        # Sequential vertical lines
        for x in [5, 15, 25, 35]:
            intersection = intersector.intersect(np.array([x, -10]), np.array([x, 10]))
            assert intersection[0] == pytest.approx(float(x))
            assert intersection[1] == pytest.approx(0.0)

    def test_invalid_polyline_too_short(self):
        """Polyline with fewer than 2 points raises error."""
        with pytest.raises(ValueError, match="at least 2"):
            PolylineIntersector(np.array([[0, 0]]))

    def test_invalid_polyline_wrong_shape(self):
        """Polyline with wrong shape raises error."""
        with pytest.raises(ValueError, match="shape"):
            PolylineIntersector(np.array([0, 0, 10, 0]))

    def test_num_segments(self):
        """Number of segments property."""
        polyline = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float64)
        intersector = PolylineIntersector(polyline)
        assert intersector.num_segments == 2


class TestProjectPointToLine:
    """Tests for project_point_to_line function."""

    def test_project_point_onto_horizontal_line(self):
        """Project point onto horizontal line."""
        point = np.array([5, 10])
        line_p1 = np.array([0, 0])
        line_p2 = np.array([10, 0])

        projection, scalar = project_point_to_line(point, line_p1, line_p2)

        assert projection[0] == pytest.approx(5.0)
        assert projection[1] == pytest.approx(0.0)
        assert scalar == pytest.approx(0.5)

    def test_project_point_onto_vertical_line(self):
        """Project point onto vertical line."""
        point = np.array([10, 5])
        line_p1 = np.array([0, 0])
        line_p2 = np.array([0, 10])

        projection, scalar = project_point_to_line(point, line_p1, line_p2)

        assert projection[0] == pytest.approx(0.0)
        assert projection[1] == pytest.approx(5.0)
        assert scalar == pytest.approx(0.5)

    def test_project_point_onto_diagonal_line(self):
        """Project point onto diagonal line."""
        point = np.array([10, 0])
        line_p1 = np.array([0, 0])
        line_p2 = np.array([10, 10])

        projection, scalar = project_point_to_line(point, line_p1, line_p2)

        # Projection onto y=x line from (10,0) is (5,5)
        assert projection[0] == pytest.approx(5.0)
        assert projection[1] == pytest.approx(5.0)
        assert scalar == pytest.approx(0.5)

    def test_scalar_at_endpoints(self):
        """Scalar is 0 at p1 and 1 at p2."""
        line_p1 = np.array([0, 0])
        line_p2 = np.array([10, 0])

        _, scalar1 = project_point_to_line(line_p1, line_p1, line_p2)
        assert scalar1 == pytest.approx(0.0)

        _, scalar2 = project_point_to_line(line_p2, line_p1, line_p2)
        assert scalar2 == pytest.approx(1.0)

    def test_scalar_beyond_segment(self):
        """Scalar can be < 0 or > 1 for points beyond segment."""
        line_p1 = np.array([0, 0])
        line_p2 = np.array([10, 0])

        # Point before p1
        _, scalar1 = project_point_to_line(np.array([-5, 0]), line_p1, line_p2)
        assert scalar1 == pytest.approx(-0.5)

        # Point after p2
        _, scalar2 = project_point_to_line(np.array([15, 0]), line_p1, line_p2)
        assert scalar2 == pytest.approx(1.5)
