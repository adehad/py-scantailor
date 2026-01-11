"""Tests for polygon utilities."""

import numpy as np
import pytest

from scantailor.math import Polygon, polygon_area, polygon_contains_point


class TestPolygon:
    """Tests for Polygon class."""

    def test_create_polygon(self):
        """Create a basic polygon."""
        vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        poly = Polygon(vertices=vertices)
        assert len(poly.vertices) == 4

    def test_too_few_vertices_raises(self):
        """Less than 3 vertices should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            Polygon(vertices=np.array([[0, 0], [1, 1]]))

    def test_wrong_shape_raises(self):
        """Wrong array shape should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            Polygon(vertices=np.array([0, 1, 2, 3]))

    def test_from_rectangle(self):
        """Create rectangle polygon."""
        poly = Polygon.from_rectangle(10, 20, 100, 50)
        assert poly.unsigned_area() == 5000

    def test_from_rectangle_invalid_raises(self):
        """Zero or negative dimensions should raise."""
        with pytest.raises(ValueError, match="positive"):
            Polygon.from_rectangle(0, 0, 0, 10)
        with pytest.raises(ValueError, match="positive"):
            Polygon.from_rectangle(0, 0, 10, -5)

    def test_area_square(self):
        """Area of a unit square."""
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        poly = Polygon(vertices=vertices)
        assert pytest.approx(poly.unsigned_area(), abs=1e-6) == 1

    def test_area_rectangle(self):
        """Area of a rectangle."""
        poly = Polygon.from_rectangle(0, 0, 10, 5)
        assert pytest.approx(poly.unsigned_area(), abs=1e-6) == 50

    def test_area_triangle(self):
        """Area of a triangle."""
        vertices = np.array([[0, 0], [10, 0], [5, 10]])
        poly = Polygon(vertices=vertices)
        assert pytest.approx(poly.unsigned_area(), abs=1e-6) == 50

    def test_signed_area_ccw(self):
        """Counter-clockwise polygon has positive area."""
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        poly = Polygon(vertices=vertices)
        assert poly.area() > 0

    def test_signed_area_cw(self):
        """Clockwise polygon has negative area."""
        vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        poly = Polygon(vertices=vertices)
        assert poly.area() < 0

    def test_centroid_square(self):
        """Centroid of a square."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        c = poly.centroid()
        assert pytest.approx(c[0], abs=1e-6) == 5
        assert pytest.approx(c[1], abs=1e-6) == 5

    def test_centroid_triangle(self):
        """Centroid of a triangle."""
        vertices = np.array([[0, 0], [9, 0], [0, 9]])
        poly = Polygon(vertices=vertices)
        c = poly.centroid()
        assert pytest.approx(c[0], abs=1e-6) == 3
        assert pytest.approx(c[1], abs=1e-6) == 3

    def test_contains_point_inside(self):
        """Point inside polygon."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        assert poly.contains(np.array([5, 5]))

    def test_contains_point_outside(self):
        """Point outside polygon."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        assert not poly.contains(np.array([15, 5]))

    def test_contains_point_on_edge(self):
        """Point on polygon edge."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        assert poly.contains(np.array([5, 0]))

    def test_contains_point_on_vertex(self):
        """Point on polygon vertex."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        assert poly.contains(np.array([0, 0]))

    def test_bounding_box(self):
        """Bounding box of polygon."""
        vertices = np.array([[5, 10], [15, 20], [10, 30]])
        poly = Polygon(vertices=vertices)
        x, y, w, h = poly.bounding_box()
        assert x == 5
        assert y == 10
        assert w == 10
        assert h == 20

    def test_intersects_overlapping(self):
        """Two overlapping polygons."""
        poly1 = Polygon.from_rectangle(0, 0, 10, 10)
        poly2 = Polygon.from_rectangle(5, 5, 10, 10)
        assert poly1.intersects(poly2)
        assert poly2.intersects(poly1)

    def test_intersects_disjoint(self):
        """Two disjoint polygons."""
        poly1 = Polygon.from_rectangle(0, 0, 10, 10)
        poly2 = Polygon.from_rectangle(20, 20, 10, 10)
        assert not poly1.intersects(poly2)

    def test_intersects_contained(self):
        """One polygon contained in another."""
        poly1 = Polygon.from_rectangle(0, 0, 20, 20)
        poly2 = Polygon.from_rectangle(5, 5, 5, 5)
        assert poly1.intersects(poly2)
        assert poly2.intersects(poly1)

    def test_translate(self):
        """Translate polygon."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        translated = poly.translate(5, 3)
        x, y, w, h = translated.bounding_box()
        assert x == 5
        assert y == 3
        assert w == 10
        assert h == 10

    def test_scale_uniform(self):
        """Scale polygon uniformly."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        scaled = poly.scale(2)
        assert pytest.approx(scaled.unsigned_area(), abs=1e-6) == 400

    def test_scale_non_uniform(self):
        """Scale polygon non-uniformly."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        scaled = poly.scale(2, 3)
        _x, _y, w, h = scaled.bounding_box()
        assert pytest.approx(w, abs=1e-6) == 20
        assert pytest.approx(h, abs=1e-6) == 30

    def test_scale_around_center(self):
        """Scale polygon around custom center."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        scaled = poly.scale(2, center=np.array([5, 5]))
        x, y, w, h = scaled.bounding_box()
        # Should be centered at (5, 5) but twice as big
        assert pytest.approx(x, abs=1e-6) == -5
        assert pytest.approx(y, abs=1e-6) == -5
        assert pytest.approx(w, abs=1e-6) == 20
        assert pytest.approx(h, abs=1e-6) == 20

    def test_to_cv2_contour(self):
        """Convert to OpenCV contour format."""
        poly = Polygon.from_rectangle(0, 0, 10, 10)
        contour = poly.to_cv2_contour()
        assert contour.shape == (4, 1, 2)
        assert contour.dtype == np.int32


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_polygon_area(self):
        """polygon_area function."""
        vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert pytest.approx(polygon_area(vertices), abs=1e-6) == 100

    def test_polygon_contains_point(self):
        """polygon_contains_point function."""
        vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert polygon_contains_point(vertices, np.array([5, 5]))
        assert not polygon_contains_point(vertices, np.array([15, 5]))
