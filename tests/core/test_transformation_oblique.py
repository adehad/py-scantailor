"""Tests for the oblique (shear) step of ImageTransformation."""

import math

import numpy as np
import pytest

from scantailor.core.models import Dpi, OrthogonalRotation
from scantailor.core.transformation import ImageTransformation, render_transformed

WIDTH = 400
HEIGHT = 600
DPI = Dpi(horizontal=300, vertical=300)


def _transformation() -> ImageTransformation:
    return ImageTransformation.from_size(WIDTH, HEIGHT, DPI)


def _shear_factor(degrees: float) -> float:
    """Horizontal shift per pixel going down; the angle counts half."""
    return math.tan(math.radians(degrees) * 0.5)


class TestObliqueState:
    """Where the oblique angle lives among the steps."""

    def test_defaults_to_no_oblique(self):
        assert _transformation().post_oblique == 0.0

    def test_set_post_oblique_is_readable(self):
        xform = _transformation()

        xform.set_post_oblique(2.5)

        assert xform.post_oblique == 2.5

    def test_zero_oblique_leaves_the_transform_alone(self):
        xform = _transformation()
        before = xform.transform

        xform.set_post_oblique(3.0)
        xform.set_post_oblique(0.0)

        assert np.allclose(xform.transform, before)

    def test_rotation_and_oblique_are_independent_settings(self):
        xform = _transformation()

        xform.set_post_oblique(2.0)
        xform.set_post_rotation(1.5)

        assert xform.post_oblique == 2.0
        assert xform.post_rotation == 1.5

    def test_changing_the_pre_crop_resets_the_oblique(self):
        xform = _transformation()
        xform.set_post_oblique(2.0)

        xform.set_pre_crop_area(
            np.array([[10, 10], [200, 10], [200, 300], [10, 300]], dtype=np.float64)
        )

        assert xform.post_oblique == 0.0

    def test_changing_the_pre_rotation_resets_the_oblique(self):
        xform = _transformation()
        xform.set_post_oblique(2.0)

        xform.set_pre_rotation(OrthogonalRotation(degrees=90))

        assert xform.post_oblique == 0.0

    def test_rescaling_the_dpi_keeps_the_oblique(self):
        xform = _transformation()
        xform.set_post_oblique(2.0)

        xform.pre_scale_to_dpi(Dpi(horizontal=150, vertical=150))

        assert xform.post_oblique == 2.0
        assert xform.resulting_rect.height == pytest.approx(HEIGHT / 2)
        expected_width = WIDTH / 2 + _shear_factor(2.0) * HEIGHT / 2
        assert xform.resulting_rect.width == pytest.approx(expected_width)


class TestObliqueGeometry:
    """The shear is x' = x - tan(angle / 2) * y, applied after the rotation."""

    @pytest.mark.parametrize("degrees", [-5.0, -2.0, 1.0, 5.0])
    def test_points_slide_sideways_in_proportion_to_their_height(self, degrees: float):
        xform = _transformation()
        xform.set_post_oblique(degrees)

        x0, y0 = xform.transform_point(100, 100)
        x1, y1 = xform.transform_point(100, 400)

        assert y1 - y0 == pytest.approx(300)
        assert x1 - x0 == pytest.approx(-_shear_factor(degrees) * 300)

    def test_horizontal_distances_are_kept(self):
        xform = _transformation()
        xform.set_post_oblique(4.0)

        left = xform.transform_point(50, 200)
        right = xform.transform_point(350, 200)

        assert right[0] - left[0] == pytest.approx(300)
        assert right[1] == pytest.approx(left[1])

    @pytest.mark.parametrize("degrees", [-4.0, 3.0])
    def test_the_top_left_of_the_bounds_stays_at_the_origin(self, degrees: float):
        xform = _transformation()
        xform.set_post_oblique(degrees)

        corners = xform.map_polygon(
            np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype=float)
        )

        assert corners[:, 0].min() == pytest.approx(0.0, abs=1e-9)
        assert corners[:, 1].min() == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("degrees", [-5.0, 3.0])
    def test_the_canvas_widens_by_the_shear_over_the_height(self, degrees: float):
        xform = _transformation()

        xform.set_post_oblique(degrees)

        rect = xform.resulting_rect
        assert rect.width == pytest.approx(WIDTH + abs(_shear_factor(degrees)) * HEIGHT)
        assert rect.height == pytest.approx(HEIGHT)

    def test_points_round_trip_through_the_transformation(self):
        xform = _transformation()
        xform.set_post_rotation(2.0)
        xform.set_post_oblique(-3.0)

        for point in ((0.0, 0.0), (123.0, 456.0), (399.0, 599.0)):
            mapped = xform.transform_point(*point)
            assert xform.transform_point_back(*mapped) == pytest.approx(point)

    def test_polygons_round_trip_through_the_transformation(self):
        xform = _transformation()
        xform.set_post_oblique(4.0)
        polygon = np.array([[10, 20], [300, 40], [280, 500]], dtype=np.float64)

        back = xform.map_polygon_back(xform.map_polygon(polygon))

        assert np.allclose(back, polygon)

    def test_shear_comes_after_rotation(self):
        rotation, oblique = 6.0, 4.0
        xform = _transformation()
        xform.set_post_rotation(rotation)
        xform.set_post_oblique(oblique)
        reference = _transformation()
        reference.set_post_rotation(rotation)

        # A displacement between two points ignores the translation part, so it
        # follows the linear map: first rotate, then shear.
        a, b = (50.0, 80.0), (320.0, 410.0)
        rotated_a = reference.transform_point(*a)
        rotated_b = reference.transform_point(*b)
        dx = rotated_b[0] - rotated_a[0]
        dy = rotated_b[1] - rotated_a[1]
        expected = (dx - _shear_factor(oblique) * dy, dy)

        mapped_a = xform.transform_point(*a)
        mapped_b = xform.transform_point(*b)

        assert (mapped_b[0] - mapped_a[0], mapped_b[1] - mapped_a[1]) == pytest.approx(
            expected
        )

    def test_the_shear_is_taken_about_the_pre_cropped_area(self):
        xform = _transformation()
        xform.set_pre_crop_area(
            np.array([[100, 100], [300, 100], [300, 500], [100, 500]], dtype=float)
        )
        xform.set_post_oblique(4.0)

        rect = xform.resulting_rect

        assert rect.height == pytest.approx(400)
        assert rect.width == pytest.approx(200 + abs(_shear_factor(4.0)) * 400)


class TestObliqueRendering:
    """The image follows the same shear as the points."""

    @staticmethod
    def _column_of_line(row: np.ndarray) -> float:
        return float(np.flatnonzero(row < 128).mean())

    def test_a_vertical_line_leans_by_the_shear(self):
        source = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        source[:, 200:203] = 0
        xform = _transformation()
        xform.set_post_oblique(4.0)

        result = render_transformed(source, xform)

        top, bottom = 50, HEIGHT - 50
        lean = self._column_of_line(result[bottom]) - self._column_of_line(result[top])
        assert lean == pytest.approx(-_shear_factor(4.0) * (bottom - top), abs=1.0)

    def test_a_marked_pixel_lands_where_the_points_say(self):
        source = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        source[300:304, 150:154] = 0
        xform = _transformation()
        xform.set_post_rotation(1.0)
        xform.set_post_oblique(-3.0)

        result = render_transformed(source, xform)

        rect = xform.resulting_rect
        expected_x, expected_y = xform.transform_point(152.0, 302.0)
        ys, xs = np.nonzero(result < 128)
        assert xs.mean() + 0.5 == pytest.approx(expected_x - rect.x, abs=1.0)
        assert ys.mean() + 0.5 == pytest.approx(expected_y - rect.y, abs=1.0)

    def test_rendering_gives_the_widened_canvas(self):
        source = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        xform = _transformation()
        xform.set_post_oblique(5.0)

        result = render_transformed(source, xform)

        assert result.shape == (
            HEIGHT,
            round(WIDTH + _shear_factor(5.0) * HEIGHT),
        )
