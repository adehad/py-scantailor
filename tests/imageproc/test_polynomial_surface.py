"""Tests for polynomial surface and polynomial line fitting."""

import numpy as np
import pytest

from scantailor.imageproc.polynomial_line import fit_polynomial_lines
from scantailor.imageproc.polynomial_surface import PolynomialSurface


class TestPolynomialSurface:
    """Least-squares surface fitting and rendering."""

    def test_recovers_a_planar_ramp(self):
        ys, xs = np.mgrid[0:60, 0:80]
        image = (40 + 1.5 * xs + 1.0 * ys).astype(np.uint8)

        rendered = PolynomialSurface.fit(image, 1, 1).render(image.shape)

        assert np.abs(rendered.astype(int) - image.astype(int)).max() <= 2

    def test_constant_image_gives_constant_render(self):
        image = np.full((20, 30), 77, dtype=np.uint8)

        rendered = PolynomialSurface.fit(image, 3, 3).render((20, 30))

        assert np.abs(rendered.astype(int) - 77).max() <= 1

    def test_mask_excludes_outliers_from_the_fit(self):
        image = np.full((40, 40), 200, dtype=np.uint8)
        image[:, :10] = 0
        mask = np.ones(image.shape, dtype=bool)
        mask[:, :10] = False

        rendered = PolynomialSurface.fit(image, 2, 2, mask).render(image.shape)

        assert abs(int(rendered[20, 30]) - 200) <= 2

    def test_render_at_other_size_keeps_the_same_function(self):
        ys, xs = np.mgrid[0:30, 0:30]
        image = (50 + 4 * xs).astype(np.uint8)
        surface = PolynomialSurface.fit(image, 1, 0)

        large = surface.render((90, 90))

        assert large.shape == (90, 90)
        assert large[0, 0] == pytest.approx(50, abs=2)
        assert large[0, -1] == pytest.approx(50 + 4 * 29, abs=3)

    def test_degrees_are_reduced_when_data_is_scarce(self):
        image = np.full((4, 4), 100, dtype=np.uint8)
        mask = np.zeros(image.shape, dtype=bool)
        mask[0, :2] = True

        surface = PolynomialSurface.fit(image, 8, 5, mask)

        assert (surface.hor_degree + 1) * (surface.vert_degree + 1) <= 2

    def test_empty_mask_gives_zero_surface(self):
        image = np.full((5, 5), 100, dtype=np.uint8)
        mask = np.zeros(image.shape, dtype=bool)

        rendered = PolynomialSurface.fit(image, 2, 2, mask).render((5, 5))

        assert not rendered.any()

    def test_fit_does_not_modify_inputs(self):
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        mask = np.ones(image.shape, dtype=bool)
        image_copy, mask_copy = image.copy(), mask.copy()

        PolynomialSurface.fit(image, 2, 2, mask)

        np.testing.assert_array_equal(image, image_copy)
        np.testing.assert_array_equal(mask, mask_copy)

    def test_render_output_is_clipped_to_uint8(self):
        bright = PolynomialSurface(coeffs=np.array([[2.0]]))
        negative = PolynomialSurface(coeffs=np.array([[-1.0]]))

        assert bright.render((3, 4)).dtype == np.uint8
        assert (bright.render((3, 4)) == 255).all()
        assert (negative.render((3, 4)) == 0).all()

    @pytest.mark.parametrize(("hor", "vert"), [(-1, 0), (0, -1)])
    def test_negative_degree_is_rejected(self, hor, vert):
        with pytest.raises(ValueError):
            PolynomialSurface.fit(np.zeros((4, 4), np.uint8), hor, vert)

    def test_mask_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            PolynomialSurface.fit(
                np.zeros((4, 4), np.uint8), 1, 1, np.ones((3, 3), bool)
            )

    def test_render_of_empty_shape_is_empty(self):
        surface = PolynomialSurface.fit(np.full((4, 4), 9, np.uint8), 1, 1)

        assert surface.render((0, 5)).shape == (0, 5)


class TestFitPolynomialLines:
    """Batched per-line polynomial fits."""

    def test_column_fit_smooths_noise_around_a_parabola(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 1, 100)
        clean = (60 + 120 * (t - 0.5) ** 2 * 4)[:, None] * np.ones((1, 8))
        noisy = np.clip(clean + rng.normal(0, 5, clean.shape), 0, 255).astype(np.uint8)

        fitted = fit_polynomial_lines(noisy, 2, axis=0)

        assert np.abs(fitted.astype(float) - clean).mean() < 3

    def test_row_fit_matches_column_fit_of_transpose(self):
        rng = np.random.default_rng(1)
        data = rng.integers(0, 256, (30, 40), dtype=np.uint8)

        by_row = fit_polynomial_lines(data, 3, axis=1)
        by_column = fit_polynomial_lines(np.ascontiguousarray(data.T), 3, axis=0)

        np.testing.assert_array_equal(by_row, by_column.T)

    def test_degree_is_reduced_for_short_lines(self):
        data = np.array([[10], [200]], dtype=np.uint8)

        fitted = fit_polynomial_lines(data, 4, axis=0)

        assert fitted.shape == (2, 1)
        assert abs(int(fitted[0, 0]) - 10) <= 1
        assert abs(int(fitted[1, 0]) - 200) <= 1

    def test_single_sample_lines_are_returned_unchanged(self):
        data = np.array([[42, 7]], dtype=np.uint8)

        np.testing.assert_array_equal(fit_polynomial_lines(data, 2, axis=0), data)

    def test_input_is_not_modified_and_dtype_is_uint8(self):
        data = np.arange(50, dtype=np.uint8).reshape(5, 10)
        original = data.copy()

        fitted = fit_polynomial_lines(data, 2, axis=1)

        np.testing.assert_array_equal(data, original)
        assert fitted.dtype == np.uint8
        assert fitted.shape == data.shape

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            fit_polynomial_lines(np.zeros((3, 3), np.uint8), -1, axis=0)
        with pytest.raises(ValueError):
            fit_polynomial_lines(np.zeros((0, 3), np.uint8), 1, axis=0)
