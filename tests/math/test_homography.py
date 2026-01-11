"""Tests for homography utilities."""

import numpy as np
import pytest

from scantailor.math import Homography, warp_perspective


class TestHomography:
    """Tests for Homography class."""

    def test_identity(self):
        """Identity homography leaves points unchanged."""
        h = Homography.identity()
        point = np.array([10, 20])
        result = h.apply_to_point(point)
        assert np.allclose(result, point)

    def test_from_four_points_identity(self):
        """Four identical point pairs create identity-like transform."""
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = src.copy()
        h = Homography.from_four_points(src, dst)
        result = h.apply_to_points(src)
        assert np.allclose(result, dst, atol=1e-3)

    def test_from_four_points_translation(self):
        """Homography can represent translation."""
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        dst = src + 5
        h = Homography.from_four_points(src, dst)
        result = h.apply_to_points(src)
        assert np.allclose(result, dst, atol=1e-3)

    def test_from_four_points_scaling(self):
        """Homography can represent scaling."""
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        dst = src * 2
        h = Homography.from_four_points(src, dst)
        result = h.apply_to_points(src)
        assert np.allclose(result, dst, atol=1e-3)

    def test_from_four_points_wrong_shape_raises(self):
        """Wrong point array shape should raise."""
        with pytest.raises(ValueError, match="shape"):
            Homography.from_four_points(
                np.array([[0, 0], [1, 0], [1, 1]]),
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            )

    def test_from_points_many_points(self):
        """Create homography from more than 4 points."""
        # Create a simple translation with extra points
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=np.float32)
        dst = src + 10
        h = Homography.from_points(src, dst, method=0)  # Use least squares
        result = h.apply_to_points(src)
        assert np.allclose(result, dst, atol=0.1)

    def test_from_points_too_few_raises(self):
        """Less than 4 points should raise."""
        with pytest.raises(ValueError, match=r"(?i)at least 4"):
            Homography.from_points(
                np.array([[0, 0], [1, 0], [1, 1]]),
                np.array([[0, 0], [1, 0], [1, 1]]),
            )

    def test_from_rectangle_to_quad(self):
        """Map rectangle to quadrilateral."""
        rect = (0, 0, 100, 100)
        quad = np.array([[10, 10], [90, 20], [80, 90], [20, 80]])
        h = Homography.from_rectangle_to_quad(rect, quad)
        # Check corners map correctly
        src_corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        result = h.apply_to_points(src_corners)
        assert np.allclose(result, quad, atol=1e-3)

    def test_apply_to_point(self):
        """Apply to single point."""
        h = Homography.identity()
        result = h.apply_to_point(np.array([5, 10]))
        assert np.allclose(result, [5, 10])

    def test_apply_to_point_wrong_shape_raises(self):
        """Wrong point shape should raise."""
        h = Homography.identity()
        with pytest.raises(ValueError, match="shape"):
            h.apply_to_point(np.array([1, 2, 3]))

    def test_apply_to_points(self):
        """Apply to multiple points."""
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        dst = src + 5
        h = Homography.from_four_points(src, dst)
        result = h.apply_to_points(src)
        assert result.shape == src.shape
        assert np.allclose(result, dst, atol=1e-3)

    def test_compose(self):
        """Compose two homographies."""
        # Two translations
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        h1 = Homography.from_four_points(src, src + 5)
        h2 = Homography.from_four_points(src, src + 3)
        composed = h1.compose(h2)
        # h1.compose(h2)(p) = h1(h2(p))
        # h2(src) = src + 3, h1(that) = that + 5 = src + 8
        result = composed.apply_to_points(src)
        assert np.allclose(result, src + 8, atol=1e-2)

    def test_inverse(self):
        """Inverse homography."""
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        dst = src + 5
        h = Homography.from_four_points(src, dst)
        inv = h.inverse()
        # h followed by inv should return original
        points = np.array([[3, 3], [7, 7]])
        result = inv.apply_to_points(h.apply_to_points(points))
        assert np.allclose(result, points, atol=1e-3)

    def test_inverse_singular_raises(self):
        """Singular homography cannot be inverted."""
        # Create a degenerate homography (all points map to a line)
        matrix = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64)
        h = Homography(matrix=matrix)
        with pytest.raises(ValueError, match="singular"):
            h.inverse()

    def test_apply_to_image(self):
        """Apply homography to an image."""
        h = Homography.identity()
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255
        result = h.apply_to_image(image)
        assert result.shape == image.shape
        # Identity should preserve the image
        assert np.array_equal(result, image)

    def test_matrix_validation(self):
        """Invalid matrix shape should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            Homography(matrix=np.array([[1, 0], [0, 1]]))


class TestWarpPerspective:
    """Tests for warp_perspective function."""

    def test_warp_identity(self):
        """Warp with identity mapping."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = src.copy()
        result = warp_perspective(image, src, dst)
        assert result.shape == image.shape

    def test_warp_translation(self):
        """Warp with translation."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[10:20, 10:20] = 255
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = src + 10
        result = warp_perspective(image, src, dst, output_size=(120, 120))
        # Content should be shifted
        assert result.shape == (120, 120)
