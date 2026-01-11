"""Tests for geometric transformations."""

import numpy as np

from scantailor.imageproc import rotate_orthogonal, scale


class TestRotateOrthogonal:
    """Tests for orthogonal rotation."""

    def test_rotate_0_degrees_unchanged(self):
        """0 degree rotation should return copy of input."""
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        result = rotate_orthogonal(image, 0)
        np.testing.assert_array_equal(result, image)
        # Should be a copy, not the same array
        assert result is not image

    def test_rotate_90_degrees(self):
        """90 degree clockwise rotation."""
        # Create image with distinct pattern
        # Original (4 rows, 3 cols):
        #   [255, 0, 0]
        #   [0, 0, 0]
        #   [0, 0, 0]
        #   [0, 0, 0]
        image = np.zeros((4, 3), dtype=np.uint8)
        image[0, 0] = 255  # Top-left marker

        result = rotate_orthogonal(image, 90)

        # Shape should be transposed (3 rows, 4 cols)
        assert result.shape == (3, 4)
        # After 90° clockwise: top-left goes to top-right
        # Position (0,0) -> (0, height-1) = (0, 3)
        assert result[0, 3] == 255
        assert result[0, 0] == 0

    def test_rotate_180_degrees(self):
        """180 degree rotation."""
        image = np.zeros((4, 3), dtype=np.uint8)
        image[0, 0] = 255  # Top-left marker

        result = rotate_orthogonal(image, 180)

        # Shape should be same
        assert result.shape == (4, 3)
        # Top-left should now be bottom-right
        assert result[3, 2] == 255
        assert result[0, 0] == 0

    def test_rotate_270_degrees(self):
        """270 degree clockwise (= 90 counter-clockwise) rotation."""
        image = np.zeros((4, 3), dtype=np.uint8)
        image[0, 0] = 255  # Top-left marker

        result = rotate_orthogonal(image, 270)

        # Shape should be transposed (3 rows, 4 cols)
        assert result.shape == (3, 4)
        # After 270° clockwise: top-left goes to bottom-left
        # Position (0,0) -> (cols-1, 0) = (2, 0)
        assert result[2, 0] == 255
        assert result[0, 0] == 0

    def test_four_rotations_returns_original(self):
        """Four 90-degree rotations should return to original."""
        image = np.random.randint(0, 256, (10, 15), dtype=np.uint8)

        result = image.copy()
        for _ in range(4):
            result = rotate_orthogonal(result, 90)

        np.testing.assert_array_equal(result, image)

    def test_preserves_dtype(self):
        """Output should preserve uint8 dtype."""
        image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        for degrees in (0, 90, 180, 270):
            result = rotate_orthogonal(image, degrees)  # type: ignore[arg-type]
            assert result.dtype == np.uint8


class TestScale:
    """Tests for scaling."""

    def test_scale_1x_unchanged(self):
        """1x scale should return copy of input."""
        image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = scale(image, 1.0)
        np.testing.assert_array_equal(result, image)
        assert result is not image

    def test_scale_2x_doubles_size(self):
        """2x scale should double dimensions."""
        image = np.random.randint(0, 256, (10, 20), dtype=np.uint8)
        result = scale(image, 2.0)
        assert result.shape == (20, 40)

    def test_scale_half_halves_size(self):
        """0.5x scale should halve dimensions."""
        image = np.random.randint(0, 256, (20, 40), dtype=np.uint8)
        result = scale(image, 0.5)
        assert result.shape == (10, 20)

    def test_scale_different_x_y(self):
        """Different x and y scales."""
        image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = scale(image, scale_x=2.0, scale_y=3.0)
        assert result.shape == (30, 20)

    def test_scale_y_defaults_to_x(self):
        """If scale_y is None, it should default to scale_x."""
        image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = scale(image, 2.0)
        assert result.shape == (20, 20)

    def test_different_interpolation_methods(self):
        """Different interpolation methods should work."""
        image = np.random.randint(0, 256, (20, 20), dtype=np.uint8)

        for method in ("nearest", "linear", "cubic", "area", "lanczos"):
            result = scale(image, 0.5, interpolation=method)  # type: ignore[arg-type]
            assert result.shape == (10, 10)

    def test_preserves_dtype(self):
        """Output should preserve uint8 dtype."""
        image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = scale(image, 2.0)
        assert result.dtype == np.uint8
