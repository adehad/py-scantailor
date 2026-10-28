"""Tests for geometric transformations."""

import numpy as np

from scantailor.imageproc.geometry import rotate_orthogonal


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
