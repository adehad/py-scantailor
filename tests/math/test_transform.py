"""Tests for affine transform utilities."""

import numpy as np
import pytest

from scantailor.math.transform import AffineTransform, compose_transforms


class TestAffineTransform:
    """Tests for AffineTransform class."""

    def test_identity(self):
        """Identity transform leaves points unchanged."""
        t = AffineTransform.identity()
        point = np.array([10, 20])
        result = t.apply_to_point(point)
        assert np.allclose(result, point)

    def test_translation(self):
        """Translation moves points."""
        t = AffineTransform.translation(5, 10)
        point = np.array([0, 0])
        result = t.apply_to_point(point)
        assert np.allclose(result, [5, 10])

    def test_rotation_90(self):
        """90 degree rotation (counter-clockwise in image coordinates)."""
        t = AffineTransform.rotation(90)
        point = np.array([10, 0])
        result = t.apply_to_point(point)
        # In image coordinates (Y down), 90 CCW: (10, 0) -> (0, -10)
        assert pytest.approx(result[0], abs=1e-6) == 0
        assert pytest.approx(result[1], abs=1e-6) == -10

    def test_rotation_180(self):
        """180 degree rotation."""
        t = AffineTransform.rotation(180)
        point = np.array([10, 5])
        result = t.apply_to_point(point)
        assert pytest.approx(result[0], abs=1e-6) == -10
        assert pytest.approx(result[1], abs=1e-6) == -5

    def test_rotation_around_center(self):
        """Rotation around a custom center."""
        t = AffineTransform.rotation(90, center=(5, 5))
        point = np.array([10, 5])
        result = t.apply_to_point(point)
        # Rotating (10, 5) around (5, 5) by 90 CCW:
        # Offset from center: (5, 0) -> after rotation: (0, -5)
        # Result: (5+0, 5-5) = (5, 0)
        assert pytest.approx(result[0], abs=1e-6) == 5
        assert pytest.approx(result[1], abs=1e-6) == 0

    def test_scaling_uniform(self):
        """Uniform scaling."""
        t = AffineTransform.scaling(2)
        point = np.array([5, 10])
        result = t.apply_to_point(point)
        assert np.allclose(result, [10, 20])

    def test_scaling_non_uniform(self):
        """Non-uniform scaling."""
        t = AffineTransform.scaling(2, 3)
        point = np.array([5, 10])
        result = t.apply_to_point(point)
        assert np.allclose(result, [10, 30])

    def test_scaling_around_center(self):
        """Scaling around a custom center."""
        t = AffineTransform.scaling(2, center=(5, 5))
        point = np.array([10, 10])
        result = t.apply_to_point(point)
        # (10-5)*2 + 5 = 15
        assert pytest.approx(result[0], abs=1e-6) == 15
        assert pytest.approx(result[1], abs=1e-6) == 15

    def test_shear_horizontal(self):
        """Horizontal shearing."""
        t = AffineTransform.shear(shx=1)
        point = np.array([0, 10])
        result = t.apply_to_point(point)
        assert pytest.approx(result[0], abs=1e-6) == 10  # x + shx*y
        assert pytest.approx(result[1], abs=1e-6) == 10

    def test_shear_vertical(self):
        """Vertical shearing."""
        t = AffineTransform.shear(shy=0.5)
        point = np.array([10, 0])
        result = t.apply_to_point(point)
        assert pytest.approx(result[0], abs=1e-6) == 10
        assert pytest.approx(result[1], abs=1e-6) == 5  # shy*x + y

    def test_apply_to_points(self):
        """Apply transform to multiple points."""
        t = AffineTransform.translation(10, 20)
        points = np.array([[0, 0], [5, 5], [10, 10]])
        result = t.apply_to_points(points)
        expected = np.array([[10, 20], [15, 25], [20, 30]])
        assert np.allclose(result, expected)

    def test_compose(self):
        """Compose two transforms."""
        t1 = AffineTransform.translation(10, 0)
        t2 = AffineTransform.translation(0, 5)
        composed = t1.compose(t2)
        # composed(p) = t1(t2(p))
        point = np.array([0, 0])
        result = composed.apply_to_point(point)
        assert np.allclose(result, [10, 5])

    def test_compose_order_matters(self):
        """Composition order matters for non-commutative transforms."""
        # Scale then translate vs translate then scale
        scale = AffineTransform.scaling(2)
        translate = AffineTransform.translation(10, 0)

        # scale.compose(translate): translate first, then scale
        st = scale.compose(translate)
        # translate.compose(scale): scale first, then translate
        ts = translate.compose(scale)

        point = np.array([5, 0])

        # st: translate(5,0) -> (15,0), scale -> (30, 0)
        assert np.allclose(st.apply_to_point(point), [30, 0])

        # ts: scale(5,0) -> (10,0), translate -> (20, 0)
        assert np.allclose(ts.apply_to_point(point), [20, 0])

    def test_inverse(self):
        """Inverse transform."""
        t = AffineTransform.translation(10, 20)
        inv = t.inverse()
        point = np.array([5, 5])
        # Apply t then inv should return original
        result = inv.apply_to_point(t.apply_to_point(point))
        assert np.allclose(result, point)

    def test_inverse_rotation(self):
        """Inverse of rotation."""
        t = AffineTransform.rotation(45)
        inv = t.inverse()
        point = np.array([10, 5])
        result = inv.apply_to_point(t.apply_to_point(point))
        assert np.allclose(result, point)

    def test_inverse_singular_raises(self):
        """Singular matrix cannot be inverted."""
        # Create a degenerate transform (all points map to a line)
        t = AffineTransform(matrix=np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float64))
        with pytest.raises(ValueError, match="singular"):
            t.inverse()

    def test_apply_to_image(self):
        """Apply transform to an image."""
        t = AffineTransform.identity()
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255
        result = t.apply_to_image(image)
        assert result.shape == image.shape
        # Identity should preserve the image
        assert np.array_equal(result, image)


class TestComposeTransforms:
    """Tests for compose_transforms function."""

    def test_compose_single(self):
        """Compose single transform returns same transform."""
        t = AffineTransform.translation(5, 10)
        result = compose_transforms(t)
        point = np.array([0, 0])
        assert np.allclose(result.apply_to_point(point), t.apply_to_point(point))

    def test_compose_multiple(self):
        """Compose multiple transforms."""
        t1 = AffineTransform.translation(10, 0)
        t2 = AffineTransform.translation(0, 5)
        t3 = AffineTransform.scaling(2)
        composed = compose_transforms(t1, t2, t3)
        point = np.array([1, 1])
        # Right-to-left: scale(1,1)->(2,2), translate->(2,7), translate->(12,7)
        result = composed.apply_to_point(point)
        assert np.allclose(result, [12, 7])

    def test_compose_empty_raises(self):
        """Empty compose should raise ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            compose_transforms()

    def test_matrix_validation(self):
        """Invalid matrix shape should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            AffineTransform(matrix=np.array([[1, 0], [0, 1]]))
