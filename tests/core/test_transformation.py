"""Tests for ImageTransformation."""

import numpy as np
import pytest

from scantailor.core.models import Dpi, OrthogonalRotation
from scantailor.core.transformation import ImageTransformation, Rect


class TestRect:
    """Tests for Rect class."""

    def test_from_size(self) -> None:
        """Test creating rect from size."""
        r = Rect.from_size(100, 200)
        assert r.x == 0
        assert r.y == 0
        assert r.width == 100
        assert r.height == 200

    def test_properties(self) -> None:
        """Test rect properties."""
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.left == 10
        assert r.top == 20
        assert r.right == 110
        assert r.bottom == 70

    def test_center(self) -> None:
        """Test center calculation."""
        r = Rect(x=0, y=0, width=100, height=100)
        cx, cy = r.center()
        assert cx == 50
        assert cy == 50

    def test_size(self) -> None:
        """Test size method."""
        r = Rect(x=10, y=20, width=30, height=40)
        w, h = r.size()
        assert w == 30
        assert h == 40


class TestImageTransformation:
    """Tests for ImageTransformation class."""

    def test_creation(self) -> None:
        """Test basic creation."""
        xform = ImageTransformation.from_size(800, 600)
        assert xform.orig_rect.width == 800
        assert xform.orig_rect.height == 600

    def test_creation_with_dpi(self) -> None:
        """Test creation with DPI."""
        xform = ImageTransformation.from_size(
            800, 600, Dpi(horizontal=300, vertical=300)
        )
        assert xform.orig_dpi.horizontal == 300
        assert xform.orig_dpi.vertical == 300

    def test_identity_transform(self) -> None:
        """Test identity transform preserves coordinates."""
        xform = ImageTransformation.from_size(800, 600)
        x, y = xform.transform_point(100, 200)
        assert pytest.approx(x, abs=0.1) == 100
        assert pytest.approx(y, abs=0.1) == 200

    def test_transform_roundtrip(self) -> None:
        """Test forward and back transforms are inverses."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_post_rotation(5.0)

        orig_x, orig_y = 400, 300
        new_x, new_y = xform.transform_point(orig_x, orig_y)
        back_x, back_y = xform.transform_point_back(new_x, new_y)

        assert pytest.approx(back_x, abs=0.01) == orig_x
        assert pytest.approx(back_y, abs=0.01) == orig_y

    def test_equalize_dpi_symmetric(self) -> None:
        """Test that symmetric DPI stays unchanged."""
        xform = ImageTransformation.from_size(
            800, 600, Dpi(horizontal=300, vertical=300)
        )
        assert xform.pre_scaled_dpi.horizontal == 300
        assert xform.pre_scaled_dpi.vertical == 300

    def test_equalize_dpi_asymmetric(self) -> None:
        """Test that asymmetric DPI is equalized to minimum."""
        xform = ImageTransformation.from_size(
            800, 600, Dpi(horizontal=300, vertical=600)
        )
        assert xform.pre_scaled_dpi.horizontal == 300
        assert xform.pre_scaled_dpi.vertical == 300

    def test_pre_rotation_90(self) -> None:
        """Test 90 degree pre-rotation."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_pre_rotation(OrthogonalRotation(degrees=90))

        # After 90 CW rotation, width and height should be swapped
        result = xform.resulting_rect
        assert pytest.approx(result.width, abs=1) == 600
        assert pytest.approx(result.height, abs=1) == 800

    def test_pre_rotation_180(self) -> None:
        """Test 180 degree pre-rotation."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_pre_rotation(OrthogonalRotation(degrees=180))

        # 180 rotation preserves dimensions
        result = xform.resulting_rect
        assert pytest.approx(result.width, abs=1) == 800
        assert pytest.approx(result.height, abs=1) == 600

    def test_post_rotation_small_angle(self) -> None:
        """Test small post-rotation (deskew)."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_post_rotation(2.0)  # 2 degrees

        # After rotation, result should be slightly larger
        result = xform.resulting_rect
        assert result.width >= 800
        assert result.height >= 600

    def test_post_rotation_zero(self) -> None:
        """Test zero post-rotation has no effect."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_post_rotation(0.0)

        result = xform.resulting_rect
        assert pytest.approx(result.width, abs=1) == 800
        assert pytest.approx(result.height, abs=1) == 600

    def test_pre_crop_reduces_size(self) -> None:
        """Test that pre-crop reduces resulting size."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))

        # Crop to center 400x300
        crop_area = np.array(
            [
                [200, 150],
                [600, 150],
                [600, 450],
                [200, 450],
            ],
            dtype=np.float64,
        )
        xform.set_pre_crop_area(crop_area)

        result = xform.resulting_rect
        assert pytest.approx(result.width, abs=1) == 400
        assert pytest.approx(result.height, abs=1) == 300

    def test_pipeline_order(self) -> None:
        """Test that steps execute in correct order."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))

        # Set up pipeline
        xform.set_pre_rotation(OrthogonalRotation(degrees=90))
        xform.set_post_rotation(5.0)

        # After 90 CW, dimensions should be swapped
        # Post-rotation should then expand slightly
        result = xform.resulting_rect
        # Should be taller than wide after 90 rotation
        assert result.height > result.width

    def test_resetting_earlier_step_clears_later(self) -> None:
        """Test that changing an earlier step resets later steps."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))

        # Set post-rotation
        xform.set_post_rotation(10.0)
        assert xform.post_rotation == 10.0

        # Change pre-rotation - should reset post-rotation
        xform.set_pre_rotation(OrthogonalRotation(degrees=90))
        assert xform.post_rotation == 0.0

    def test_resulting_pre_crop_area(self) -> None:
        """Test resulting pre-crop area is computed."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        area = xform.resulting_pre_crop_area

        # Should be a 4-point polygon
        assert area.shape == (4, 2)

    def test_resulting_post_crop_area(self) -> None:
        """Test resulting post-crop area is computed."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        area = xform.resulting_post_crop_area

        # Should be a 4-point polygon
        assert area.shape == (4, 2)

    def test_transform_matrix_shape(self) -> None:
        """Test transform matrix has correct shape."""
        xform = ImageTransformation.from_size(800, 600)
        assert xform.transform.shape == (3, 3)
        assert xform.transform_back.shape == (3, 3)

    def test_null_dpi_handling(self) -> None:
        """Test handling of null (zero) DPI."""
        xform = ImageTransformation.from_size(800, 600, Dpi())
        # Should not crash, just use identity-like transforms
        x, y = xform.transform_point(100, 100)
        assert pytest.approx(x, abs=1) == 100
        assert pytest.approx(y, abs=1) == 100
