"""Tests for ImageTransformation."""

import numpy as np
import pytest

from scantailor.core.models import Dpi, OrthogonalRotation
from scantailor.core.transformation import (
    ImageTransformation,
    Rect,
    render_transformed,
)


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


class TestTransformComposition:
    """Steps are applied in pipeline order and keep the result at the origin."""

    def test_pre_rotation_90_maps_corners_clockwise(self) -> None:
        """Top-left goes to top-right and the canvas becomes height x width."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_pre_rotation(OrthogonalRotation(degrees=90))

        assert xform.transform_point(0, 0) == pytest.approx((600, 0))
        assert xform.transform_point(800, 0) == pytest.approx((600, 800))
        assert xform.resulting_rect.x == pytest.approx(0)
        assert xform.resulting_rect.y == pytest.approx(0)

    def test_pre_rotation_270_maps_corners_counter_clockwise(self) -> None:
        """Top-left goes to bottom-left."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_pre_rotation(OrthogonalRotation(degrees=270))

        assert xform.transform_point(0, 0) == pytest.approx((0, 800))
        assert xform.transform_point(800, 600) == pytest.approx((600, 0))

    def test_post_rotation_turns_clockwise_about_the_centre(self) -> None:
        """The centre is fixed and the canvas starts at the origin."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_post_rotation(10.0)

        rect = xform.resulting_rect
        assert (rect.x, rect.y) == pytest.approx((0, 0), abs=1e-9)
        assert xform.transform_point(400, 300) == pytest.approx(rect.center())
        top_right_x, top_right_y = xform.transform_point(800, 0)
        assert top_right_y > 0
        assert top_right_x == pytest.approx(rect.right)

    def test_crop_then_rotation_stays_inside_resulting_rect(self) -> None:
        """Rotating a cropped area leaves its corners on the canvas edges."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_pre_crop_area(
            np.array([[200, 150], [600, 150], [600, 450], [200, 450]], dtype=float)
        )
        xform.set_post_rotation(10.0)

        corners = xform.resulting_pre_crop_area
        rect = xform.resulting_rect
        assert corners.min(axis=0) == pytest.approx((0, 0), abs=1e-9)
        assert corners.max(axis=0) == pytest.approx((rect.right, rect.bottom))

    def test_post_crop_area_sets_resulting_size_and_origin(self) -> None:
        """A post-crop polygon becomes the new canvas."""
        xform = ImageTransformation.from_size(800, 600, Dpi.uniform(300))
        xform.set_post_crop_area(
            np.array([[100, 50], [300, 50], [300, 250], [100, 250]], dtype=float)
        )

        rect = xform.resulting_rect
        assert (rect.width, rect.height) == pytest.approx((200, 200))
        assert xform.transform_point(100, 50) == pytest.approx((0, 0))


class TestRenderTransformed:
    """render_transformed resamples once through the whole transformation."""

    @staticmethod
    def _gradient() -> np.ndarray:
        return np.arange(60 * 80, dtype=np.uint32).reshape(60, 80).astype(np.uint8)

    def test_identity_returns_equal_copy(self) -> None:
        """No steps give back identical pixels in a new array."""
        image = self._gradient()
        xform = ImageTransformation.from_size(80, 60, Dpi.uniform(300))

        result = render_transformed(image, xform)

        np.testing.assert_array_equal(result, image)
        assert not np.shares_memory(result, image)

    def test_crop_returns_exact_window(self) -> None:
        """A whole-pixel crop is a bit-exact slice."""
        image = self._gradient()
        xform = ImageTransformation.from_size(80, 60, Dpi.uniform(300))
        xform.set_pre_crop_area(
            np.array([[10, 5], [50, 5], [50, 45], [10, 45]], dtype=float)
        )

        result = render_transformed(image, xform)

        np.testing.assert_array_equal(result, image[5:45, 10:50])

    def test_pre_crop_outside_the_area_is_white(self) -> None:
        """A post-crop reaching past the pre-crop area shows white there."""
        image = np.zeros((60, 80), dtype=np.uint8)
        xform = ImageTransformation.from_size(80, 60, Dpi.uniform(300))
        xform.set_pre_crop_area(
            np.array([[0, 0], [40, 0], [40, 60], [0, 60]], dtype=float)
        )
        xform.set_post_crop_area(
            np.array([[0, 0], [60, 0], [60, 60], [0, 60]], dtype=float)
        )

        result = render_transformed(image, xform)

        assert result.shape == (60, 60)
        assert np.all(result[:, :40] == 0)
        assert np.all(result[:, 40:] == 255)

    def test_rotation_expands_canvas_and_fills_corners_white(self) -> None:
        """Rotating a black page leaves white wedges at the corners."""
        image = np.zeros((60, 80), dtype=np.uint8)
        xform = ImageTransformation.from_size(80, 60, Dpi.uniform(300))
        xform.set_post_rotation(10.0)

        result = render_transformed(image, xform)

        assert result.shape[1] > 80
        assert result.shape[0] > 60
        assert result[0, 0] == 255
        assert result[result.shape[0] // 2, result.shape[1] // 2] == 0

    def test_color_image_keeps_channels(self) -> None:
        """Three-channel images stay three-channel and pad with white."""
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        xform = ImageTransformation.from_size(80, 60, Dpi.uniform(300))
        xform.set_post_rotation(10.0)

        result = render_transformed(image, xform)

        assert result.ndim == 3
        assert result.shape[2] == 3
        assert tuple(result[0, 0]) == (255, 255, 255)

    def test_source_is_not_modified(self) -> None:
        """Rendering never writes into the input array."""
        image = self._gradient()
        original = image.copy()
        xform = ImageTransformation.from_size(80, 60, Dpi.uniform(300))
        xform.set_post_rotation(7.0)

        render_transformed(image, xform)

        np.testing.assert_array_equal(image, original)

    def test_single_channel_3d_image_keeps_its_channel_axis(self) -> None:
        """An (H, W, 1) input comes back as (H, W, 1), not (H, W)."""
        image = np.zeros((60, 80, 1), dtype=np.uint8)
        xform = ImageTransformation.from_size(80, 60, Dpi.uniform(300))
        xform.set_post_rotation(10.0)

        result = render_transformed(image, xform)

        assert result.ndim == 3
        assert result.shape[2] == 1
        assert result[0, 0, 0] == 255

    def test_one_pixel_image_renders_at_least_one_pixel(self) -> None:
        """The smallest image survives rotation without an empty result."""
        image = np.zeros((1, 1), dtype=np.uint8)
        xform = ImageTransformation.from_size(1, 1, Dpi.uniform(300))
        xform.set_post_rotation(10.0)

        result = render_transformed(image, xform)

        assert result.shape[0] >= 1
        assert result.shape[1] >= 1
