"""Tests for Fix Orientation filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scantailor.core import ImageId, OrthogonalRotation, PageId, SubPage
from scantailor.filters.fix_orientation import Filter, Settings


class TestFilter:
    """Tests for Filter class."""

    def test_filter_name(self):
        """Filter should have correct name."""
        filter = Filter()
        assert filter.name == "Fix Orientation"

    def test_creates_default_settings(self):
        """Filter should create settings if not provided."""
        filter = Filter()
        assert filter.settings is not None

    def test_uses_provided_settings(self):
        """Filter should use provided settings."""
        settings = Settings()
        filter = Filter(settings)
        assert filter.settings is settings

    def test_get_rotation_default(self):
        """get_rotation should return 0 degrees by default."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))

        rotation = filter.get_rotation(image_id)

        assert rotation.degrees == 0

    def test_set_rotation(self):
        """set_rotation should store the rotation."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))

        filter.set_rotation(image_id, OrthogonalRotation(degrees=180))

        assert filter.get_rotation(image_id).degrees == 180

    def test_rotate_clockwise(self):
        """rotate_clockwise should increment rotation by 90 degrees."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))

        # Start at 0, rotate to 90
        new_rotation = filter.rotate_clockwise(image_id)
        assert new_rotation.degrees == 90
        assert filter.get_rotation(image_id).degrees == 90

        # Rotate to 180
        new_rotation = filter.rotate_clockwise(image_id)
        assert new_rotation.degrees == 180

        # Rotate to 270
        new_rotation = filter.rotate_clockwise(image_id)
        assert new_rotation.degrees == 270

        # Rotate back to 0
        new_rotation = filter.rotate_clockwise(image_id)
        assert new_rotation.degrees == 0

    def test_rotate_counter_clockwise(self):
        """rotate_counter_clockwise should decrement rotation by 90 degrees."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))

        # Start at 0, rotate to 270
        new_rotation = filter.rotate_counter_clockwise(image_id)
        assert new_rotation.degrees == 270
        assert filter.get_rotation(image_id).degrees == 270

        # Rotate to 180
        new_rotation = filter.rotate_counter_clockwise(image_id)
        assert new_rotation.degrees == 180

        # Rotate to 90
        new_rotation = filter.rotate_counter_clockwise(image_id)
        assert new_rotation.degrees == 90

        # Rotate back to 0
        new_rotation = filter.rotate_counter_clockwise(image_id)
        assert new_rotation.degrees == 0

    def test_reset_rotation(self):
        """reset_rotation should set rotation to 0 degrees."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))
        filter.set_rotation(image_id, OrthogonalRotation(degrees=180))

        filter.reset_rotation(image_id)

        assert filter.get_rotation(image_id).degrees == 0

    def test_apply_to_pages(self):
        """apply_to_pages should set rotation for multiple pages."""
        filter = Filter()
        image_id1 = ImageId(file_path=Path("/scan1.tiff"))
        image_id2 = ImageId(file_path=Path("/scan2.tiff"))
        pages = [
            PageId(image_id=image_id1, sub_page=SubPage.SINGLE_PAGE),
            PageId(image_id=image_id2, sub_page=SubPage.SINGLE_PAGE),
        ]

        filter.apply_to_pages(pages, OrthogonalRotation(degrees=90))

        assert filter.get_rotation(image_id1).degrees == 90
        assert filter.get_rotation(image_id2).degrees == 90


class TestFilterProcessImage:
    """Tests for image processing functionality."""

    def test_process_image_no_rotation(self):
        """process_image with 0 degrees should return copy of input."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))
        image = np.arange(20, dtype=np.uint8).reshape(4, 5)

        result = filter.process_image(image, image_id)

        np.testing.assert_array_equal(result, image)
        assert result is not image  # Should be a copy

    def test_process_image_90_degrees(self):
        """process_image with 90 degrees should rotate the image."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))
        filter.set_rotation(image_id, OrthogonalRotation(degrees=90))

        # Create image with marker at top-left
        image = np.zeros((4, 3), dtype=np.uint8)
        image[0, 0] = 255

        result = filter.process_image(image, image_id)

        # After 90° clockwise, top-left goes to top-right
        assert result.shape == (3, 4)
        assert result[0, 3] == 255
        assert result[0, 0] == 0

    def test_process_image_180_degrees(self):
        """process_image with 180 degrees should rotate the image."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))
        filter.set_rotation(image_id, OrthogonalRotation(degrees=180))

        # Create image with marker at top-left
        image = np.zeros((4, 3), dtype=np.uint8)
        image[0, 0] = 255

        result = filter.process_image(image, image_id)

        # After 180°, top-left goes to bottom-right
        assert result.shape == (4, 3)
        assert result[3, 2] == 255
        assert result[0, 0] == 0

    def test_process_image_270_degrees(self):
        """process_image with 270 degrees should rotate the image."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))
        filter.set_rotation(image_id, OrthogonalRotation(degrees=270))

        # Create image with marker at top-left
        image = np.zeros((4, 3), dtype=np.uint8)
        image[0, 0] = 255

        result = filter.process_image(image, image_id)

        # After 270° clockwise, top-left goes to bottom-left
        assert result.shape == (3, 4)
        assert result[2, 0] == 255
        assert result[0, 0] == 0

    def test_process_image_preserves_dtype(self):
        """process_image should preserve uint8 dtype."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))
        filter.set_rotation(image_id, OrthogonalRotation(degrees=90))

        image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = filter.process_image(image, image_id)

        assert result.dtype == np.uint8

    def test_process_image_four_rotations_returns_original(self):
        """Four 90-degree rotations should return to original."""
        filter = Filter()
        image_id = ImageId(file_path=Path("/test.tiff"))
        image = np.random.randint(0, 256, (10, 15), dtype=np.uint8)

        # Apply 4 rotations
        result = image.copy()
        for _ in range(4):
            filter.rotate_clockwise(image_id)
            result = filter.process_image(result, image_id)
            filter.reset_rotation(image_id)  # Reset for next iteration

        # After 4 rotations (0->90->180->270->0), we're back to original
        np.testing.assert_array_equal(result, image)
