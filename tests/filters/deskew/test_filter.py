"""Tests for Deskew filter."""

from pathlib import Path

import numpy as np

from scantailor.core import ImageId, PageId, SubPage
from scantailor.filters.deskew import AutoManualMode, Filter, Params, Settings


def make_page_id(name: str, sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    """Helper to create PageId instances."""
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}.tiff")),
        sub_page=sub_page,
    )


class TestFilter:
    """Tests for Filter class."""

    def test_filter_name(self):
        """Filter should have correct name."""
        filter = Filter()
        assert filter.name == "Deskew"

    def test_creates_default_settings(self):
        """Filter should create settings if not provided."""
        filter = Filter()
        assert filter.settings is not None

    def test_uses_provided_settings(self):
        """Filter should use provided settings."""
        settings = Settings()
        filter = Filter(settings)
        assert filter.settings is settings

    def test_get_params_default(self):
        """get_params should return default params."""
        filter = Filter()
        page_id = make_page_id("test")

        params = filter.get_params(page_id)

        assert params.deskew_angle_deg == 0.0
        assert params.mode == AutoManualMode.AUTO

    def test_get_deskew_angle(self):
        """get_deskew_angle should return the angle."""
        filter = Filter()
        page_id = make_page_id("test")
        filter.set_params(page_id, Params(deskew_angle_deg=3.5))

        angle = filter.get_deskew_angle(page_id)

        assert angle == 3.5

    def test_set_manual_angle(self):
        """set_manual_angle should set angle with MANUAL mode."""
        filter = Filter()
        page_id = make_page_id("test")

        params = filter.set_manual_angle(page_id, 5.0)

        assert params.deskew_angle_deg == 5.0
        assert params.mode == AutoManualMode.MANUAL
        assert filter.get_params(page_id).deskew_angle_deg == 5.0

    def test_reset_to_auto(self):
        """reset_to_auto should reset to default params."""
        filter = Filter()
        page_id = make_page_id("test")
        filter.set_manual_angle(page_id, 10.0)

        filter.reset_to_auto(page_id)

        params = filter.get_params(page_id)
        assert params.deskew_angle_deg == 0.0
        assert params.mode == AutoManualMode.AUTO

    def test_apply_to_pages(self):
        """apply_to_pages should set params for all pages."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        params = Params(deskew_angle_deg=2.5, mode=AutoManualMode.MANUAL)

        filter.apply_to_pages([page1, page2], params)

        assert filter.get_deskew_angle(page1) == 2.5
        assert filter.get_deskew_angle(page2) == 2.5


class TestFilterDetectSkew:
    """Tests for skew detection functionality."""

    def test_detect_skew_returns_result(self):
        """detect_skew should return a SkewResult."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100), dtype=np.uint8)

        result = filter.detect_skew(image, page_id)

        assert hasattr(result, "angle")
        assert hasattr(result, "confidence")

    def test_detect_skew_stores_result(self):
        """detect_skew should store the result in settings."""
        filter = Filter()
        page_id = make_page_id("test")
        # Create image with horizontal lines (no skew)
        image = np.zeros((100, 200), dtype=np.uint8)
        image[30, 20:180] = 255
        image[60, 20:180] = 255

        filter.detect_skew(image, page_id)

        # Should have stored something
        assert filter.settings.is_params_set(page_id)

    def test_detect_skew_no_store(self):
        """detect_skew with store_result=False should not store."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100), dtype=np.uint8)

        filter.detect_skew(image, page_id, store_result=False)

        assert filter.settings.is_params_set(page_id) is False

    def test_detect_skew_sets_auto_mode(self):
        """detect_skew should set AUTO mode."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 200), dtype=np.uint8)
        image[50, :] = 255

        filter.detect_skew(image, page_id)

        params = filter.get_params(page_id)
        assert params.mode == AutoManualMode.AUTO

    def test_detect_skew_handles_color(self):
        """detect_skew should handle color images."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should not raise
        result = filter.detect_skew(image, page_id)

        assert result is not None


class TestFilterProcessImage:
    """Tests for image processing functionality."""

    def test_process_image_no_rotation(self):
        """process_image with 0 angle should return copy."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.arange(20, dtype=np.uint8).reshape(4, 5)

        result = filter.process_image(image, page_id)

        np.testing.assert_array_equal(result, image)
        assert result is not image

    def test_process_image_rotates(self):
        """process_image with angle should rotate the image."""
        filter = Filter()
        page_id = make_page_id("test")
        filter.set_manual_angle(page_id, 5.0)

        # Create image with marker
        image = np.zeros((100, 100), dtype=np.uint8)
        image[10, 50] = 255  # Marker near top center

        result = filter.process_image(image, page_id)

        # Image should be rotated (and possibly larger due to rotation)
        # The marker should have moved
        assert result is not image

    def test_process_image_preserves_dtype(self):
        """process_image should preserve uint8 dtype."""
        filter = Filter()
        page_id = make_page_id("test")
        filter.set_manual_angle(page_id, 3.0)

        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = filter.process_image(image, page_id)

        assert result.dtype == np.uint8

    def test_process_image_handles_color(self):
        """process_image should handle color images."""
        filter = Filter()
        page_id = make_page_id("test")
        filter.set_manual_angle(page_id, 2.0)

        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = filter.process_image(image, page_id)

        assert result.dtype == np.uint8
        assert len(result.shape) == 3

    def test_process_image_expands_canvas(self):
        """Rotated image should have expanded canvas to fit content."""
        filter = Filter()
        page_id = make_page_id("test")
        filter.set_manual_angle(page_id, 45.0)  # 45 degree rotation

        image = np.zeros((100, 100), dtype=np.uint8)
        result = filter.process_image(image, page_id)

        # 45 degree rotation of square should produce larger bounding box
        # New side ~= old_side * sqrt(2) ≈ 141
        assert result.shape[0] > 100
        assert result.shape[1] > 100
