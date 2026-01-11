"""Tests for Select Content filter."""

from pathlib import Path

import numpy as np

from scantailor.core import ImageId, PageId, SubPage
from scantailor.filters.select_content import (
    ContentBox,
    ContentDetectionMode,
    Filter,
    PageBox,
    PageDetectionMode,
    Params,
    Settings,
)


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
        assert filter.name == "Select Content"

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

        assert params.content_box.is_empty()
        assert params.content_detection_mode == ContentDetectionMode.AUTO

    def test_set_params(self):
        """set_params should store parameters."""
        filter = Filter()
        page_id = make_page_id("test")
        params = Params(
            content_box=ContentBox(x=10, y=20, width=100, height=50),
        )

        filter.set_params(page_id, params)

        assert filter.get_params(page_id).content_box.x == 10

    def test_is_params_set(self):
        """is_params_set should reflect storage state."""
        filter = Filter()
        page_id = make_page_id("test")

        assert filter.is_params_set(page_id) is False

        filter.set_params(page_id, Params())
        assert filter.is_params_set(page_id) is True


class TestFilterDetectContent:
    """Tests for content detection functionality."""

    def test_detect_content_returns_result(self):
        """detect_content should return a ContentDetectionResult."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100), dtype=np.uint8)

        result = filter.detect_content(image, page_id)

        assert hasattr(result, "content_box")
        assert hasattr(result, "confidence")

    def test_detect_content_stores_result(self):
        """detect_content should store result in settings."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.ones((100, 100), dtype=np.uint8) * 255
        image[30:70, 30:70] = 0  # Add content

        filter.detect_content(image, page_id)

        # Should have stored something
        assert filter.is_params_set(page_id)

    def test_detect_content_no_store(self):
        """detect_content with store_result=False should not store."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100), dtype=np.uint8)

        filter.detect_content(image, page_id, store_result=False)

        assert filter.is_params_set(page_id) is False

    def test_detect_content_sets_auto_mode(self):
        """detect_content should set AUTO mode."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.ones((100, 100), dtype=np.uint8) * 255
        image[30:70, 30:70] = 0

        filter.detect_content(image, page_id)

        params = filter.get_params(page_id)
        assert params.content_detection_mode == ContentDetectionMode.AUTO

    def test_detect_content_handles_color(self):
        """detect_content should handle color images."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = filter.detect_content(image, page_id)

        assert result is not None


class TestFilterDetectPage:
    """Tests for page detection functionality."""

    def test_detect_page_returns_page_box(self):
        """detect_page should return a PageBox."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100), dtype=np.uint8)

        result = filter.detect_page(image, page_id)

        assert isinstance(result, PageBox)

    def test_detect_page_stores_result(self):
        """detect_page should store result in settings."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100), dtype=np.uint8)

        filter.detect_page(image, page_id)

        assert filter.is_params_set(page_id)

    def test_detect_page_no_store(self):
        """detect_page with store_result=False should not store."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100), dtype=np.uint8)

        filter.detect_page(image, page_id, store_result=False)

        assert filter.is_params_set(page_id) is False

    def test_detect_page_fallback_to_full_image(self):
        """detect_page should fall back to full image when no edges found."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((200, 300), dtype=np.uint8)  # Blank image

        result = filter.detect_page(image, page_id)

        # Should return full image size
        assert result.width == 300
        assert result.height == 200


class TestFilterProcess:
    """Tests for full processing functionality."""

    def test_process_returns_params(self):
        """process should return updated params."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((100, 100), dtype=np.uint8)

        result = filter.process(image, page_id)

        assert isinstance(result, Params)

    def test_process_respects_disabled_page(self):
        """process should use full image when page detection disabled."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((200, 300), dtype=np.uint8)

        # Pre-set to disabled
        filter.set_params(
            page_id, Params(page_detection_mode=PageDetectionMode.DISABLED)
        )

        result = filter.process(image, page_id)

        assert result.page_box.width == 300
        assert result.page_box.height == 200

    def test_process_respects_disabled_content(self):
        """process should use page box when content detection disabled."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((200, 300), dtype=np.uint8)

        # Pre-set to disabled
        filter.set_params(
            page_id,
            Params(content_detection_mode=ContentDetectionMode.DISABLED),
        )

        result = filter.process(image, page_id)

        # Content should equal page (minus any clipping)
        assert result.content_box.width == result.page_box.width

    def test_process_calculates_physical_size(self):
        """process should calculate physical size in mm."""
        filter = Filter()
        page_id = make_page_id("test")
        image = np.zeros((300, 300), dtype=np.uint8)

        result = filter.process(image, page_id, dpi=300.0)

        # At 300 DPI, 300 pixels should be about 25.4 mm
        assert result.content_size_mm.width_mm > 0


class TestFilterManualOperations:
    """Tests for manual content/page operations."""

    def test_set_manual_content(self):
        """set_manual_content should set box and MANUAL mode."""
        filter = Filter()
        page_id = make_page_id("test")
        box = ContentBox(x=10, y=20, width=100, height=50)

        params = filter.set_manual_content(page_id, box)

        assert params.content_box == box
        assert params.content_detection_mode == ContentDetectionMode.MANUAL
        assert filter.get_params(page_id).content_box == box

    def test_set_manual_page(self):
        """set_manual_page should set box and MANUAL mode."""
        filter = Filter()
        page_id = make_page_id("test")
        box = PageBox(x=0, y=0, width=800, height=600)

        params = filter.set_manual_page(page_id, box)

        assert params.page_box == box
        assert params.page_detection_mode == PageDetectionMode.MANUAL

    def test_reset_content_to_auto(self):
        """reset_content_to_auto should set AUTO mode."""
        filter = Filter()
        page_id = make_page_id("test")

        filter.set_manual_content(page_id, ContentBox(x=10, y=10, width=50, height=50))
        filter.reset_content_to_auto(page_id)

        assert (
            filter.get_params(page_id).content_detection_mode
            == ContentDetectionMode.AUTO
        )

    def test_reset_page_to_auto(self):
        """reset_page_to_auto should set AUTO mode."""
        filter = Filter()
        page_id = make_page_id("test")

        filter.set_manual_page(page_id, PageBox(x=0, y=0, width=800, height=600))
        filter.reset_page_to_auto(page_id)

        assert filter.get_params(page_id).page_detection_mode == PageDetectionMode.AUTO

    def test_disable_content_detection(self):
        """disable_content_detection should set DISABLED mode."""
        filter = Filter()
        page_id = make_page_id("test")

        filter.disable_content_detection(page_id)

        assert (
            filter.get_params(page_id).content_detection_mode
            == ContentDetectionMode.DISABLED
        )

    def test_disable_page_detection(self):
        """disable_page_detection should set DISABLED mode."""
        filter = Filter()
        page_id = make_page_id("test")

        filter.disable_page_detection(page_id)

        assert (
            filter.get_params(page_id).page_detection_mode == PageDetectionMode.DISABLED
        )

    def test_get_content_box(self):
        """get_content_box should return the content box."""
        filter = Filter()
        page_id = make_page_id("test")
        box = ContentBox(x=10, y=20, width=100, height=50)
        filter.set_manual_content(page_id, box)

        assert filter.get_content_box(page_id) == box

    def test_get_page_box(self):
        """get_page_box should return the page box."""
        filter = Filter()
        page_id = make_page_id("test")
        box = PageBox(x=0, y=0, width=800, height=600)
        filter.set_manual_page(page_id, box)

        assert filter.get_page_box(page_id) == box


class TestFilterBatchOperations:
    """Tests for batch operations."""

    def test_apply_to_pages(self):
        """apply_to_pages should set params for all pages."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        params = Params(
            content_box=ContentBox(x=20, y=20, width=150, height=150),
            content_detection_mode=ContentDetectionMode.MANUAL,
        )

        filter.apply_to_pages([page1, page2], params)

        assert filter.get_params(page1).content_box.x == 20
        assert filter.get_params(page2).content_box.x == 20

    def test_set_fine_tune(self):
        """set_fine_tune should update fine-tune setting."""
        filter = Filter()
        page_id = make_page_id("test")

        filter.set_fine_tune(page_id, True)

        assert filter.get_params(page_id).fine_tune_corners is True

        filter.set_fine_tune(page_id, False)

        assert filter.get_params(page_id).fine_tune_corners is False
