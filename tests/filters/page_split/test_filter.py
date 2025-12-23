"""Tests for page split Filter class."""

from pathlib import Path

import numpy as np

from scantailor.core import ImageId, PageId, SubPage
from scantailor.filters.page_split import (
    AutoManualMode,
    Filter,
    LayoutType,
    PageLayoutType,
    Params,
    Settings,
)


def make_page_id(name: str = "test.tiff") -> PageId:
    """Create a test PageId."""
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}")),
        sub_page=SubPage.SINGLE_PAGE,
    )


class TestFilter:
    """Tests for Filter class."""

    def test_init_default_settings(self):
        """Filter creates default settings if none provided."""
        filter = Filter()
        assert filter.settings is not None

    def test_init_with_settings(self):
        """Filter uses provided settings."""
        settings = Settings()
        filter = Filter(settings)
        assert filter.settings is settings

    def test_name(self):
        """Filter has correct name."""
        filter = Filter()
        assert filter.name == "Page Split"

    def test_get_default_params(self):
        """Get default params for unknown page."""
        filter = Filter()
        page_id = make_page_id()
        params = filter.get_params(page_id)
        assert params.layout_type == LayoutType.AUTO_LAYOUT_TYPE

    def test_set_params(self):
        """Set and get params."""
        filter = Filter()
        page_id = make_page_id()
        params = Params(layout_type=LayoutType.TWO_PAGES)
        filter.set_params(page_id, params)
        assert filter.get_params(page_id) == params

    def test_is_params_set(self):
        """Check if params have been set."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.is_params_set(page_id) is False

        filter.set_params(page_id, Params())
        assert filter.is_params_set(page_id) is True

    def test_get_layout_type(self):
        """Get layout type for page."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.get_layout_type(page_id) == LayoutType.AUTO_LAYOUT_TYPE

    def test_set_layout_type(self):
        """Set layout type for page."""
        filter = Filter()
        page_id = make_page_id()
        params = filter.set_layout_type(page_id, LayoutType.TWO_PAGES)
        assert params.layout_type == LayoutType.TWO_PAGES
        assert filter.get_layout_type(page_id) == LayoutType.TWO_PAGES

    def test_get_page_layout_none_initially(self):
        """Page layout is None before detection."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.get_page_layout(page_id) is None

    def test_detect_split(self):
        """Detect split stores result."""
        filter = Filter()
        page_id = make_page_id()
        image = np.zeros((100, 100), dtype=np.uint8)

        result = filter.detect_split(image, page_id)

        assert result.layout is not None
        assert filter.get_page_layout(page_id) is not None

    def test_detect_split_no_store(self):
        """Detect split without storing."""
        filter = Filter()
        page_id = make_page_id()
        image = np.zeros((100, 100), dtype=np.uint8)

        result = filter.detect_split(image, page_id, store_result=False)

        assert result.layout is not None
        assert filter.get_page_layout(page_id) is None

    def test_set_manual_split(self):
        """Set manual split line."""
        filter = Filter()
        page_id = make_page_id()

        params = filter.set_manual_split(page_id, split_x=100, width=200, height=100)

        assert params.split_line_mode == AutoManualMode.MANUAL
        assert params.layout_type == LayoutType.TWO_PAGES
        assert params.page_layout is not None
        assert params.page_layout.get_split_line_x() == 100

    def test_set_manual_cutters(self):
        """Set manual cutter lines."""
        filter = Filter()
        page_id = make_page_id()

        params = filter.set_manual_cutters(
            page_id, left_x=20, right_x=180, width=200, height=100
        )

        assert params.split_line_mode == AutoManualMode.MANUAL
        assert params.layout_type == LayoutType.PAGE_PLUS_OFFCUT
        assert params.page_layout is not None
        assert params.page_layout.layout_type == PageLayoutType.SINGLE_PAGE_CUT

    def test_reset_to_auto(self):
        """Reset page to auto mode."""
        filter = Filter()
        page_id = make_page_id()

        # Set manual first
        filter.set_manual_split(page_id, split_x=100, width=200, height=100)
        assert filter.get_params(page_id).is_manual()

        # Reset to auto
        filter.reset_to_auto(page_id)
        params = filter.get_params(page_id)
        assert params.is_auto()
        assert params.layout_type == LayoutType.AUTO_LAYOUT_TYPE

    def test_apply_to_pages(self):
        """Apply params to multiple pages."""
        filter = Filter()
        page_ids = [make_page_id(f"test{i}.tiff") for i in range(3)]
        params = Params(layout_type=LayoutType.TWO_PAGES)

        filter.apply_to_pages(page_ids, params)

        for page_id in page_ids:
            assert filter.get_params(page_id) == params

    def test_process_auto_mode(self):
        """Process in auto mode detects split."""
        filter = Filter()
        page_id = make_page_id()
        image = np.zeros((100, 100), dtype=np.uint8)

        params = filter.process(image, page_id)

        assert params.page_layout is not None

    def test_process_manual_mode_preserves_layout(self):
        """Process in manual mode preserves existing layout."""
        filter = Filter()
        page_id = make_page_id()
        image = np.zeros((100, 100), dtype=np.uint8)

        # Set manual layout first
        filter.set_manual_split(page_id, split_x=50, width=100, height=100)
        original_layout = filter.get_page_layout(page_id)

        # Process should not change manual layout
        params = filter.process(image, page_id)

        assert params.page_layout == original_layout

    def test_get_sub_page_count_default(self):
        """Default sub-page count is 1."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.get_sub_page_count(page_id) == 1

    def test_get_sub_page_count_two_pages(self):
        """Two pages layout has count 2."""
        filter = Filter()
        page_id = make_page_id()
        filter.set_manual_split(page_id, split_x=100, width=200, height=100)
        assert filter.get_sub_page_count(page_id) == 2

    def test_detect_split_respects_layout_type(self):
        """Detection respects the set layout type."""
        filter = Filter()
        page_id = make_page_id()
        image = np.zeros((100, 100), dtype=np.uint8)

        # Set to single page uncut
        filter.set_layout_type(page_id, LayoutType.SINGLE_PAGE_UNCUT)
        result = filter.detect_split(image, page_id)

        assert result.layout.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT
