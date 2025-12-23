"""Tests for page split Params class."""

import pytest

from scantailor.filters.page_split import (
    AutoManualMode,
    LayoutType,
    PageLayout,
    Params,
)


class TestParams:
    """Tests for Params class."""

    def test_default_params(self):
        """Default params have expected values."""
        params = Params()
        assert params.layout_type == LayoutType.AUTO_LAYOUT_TYPE
        assert params.page_layout is None
        assert params.split_line_mode == AutoManualMode.AUTO

    def test_with_layout_type(self):
        """Create params with different layout type."""
        params = Params()
        new_params = params.with_layout_type(LayoutType.TWO_PAGES)
        assert new_params.layout_type == LayoutType.TWO_PAGES
        # Original unchanged
        assert params.layout_type == LayoutType.AUTO_LAYOUT_TYPE

    def test_with_page_layout(self):
        """Create params with page layout."""
        params = Params()
        layout = PageLayout.single_page_uncut(100, 100)
        new_params = params.with_page_layout(layout)
        assert new_params.page_layout == layout
        assert new_params.split_line_mode == AutoManualMode.AUTO

    def test_with_manual_layout(self):
        """Create params with manual layout."""
        params = Params()
        layout = PageLayout.two_pages(200, 100, split_x=100)
        new_params = params.with_manual_layout(layout)
        assert new_params.page_layout == layout
        assert new_params.split_line_mode == AutoManualMode.MANUAL

    def test_with_auto_layout(self):
        """Create params with auto layout."""
        params = Params(split_line_mode=AutoManualMode.MANUAL)
        layout = PageLayout.single_page_uncut(100, 100)
        new_params = params.with_auto_layout(layout)
        assert new_params.page_layout == layout
        assert new_params.split_line_mode == AutoManualMode.AUTO

    def test_is_auto(self):
        """Check if params is in auto mode."""
        params = Params()
        assert params.is_auto() is True

        manual_params = Params(split_line_mode=AutoManualMode.MANUAL)
        assert manual_params.is_auto() is False

    def test_is_manual(self):
        """Check if params is in manual mode."""
        params = Params()
        assert params.is_manual() is False

        manual_params = Params(split_line_mode=AutoManualMode.MANUAL)
        assert manual_params.is_manual() is True

    def test_frozen_model(self):
        """Params is immutable."""
        params = Params()
        with pytest.raises(Exception):
            params.layout_type = LayoutType.TWO_PAGES  # type: ignore[misc]
