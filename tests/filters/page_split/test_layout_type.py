"""Tests for LayoutType enum."""

from scantailor.filters.page_split.layout_type import LayoutType


class TestLayoutType:
    """Tests for LayoutType enum."""

    def test_auto_layout_type_value(self):
        """Auto layout type has correct string value."""
        assert LayoutType.AUTO_LAYOUT_TYPE.value == "auto-detect"

    def test_single_page_uncut_value(self):
        """Single page uncut has correct string value."""
        assert LayoutType.SINGLE_PAGE_UNCUT.value == "single-uncut"

    def test_page_plus_offcut_value(self):
        """Page plus offcut has correct string value."""
        assert LayoutType.PAGE_PLUS_OFFCUT.value == "single-cut"

    def test_two_pages_value(self):
        """Two pages has correct string value."""
        assert LayoutType.TWO_PAGES.value == "two-pages"

    def test_from_string_auto(self):
        """Parse auto-detect string."""
        assert LayoutType.from_string("auto-detect") == LayoutType.AUTO_LAYOUT_TYPE

    def test_from_string_single_uncut(self):
        """Parse single-uncut string."""
        assert LayoutType.from_string("single-uncut") == LayoutType.SINGLE_PAGE_UNCUT

    def test_from_string_single_cut(self):
        """Parse single-cut string."""
        assert LayoutType.from_string("single-cut") == LayoutType.PAGE_PLUS_OFFCUT

    def test_from_string_two_pages(self):
        """Parse two-pages string."""
        assert LayoutType.from_string("two-pages") == LayoutType.TWO_PAGES

    def test_from_string_unknown_defaults_to_auto(self):
        """Unknown string defaults to auto."""
        assert LayoutType.from_string("unknown") == LayoutType.AUTO_LAYOUT_TYPE

    def test_str_representation(self):
        """String representation returns value."""
        assert str(LayoutType.TWO_PAGES) == "two-pages"
