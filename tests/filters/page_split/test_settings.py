"""Tests for page split Settings class."""

from pathlib import Path

from scantailor.core import ImageId, PageId, SubPage
from scantailor.filters.page_split import LayoutType, Params, Settings


def make_page_id(name: str = "test.tiff") -> PageId:
    """Create a test PageId."""
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}")),
        sub_page=SubPage.SINGLE_PAGE,
    )


class TestSettings:
    """Tests for Settings class."""

    def test_default_params(self):
        """Get default params for unknown page."""
        settings = Settings()
        page_id = make_page_id()
        params = settings.get_params(page_id)
        assert params.layout_type == LayoutType.AUTO_LAYOUT_TYPE

    def test_set_and_get_params(self):
        """Set and retrieve params."""
        settings = Settings()
        page_id = make_page_id()
        params = Params(layout_type=LayoutType.TWO_PAGES)
        settings.set_params(page_id, params)
        assert settings.get_params(page_id) == params

    def test_is_params_set(self):
        """Check if params have been set."""
        settings = Settings()
        page_id = make_page_id()
        assert settings.is_params_set(page_id) is False

        settings.set_params(page_id, Params())
        assert settings.is_params_set(page_id) is True

    def test_clear_params(self):
        """Clear params for a page."""
        settings = Settings()
        page_id = make_page_id()
        settings.set_params(page_id, Params(layout_type=LayoutType.TWO_PAGES))
        assert settings.is_params_set(page_id) is True

        settings.clear_params(page_id)
        assert settings.is_params_set(page_id) is False

    def test_clear_params_nonexistent(self):
        """Clear params for page that doesn't exist is no-op."""
        settings = Settings()
        page_id = make_page_id()
        settings.clear_params(page_id)  # Should not raise

    def test_default_layout_type(self):
        """Get and set default layout type."""
        settings = Settings()
        assert settings.get_default_layout_type() == LayoutType.AUTO_LAYOUT_TYPE

        settings.set_default_layout_type(LayoutType.TWO_PAGES)
        assert settings.get_default_layout_type() == LayoutType.TWO_PAGES

        # New pages get the new default
        page_id = make_page_id()
        params = settings.get_params(page_id)
        assert params.layout_type == LayoutType.TWO_PAGES

    def test_apply_params_to_pages(self):
        """Apply same params to multiple pages."""
        settings = Settings()
        page_ids = [make_page_id(f"test{i}.tiff") for i in range(3)]
        params = Params(layout_type=LayoutType.SINGLE_PAGE_UNCUT)

        settings.apply_params_to_pages(page_ids, params)

        for page_id in page_ids:
            assert settings.get_params(page_id) == params

    def test_clear_all(self):
        """Clear all stored params."""
        settings = Settings()
        page_ids = [make_page_id(f"test{i}.tiff") for i in range(3)]
        params = Params(layout_type=LayoutType.TWO_PAGES)

        for page_id in page_ids:
            settings.set_params(page_id, params)

        settings.clear_all()

        for page_id in page_ids:
            assert settings.is_params_set(page_id) is False

    def test_different_pages_independent(self):
        """Different pages have independent params."""
        settings = Settings()
        page1 = make_page_id("page1.tiff")
        page2 = make_page_id("page2.tiff")

        settings.set_params(page1, Params(layout_type=LayoutType.TWO_PAGES))
        settings.set_params(page2, Params(layout_type=LayoutType.SINGLE_PAGE_UNCUT))

        assert settings.get_params(page1).layout_type == LayoutType.TWO_PAGES
        assert settings.get_params(page2).layout_type == LayoutType.SINGLE_PAGE_UNCUT
