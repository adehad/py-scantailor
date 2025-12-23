"""Tests for output Settings class."""

from pathlib import Path

from scantailor.core import Dpi, ImageId, PageId, SubPage
from scantailor.filters.output import ColorMode, DespeckleLevel, Params, Settings


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
        assert params.output_dpi == Dpi.uniform(600)
        assert params.color_mode == ColorMode.BLACK_AND_WHITE
        assert params.despeckle_level == DespeckleLevel.NORMAL

    def test_set_and_get_params(self):
        """Set and retrieve params."""
        settings = Settings()
        page_id = make_page_id()
        params = Params(color_mode=ColorMode.COLOR_GRAYSCALE)
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
        settings.set_params(page_id, Params(color_mode=ColorMode.MIXED))
        assert settings.is_params_set(page_id) is True

        settings.clear_params(page_id)
        assert settings.is_params_set(page_id) is False

    def test_default_dpi(self):
        """Get and set default DPI."""
        settings = Settings()
        assert settings.get_default_dpi() == Dpi.uniform(600)

        settings.set_default_dpi(Dpi.uniform(300))
        assert settings.get_default_dpi() == Dpi.uniform(300)

        # New pages get the new default
        page_id = make_page_id()
        params = settings.get_params(page_id)
        assert params.output_dpi == Dpi.uniform(300)

    def test_default_color_mode(self):
        """Get and set default color mode."""
        settings = Settings()
        assert settings.get_default_color_mode() == ColorMode.BLACK_AND_WHITE

        settings.set_default_color_mode(ColorMode.COLOR_GRAYSCALE)
        assert settings.get_default_color_mode() == ColorMode.COLOR_GRAYSCALE

    def test_default_despeckle_level(self):
        """Get and set default despeckle level."""
        settings = Settings()
        assert settings.get_default_despeckle_level() == DespeckleLevel.NORMAL

        settings.set_default_despeckle_level(DespeckleLevel.AGGRESSIVE)
        assert settings.get_default_despeckle_level() == DespeckleLevel.AGGRESSIVE

    def test_apply_params_to_pages(self):
        """Apply same params to multiple pages."""
        settings = Settings()
        page_ids = [make_page_id(f"test{i}.tiff") for i in range(3)]
        params = Params(color_mode=ColorMode.MIXED)

        settings.apply_params_to_pages(page_ids, params)

        for page_id in page_ids:
            assert settings.get_params(page_id) == params

    def test_clear_all(self):
        """Clear all stored params."""
        settings = Settings()
        page_ids = [make_page_id(f"test{i}.tiff") for i in range(3)]
        params = Params(color_mode=ColorMode.MIXED)

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

        settings.set_params(page1, Params(color_mode=ColorMode.BLACK_AND_WHITE))
        settings.set_params(page2, Params(color_mode=ColorMode.COLOR_GRAYSCALE))

        assert settings.get_params(page1).color_mode == ColorMode.BLACK_AND_WHITE
        assert settings.get_params(page2).color_mode == ColorMode.COLOR_GRAYSCALE
