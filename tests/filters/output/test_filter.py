"""Tests for output Filter class."""

from pathlib import Path

import numpy as np

from scantailor.core import Dpi, ImageId, PageId, SubPage
from scantailor.filters.output import (
    BinarizationMethod,
    BinarizationOptions,
    ColorMode,
    DespeckleLevel,
    Filter,
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
        assert filter.name == "Output"

    def test_get_default_params(self):
        """Get default params for unknown page."""
        filter = Filter()
        page_id = make_page_id()
        params = filter.get_params(page_id)
        assert params.output_dpi == Dpi.uniform(600)
        assert params.color_mode == ColorMode.BLACK_AND_WHITE

    def test_set_params(self):
        """Set and get params."""
        filter = Filter()
        page_id = make_page_id()
        params = Params(color_mode=ColorMode.MIXED)
        filter.set_params(page_id, params)
        assert filter.get_params(page_id) == params

    def test_is_params_set(self):
        """Check if params have been set."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.is_params_set(page_id) is False

        filter.set_params(page_id, Params())
        assert filter.is_params_set(page_id) is True

    def test_get_output_dpi(self):
        """Get output DPI for page."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.get_output_dpi(page_id) == Dpi.uniform(600)

    def test_set_output_dpi(self):
        """Set output DPI for page."""
        filter = Filter()
        page_id = make_page_id()
        params = filter.set_output_dpi(page_id, Dpi.uniform(300))
        assert params.output_dpi == Dpi.uniform(300)
        assert filter.get_output_dpi(page_id) == Dpi.uniform(300)

    def test_get_color_mode(self):
        """Get color mode for page."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.get_color_mode(page_id) == ColorMode.BLACK_AND_WHITE

    def test_set_color_mode(self):
        """Set color mode for page."""
        filter = Filter()
        page_id = make_page_id()
        params = filter.set_color_mode(page_id, ColorMode.COLOR_GRAYSCALE)
        assert params.color_mode == ColorMode.COLOR_GRAYSCALE
        assert filter.get_color_mode(page_id) == ColorMode.COLOR_GRAYSCALE

    def test_get_despeckle_level(self):
        """Get despeckle level for page."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.get_despeckle_level(page_id) == DespeckleLevel.NORMAL

    def test_set_despeckle_level(self):
        """Set despeckle level for page."""
        filter = Filter()
        page_id = make_page_id()
        params = filter.set_despeckle_level(page_id, DespeckleLevel.AGGRESSIVE)
        assert params.despeckle_level == DespeckleLevel.AGGRESSIVE
        assert filter.get_despeckle_level(page_id) == DespeckleLevel.AGGRESSIVE

    def test_get_binarization_method(self):
        """Get binarization method for page."""
        filter = Filter()
        page_id = make_page_id()
        assert filter.get_binarization_method(page_id) == BinarizationMethod.OTSU

    def test_set_binarization_method(self):
        """Set binarization method for page."""
        filter = Filter()
        page_id = make_page_id()
        params = filter.set_binarization_method(page_id, BinarizationMethod.SAUVOLA)
        assert params.binarization.method == BinarizationMethod.SAUVOLA
        assert filter.get_binarization_method(page_id) == BinarizationMethod.SAUVOLA

    def test_set_binarization_options(self):
        """Set binarization options for page."""
        filter = Filter()
        page_id = make_page_id()
        options = BinarizationOptions(
            method=BinarizationMethod.WOLF,
            window_size=300,
        )
        params = filter.set_binarization_options(page_id, options)
        assert params.binarization == options

    def test_apply_to_pages(self):
        """Apply params to multiple pages."""
        filter = Filter()
        page_ids = [make_page_id(f"test{i}.tiff") for i in range(3)]
        params = Params(color_mode=ColorMode.MIXED)

        filter.apply_to_pages(page_ids, params)

        for page_id in page_ids:
            assert filter.get_params(page_id) == params

    def test_process(self):
        """Process an image."""
        filter = Filter()
        page_id = make_page_id()
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200
        image[50:, :] = 50

        result = filter.process(image, page_id)

        assert result.image is not None
        assert result.is_binary is True

    def test_process_with_params(self):
        """Process an image with specific params."""
        filter = Filter()
        image = np.zeros((100, 100), dtype=np.uint8)
        image[25:75, 25:75] = 128

        params = Params(color_mode=ColorMode.COLOR_GRAYSCALE)
        result = filter.process_with_params(image, params)

        assert result.is_binary is False
        np.testing.assert_array_equal(result.image, image)

    def test_process_binary_output(self):
        """Process produces binary output for B&W mode."""
        filter = Filter()
        page_id = make_page_id()
        filter.set_color_mode(page_id, ColorMode.BLACK_AND_WHITE)

        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 200
        image[50:, :] = 50

        result = filter.process(image, page_id)

        assert result.is_binary is True
        unique_values = np.unique(result.image)
        assert len(unique_values) <= 2
