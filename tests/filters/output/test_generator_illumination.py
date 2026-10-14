"""Illumination normalisation, grayscale/colour output and margin filling."""

from typing import ClassVar

import numpy as np
import pytest

from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import ColorCommonOptions, FillingColor
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params


def _uneven_gray_page() -> np.ndarray:
    """Gray page with a horizontal lighting falloff and rows of dark text."""
    height, width = 300, 240
    xs = np.linspace(0.0, 1.0, width)[None, :]
    falloff = 1.0 - 0.4 * xs
    page = np.full((height, width), 230.0)
    for y in range(40, 260, 8):
        page[y : y + 3, 30:210] = 40.0
    return np.clip(page * falloff, 0, 255).astype(np.uint8)


def _gray_params(**kwargs) -> Params:
    return Params(color_mode=ColorMode.COLOR_GRAYSCALE, **kwargs)


class TestGrayscaleOutput:
    """Grayscale output with and without normalisation."""

    def test_defaults_leave_pixels_unchanged(self):
        page = _uneven_gray_page()

        result = generate_output(page, _gray_params())

        np.testing.assert_array_equal(result.image, page)
        assert result.is_binary is False

    def test_output_is_a_copy_of_the_input(self):
        page = _uneven_gray_page()

        result = generate_output(page, _gray_params())

        assert result.image is not page
        assert not np.shares_memory(result.image, page)

    def test_normalisation_changes_the_image_and_evens_out_paper(self):
        page = _uneven_gray_page()
        options = ColorCommonOptions(normalize_illumination=True)

        result = generate_output(page, _gray_params(), color_options=options)

        left = result.image[:, 5:25][::8].mean()
        right = result.image[:, 215:235][::8].mean()
        assert not np.array_equal(result.image, page)
        assert abs(left - right) < 10
        assert abs(float(page[:, 5:25].mean()) - float(page[:, 215:235].mean())) > 40

    def test_normalisation_preserves_shape_and_dtype_and_input(self):
        page = _uneven_gray_page()
        original = page.copy()
        options = ColorCommonOptions(normalize_illumination=True)

        result = generate_output(page, _gray_params(), color_options=options)

        assert result.image.shape == page.shape
        assert result.image.dtype == np.uint8
        np.testing.assert_array_equal(page, original)

    def test_white_on_black_page_is_unchanged_without_processing(self):
        page = _uneven_gray_page()

        result = generate_output(page, _gray_params(black_on_white=False))

        np.testing.assert_array_equal(result.image, page)

    def test_white_on_black_page_is_normalised_in_inverted_space(self):
        page = _uneven_gray_page()
        options = ColorCommonOptions(normalize_illumination=True)

        expected = generate_output(page, _gray_params(), color_options=options)
        result = generate_output(
            255 - page, _gray_params(black_on_white=False), color_options=options
        )

        np.testing.assert_array_equal(result.image, 255 - expected.image)


class TestColorOutput:
    """Colour output keeps chroma while normalising brightness."""

    @staticmethod
    def _color_page() -> np.ndarray:
        gray = _uneven_gray_page().astype(np.float64)
        tint = np.array([0.85, 1.0, 0.9])  # BGR tint on the paper
        return np.clip(gray[..., None] * tint, 0, 255).astype(np.uint8)

    def test_colour_defaults_leave_pixels_unchanged(self):
        page = self._color_page()

        result = generate_output(page, _gray_params())

        np.testing.assert_array_equal(result.image, page)

    def test_colour_normalisation_keeps_three_channels_and_lifts_dark_side(self):
        page = self._color_page()
        options = ColorCommonOptions(normalize_illumination=True)

        result = generate_output(page, _gray_params(), color_options=options)

        assert result.image.shape == page.shape
        assert result.image.dtype == np.uint8
        assert result.image[:, 215:235].mean() > page[:, 215:235].mean() + 20

    def test_colour_normalisation_preserves_alpha_channel(self):
        rgb = self._color_page()
        alpha = np.full((*rgb.shape[:2], 1), 77, dtype=np.uint8)
        page = np.concatenate([rgb, alpha], axis=2)
        options = ColorCommonOptions(normalize_illumination=True)

        result = generate_output(page, _gray_params(), color_options=options)

        assert result.image.shape == page.shape
        assert (result.image[..., 3] == 77).all()

    def test_white_on_black_colour_page_keeps_channels_and_alpha(self):
        rgb = self._color_page()
        alpha = np.full((*rgb.shape[:2], 1), 77, dtype=np.uint8)
        page = np.concatenate([rgb, alpha], axis=2)

        result = generate_output(page, _gray_params(black_on_white=False))

        np.testing.assert_array_equal(result.image, page)


class TestMarginFilling:
    """Content polygon and filling colour."""

    POLYGON: ClassVar[list[tuple[float, float]]] = [
        (20.0, 20.0),
        (100.0, 20.0),
        (100.0, 80.0),
        (20.0, 80.0),
    ]

    def _page(self) -> np.ndarray:
        return np.full((100, 120), 150, dtype=np.uint8)

    def test_white_filling_colour_whitens_margins_only(self):
        options = ColorCommonOptions(filling_color=FillingColor.WHITE)

        result = generate_output(
            self._page(),
            _gray_params(),
            color_options=options,
            content_polygon=self.POLYGON,
        )

        assert result.image[5, 5] == 255
        assert result.image[50, 60] == 150

    def test_black_filling_colour_stays_black_on_white_on_black_output(self):
        options = ColorCommonOptions(filling_color=FillingColor.BLACK)

        result = generate_output(
            self._page(),
            _gray_params(black_on_white=False),
            color_options=options,
            content_polygon=self.POLYGON,
        )

        assert result.image[5, 5] == 0

    def test_background_filling_colour_uses_the_dominant_content_value(self):
        page = self._page()
        page[30:40, 30:40] = 10
        options = ColorCommonOptions(filling_color=FillingColor.BACKGROUND)

        result = generate_output(
            page, _gray_params(), color_options=options, content_polygon=self.POLYGON
        )

        assert result.image[5, 5] == 150

    def test_fill_margins_disabled_keeps_margin_pixels(self):
        page = self._page()
        page[5, 5] = 33
        options = ColorCommonOptions(
            fill_margins=False, filling_color=FillingColor.WHITE
        )

        result = generate_output(
            page, _gray_params(), color_options=options, content_polygon=self.POLYGON
        )

        assert result.image[5, 5] == 33

    def test_no_polygon_means_no_margin_fill(self):
        options = ColorCommonOptions(filling_color=FillingColor.WHITE)

        result = generate_output(self._page(), _gray_params(), color_options=options)

        assert (result.image == 150).all()

    def test_colour_margin_fill_sets_every_channel(self):
        page = np.full((100, 120, 3), 120, dtype=np.uint8)
        options = ColorCommonOptions(filling_color=FillingColor.WHITE)

        result = generate_output(
            page, _gray_params(), color_options=options, content_polygon=self.POLYGON
        )

        assert (result.image[5, 5] == 255).all()
        assert (result.image[50, 60] == 120).all()

    def test_binary_output_margins_follow_filling_colour(self):
        page = np.full((100, 120), 200, dtype=np.uint8)
        page[40:60, 40:60] = 20
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(normalize_illumination=False),
        )

        white = generate_output(
            page,
            params,
            color_options=ColorCommonOptions(filling_color=FillingColor.WHITE),
            content_polygon=self.POLYGON,
        )
        black = generate_output(
            page,
            params,
            color_options=ColorCommonOptions(filling_color=FillingColor.BLACK),
            content_polygon=self.POLYGON,
        )

        assert white.image[5, 5] == 255
        assert black.image[5, 5] == 0


class TestBinaryIlluminationNormalisation:
    """Normalisation before binarisation keeps dense text."""

    def test_dense_text_survives_default_binarisation(self):
        page = _uneven_gray_page()
        for y in range(40, 260, 8):
            page[y + 4 : y + 7, 30:210] = 60
        params = Params(color_mode=ColorMode.BLACK_AND_WHITE)

        result = generate_output(page, params)

        assert result.is_binary is True
        ink_pixels = int(np.count_nonzero(result.image == 0))
        text_pixels = int(np.count_nonzero(page < 100))
        assert ink_pixels >= 0.7 * text_pixels

    @pytest.mark.parametrize("normalize", [True, False])
    def test_binary_output_is_two_valued(self, normalize):
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(normalize_illumination=normalize),
        )

        result = generate_output(_uneven_gray_page(), params)

        assert set(np.unique(result.image)) <= {0, 255}

    def test_white_on_black_page_keeps_white_text_on_black(self):
        page = _uneven_gray_page()
        for y in range(40, 260, 8):
            page[y + 4 : y + 7, 30:210] = 60
        params = Params(color_mode=ColorMode.BLACK_AND_WHITE, black_on_white=False)

        result = generate_output(255 - page, params)
        reference = generate_output(page, Params(color_mode=ColorMode.BLACK_AND_WHITE))

        white_pixels = int(np.count_nonzero(result.image == 255))
        reference_ink = int(np.count_nonzero(reference.image == 0))
        assert white_pixels == pytest.approx(reference_ink, rel=0.05)
