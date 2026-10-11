"""Tests for the mixed layer choice, ink colouring and original background."""

import numpy as np
import pytest

from scantailor.filters.output.binarization import ColorSegmenterOptions
from scantailor.filters.output.color_options import (
    ColorCommonOptions,
    PosterizationOptions,
)
from scantailor.filters.output.mixed import (
    background_lightness,
    build_mixed_layer,
    colorize_ink,
    compose_mixed_colored,
    compose_over_original,
)
from scantailor.filters.output.render_params import RenderParams
from scantailor.imageproc.filters import wiener_color_filter

_SHAPE = (40, 60)
_WIENER = ColorCommonOptions(wiener_coef=0.6, wiener_window_size=5)


def _noisy_page(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, (*_SHAPE, 3), dtype=np.uint8)


def _render(*, binarize: bool, color_norm: bool, bin_norm: bool) -> RenderParams:
    return RenderParams(
        need_binarization=binarize,
        mixed_output=True,
        normalize_illumination_color=color_norm,
        normalize_illumination=bin_norm,
    )


class TestBuildMixedLayer:
    """With binarisation on, the layer depends on both normalisation flags."""

    @pytest.fixture
    def original(self):
        return _noisy_page(1)

    @pytest.fixture
    def normalized(self):
        return np.clip(_noisy_page(1).astype(int) + 20, 0, 255).astype(np.uint8)

    @pytest.mark.parametrize("bin_norm", [False, True])
    def test_colour_normalisation_off_gives_the_untouched_original(
        self, original, normalized, bin_norm
    ):
        render = _render(binarize=True, color_norm=False, bin_norm=bin_norm)
        layer = build_mixed_layer(original, normalized, render, _WIENER)
        np.testing.assert_array_equal(layer, original)

    def test_both_normalisations_on_gives_the_filtered_normalised_page(
        self, original, normalized
    ):
        render = _render(binarize=True, color_norm=True, bin_norm=True)
        layer = build_mixed_layer(original, normalized, render, _WIENER)
        np.testing.assert_array_equal(layer, wiener_color_filter(normalized, 5, 0.6))
        assert not np.array_equal(layer, normalized)

    def test_only_colour_normalisation_on_filters_the_original(
        self, original, normalized
    ):
        render = _render(binarize=True, color_norm=True, bin_norm=False)
        layer = build_mixed_layer(original, normalized, render, _WIENER)
        np.testing.assert_array_equal(layer, wiener_color_filter(original, 5, 0.6))

    def test_without_binarisation_the_layer_is_always_filtered(
        self, original, normalized
    ):
        plain = build_mixed_layer(
            original,
            normalized,
            _render(binarize=False, color_norm=False, bin_norm=False),
            _WIENER,
        )
        flat = build_mixed_layer(
            original,
            normalized,
            _render(binarize=False, color_norm=False, bin_norm=True),
            _WIENER,
        )
        np.testing.assert_array_equal(plain, wiener_color_filter(original, 5, 0.6))
        np.testing.assert_array_equal(flat, wiener_color_filter(normalized, 5, 0.6))

    def test_missing_normalised_page_is_rejected_when_needed(self, original):
        render = _render(binarize=True, color_norm=True, bin_norm=True)
        with pytest.raises(ValueError, match="normalized"):
            build_mixed_layer(original, None, render, _WIENER)

    def test_normalised_page_is_not_needed_when_the_layer_is_the_original(
        self, original
    ):
        render = _render(binarize=True, color_norm=False, bin_norm=True)
        np.testing.assert_array_equal(
            build_mixed_layer(original, None, render, _WIENER), original
        )

    def test_inputs_are_not_modified(self, original, normalized):
        before = original.copy(), normalized.copy()
        render = _render(binarize=True, color_norm=True, bin_norm=True)
        build_mixed_layer(original, normalized, render, _WIENER)
        np.testing.assert_array_equal(original, before[0])
        np.testing.assert_array_equal(normalized, before[1])


class TestBackgroundLightness:
    def test_gray_level_is_returned_as_is(self):
        assert background_lightness(200) == 200

    def test_colour_is_the_mean_of_extreme_channels_rounded_up(self):
        assert background_lightness((10, 20, 31)) == 21

    def test_alpha_is_ignored(self):
        assert background_lightness((100, 100, 100, 0)) == 100


def _inked_page():
    image = np.full((50, 80, 3), 250, dtype=np.uint8)
    ink = np.zeros((50, 80), dtype=bool)
    image[10:30, 10:40] = (40, 40, 190)
    image[10:30, 50:70] = (190, 40, 40)
    ink[10:30, 10:40] = ink[10:30, 50:70] = True
    return image, ink


class TestColorizeInk:
    def test_ink_takes_original_colours_on_white_paper(self):
        image, ink = _inked_page()
        result = colorize_ink(
            image, ink, (300, 300), ColorSegmenterOptions(enabled=True)
        )
        assert tuple(result[20, 20]) == (40, 40, 190)
        assert tuple(result[20, 60]) == (190, 40, 40)
        assert (result[~ink] == 255).all()

    def test_disabled_posterization_leaves_colours_exact(self):
        image, ink = _inked_page()
        result = colorize_ink(
            image,
            ink,
            (300, 300),
            ColorSegmenterOptions(enabled=True),
            PosterizationOptions(enabled=False),
        )
        assert tuple(result[20, 20]) == (40, 40, 190)

    def test_posterization_reduces_the_colour_count(self):
        rng = np.random.default_rng(5)
        image = np.full((60, 90, 3), 255, dtype=np.uint8)
        ink = np.zeros((60, 90), dtype=bool)
        image[10:50, 10:80] = rng.integers(0, 90, (40, 70, 3), dtype=np.uint8)
        ink[10:50, 10:80] = True
        options = ColorSegmenterOptions(enabled=True)
        plain = colorize_ink(image, ink, (300, 300), options)
        posterized = colorize_ink(
            image,
            ink,
            (300, 300),
            options,
            PosterizationOptions(enabled=True, level=2),
            background_color=(250, 250, 250),
        )

        def distinct(img):
            return len(np.unique(img.reshape(-1, 3), axis=0))

        assert posterized.shape == plain.shape
        assert distinct(posterized) <= distinct(plain)
        assert distinct(posterized) <= 9

    def test_threshold_adjustments_reach_the_segmenter(self):
        image = np.full((50, 80, 3), 255, dtype=np.uint8)
        ink = np.zeros((50, 80), dtype=bool)
        image[10:30, 10:40] = (30, 30, 30)
        image[10:30, 40:70] = (30, 200, 30)
        ink[10:30, 10:70] = True

        plain = colorize_ink(
            image, ink, (300, 300), ColorSegmenterOptions(enabled=True)
        )
        merged = colorize_ink(
            image,
            ink,
            (300, 300),
            ColorSegmenterOptions(enabled=True, green_threshold_adjustment=255),
        )

        assert tuple(plain[20, 20]) != tuple(plain[20, 60])
        assert tuple(merged[20, 20]) == tuple(merged[20, 60])


class TestCompositions:
    def test_coloured_text_replaces_the_text_region_and_pictures_are_reserved(self):
        layer = np.full((6, 8, 3), 100, dtype=np.uint8)
        layer[0, 0] = 0
        layer[0, 1] = 255
        text_region = np.zeros((6, 8), dtype=bool)
        text_region[3:, :] = True
        colored = np.full_like(layer, 255)
        colored[4, 2] = (10, 20, 30)
        result = compose_mixed_colored(layer, text_region, colored)
        assert tuple(result[4, 2]) == (10, 20, 30)
        assert tuple(result[3, 0]) == (255, 255, 255)
        assert tuple(result[0, 0]) == (1, 1, 1)
        assert tuple(result[0, 1]) == (254, 254, 254)
        assert tuple(result[1, 5]) == (100, 100, 100)

    def test_original_background_keeps_the_page_and_draws_black_ink(self):
        layer = np.full((6, 8), 180, dtype=np.uint8)
        layer[0, 0], layer[0, 1] = 0, 255
        ink = np.zeros((6, 8), dtype=bool)
        ink[3, 3] = True
        result = compose_over_original(layer, ink)
        assert result[3, 3] == 0
        assert result[2, 2] == 180
        assert result[0, 0] == 1
        assert result[0, 1] == 254

    def test_original_background_with_colours_skips_white_ink_pixels(self):
        layer = np.full((4, 6, 3), 120, dtype=np.uint8)
        ink = np.zeros((4, 6), dtype=bool)
        ink[1, 1:3] = True
        colored = np.full_like(layer, 255)
        colored[1, 1] = (5, 6, 7)
        result = compose_over_original(layer, ink, colored)
        assert tuple(result[1, 1]) == (5, 6, 7)
        assert tuple(result[1, 2]) == (120, 120, 120)

    def test_original_background_keeps_alpha(self):
        layer = np.full((3, 3, 4), 90, dtype=np.uint8)
        layer[..., 3] = 200
        ink = np.zeros((3, 3), dtype=bool)
        ink[1, 1] = True
        result = compose_over_original(layer, ink)
        assert tuple(result[1, 1]) == (0, 0, 0, 200)

    def test_layer_is_not_modified(self):
        layer = np.zeros((3, 3), dtype=np.uint8)
        compose_over_original(layer, np.ones((3, 3), dtype=bool))
        compose_over_original(layer, np.zeros((3, 3), dtype=bool))
        assert (layer == 0).all()
