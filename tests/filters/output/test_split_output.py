"""Tests for split output: foreground and background layers of a mixed page."""

import numpy as np
import pytest
from PIL import Image

from scantailor.core.models import Dpi
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.mixed import compose_mixed, compose_over_original
from scantailor.filters.output.params import Params
from scantailor.filters.output.render_params import RenderParams, SplittingOptions
from scantailor.filters.output.split_output import (
    BACKGROUND_DIR,
    FOREGROUND_DIR,
    ORIGINAL_BACKGROUND_DIR,
    ForegroundType,
    foreground_type_for,
    merge_split_images,
    split_by_foreground_mask,
    split_mixed_output,
    split_with_original_background,
    write_split_images,
)

_H, _W = 60, 90


def _page():
    """Gray page with a picture on the right, ink lines on the left, paper 235."""
    layer = np.full((_H, _W), 235, dtype=np.uint8)
    rng = np.random.default_rng(0)
    layer[10:50, 50:85] = rng.integers(60, 200, (40, 35), dtype=np.uint8)
    text_region = np.ones((_H, _W), dtype=bool)
    text_region[5:55, 45:88] = False
    ink = np.zeros((_H, _W), dtype=bool)
    ink[10:14, 5:40] = True
    ink[20:24, 5:30] = True
    return layer, text_region, ink


def _render(**flags) -> RenderParams:
    return RenderParams(
        need_binarization=True, mixed_output=True, split_output=True, **flags
    )


class TestForegroundType:
    def test_plain_binarisation_is_binary(self):
        assert foreground_type_for(_render()) == ForegroundType.BINARY

    def test_colour_segmentation_is_indexed(self):
        render = _render(color_segmentation=True)
        assert foreground_type_for(render) == ForegroundType.INDEXED

    def test_no_binarisation_is_colour(self):
        render = RenderParams(mixed_output=True, split_output=True)
        assert foreground_type_for(render) == ForegroundType.COLOR

    def test_colour_foreground_option_turns_binarisation_off(self):
        params = Params(color_mode=ColorMode.MIXED, output_dpi=Dpi.uniform(300))
        render = RenderParams.from_params(
            params, SplittingOptions(split_output=True, color_foreground=True)
        )
        assert foreground_type_for(render) == ForegroundType.COLOR


class TestSplitByForegroundMask:
    @pytest.fixture
    def split(self):
        layer, text_region, ink = _page()
        composed = compose_mixed(layer, text_region, ink)
        result = split_mixed_output(composed, _render(), text_region, ink)
        return result, composed, text_region, ink, layer

    def test_foreground_has_only_binarised_text(self, split):
        result, _, text_region, ink, _ = split
        foreground = result.foreground
        assert set(np.unique(foreground)) <= {0, 255}
        assert ((foreground == 0) == (ink & text_region)).all()
        assert (foreground[~text_region] == 255).all()

    def test_background_has_only_the_picture(self, split):
        result, composed, text_region, _, layer = split
        background = result.background
        assert (background[text_region] == 255).all()
        np.testing.assert_array_equal(background[~text_region], composed[~text_region])
        np.testing.assert_array_equal(
            background[10:50, 50:85], np.where(layer == 255, 254, layer)[10:50, 50:85]
        )

    def test_no_original_background_layer(self, split):
        assert split[0].original_background is None

    def test_merging_the_layers_restores_the_composed_page(self, split):
        result, composed, *_ = split
        merged = merge_split_images(result.foreground, result.background)
        np.testing.assert_array_equal(merged, composed)

    def test_input_is_not_modified(self):
        layer, text_region, ink = _page()
        composed = compose_mixed(layer, text_region, ink)
        before = composed.copy()
        split_mixed_output(composed, _render(), text_region, ink)
        np.testing.assert_array_equal(composed, before)


class TestColourPages:
    @staticmethod
    def _colour_page():
        layer, text_region, ink = _page()
        colour = np.dstack([layer, layer // 2, layer // 3]).astype(np.uint8)
        return colour, text_region, ink

    def test_binary_foreground_of_a_colour_page_is_two_dimensional(self):
        colour, text_region, ink = self._colour_page()
        composed = compose_mixed(colour, text_region, ink)
        result = split_mixed_output(composed, _render(), text_region, ink)
        assert result.foreground.ndim == 2
        assert ((result.foreground == 0) == (ink & text_region)).all()

    def test_colour_foreground_keeps_colours_and_alpha(self):
        colour, text_region, _ = self._colour_page()
        rgba = np.dstack([colour, np.full((_H, _W), 77, dtype=np.uint8)])
        result = split_by_foreground_mask(rgba, text_region, ForegroundType.COLOR)
        assert result.foreground.shape == rgba.shape
        assert (result.foreground[~text_region][:, :3] == 255).all()
        assert (result.foreground[..., 3] == 77).all()
        np.testing.assert_array_equal(result.foreground[text_region], rgba[text_region])

    def test_colour_round_trip(self):
        colour, text_region, ink = self._colour_page()
        composed = compose_mixed(colour, text_region, ink)
        result = split_by_foreground_mask(composed, text_region, ForegroundType.COLOR)
        merged = merge_split_images(result.foreground, result.background)
        np.testing.assert_array_equal(merged, composed)


class TestOriginalBackground:
    @pytest.fixture
    def split(self):
        layer, text_region, ink = _page()
        composed = compose_over_original(layer, ink)
        result = split_mixed_output(
            composed, _render(original_background=True), text_region, ink
        )
        return result, composed, text_region, ink, layer

    def test_foreground_is_only_the_ink(self, split):
        result, _, text_region, ink, _ = split
        assert ((result.foreground == 0) == (ink & text_region)).all()
        assert (result.foreground[~(ink & text_region)] == 255).all()

    def test_background_is_only_the_picture(self, split):
        result, composed, text_region, _, _ = split
        assert (result.background[text_region] == 255).all()
        np.testing.assert_array_equal(
            result.background[~text_region], composed[~text_region]
        )

    def test_original_background_has_white_ink_and_black_pictures(self, split):
        result, _, text_region, ink, layer = split
        original = result.original_background
        assert original is not None
        assert (original[ink & text_region] == 255).all()
        assert (original[~text_region] == 0).all()
        paper = text_region & ~ink
        np.testing.assert_array_equal(original[paper], layer[paper])

    def test_merging_the_three_layers_restores_the_composed_page(self, split):
        result, composed, *_ = split
        merged = merge_split_images(
            result.foreground, result.background, result.original_background
        )
        np.testing.assert_array_equal(merged, composed)

    def test_ink_inside_the_picture_region_is_not_foreground(self):
        layer, text_region, ink = _page()
        ink[20:24, 60:70] = True
        composed = compose_over_original(layer, ink)
        result = split_with_original_background(
            composed, ink & text_region, text_region, ForegroundType.BINARY
        )
        assert (result.foreground[20:24, 60:70] == 255).all()


class TestColourSegmentedForeground:
    def test_indexed_foreground_keeps_the_colours_of_the_ink(self):
        layer, text_region, ink = _page()
        colour = np.dstack([layer] * 3).astype(np.uint8)
        composed = compose_mixed(colour, text_region, ink)
        composed[ink & text_region] = (10, 20, 200)
        result = split_mixed_output(
            composed, _render(color_segmentation=True), text_region, ink
        )
        assert result.foreground_type == ForegroundType.INDEXED
        assert result.foreground.ndim == 3
        assert (result.foreground[ink & text_region] == (10, 20, 200)).all()
        assert (result.foreground[text_region & ~ink] == 255).all()


def test_merging_layers_of_different_size_is_rejected():
    with pytest.raises(ValueError, match="same height and width"):
        merge_split_images(
            np.full((3, 3), 255, np.uint8), np.full((4, 3), 255, np.uint8)
        )


class TestWriteSplitImages:
    def test_layers_go_to_their_folders(self, tmp_path):
        layer, text_region, ink = _page()
        composed = compose_mixed(layer, text_region, ink)
        result = split_mixed_output(composed, _render(), text_region, ink)

        paths = write_split_images(tmp_path, "0001.tif", result, Dpi.uniform(300))

        assert [p.parent.name for p in paths] == [FOREGROUND_DIR, BACKGROUND_DIR]
        assert [p.parent.parent for p in paths] == [tmp_path, tmp_path]
        with Image.open(paths[0]) as foreground:
            assert foreground.mode == "1"
            assert foreground.size == (_W, _H)
            assert not np.asarray(foreground)[10, 10]

    def test_original_background_gets_a_third_folder(self, tmp_path):
        layer, text_region, ink = _page()
        composed = compose_over_original(layer, ink)
        result = split_mixed_output(
            composed, _render(original_background=True), text_region, ink
        )
        paths = write_split_images(tmp_path, "0001.tif", result)
        assert [p.parent.name for p in paths] == [
            FOREGROUND_DIR,
            BACKGROUND_DIR,
            ORIGINAL_BACKGROUND_DIR,
        ]

    def test_colour_layers_lose_their_alpha_channel(self, tmp_path):
        layer, text_region, _ = _page()
        rgba = np.dstack([layer] * 3 + [np.full((_H, _W), 255, np.uint8)])
        result = split_by_foreground_mask(rgba, text_region, ForegroundType.COLOR)
        paths = write_split_images(tmp_path, "p.tif", result)
        with Image.open(paths[1]) as background:
            assert "A" not in background.mode
