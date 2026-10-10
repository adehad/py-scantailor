"""Tests for mixed output, picture zones, offcut filling and the colour options."""

from typing import ClassVar

import numpy as np
import pytest

from scantailor.core.models import Dpi
from scantailor.core.zones import PictureLayer, ZoneSet, ZoneSpline
from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import (
    ColorCommonOptions,
    FillingColor,
    PosterizationOptions,
)
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions
from scantailor.filters.output.render_params import RenderParams

PAGE_HEIGHT = 600
PAGE_WIDTH = 480
PAPER = 235
PHOTO = (slice(300, 480), slice(120, 360))  # rows, columns


def _text_lines(page: np.ndarray) -> None:
    """Draw dark text-like strokes on the top third of the page."""
    for top in range(40, 220, 36):
        for left in range(40, 440, 14):
            page[top : top + 18, left : left + 4] = 25
            page[top + 7 : top + 11, left : left + 10] = 25


def _photo_block(page: np.ndarray) -> np.ndarray:
    """Paint a textured gradient block that keeps well away from 0 and 255."""
    rows, cols = PHOTO
    height = rows.stop - rows.start
    width = cols.stop - cols.start
    ramp = np.linspace(70, 190, width)[None, :] + np.linspace(0, 25, height)[:, None]
    noise = np.random.default_rng(3).integers(-22, 23, size=(height, width))
    page[rows, cols] = np.clip(ramp + noise, 40, 215).astype(np.uint8)
    return page[rows, cols].copy()


def _synthetic_page() -> tuple[np.ndarray, np.ndarray]:
    page = np.full((PAGE_HEIGHT, PAGE_WIDTH), PAPER, dtype=np.uint8)
    _text_lines(page)
    photo = _photo_block(page)
    return page, photo


_PLAIN_BINARIZATION = BinarizationOptions(
    normalize_illumination=False,
    savitzky_golay_smoothing=False,
    morphological_smoothing=False,
)


def _mixed_params(**overrides) -> Params:
    options = {
        "output_dpi": Dpi.uniform(300),
        "color_mode": ColorMode.MIXED,
        "despeckle_level": DespeckleLevel.OFF,
        "binarization": _PLAIN_BINARIZATION,
    }
    options.update(overrides)
    return Params(**options)


def _rectangle(left: float, top: float, right: float, bottom: float) -> ZoneSpline:
    return ZoneSpline(
        points=[(left, top), (right, top), (right, bottom), (left, bottom)]
    )


def _interior(block: tuple[slice, slice], inset: int = 12) -> tuple[slice, slice]:
    rows, cols = block
    return (
        slice(rows.start + inset, rows.stop - inset),
        slice(cols.start + inset, cols.stop - inset),
    )


class TestMixedOutput:
    """Mixed mode keeps pictures grey and binarises text."""

    def test_photo_keeps_gray_levels_and_text_is_pure_black_and_white(self):
        page, photo = _synthetic_page()

        result = generate_output(page, _mixed_params())

        inner = _interior(PHOTO)
        original_inner = photo[12:-12, 12:-12]
        assert np.abs(result.image[inner].astype(int) - original_inner).max() <= 1
        assert len(np.unique(result.image[inner])) > 20
        text_area = result.image[30:230, :]
        assert set(np.unique(text_area)) <= {0, 255}
        assert (text_area == 0).any()

    def test_output_is_not_reported_binary_and_exposes_picture_mask(self):
        page, _ = _synthetic_page()

        result = generate_output(page, _mixed_params())

        assert result.is_binary is False
        assert result.picture_mask is not None
        assert result.picture_mask[_interior(PHOTO)].all()
        assert not result.picture_mask[30:230, :].any()

    def test_picture_pixels_are_never_pure_black_or_white(self):
        page, _ = _synthetic_page()
        page[PHOTO[0].start + 40 : PHOTO[0].start + 50, 200:210] = 0
        page[PHOTO[0].start + 60 : PHOTO[0].start + 70, 200:210] = 255

        result = generate_output(page, _mixed_params())

        picture = result.picture_mask
        assert picture is not None
        assert not np.isin(result.image[picture], (0, 255)).any()

    def test_input_is_not_modified(self):
        page, _ = _synthetic_page()
        before = page.copy()

        generate_output(page, _mixed_params())

        assert np.array_equal(page, before)

    def test_colour_picture_keeps_its_hue(self):
        page, _ = _synthetic_page()
        colour = np.stack([page, page, page], axis=2)
        rows, cols = PHOTO
        colour[rows, cols, 2] = np.clip(colour[rows, cols, 2].astype(int) - 30, 0, 255)

        result = generate_output(colour, _mixed_params())

        inner = _interior(PHOTO)
        assert result.image.shape == colour.shape
        assert np.abs(result.image[inner].astype(int) - colour[inner]).max() <= 1

    def test_alpha_channel_stays_opaque_for_text_and_margins(self):
        page, _ = _synthetic_page()
        bgra = np.dstack([page, page, page, np.full_like(page, 255)])
        polygon = [(30.0, 30.0), (450.0, 30.0), (450.0, 500.0), (30.0, 500.0)]

        result = generate_output(
            bgra,
            _mixed_params(),
            color_options=ColorCommonOptions(filling_color=FillingColor.BLACK),
            content_polygon=polygon,
        )

        assert (result.image[..., 3] == 255).all()

    def test_detection_off_binarises_the_whole_page(self):
        page, _ = _synthetic_page()
        params = _mixed_params(
            picture_shape=PictureShapeOptions(shape=PictureShape.OFF)
        )

        result = generate_output(page, params)

        assert set(np.unique(result.image)) <= {0, 255}
        assert result.picture_mask is not None
        assert not result.picture_mask.any()

    def test_rectangular_shape_turns_the_photo_into_one_filled_box(self):
        page, _ = _synthetic_page()
        params = _mixed_params(
            picture_shape=PictureShapeOptions(shape=PictureShape.RECTANGULAR)
        )

        result = generate_output(page, params)

        mask = result.picture_mask
        assert mask is not None
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        assert mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].all()
        assert mask[_interior(PHOTO)].all()

    def test_white_on_black_page_is_inverted_back(self):
        page, _ = _synthetic_page()

        normal = generate_output(page, _mixed_params())
        inverted = generate_output(255 - page, _mixed_params(black_on_white=False))

        inner = _interior(PHOTO)
        assert (
            np.abs(inverted.image[inner].astype(int) - (255 - page)[inner]).max() <= 1
        )
        assert (normal.image[30:230] == 255 - inverted.image[30:230]).mean() > 0.98

    def test_margins_outside_content_are_white_by_default_and_black_on_request(self):
        page, _ = _synthetic_page()
        polygon = [(30.0, 30.0), (450.0, 30.0), (450.0, 500.0), (30.0, 500.0)]

        white = generate_output(page, _mixed_params(), content_polygon=polygon)
        black = generate_output(
            page,
            _mixed_params(),
            color_options=ColorCommonOptions(filling_color=FillingColor.BLACK),
            content_polygon=polygon,
        )

        assert (white.image[540:, :] == 255).all()
        assert (black.image[540:, :] == 0).all()
        assert white.picture_mask is not None
        assert not white.picture_mask[540:, :].any()


class TestManualPictureZones:
    """Manual zones change what counts as a picture."""

    def test_painter_zone_keeps_original_grays_where_nothing_was_detected(self):
        page, _ = _synthetic_page()
        zones = ZoneSet()
        zones.add_picture_zone(_rectangle(30, 30, 450, 70), layer=PictureLayer.PAINTER2)
        params = _mixed_params(
            picture_shape=PictureShapeOptions(shape=PictureShape.OFF)
        )

        result = generate_output(page, params, picture_zones=zones.picture_zones)

        inside = (slice(40, 60), slice(40, 440))
        assert result.picture_mask is not None
        assert result.picture_mask[inside].all()
        assert np.abs(result.image[inside].astype(int) - page[inside]).max() <= 1
        assert set(np.unique(result.image[100:230, :])) <= {0, 255}

    def test_eraser_zone_binarises_a_detected_photo(self):
        page, _ = _synthetic_page()
        zones = ZoneSet()
        zones.add_picture_zone(
            _rectangle(100, 280, 380, 500), layer=PictureLayer.ERASER3
        )

        result = generate_output(
            page, _mixed_params(), picture_zones=zones.picture_zones
        )

        assert set(np.unique(result.image[_interior(PHOTO)])) <= {0, 255}
        assert result.picture_mask is not None
        assert not result.picture_mask[_interior(PHOTO)].any()

    def test_foreground_zone_keeps_colours_of_ink_and_whitens_the_background(self):
        page, _ = _synthetic_page()
        zones = ZoneSet()
        zones.add_picture_zone(
            _rectangle(30, 30, 450, 230), layer=PictureLayer.FOREGROUND
        )
        params = _mixed_params(
            picture_shape=PictureShapeOptions(shape=PictureShape.OFF)
        )

        result = generate_output(page, params, picture_zones=zones.picture_zones)

        zone = (slice(30, 230), slice(30, 450))
        ink = page[zone] < 100
        assert (result.image[zone][~ink] == 255).mean() > 0.99
        assert np.abs(result.image[zone][ink].astype(int) - page[zone][ink]).max() <= 1

    def test_background_zone_blackens_ink_and_keeps_the_paper_grey(self):
        page, _ = _synthetic_page()
        zones = ZoneSet()
        zones.add_picture_zone(
            _rectangle(30, 30, 450, 230), layer=PictureLayer.BACKGROUND
        )
        params = _mixed_params(
            picture_shape=PictureShapeOptions(shape=PictureShape.OFF)
        )

        result = generate_output(page, params, picture_zones=zones.picture_zones)

        zone = (slice(30, 230), slice(30, 450))
        ink = page[zone] < 100
        assert (result.image[zone][ink] == 0).mean() > 0.99
        assert np.abs(result.image[zone][~ink].astype(int) - PAPER).max() <= 1

    def test_zone_on_the_no_op_layer_changes_nothing(self):
        page, _ = _synthetic_page()
        zones = ZoneSet()
        zones.add_picture_zone(_rectangle(30, 30, 450, 230), layer=PictureLayer.NOOP)

        without = generate_output(page, _mixed_params())
        with_zone = generate_output(
            page, _mixed_params(), picture_zones=zones.picture_zones
        )

        assert np.array_equal(without.image, with_zone.image)


class TestFillOffcut:
    """Everything outside the page outline is filled when fill_offcut is on."""

    OUTLINE: ClassVar[list[tuple[float, float]]] = [
        (100.0, 60.0),
        (400.0, 60.0),
        (400.0, 540.0),
        (100.0, 540.0),
    ]

    def _grey_params(self) -> Params:
        return Params(color_mode=ColorMode.COLOR_GRAYSCALE)

    def test_outside_the_outline_takes_the_background_colour(self):
        page = np.full((PAGE_HEIGHT, PAGE_WIDTH), 200, dtype=np.uint8)
        page[:, :100] = 30
        page[:, 400:] = 30

        result = generate_output(
            page,
            self._grey_params(),
            color_options=ColorCommonOptions(fill_margins=False),
            page_outline=self.OUTLINE,
        )

        assert (result.image[:, :95] == 200).all()
        assert (result.image[:, 405:] == 200).all()
        assert (result.image[:50, :] == 200).all()
        assert (result.image[100:500, 110:390] == 200).all()

    def test_offcut_is_kept_when_the_option_is_off(self):
        page = np.full((PAGE_HEIGHT, PAGE_WIDTH), 200, dtype=np.uint8)
        page[:, :100] = 30

        result = generate_output(
            page,
            self._grey_params(),
            color_options=ColorCommonOptions(fill_offcut=False, fill_margins=False),
            page_outline=self.OUTLINE,
        )

        assert np.array_equal(result.image, page)

    def test_white_filling_colour_fills_the_offcut_white(self):
        page = np.full((PAGE_HEIGHT, PAGE_WIDTH), 120, dtype=np.uint8)

        result = generate_output(
            page,
            self._grey_params(),
            color_options=ColorCommonOptions(
                fill_margins=False, filling_color=FillingColor.WHITE
            ),
            page_outline=self.OUTLINE,
        )

        assert (result.image[:, :90] == 255).all()
        assert (result.image[200:400, 150:350] == 120).all()

    def test_bilevel_output_fills_the_offcut_too(self):
        page = np.full((PAGE_HEIGHT, PAGE_WIDTH), 235, dtype=np.uint8)
        page[:, :100] = 20
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            despeckle_level=DespeckleLevel.OFF,
            binarization=BinarizationOptions(normalize_illumination=False),
        )

        result = generate_output(
            page,
            params,
            color_options=ColorCommonOptions(fill_margins=False),
            page_outline=self.OUTLINE,
        )

        assert (result.image[:, :95] == 255).all()

    def test_margin_and_offcut_combine(self):
        page = np.full((PAGE_HEIGHT, PAGE_WIDTH), 200, dtype=np.uint8)
        content = [(150.0, 100.0), (350.0, 100.0), (350.0, 400.0), (150.0, 400.0)]
        page[:, :] = 90
        page[100:400, 150:350] = 200

        result = generate_output(
            page,
            self._grey_params(),
            color_options=ColorCommonOptions(filling_color=FillingColor.WHITE),
            content_polygon=content,
            page_outline=self.OUTLINE,
        )

        assert (result.image[:80, :] == 255).all()
        assert (result.image[:, :140] == 255).all()
        assert (result.image[150:350, 200:300] == 200).all()


class TestPosterizeInOutput:
    """Posterization applies to grayscale and colour output."""

    def test_colour_output_has_few_distinct_colours(self):
        rng = np.random.default_rng(0)
        page = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        options = ColorCommonOptions(
            posterization=PosterizationOptions(
                enabled=True, level=3, force_black_and_white=False
            ),
            fill_margins=False,
        )

        result = generate_output(
            page,
            Params(color_mode=ColorMode.COLOR_GRAYSCALE),
            color_options=options,
        )

        colours = {tuple(c) for c in result.image.reshape(-1, 3)}
        assert len(colours) <= 4**3

    def test_posterization_off_leaves_the_page_untouched(self):
        rng = np.random.default_rng(1)
        page = rng.integers(0, 256, size=(60, 60, 3), dtype=np.uint8)

        result = generate_output(
            page,
            Params(color_mode=ColorMode.COLOR_GRAYSCALE),
            color_options=ColorCommonOptions(fill_margins=False),
        )

        assert np.array_equal(result.image, page)


class TestWienerInOutput:
    """The Wiener colour filter is opt-in."""

    def test_coefficient_zero_leaves_a_noisy_page_unchanged(self):
        rng = np.random.default_rng(2)
        page = rng.integers(100, 160, size=(80, 80), dtype=np.uint8)

        result = generate_output(
            page,
            Params(color_mode=ColorMode.COLOR_GRAYSCALE),
            color_options=ColorCommonOptions(fill_margins=False, wiener_coef=0.0),
        )

        assert np.array_equal(result.image, page)

    def test_positive_coefficient_reduces_noise(self):
        rng = np.random.default_rng(2)
        page = rng.integers(100, 160, size=(80, 80), dtype=np.uint8)

        result = generate_output(
            page,
            Params(color_mode=ColorMode.COLOR_GRAYSCALE),
            color_options=ColorCommonOptions(fill_margins=False, wiener_coef=0.5),
        )

        assert result.image.std() < page.std()


class TestFillZones:
    """Fill zones paint their colour over the finished output."""

    def test_binary_output_snaps_the_zone_colour_to_black_or_white(self):
        page, _ = _synthetic_page()
        zones = ZoneSet()
        zones.add_fill_zone(_rectangle(200, 400, 300, 450), color=(20, 20, 20))
        params = Params(
            output_dpi=Dpi.uniform(300),
            despeckle_level=DespeckleLevel.OFF,
            binarization=BinarizationOptions(normalize_illumination=False),
        )

        result = generate_output(page, params, fill_zones=zones.fill_zones)

        assert (result.image[410:440, 210:290] == 0).all()
        assert result.is_binary

    def test_colour_output_paints_the_exact_colour(self):
        page = np.full((100, 100, 3), 200, dtype=np.uint8)
        zones = ZoneSet()
        zones.add_fill_zone(_rectangle(20, 20, 60, 60), color=(255, 0, 0))

        result = generate_output(
            page,
            Params(color_mode=ColorMode.COLOR_GRAYSCALE),
            color_options=ColorCommonOptions(fill_margins=False),
            fill_zones=zones.fill_zones,
        )

        assert tuple(result.image[40, 40]) == (0, 0, 255)
        assert tuple(result.image[80, 80]) == (200, 200, 200)

    def test_colour_output_posterizes_the_zone_colour_too(self):
        page = np.full((100, 100, 3), 200, dtype=np.uint8)
        page[:, :5] = 20
        page[:, 5:10] = 240
        zones = ZoneSet()
        zones.add_fill_zone(_rectangle(20, 20, 60, 60), color=(195, 195, 195))
        options = ColorCommonOptions(
            fill_margins=False,
            posterization=PosterizationOptions(
                enabled=True, level=2, force_black_and_white=False
            ),
        )

        result = generate_output(
            page,
            Params(color_mode=ColorMode.COLOR_GRAYSCALE),
            color_options=options,
            fill_zones=zones.fill_zones,
        )

        assert tuple(result.image[40, 40]) == (200, 200, 200)


class TestDefaultsMatchCpp:
    """Defaults follow ColorCommonOptions, BlackWhiteOptions and RenderParams in C++."""

    def test_color_common_defaults(self):
        options = ColorCommonOptions()

        assert options.fill_offcut is True
        assert options.fill_margins is True
        assert options.normalize_illumination is False
        assert options.filling_color == FillingColor.BACKGROUND
        assert options.wiener_coef == 0.0
        assert options.wiener_window_size == 5

    def test_posterization_defaults(self):
        options = PosterizationOptions()

        assert options.enabled is False
        assert options.level == 4
        assert options.normalization_enabled is False
        assert options.force_black_and_white is True

    def test_black_white_defaults(self):
        options = BinarizationOptions()

        assert options.savitzky_golay_smoothing is True
        assert options.morphological_smoothing is True
        assert options.normalize_illumination is True
        assert options.color_segmenter.enabled is False
        assert options.color_segmenter.noise_reduction == 7

    def test_picture_shape_defaults(self):
        options = PictureShapeOptions()

        assert options.shape == PictureShape.FREE
        assert options.sensitivity == 100
        assert options.higher_search_sensitivity is False

    def test_render_params_set_fill_offcut_and_savitzky_golay_by_default(self):
        render = RenderParams.from_params(Params())

        assert render.fill_offcut is True
        assert render.savitzky_golay_smoothing is True
        assert render.posterize is False

    def test_posterize_flag_follows_the_option_in_colour_mode(self):
        options = ColorCommonOptions(posterization=PosterizationOptions(enabled=True))
        params = Params(color_mode=ColorMode.COLOR_GRAYSCALE, color_common=options)

        assert RenderParams.from_params(params).posterize is True

    def test_posterize_in_bilevel_mode_needs_colour_segmentation(self):
        options = ColorCommonOptions(posterization=PosterizationOptions(enabled=True))
        plain = Params(color_mode=ColorMode.BLACK_AND_WHITE, color_common=options)

        assert RenderParams.from_params(plain).posterize is False

    @pytest.mark.parametrize("level", [1, 256])
    def test_posterization_level_is_bounded(self, level):
        with pytest.raises(ValueError, match="level"):
            PosterizationOptions(level=level)
