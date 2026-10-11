"""Dewarping inside generate_output: order of steps, masks and zones."""

import numpy as np
import pytest

from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
    Zone,
    ZoneSpline,
)
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.filters.output import generator
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import ColorCommonOptions, FillingColor
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params

HEIGHT, WIDTH = 120, 160


def _page() -> np.ndarray:
    """Gray page with rows of dark text and a lighting falloff."""
    xs = np.linspace(1.0, 0.7, WIDTH)[None, :]
    page = np.full((HEIGHT, WIDTH), 230.0) * xs
    for y in range(20, 100, 10):
        page[y : y + 3, 20:140] = 40.0
    return np.clip(page, 0, 255).astype(np.uint8)


def _identity_dewarper() -> CylindricalSurfaceDewarper:
    """A model made of the page's own edges, which changes nothing."""
    top = np.array([[0.0, 0.0], [float(WIDTH), 0.0]])
    bottom = np.array([[0.0, float(HEIGHT)], [float(WIDTH), float(HEIGHT)]])
    return CylindricalSurfaceDewarper.from_directrices(top, bottom)


def _bowed_dewarper() -> CylindricalSurfaceDewarper:
    xs = np.linspace(0.0, WIDTH, 9)
    sag = 8.0 * np.sin(np.pi * xs / WIDTH)
    top = np.column_stack([xs, 10.0 + sag])
    bottom = np.column_stack([xs, HEIGHT - 10.0 + sag])
    return CylindricalSurfaceDewarper.from_directrices(top, bottom)


def _params(mode: DewarpingMode = DewarpingMode.AUTO, **kwargs) -> Params:
    return Params(
        color_mode=ColorMode.COLOR_GRAYSCALE,
        dewarping=DewarpingOptions(mode=mode, post_deskew=False),
        color_common=ColorCommonOptions(
            normalize_illumination=True, fill_margins=False
        ),
        **kwargs,
    )


class TestDewarpingOff:
    """Without dewarping a dewarper changes nothing."""

    def test_a_supplied_dewarper_is_ignored_when_mode_is_off(self) -> None:
        page = _page()
        params = _params(DewarpingMode.OFF)

        with_dewarper = generate_output(page, params, _bowed_dewarper())
        without = generate_output(page, params)

        np.testing.assert_array_equal(with_dewarper.image, without.image)
        assert with_dewarper.was_dewarped is False

    def test_the_model_of_the_whole_page_leaves_pixels_unchanged(self) -> None:
        page = _page()

        dewarped = generate_output(page, _params(), _identity_dewarper())
        plain = generate_output(page, _params(DewarpingMode.OFF))

        assert dewarped.was_dewarped is True
        np.testing.assert_array_equal(dewarped.image, plain.image)


class TestOrder:
    """Illumination is normalised on the warped page, before dewarping."""

    def test_illumination_is_normalised_once_and_before_the_dewarper_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        real_normalize = generator.normalize_illumination_gray
        real_remap = generator.remap_with_maps
        seen: dict[str, np.ndarray] = {}

        def normalize_spy(*args, **kwargs):
            events.append("normalize")
            seen["normalized"] = real_normalize(*args, **kwargs)
            return seen["normalized"]

        def remap_spy(*args, **kwargs):
            events.append("remap")
            return real_remap(*args, **kwargs)

        def factory(gray: np.ndarray):
            events.append("factory")
            seen["factory_input"] = gray
            return _bowed_dewarper()

        monkeypatch.setattr(generator, "normalize_illumination_gray", normalize_spy)
        monkeypatch.setattr(generator, "remap_with_maps", remap_spy)

        result = generate_output(_page(), _params(), factory)

        assert result.was_dewarped is True
        assert events[:2] == ["normalize", "factory"]
        assert events.count("normalize") == 1
        assert set(events[2:]) == {"remap"}
        np.testing.assert_array_equal(seen["factory_input"], seen["normalized"])

    def test_factory_returning_none_leaves_geometry_alone(self) -> None:
        page = _page()

        result = generate_output(page, _params(), lambda gray: None)
        plain = generate_output(page, _params(DewarpingMode.OFF))

        assert result.was_dewarped is False
        np.testing.assert_array_equal(result.image, plain.image)

    def test_no_normalisation_means_no_estimate_before_dewarping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            generator,
            "normalize_illumination_gray",
            lambda *args, **kwargs: calls.append("normalize"),
        )
        params = Params(
            color_mode=ColorMode.COLOR_GRAYSCALE,
            dewarping=DewarpingOptions(mode=DewarpingMode.AUTO, post_deskew=False),
            color_common=ColorCommonOptions(normalize_illumination=False),
        )

        generate_output(_page(), params, _bowed_dewarper())

        assert calls == []


class TestMasksFollowTheDewarp:
    """Masks and zones are warped with the page."""

    def test_margin_fill_uses_the_dewarped_content_area(self) -> None:
        page = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)
        params = Params(
            color_mode=ColorMode.COLOR_GRAYSCALE,
            dewarping=DewarpingOptions(mode=DewarpingMode.AUTO, post_deskew=False),
            color_common=ColorCommonOptions(
                fill_margins=True, filling_color=FillingColor.WHITE
            ),
        )
        content = [(30.0, 30.0), (130.0, 30.0), (130.0, 90.0), (30.0, 90.0)]

        result = generate_output(
            page, params, _identity_dewarper(), content_polygon=content
        )

        assert result.image[60, 80] == 100
        assert result.image[5, 5] == 255
        assert result.image[60, 10] == 255

    def test_fill_zone_is_painted_where_the_dewarped_page_shows_it(self) -> None:
        page = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)
        zone = Zone(
            spline=ZoneSpline(points=[(20, 20), (60, 20), (60, 60), (20, 60)]),
            properties=FillZoneProperties(color=(0, 0, 0)),
        )

        dewarped = generate_output(
            page, _params(), _identity_dewarper(), fill_zones=[zone]
        )
        plain = generate_output(page, _params(DewarpingMode.OFF), fill_zones=[zone])

        np.testing.assert_array_equal(dewarped.image, plain.image)
        assert dewarped.image[40, 40] == 0
        assert dewarped.image[80, 100] == plain.image[80, 100]

    def test_picture_zone_layers_are_warped_into_mixed_output(self) -> None:
        page = _page()
        zone = Zone(
            spline=ZoneSpline(points=[(20, 20), (100, 20), (100, 80), (20, 80)]),
            properties=PictureZoneProperties(layer=PictureLayer.PAINTER2),
        )
        params = Params(
            color_mode=ColorMode.MIXED,
            dewarping=DewarpingOptions(mode=DewarpingMode.AUTO, post_deskew=False),
        )

        dewarped = generate_output(
            page, params, _identity_dewarper(), picture_zones=[zone]
        )
        plain = generate_output(
            page,
            params.with_dewarping(DewarpingOptions()),
            picture_zones=[zone],
        )

        assert dewarped.picture_mask is not None and plain.picture_mask is not None
        np.testing.assert_array_equal(dewarped.picture_mask, plain.picture_mask)
        assert dewarped.picture_mask[50, 60]

    def test_curved_model_changes_pixels_but_keeps_the_size(self) -> None:
        page = _page()

        result = generate_output(page, _params(), _bowed_dewarper())
        plain = generate_output(page, _params(DewarpingMode.OFF))

        assert result.image.shape == plain.image.shape
        assert not np.array_equal(result.image, plain.image)

    def test_unusable_model_leaves_the_page_as_it_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            CylindricalSurfaceDewarper,
            "directrix_bounding_box",
            lambda self: (0.0, 0.0, 0.0, 0.0),
        )
        page = _page()

        result = generate_output(page, _params(), _bowed_dewarper())
        plain = generate_output(page, _params(DewarpingMode.OFF))

        assert result.was_dewarped is False
        np.testing.assert_array_equal(result.image, plain.image)

    def test_white_on_black_page_gets_a_dark_background_outside_the_source(
        self,
    ) -> None:
        page = np.full((HEIGHT, WIDTH), 30, dtype=np.uint8)
        params = Params(
            color_mode=ColorMode.COLOR_GRAYSCALE,
            black_on_white=False,
            dewarping=DewarpingOptions(mode=DewarpingMode.AUTO, post_deskew=False),
            color_common=ColorCommonOptions(fill_margins=False),
        )
        xs = np.linspace(-WIDTH * 0.5, WIDTH * 1.5, 5)
        small = CylindricalSurfaceDewarper.from_directrices(
            np.column_stack([xs, np.full(5, 10.0)]),
            np.column_stack([xs, np.full(5, HEIGHT - 10.0)]),
        )

        result = generate_output(page, params, small)

        assert result.image.shape == page.shape
        assert result.image[60, 80] == pytest.approx(30, abs=1)

    def test_post_deskew_keeps_the_output_size(self) -> None:
        page = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)
        params = Params(
            color_mode=ColorMode.COLOR_GRAYSCALE,
            dewarping=DewarpingOptions(
                mode=DewarpingMode.MANUAL, post_deskew=True, post_deskew_angle=3.0
            ),
            color_common=ColorCommonOptions(fill_margins=False),
        )

        result = generate_output(page, params, _identity_dewarper())

        assert result.was_dewarped is True
        assert result.image.shape == page.shape
        assert result.image[HEIGHT // 2, WIDTH // 2] == 100

    def test_positive_post_deskew_angle_turns_the_page_clockwise(self) -> None:
        page = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        page[HEIGHT // 2 - 2 : HEIGHT // 2 + 2, WIDTH // 2 + 20 : WIDTH // 2 + 40] = 0
        params = Params(
            color_mode=ColorMode.COLOR_GRAYSCALE,
            dewarping=DewarpingOptions(
                mode=DewarpingMode.MANUAL, post_deskew=True, post_deskew_angle=10.0
            ),
            color_common=ColorCommonOptions(fill_margins=False),
        )

        result = generate_output(page, params, _identity_dewarper())

        rows = np.nonzero(result.image < 128)[0]
        assert rows.mean() > HEIGHT // 2 + 3

    def test_automatic_mode_ignores_the_post_deskew_angle(self) -> None:
        page = _page()
        params = Params(
            color_mode=ColorMode.COLOR_GRAYSCALE,
            dewarping=DewarpingOptions(
                mode=DewarpingMode.AUTO, post_deskew=True, post_deskew_angle=3.0
            ),
            color_common=ColorCommonOptions(fill_margins=False),
        )
        level = Params(
            color_mode=ColorMode.COLOR_GRAYSCALE,
            dewarping=DewarpingOptions(mode=DewarpingMode.AUTO, post_deskew=False),
            color_common=ColorCommonOptions(fill_margins=False),
        )

        result = generate_output(page, params, _identity_dewarper())
        expected = generate_output(page, level, _identity_dewarper())

        np.testing.assert_array_equal(result.image, expected.image)
