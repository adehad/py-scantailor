"""The generator can hand back the page as it was before despeckling."""

import numpy as np
import pytest

from scantailor.core.zones import FillZoneProperties, Zone, ZoneSpline
from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions

SPECK = (slice(20, 22), slice(20, 22))
_PLAIN = BinarizationOptions(
    normalize_illumination=False,
    savitzky_golay_smoothing=False,
    morphological_smoothing=False,
)


def _page() -> np.ndarray:
    """White page with two big dark bars and one two pixel speck."""
    page = np.full((320, 320), 240, dtype=np.uint8)
    page[120:180, 60:260] = 20
    page[220:280, 60:260] = 20
    page[SPECK] = 20
    return page


def _params(mode: ColorMode, level: DespeckleLevel) -> Params:
    return Params(
        color_mode=mode,
        binarization=_PLAIN,
        despeckle_level=level,
        picture_shape=PictureShapeOptions(shape=PictureShape.OFF),
    )


def test_black_and_white_returns_the_page_before_despeckling() -> None:
    result = generate_output(
        _page(),
        _params(ColorMode.BLACK_AND_WHITE, DespeckleLevel.AGGRESSIVE),
        keep_despeckle_source=True,
    )

    source = result.despeckle_source
    assert source is not None
    assert source.shape == result.image.shape
    assert set(np.unique(source)) <= {0, 255}
    assert np.all(source[SPECK] == 0)
    assert np.all(result.image[SPECK] == 255)
    assert np.all(source[120:180, 60:260] == 0)


def test_the_source_is_kept_even_when_despeckling_is_off() -> None:
    result = generate_output(
        _page(),
        _params(ColorMode.BLACK_AND_WHITE, DespeckleLevel.OFF),
        keep_despeckle_source=True,
    )

    assert result.despeckle_source is not None
    assert np.array_equal(result.despeckle_source, result.image)


def test_nothing_is_kept_unless_asked_for() -> None:
    result = generate_output(
        _page(), _params(ColorMode.BLACK_AND_WHITE, DespeckleLevel.AGGRESSIVE)
    )

    assert result.despeckle_source is None


def test_mixed_output_returns_the_text_before_despeckling() -> None:
    result = generate_output(
        _page(),
        _params(ColorMode.MIXED, DespeckleLevel.AGGRESSIVE),
        keep_despeckle_source=True,
    )

    source = result.despeckle_source
    assert source is not None
    assert np.all(source[SPECK] == 0)
    assert result.ink_mask is not None
    assert not result.ink_mask[SPECK].any()


def test_a_page_that_is_not_binarised_has_no_source() -> None:
    result = generate_output(
        _page(),
        _params(ColorMode.COLOR_GRAYSCALE, DespeckleLevel.NORMAL),
        keep_despeckle_source=True,
    )

    assert result.despeckle_source is None


def test_the_source_does_not_change_the_output() -> None:
    params = _params(ColorMode.BLACK_AND_WHITE, DespeckleLevel.NORMAL)

    with_source = generate_output(_page(), params, keep_despeckle_source=True)
    without = generate_output(_page(), params)

    assert np.array_equal(with_source.image, without.image)


@pytest.mark.parametrize("black_on_white", [True, False])
def test_the_source_keeps_ink_as_zero_whatever_the_page_polarity(
    black_on_white: bool,
) -> None:
    page = _page() if black_on_white else 255 - _page()
    params = _params(
        ColorMode.BLACK_AND_WHITE, DespeckleLevel.AGGRESSIVE
    ).with_black_on_white(black_on_white)

    result = generate_output(page, params, keep_despeckle_source=True)

    assert result.despeckle_source is not None
    assert np.all(result.despeckle_source[120:180, 60:260] == 0)


def test_a_fill_zone_does_not_take_the_pre_despeckle_page_away() -> None:
    zone = Zone(
        spline=ZoneSpline(points=[(50, 110), (270, 110), (270, 190), (50, 190)]),
        properties=FillZoneProperties(color=(255, 255, 255)),
    )

    result = generate_output(
        _page(),
        _params(ColorMode.BLACK_AND_WHITE, DespeckleLevel.NORMAL),
        fill_zones=[zone],
        keep_despeckle_source=True,
    )

    assert result.despeckle_source is not None
    assert result.despeckle_source[150, 100] == 0
    assert result.image[150, 100] == 255
