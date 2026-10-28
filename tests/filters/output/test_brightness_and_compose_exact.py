"""Brightness replacement and mixed composition match their float definitions."""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.output.generator import _replace_brightness
from scantailor.filters.output.mixed import compose_mixed

_RNG_SEED = 11


def _reference_brightness(
    image: NDArray[np.uint8], brightness: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """``floor(c + b - (11 R + 16 G + 5 B) / 32 + 0.5)``, clipped, in float32."""
    result = image.copy()
    color = image[..., :3].astype(np.float32)
    luma = (11 * color[..., 2] + 16 * color[..., 1] + 5 * color[..., 0]) / 32
    shifted = color + (brightness.astype(np.float32) - luma)[..., None]
    result[..., :3] = np.clip(np.floor(shifted + 0.5), 0, 255).astype(np.uint8)
    return result


@pytest.fixture(scope="module")
def every_colour() -> NDArray[np.uint8]:
    """A 4096 x 4096 BGR image holding each of the 2^24 colours once."""
    codes = np.arange(1 << 24, dtype=np.uint32)
    channels = [(codes >> shift).astype(np.uint8) for shift in (16, 8, 0)]
    return np.stack(channels, axis=-1).reshape(4096, 4096, 3)


@pytest.mark.parametrize("quarter", range(4))
def test_every_colour_takes_a_random_brightness_as_the_float_formula_does(
    every_colour: NDArray[np.uint8], quarter: int
) -> None:
    """All 2^24 colours, each with its own random target, match the float result."""
    colours = every_colour[quarter::4]
    rng = np.random.default_rng(_RNG_SEED + quarter)
    brightness = rng.integers(0, 256, colours.shape[:2], dtype=np.uint8)

    np.testing.assert_array_equal(
        _replace_brightness(colours, brightness),
        _reference_brightness(colours, brightness),
    )


@pytest.mark.parametrize("level", [0, 1, 15, 16, 17, 127, 128, 200, 254, 255])
def test_a_uniform_brightness_level_matches_the_float_formula(
    every_colour: NDArray[np.uint8], level: int
) -> None:
    """Rounding edges and clipping at both ends agree for these target levels."""
    sample = every_colour[::7, ::3]
    brightness = np.full(sample.shape[:2], level, dtype=np.uint8)

    np.testing.assert_array_equal(
        _replace_brightness(sample, brightness),
        _reference_brightness(sample, brightness),
    )


def _random_bgra() -> NDArray[np.uint8]:
    return np.random.default_rng(_RNG_SEED).integers(0, 256, (37, 53, 4), np.uint8)


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_random_bgra(), id="bgra-keeps-alpha"),
        pytest.param(_random_bgra()[3:20, 5:40, :3], id="cropped-view"),
        pytest.param(_random_bgra()[::-1, ::-1, :3], id="reversed-view"),
        pytest.param(_random_bgra()[:1, :, :3], id="one-row"),
        pytest.param(_random_bgra()[:0, :, :3], id="empty"),
    ],
)
def test_alpha_views_and_thin_images_match_the_float_formula(
    image: NDArray[np.uint8],
) -> None:
    """Alpha is carried over and strides or a single row change nothing."""
    rng = np.random.default_rng(_RNG_SEED)
    brightness = rng.integers(0, 256, image.shape[:2], dtype=np.uint8)
    before = image.copy()

    np.testing.assert_array_equal(
        _replace_brightness(image, brightness),
        _reference_brightness(image, brightness),
    )
    np.testing.assert_array_equal(image, before)


def _reference_compose(
    layer: NDArray[np.uint8],
    text_region: NDArray[np.bool_],
    ink: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Reserve pure black and white outside the text, then paint 0/255 inside it."""
    result = layer.copy()
    outside = ~text_region
    if layer.ndim == 2:
        result[outside & (layer == 0)] = 1
        result[outside & (layer == 255)] = 254
        result[text_region] = np.where(ink, 0, 255)[text_region]
        return result
    color = layer[..., :3]
    result[outside & (color == 0).all(axis=2), :3] = 1
    result[outside & (color == 255).all(axis=2), :3] = 254
    result[text_region, :3] = np.where(ink, 0, 255)[text_region][:, None]
    return result


@pytest.mark.parametrize("channels", [None, 3, 4], ids=["gray", "bgr", "bgra"])
def test_composed_text_over_a_picture_matches_the_masked_assignment(
    channels: int | None,
) -> None:
    """Text pixels become 0 or 255 in each colour channel; alpha and the rest keep."""
    rng = np.random.default_rng(_RNG_SEED)
    shape = (41, 29) if channels is None else (41, 29, channels)
    # Levels 0 and 255 are frequent so the reserved values are exercised.
    layer = rng.choice(np.array([0, 1, 128, 254, 255], np.uint8), shape)
    text_region = rng.random(shape[:2]) < 0.5
    ink = rng.random(shape[:2]) < 0.5
    before = layer.copy()

    np.testing.assert_array_equal(
        compose_mixed(layer, text_region, ink),
        _reference_compose(layer, text_region, ink),
    )
    np.testing.assert_array_equal(layer, before)
