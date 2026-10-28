"""Decoding 16-bit, 32-bit and colour images gives the documented 8-bit pixels."""

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from scantailor.core.image_io import _to_array

_RNG = np.random.default_rng(5)


def _reference_16_bit(samples: NDArray[np.integer]) -> NDArray[np.uint8]:
    """Clip to 0..65535, then ``(s * 255 + 32767) // 65535`` in wide integers."""
    clipped = np.clip(samples.astype(np.int64), 0, 65535)
    return ((clipped * 255 + 32767) // 65535).astype(np.uint8)


def test_every_16_bit_sample_scales_to_the_rounded_8_bit_level() -> None:
    """All 65536 levels of an I;16 image round to the nearest 8-bit level."""
    samples = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256)

    image = Image.fromarray(samples)
    assert image.mode == "I;16"

    decoded = _to_array(image)

    np.testing.assert_array_equal(decoded, _reference_16_bit(samples))


def test_32_bit_samples_outside_the_16_bit_range_clip_to_black_and_white() -> None:
    """Mode I is signed: negatives become 0 and values past 65535 become 255."""
    samples = np.array(
        [[-(2**31), -1, 0, 1], [65534, 65535, 65536, 2**31 - 1]], dtype=np.int32
    )

    image = Image.fromarray(samples)
    assert image.mode == "I"

    decoded = _to_array(image)

    np.testing.assert_array_equal(decoded, _reference_16_bit(samples))
    np.testing.assert_array_equal(decoded, [[0, 0, 0, 0], [255, 255, 255, 255]])


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "CMYK", "YCbCr"])
def test_a_colour_image_decodes_to_bgr_of_its_rgb_conversion(mode: str) -> None:
    """Colour modes decode to the channel-reversed RGB conversion, contiguous."""
    rgb = _RNG.integers(0, 256, (13, 17, 3), dtype=np.uint8)
    image = Image.fromarray(rgb).convert(mode)

    decoded = _to_array(image)

    expected = np.asarray(image.convert("RGB"))[..., ::-1]
    assert decoded.flags.c_contiguous
    np.testing.assert_array_equal(decoded, expected)


def test_a_colour_palette_decodes_to_bgr() -> None:
    """A palette with colours decodes through RGB to BGR."""
    image = Image.fromarray(_RNG.integers(0, 256, (9, 11, 3), np.uint8)).convert(
        "P", palette=Image.Palette.ADAPTIVE, colors=8
    )

    expected = np.asarray(image.convert("RGB"))[..., ::-1]
    np.testing.assert_array_equal(_to_array(image), expected)
