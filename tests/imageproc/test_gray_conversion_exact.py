"""Both gray conversions give qGray's integer result for every 8-bit colour."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc.binary_threshold import to_gray
from scantailor.imageproc.filters import bgr_to_gray

GrayConversion = Callable[[NDArray[np.uint8]], NDArray[np.uint8]]
CONVERSIONS = pytest.mark.parametrize(
    "convert", [bgr_to_gray, to_gray], ids=["bgr_to_gray", "to_gray"]
)


def _reference_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """``(11 R + 16 G + 5 B) // 32`` in wide integers, the definition both follow."""
    blue = image[..., 0].astype(np.int64)
    green = image[..., 1].astype(np.int64)
    red = image[..., 2].astype(np.int64)
    return ((11 * red + 16 * green + 5 * blue) // 32).astype(np.uint8)


@pytest.fixture(scope="module")
def every_colour() -> NDArray[np.uint8]:
    """A 4096 x 4096 BGR image holding each of the 2^24 colours once."""
    codes = np.arange(1 << 24, dtype=np.uint32)
    channels = [(codes >> shift).astype(np.uint8) for shift in (16, 8, 0)]
    return np.stack(channels, axis=-1).reshape(4096, 4096, 3)


@CONVERSIONS
def test_every_8_bit_colour_converts_to_the_integer_gray(
    convert: GrayConversion, every_colour: NDArray[np.uint8]
) -> None:
    """All 2^24 BGR colours give exactly ``(11 R + 16 G + 5 B) // 32``."""
    np.testing.assert_array_equal(convert(every_colour), _reference_gray(every_colour))


def _random_bgra() -> NDArray[np.uint8]:
    return np.random.default_rng(3).integers(0, 256, (37, 53, 4), dtype=np.uint8)


@CONVERSIONS
@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_random_bgra(), id="bgra"),
        pytest.param(_random_bgra()[..., :3], id="bgr-view-of-bgra"),
        pytest.param(_random_bgra()[3:20, 5:40, :3], id="cropped-view"),
        pytest.param(_random_bgra()[::-1, ::-1, :3], id="reversed-view"),
        pytest.param(_random_bgra()[::-1], id="reversed-bgra"),
        pytest.param(_random_bgra()[:1, :, :3], id="one-row"),
        pytest.param(_random_bgra()[:, :1, :3], id="one-column"),
        pytest.param(_random_bgra()[:1, :1, :3], id="one-pixel"),
    ],
)
def test_alpha_and_views_convert_like_a_plain_bgr_copy(
    convert: GrayConversion, image: NDArray[np.uint8]
) -> None:
    """Alpha is ignored and strides do not change the result or its shape."""
    gray = convert(image)

    assert gray.dtype == np.uint8
    np.testing.assert_array_equal(gray, _reference_gray(image))


@CONVERSIONS
def test_a_read_only_image_converts(convert: GrayConversion) -> None:
    """A read-only input is read, never written."""
    image = _random_bgra()[..., :3].copy()
    image.flags.writeable = False

    np.testing.assert_array_equal(convert(image), _reference_gray(image))


@CONVERSIONS
@pytest.mark.parametrize("shape", [(0, 5, 3), (5, 0, 3), (0, 0, 4)])
def test_an_empty_image_gives_an_empty_gray_image(
    convert: GrayConversion, shape: tuple[int, int, int]
) -> None:
    """An image with no pixels gives a uint8 gray image of the same height and width."""
    gray = convert(np.zeros(shape, dtype=np.uint8))

    assert gray.shape == shape[:2]
    assert gray.dtype == np.uint8


@CONVERSIONS
def test_a_gray_image_is_returned_as_is(convert: GrayConversion) -> None:
    """2-D input is already gray and comes back as the same array."""
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)

    assert convert(gray) is gray
