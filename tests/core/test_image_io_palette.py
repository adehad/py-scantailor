"""``write_image`` stores a BGR page of few colours as a palette TIFF."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from scantailor.core.image_io import load_image, write_image
from scantailor.core.models import ImageId
from scantailor.core.settings import TiffCompression, TiffCompressions
from scantailor.imageproc.posterizer import posterize

BITS_PER_SAMPLE_TAG = 258
PHOTOMETRIC_TAG = 262
PHOTOMETRIC_PALETTE = 3


def _scan(height: int = 120, width: int = 90) -> NDArray[np.uint8]:
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:height, 0:width]
    bgr = np.dstack([x * 255 // width, y * 255 // height, (x + y) % 256])
    noisy = bgr + rng.integers(-15, 15, bgr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _colours(height: int, width: int, count: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(9)
    table = rng.integers(0, 256, (count, 3), dtype=np.uint8)
    return table[rng.integers(0, count, (height, width))]


def _write_and_open(
    tmp_path: Path,
    pixels: NDArray[np.uint8],
    compressions: TiffCompressions | None = None,
) -> Image.Image:
    path = tmp_path / "page.tif"
    write_image(path, pixels, compressions=compressions)
    return Image.open(path)


def test_a_posterized_page_is_written_as_an_8_bit_palette_tiff(tmp_path: Path) -> None:
    pixels = posterize(_scan(), 4)

    with _write_and_open(tmp_path, pixels) as image:
        assert image.mode == "P"
        assert image.tag_v2[PHOTOMETRIC_TAG] == PHOTOMETRIC_PALETTE
        assert image.tag_v2[BITS_PER_SAMPLE_TAG] in (8, (8,))


def test_a_palette_tiff_loads_back_as_the_same_bgr_pixels(tmp_path: Path) -> None:
    pixels = posterize(_scan(), 4)
    path = tmp_path / "page.tif"

    write_image(path, pixels)
    loaded, _ = load_image(ImageId(file_path=path))

    assert np.array_equal(loaded, pixels)


def test_a_page_of_exactly_256_colours_still_gets_a_palette(tmp_path: Path) -> None:
    pixels = _colours(64, 64, 256)
    assert len(np.unique(pixels.reshape(-1, 3), axis=0)) == 256

    with _write_and_open(tmp_path, pixels) as image:
        assert image.mode == "P"
        assert np.array_equal(np.asarray(image.convert("RGB"))[..., ::-1], pixels)


def test_a_page_of_257_colours_stays_rgb(tmp_path: Path) -> None:
    pixels = _colours(64, 64, 257)
    assert len(np.unique(pixels.reshape(-1, 3), axis=0)) == 257

    with _write_and_open(tmp_path, pixels) as image:
        assert image.mode == "RGB"


def test_a_continuous_tone_page_stays_rgb(tmp_path: Path) -> None:
    with _write_and_open(tmp_path, _scan()) as image:
        assert image.mode == "RGB"


def test_a_page_of_grey_colours_is_written_as_grey(tmp_path: Path) -> None:
    levels = np.array([0, 85, 170, 255], dtype=np.uint8)
    grey = levels[np.random.default_rng(2).integers(0, 4, (40, 30))]
    pixels = np.dstack([grey, grey, grey])

    with _write_and_open(tmp_path, pixels) as image:
        assert image.mode == "L"
        assert np.array_equal(np.asarray(image), grey)


def test_jpeg_compression_keeps_the_page_rgb(tmp_path: Path) -> None:
    pixels = posterize(_scan(), 4)
    jpeg = TiffCompressions(color=TiffCompression.JPEG)

    with _write_and_open(tmp_path, pixels, compressions=jpeg) as image:
        assert image.mode == "RGB"


@pytest.mark.parametrize("method", [TiffCompression.LZW, TiffCompression.DEFLATE])
def test_the_palette_tiff_uses_the_colour_method(
    tmp_path: Path, method: TiffCompression
) -> None:
    pixels = posterize(_scan(), 4)

    with _write_and_open(
        tmp_path, pixels, compressions=TiffCompressions(color=method)
    ) as image:
        assert image.mode == "P"
        assert (
            image.info["compression"]
            == {
                TiffCompression.LZW: "tiff_lzw",
                TiffCompression.DEFLATE: "tiff_adobe_deflate",
            }[method]
        )


def test_a_posterized_png_stays_rgb(tmp_path: Path) -> None:
    path = tmp_path / "page.png"

    write_image(path, posterize(_scan(), 4))

    with Image.open(path) as image:
        assert image.mode == "RGB"
