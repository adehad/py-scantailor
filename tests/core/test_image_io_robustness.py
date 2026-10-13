"""Decompression limit, atomic writes, alpha, resolution and mode I handling."""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from scantailor.core import image_io
from scantailor.core.image_io import (
    ImageLoadError,
    list_image_pages,
    load_image,
    write_image,
)
from scantailor.core.models import Dpi, ImageId


def _tiff_with_resolution(path: Path, x_res: tuple[int, int], y_res: tuple[int, int]):
    directory = TiffImagePlugin.ImageFileDirectory_v2()
    directory[282] = TiffImagePlugin.IFDRational(*x_res)
    directory[283] = TiffImagePlugin.IFDRational(*y_res)
    directory[296] = 2
    Image.new("L", (8, 8), 128).save(path, tiffinfo=directory)


def test_decompression_limit_is_large_but_finite() -> None:
    limit = Image.MAX_IMAGE_PIXELS

    assert limit is not None
    assert limit == image_io.MAX_IMAGE_PIXELS
    assert limit > 89_478_485


def test_image_beyond_twice_the_limit_is_an_image_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.png"
    write_image(path, np.zeros((40, 40), dtype=np.uint8))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ImageLoadError):
        load_image(ImageId(file_path=path))
    with pytest.raises(ImageLoadError):
        list_image_pages(path)


@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_failed_write_leaves_the_earlier_file_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    path = tmp_path / f"page{suffix}"
    original = np.full((10, 10), 7, dtype=np.uint8)
    write_image(path, original)
    before = path.read_bytes()

    def fail(self: Image.Image, fp: object, *args: object, **kwargs: object) -> None:
        Path(str(fp)).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        write_image(path, np.zeros((10, 10), dtype=np.uint8))

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_write_replaces_an_existing_file_with_the_new_pixels(tmp_path: Path) -> None:
    path = tmp_path / "page.tif"
    write_image(path, np.full((6, 6), 1, dtype=np.uint8))
    write_image(path, np.full((6, 6), 200, dtype=np.uint8))

    loaded, _ = load_image(ImageId(file_path=path))

    assert np.all(loaded == 200)
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_bgra_is_written_with_its_alpha_channel(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"a{suffix}"
    bgra = np.zeros((4, 5, 4), dtype=np.uint8)
    bgra[..., 0] = 10  # blue
    bgra[..., 1] = 20  # green
    bgra[..., 2] = 30  # red
    bgra[..., 3] = np.arange(20, dtype=np.uint8).reshape(4, 5) * 10

    write_image(path, bgra, Dpi.uniform(150))

    with Image.open(path) as raw:
        assert raw.mode == "RGBA"
        rgba = np.asarray(raw)
    assert np.array_equal(rgba[..., :3], bgra[..., [2, 1, 0]])
    assert np.array_equal(rgba[..., 3], bgra[..., 3])
    loaded, meta = load_image(ImageId(file_path=path))
    assert loaded.shape == (4, 5, 3)
    assert meta.dpi == Dpi.uniform(150)


def test_bgra_tiff_with_jpeg_colour_method_still_keeps_alpha(tmp_path: Path) -> None:
    from scantailor.core.settings import (
        TiffCompression,
        TiffCompressions,
    )

    path = tmp_path / "a.tif"
    bgra = np.full((4, 4, 4), 90, dtype=np.uint8)

    write_image(path, bgra, compressions=TiffCompressions(color=TiffCompression.JPEG))

    with Image.open(path) as raw:
        assert raw.mode == "RGBA"


def test_16_bit_colour_is_refused_with_a_message_naming_the_reason(
    tmp_path: Path,
) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint16)

    with pytest.raises(ValueError, match="16-bit colour"):
        write_image(tmp_path / "x.tif", pixels)
    assert list(tmp_path.iterdir()) == []


def test_16_bit_colour_tiff_loads_as_8_bit(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "deep.tif"
    deep = np.zeros((5, 6, 3), dtype=np.uint16)
    deep[..., 0] = 65535
    assert cv2.imwrite(str(path), deep)

    loaded, _ = load_image(ImageId(file_path=path))

    assert loaded.shape == (5, 6, 3)
    assert np.all(loaded[..., 0] == 255)
    assert np.all(loaded[..., 1:] == 0)


@pytest.mark.parametrize(
    ("x_res", "y_res"),
    [((300, 0), (300, 1)), ((300, 1), (0, 0)), ((0, 1), (0, 1))],
)
def test_zero_denominator_or_zero_tiff_resolution_is_unknown_dpi(
    tmp_path: Path, x_res: tuple[int, int], y_res: tuple[int, int]
) -> None:
    path = tmp_path / "bad.tif"
    _tiff_with_resolution(path, x_res, y_res)

    _, meta = load_image(ImageId(file_path=path))
    ((_, listed),) = list_image_pages(path)

    assert meta.dpi == Dpi()
    assert listed.dpi == Dpi()


def test_valid_rational_resolution_is_still_read(tmp_path: Path) -> None:
    path = tmp_path / "ok.tif"
    _tiff_with_resolution(path, (300, 1), (600, 1))

    _, meta = load_image(ImageId(file_path=path))

    assert meta.dpi == Dpi(horizontal=300, vertical=600)


def test_dpi_value_rejects_non_finite_numbers() -> None:
    assert image_io._dpi_value(math.nan) == 0
    assert image_io._dpi_value(math.inf) == 0
    assert image_io._dpi_value(-5) == 0
    assert image_io._dpi_value(299.6) == 300


def test_mode_i_samples_are_clipped_before_scaling_to_8_bits(tmp_path: Path) -> None:
    path = tmp_path / "deep.tif"
    samples = np.array([[-5, 0, 65535, 70000, 32768]], dtype=np.int32)
    Image.fromarray(samples, mode="I").save(path)

    loaded, _ = load_image(ImageId(file_path=path))

    assert loaded.tolist() == [[0, 0, 255, 255, 128]]
