"""Tests for scantailor.core.image_io."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scantailor.core.image_io import (
    ImageLoadError,
    UnsupportedImageFormatError,
    list_image_pages,
    load_image,
    write_image,
)
from scantailor.core.models import Dpi, ImageId


def _gray(h: int = 20, w: int = 30) -> np.ndarray:
    return (np.arange(h * w).reshape(h, w) % 251).astype(np.uint8)


def _bgr(h: int = 20, w: int = 30) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_gray_round_trip_preserves_pixels_and_dpi(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"g{suffix}"
    pixels = _gray()
    write_image(path, pixels, Dpi(horizontal=300, vertical=600))
    loaded, meta = load_image(ImageId(file_path=path))
    assert np.array_equal(loaded, pixels)
    assert (meta.width, meta.height) == (30, 20)
    assert meta.dpi == Dpi(horizontal=300, vertical=600)


@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_bgr_round_trip_preserves_channel_order(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"c{suffix}"
    pixels = _bgr()
    write_image(path, pixels, Dpi.uniform(200))
    loaded, meta = load_image(ImageId(file_path=path))
    assert np.array_equal(loaded, pixels)
    assert meta.dpi == Dpi.uniform(200)


def test_bilevel_tiff_is_one_bit_group4_and_loads_black_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "b.tif"
    black = np.zeros((16, 40), dtype=bool)
    black[4:8, 5:30] = True
    write_image(path, black, Dpi.uniform(600))
    with Image.open(path) as raw:
        assert raw.mode == "1"
        assert raw.info["compression"] == "group4"
    loaded, meta = load_image(ImageId(file_path=path))
    assert np.array_equal(loaded, np.where(black, 0, 255).astype(np.uint8))
    assert meta.dpi == Dpi.uniform(600)


def test_color_tiff_uses_lzw(tmp_path: Path) -> None:
    path = tmp_path / "c.tif"
    write_image(path, _bgr())
    with Image.open(path) as raw:
        assert raw.info["compression"] == "tiff_lzw"


@pytest.mark.parametrize("suffix", [".tif", ".png"])
def test_one_by_one_image_round_trips(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"p{suffix}"
    write_image(path, np.array([[77]], dtype=np.uint8))
    loaded, meta = load_image(ImageId(file_path=path))
    assert loaded.tolist() == [[77]]
    assert (meta.width, meta.height) == (1, 1)


def test_empty_array_rejected_without_leaving_a_file(tmp_path: Path) -> None:
    path = tmp_path / "e.tif"
    with pytest.raises(ValueError, match="empty"):
        write_image(path, np.zeros((0, 0), dtype=np.uint8))
    assert not path.exists()


def test_sixteen_bit_tiff_scaled_to_eight_bit(tmp_path: Path) -> None:
    path = tmp_path / "s16.tif"
    Image.fromarray(np.array([[0, 1000, 65535]], dtype=np.uint16)).save(path)
    pixels, _ = load_image(ImageId(file_path=path))
    assert pixels.tolist() == [[0, 4, 255]]


def test_write_does_not_mutate_input(tmp_path: Path) -> None:
    pixels = _bgr()
    before = pixels.copy()
    write_image(tmp_path / "x.tif", pixels)
    assert np.array_equal(pixels, before)


def test_missing_dpi_gives_null_dpi(tmp_path: Path) -> None:
    path = tmp_path / "n.png"
    write_image(path, _gray())
    _, meta = load_image(ImageId(file_path=path))
    assert meta.dpi.is_null()
    assert not meta.is_dpi_ok()


def test_jpeg_dpi_and_size(tmp_path: Path) -> None:
    path = tmp_path / "j.jpg"
    Image.fromarray(_gray(), mode="L").save(path, dpi=(300, 300))
    pages = list_image_pages(path)
    assert pages[0][1].dpi == Dpi.uniform(300)
    pixels, _ = load_image(ImageId(file_path=path))
    assert pixels.shape == (20, 30)


def test_jpeg_exif_resolution_without_jfif_density_gives_null_dpi(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exif_only.jpg"
    exif = Image.Exif()
    exif[282] = 200.0
    exif[283] = 200.0
    exif[296] = 2
    Image.fromarray(_gray(), mode="L").save(path, exif=exif)
    assert list_image_pages(path)[0][1].dpi.is_null()


def test_tiff_centimetre_resolution_unit_converted_to_dpi(tmp_path: Path) -> None:
    path = tmp_path / "cm.tif"
    Image.fromarray(_gray(), mode="L").save(
        path, resolution=118.11, resolution_unit="cm"
    )
    _, meta = load_image(ImageId(file_path=path))
    assert meta.dpi == Dpi.uniform(300)


def test_multipage_tiff_page_selection_and_enumeration(tmp_path: Path) -> None:
    path = tmp_path / "m.tif"
    frames = [np.full((10 + i, 12), 40 * i, dtype=np.uint8) for i in range(3)]
    imgs = [Image.fromarray(f, mode="L") for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], dpi=(300, 300))

    pages = list_image_pages(path)
    assert [p.page for p, _ in pages] == [1, 2, 3]
    assert [m.height for _, m in pages] == [10, 11, 12]

    for image_id, _ in pages:
        pixels, meta = load_image(image_id)
        assert np.array_equal(pixels, frames[image_id.page - 1])
        assert meta.height == 9 + image_id.page


def test_single_page_file_enumerates_as_page_zero(tmp_path: Path) -> None:
    path = tmp_path / "s.png"
    write_image(path, _gray())
    assert [i.page for i, _ in list_image_pages(path)] == [0]


def test_page_out_of_range_raises(tmp_path: Path) -> None:
    path = tmp_path / "s.tif"
    write_image(path, _gray())
    with pytest.raises(ImageLoadError, match="out of range"):
        load_image(ImageId(file_path=path, page=2))


def test_page_on_non_tiff_raises(tmp_path: Path) -> None:
    path = tmp_path / "s.png"
    write_image(path, _gray())
    with pytest.raises(ImageLoadError, match="only TIFF"):
        load_image(ImageId(file_path=path, page=2))


def test_sixteen_bit_scaled_to_eight_bit(tmp_path: Path) -> None:
    path = tmp_path / "s16.png"
    samples = np.array([[0, 257, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(samples).save(path)
    pixels, _ = load_image(ImageId(file_path=path))
    assert pixels.dtype == np.uint8
    assert pixels.tolist() == [[0, 1, 128, 255]]


def test_gray_palette_loads_as_gray_and_colour_palette_as_bgr(tmp_path: Path) -> None:
    index = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    gray_path = tmp_path / "pg.png"
    img = Image.fromarray(index, mode="P")
    img.putpalette([10, 10, 10, 200, 200, 200])
    img.save(gray_path)
    gray, _ = load_image(ImageId(file_path=gray_path))
    assert gray.tolist() == [[10, 200], [200, 10]]

    color_path = tmp_path / "pc.png"
    img = Image.fromarray(index, mode="P")
    img.putpalette([255, 0, 0, 0, 0, 255])  # RGB red, blue
    img.save(color_path)
    color, _ = load_image(ImageId(file_path=color_path))
    assert color.shape == (2, 2, 3)
    assert color[0, 0].tolist() == [0, 0, 255]  # BGR red
    assert color[0, 1].tolist() == [255, 0, 0]  # BGR blue


def test_alpha_channel_discarded(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    Image.new("RGBA", (4, 4), (1, 2, 3, 128)).save(path)
    pixels, _ = load_image(ImageId(file_path=path))
    assert pixels.shape == (4, 4, 3)
    assert pixels[0, 0].tolist() == [3, 2, 1]


def test_unsupported_file_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "notes.png"
    path.write_text("not an image")
    with pytest.raises(UnsupportedImageFormatError):
        load_image(ImageId(file_path=path))
    with pytest.raises(UnsupportedImageFormatError):
        list_image_pages(path)


def test_truncated_file_raises_load_error(tmp_path: Path) -> None:
    good = tmp_path / "g.png"
    write_image(good, _bgr(64, 64))
    bad = tmp_path / "bad.png"
    bad.write_bytes(good.read_bytes()[:80])
    with pytest.raises(ImageLoadError):
        load_image(ImageId(file_path=bad))


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(ImageId(file_path=tmp_path / "nope.tif"))


@pytest.mark.parametrize(
    ("name", "array"),
    [
        ("x.bmp", np.zeros((2, 2), np.uint8)),
        ("x.tif", np.zeros((2, 2), np.float32)),
        ("x.tif", np.zeros((2, 2, 4), np.uint8)),
    ],
)
def test_write_rejects_unsupported_input(
    tmp_path: Path, name: str, array: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        write_image(tmp_path / name, array)
