"""Tests for EXIF orientation handling in scantailor.core.image_io."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scantailor.core.image_io import ImageLoadError, list_image_pages, load_image
from scantailor.core.models import Dpi, ImageId

_EXIF_ORIENTATION_TAG = 0x0112
_HEIGHT = 16
_WIDTH = 24
_BLOCK = 8


def _stored_pixels() -> np.ndarray:
    """A 16x24 gray image of 8x8 blocks, so every block has its own value."""
    values = np.arange(1, (_HEIGHT // _BLOCK) * (_WIDTH // _BLOCK) + 1) * 35
    blocks = values.reshape(_HEIGHT // _BLOCK, _WIDTH // _BLOCK).astype(np.uint8)
    return np.kron(blocks, np.ones((_BLOCK, _BLOCK), dtype=np.uint8))


def _upright(stored: np.ndarray, orientation: int) -> np.ndarray:
    """What an EXIF-aware viewer shows for pixels stored with this orientation."""
    return {
        1: stored,
        2: stored[:, ::-1],
        3: stored[::-1, ::-1],
        4: stored[::-1, :],
        5: stored.T,
        6: np.rot90(stored, -1),
        7: stored.T[::-1, ::-1],
        8: np.rot90(stored, 1),
    }[orientation]


def _write_jpeg(path: Path, orientation: int, dpi: tuple[int, int]) -> None:
    exif = Image.Exif()
    exif[_EXIF_ORIENTATION_TAG] = orientation
    Image.fromarray(_stored_pixels()).save(
        path, quality=100, subsampling=0, exif=exif, dpi=dpi
    )


def _write_tiff(path: Path, orientation: int, dpi: tuple[int, int]) -> None:
    tags = Image.Exif()
    tags[_EXIF_ORIENTATION_TAG] = orientation
    Image.fromarray(_stored_pixels()).save(
        path, tiffinfo=tags, dpi=dpi, compression="raw"
    )


@pytest.mark.parametrize(
    ("writer", "suffix"),
    [(_write_jpeg, ".jpg"), (_write_tiff, ".tif")],
    ids=["jpeg", "tiff"],
)
@pytest.mark.parametrize("orientation", range(1, 9))
def test_load_image_applies_exif_orientation_to_pixels(
    tmp_path: Path, writer, suffix: str, orientation: int
) -> None:
    path = tmp_path / f"photo{suffix}"
    writer(path, orientation, (300, 300))

    pixels, metadata = load_image(ImageId(file_path=path))

    expected = _upright(_stored_pixels(), orientation)
    assert pixels.shape == expected.shape
    assert (metadata.height, metadata.width) == expected.shape
    assert np.abs(pixels.astype(int) - expected.astype(int)).max() <= 6


@pytest.mark.parametrize(
    ("writer", "suffix"),
    [(_write_jpeg, ".jpg"), (_write_tiff, ".tif")],
    ids=["jpeg", "tiff"],
)
@pytest.mark.parametrize("orientation", range(1, 9))
def test_list_image_pages_reports_the_upright_size_and_dpi(
    tmp_path: Path, writer, suffix: str, orientation: int
) -> None:
    path = tmp_path / f"photo{suffix}"
    writer(path, orientation, (200, 400))

    [(_, metadata)] = list_image_pages(path)

    swapped = orientation >= 5
    assert (metadata.width, metadata.height) == (
        (_HEIGHT, _WIDTH) if swapped else (_WIDTH, _HEIGHT)
    )
    assert metadata.dpi == (
        Dpi(horizontal=400, vertical=200)
        if swapped
        else Dpi(horizontal=200, vertical=400)
    )


@pytest.mark.parametrize("orientation", [6, 8])
def test_load_image_swaps_dpi_for_quarter_turn_orientations(
    tmp_path: Path, orientation: int
) -> None:
    path = tmp_path / "photo.jpg"
    _write_jpeg(path, orientation, (200, 400))

    _, metadata = load_image(ImageId(file_path=path))

    assert metadata.dpi == Dpi(horizontal=400, vertical=200)


def test_load_image_keeps_dpi_for_half_turn_orientation(tmp_path: Path) -> None:
    path = tmp_path / "photo.tif"
    _write_tiff(path, 3, (200, 400))

    _, metadata = load_image(ImageId(file_path=path))

    assert metadata.dpi == Dpi(horizontal=200, vertical=400)


def test_load_image_dpi_override_applies_to_the_upright_image(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    _write_jpeg(path, 6, (200, 400))

    _, metadata = load_image(ImageId(file_path=path), Dpi(horizontal=111, vertical=222))

    assert metadata.dpi == Dpi(horizontal=111, vertical=222)


def test_image_without_orientation_tag_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "plain.tif"
    Image.fromarray(_stored_pixels()).save(path, compression="raw")

    pixels, _ = load_image(ImageId(file_path=path))

    assert np.array_equal(pixels, _stored_pixels())


def test_multipage_tiff_applies_each_frames_own_orientation(tmp_path: Path) -> None:
    path = tmp_path / "book.tif"
    frames = []
    for orientation in (1, 6):
        frame = Image.fromarray(_stored_pixels())
        tags = Image.Exif()
        tags[_EXIF_ORIENTATION_TAG] = orientation
        frame.encoderinfo = {"tiffinfo": tags, "dpi": (200, 400), "compression": "raw"}
        frames.append(frame)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], **frames[0].encoderinfo
    )

    pages = list_image_pages(path)
    sizes = [(info.width, info.height) for _, info in pages]
    assert sizes == [(_WIDTH, _HEIGHT), (_HEIGHT, _WIDTH)]

    first, _ = load_image(pages[0][0])
    second, second_meta = load_image(pages[1][0])
    assert np.array_equal(first, _stored_pixels())
    assert np.array_equal(second, _upright(_stored_pixels(), 6))
    assert second_meta.dpi == Dpi(horizontal=400, vertical=200)


@pytest.mark.parametrize("orientation", [3, 5, 6, 7, 8])
@pytest.mark.parametrize(
    "compression", ["raw", "tiff_lzw", "tiff_adobe_deflate", "packbits"]
)
def test_tiff_orientation_is_applied_once_whatever_the_compression(
    tmp_path: Path, compression: str, orientation: int
) -> None:
    path = tmp_path / "photo.tif"
    tags = Image.Exif()
    tags[_EXIF_ORIENTATION_TAG] = orientation
    Image.fromarray(_stored_pixels()).save(path, tiffinfo=tags, compression=compression)

    pixels, metadata = load_image(ImageId(file_path=path))

    assert np.array_equal(pixels, _upright(_stored_pixels(), orientation))
    assert (metadata.height, metadata.width) == pixels.shape


def test_bilevel_group4_tiff_with_orientation_loads_upright(tmp_path: Path) -> None:
    path = tmp_path / "bilevel.tif"
    stored = _stored_pixels() > 100
    tags = Image.Exif()
    tags[_EXIF_ORIENTATION_TAG] = 6
    Image.fromarray(stored).convert("1").save(path, tiffinfo=tags, compression="group4")

    pixels, _ = load_image(ImageId(file_path=path))

    expected = np.where(np.rot90(stored, -1), 255, 0)
    assert np.array_equal(pixels, expected.astype(np.uint8))


@pytest.mark.filterwarnings("ignore:Corrupt EXIF")
def test_unreadable_file_still_raises_image_load_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.tif"
    path.write_bytes(b"II*\x00" + b"\xff" * 12)

    with pytest.raises(ImageLoadError):
        load_image(ImageId(file_path=path))
