"""The decoded source of a page is shared by everything that shows or renders it."""

import threading
from pathlib import Path

import numpy as np
import pytest

from scantailor.app.ui import source_cache
from scantailor.app.ui.source_cache import SourceCache
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId

FILE_DPI = 72
FIXED_DPI = 300


def _image(path: Path, level: int = 40, dpi: int = FILE_DPI) -> ImageId:
    pixels = np.full((80, 60), 220, dtype=np.uint8)
    pixels[20:60, 15:45] = level
    write_image(path, pixels, Dpi.uniform(dpi))
    return ImageId(file_path=path)


@pytest.fixture
def decodes(monkeypatch: pytest.MonkeyPatch) -> list[ImageId]:
    """Every decode that reaches the image loader."""
    seen: list[ImageId] = []
    real = source_cache.load_page

    def counting(image_id, dpi_overrides=None):
        seen.append(image_id)
        return real(image_id, dpi_overrides)

    monkeypatch.setattr(source_cache, "load_page", counting)
    return seen


def test_a_page_is_decoded_once_however_often_it_is_loaded(
    tmp_path: Path, decodes: list[ImageId]
) -> None:
    cache = SourceCache()
    image = _image(tmp_path / "a.tif")

    first = cache.load(image)
    second = cache.load(image)

    assert second.pixels is first.pixels
    assert decodes == [image]


def test_shared_pixels_cannot_be_modified(tmp_path: Path) -> None:
    cache = SourceCache()

    page = cache.load(_image(tmp_path / "a.tif"))

    assert not page.pixels.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        page.pixels[0, 0] = 1


def test_peek_returns_only_what_was_loaded(tmp_path: Path) -> None:
    cache = SourceCache()
    image = _image(tmp_path / "a.tif")

    assert cache.peek(image) is None
    loaded = cache.load(image)

    assert cache.peek(image) is loaded


def test_a_page_loaded_without_keeping_is_not_cached(
    tmp_path: Path, decodes: list[ImageId]
) -> None:
    cache = SourceCache()
    image = _image(tmp_path / "a.tif")

    cache.load(image, keep=False)

    assert cache.peek(image) is None
    cache.load(image)
    assert decodes == [image, image]


def test_a_cached_page_is_served_to_a_load_that_does_not_keep(
    tmp_path: Path, decodes: list[ImageId]
) -> None:
    cache = SourceCache()
    image = _image(tmp_path / "a.tif")
    kept = cache.load(image)

    passing = cache.load(image, keep=False)

    assert passing.pixels is kept.pixels
    assert decodes == [image]


def test_the_oldest_page_is_dropped_beyond_capacity(
    tmp_path: Path, decodes: list[ImageId]
) -> None:
    cache = SourceCache(capacity=2)
    first, second, third = (_image(tmp_path / f"{n}.tif") for n in "abc")

    for image in (first, second, third):
        cache.load(image)

    assert cache.peek(first) is None
    assert cache.peek(second) is not None
    assert cache.peek(third) is not None


def test_a_project_resolution_gets_its_own_decode(tmp_path: Path) -> None:
    cache = SourceCache()
    image = _image(tmp_path / "a.tif")

    unfixed = cache.load(image)
    fixed = cache.load(image, {image: Dpi.uniform(FIXED_DPI)})

    assert unfixed.horizontal_dpi() == FILE_DPI
    assert fixed.horizontal_dpi() == FIXED_DPI
    assert cache.peek(image, {image: Dpi.uniform(FIXED_DPI)}) is fixed


def test_a_file_that_changed_on_disk_is_decoded_again(tmp_path: Path) -> None:
    cache = SourceCache()
    path = tmp_path / "a.tif"
    image = _image(path, level=40)
    before = cache.load(image)

    _image(path, level=90)
    after = cache.load(image)

    assert before.pixels[40, 30] == 40
    assert after.pixels[40, 30] == 90


def test_a_missing_file_raises_instead_of_being_cached(tmp_path: Path) -> None:
    cache = SourceCache()
    image = ImageId(file_path=tmp_path / "missing.tif")

    with pytest.raises(FileNotFoundError):
        cache.load(image)

    assert cache.peek(image) is None


def test_clear_forgets_every_page(tmp_path: Path) -> None:
    cache = SourceCache()
    image = _image(tmp_path / "a.tif")
    cache.load(image)

    cache.clear()

    assert cache.peek(image) is None


def test_threads_loading_one_page_decode_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = SourceCache()
    image = _image(tmp_path / "a.tif")
    real = source_cache.load_page
    started = threading.Event()
    release = threading.Event()
    decodes: list[ImageId] = []

    def slow(image_id, dpi_overrides=None):
        decodes.append(image_id)
        started.set()
        assert release.wait(timeout=10)
        return real(image_id, dpi_overrides)

    monkeypatch.setattr(source_cache, "load_page", slow)
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.load(image).pixels))
        for _ in range(3)
    ]
    threads[0].start()
    assert started.wait(timeout=10)
    for thread in threads[1:]:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(timeout=10)

    assert len(decodes) == 1
    assert len(results) == 3
    assert all(pixels is results[0] for pixels in results)
