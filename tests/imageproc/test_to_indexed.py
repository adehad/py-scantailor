"""Splitting a page into a palette and an index image, chunk by chunk."""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc import posterizer
from scantailor.imageproc.posterizer import to_indexed

# The sample to_indexed takes first reads every 97th pixel.
SAMPLE_STRIDE = 97


def _rebuilt(palette: NDArray[np.uint8], indices: NDArray[np.uint8]) -> NDArray:
    return palette[indices][..., ::-1]


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunks of 1000 pixels, so that a small page spans several of them."""
    monkeypatch.setattr(posterizer, "_CHUNK_PIXELS", 1000)


def test_colours_found_in_a_later_chunk_keep_earlier_indices_right(
    small_chunks: None,
) -> None:
    pixels = np.zeros((60, 100, 3), dtype=np.uint8)
    # Colours of increasing value, each first seen in a later chunk and missed
    # by the sample, so the palette grows and renumbers the indices behind it.
    for row, value in ((15, 200), (35, 50), (55, 120)):
        pixels[row, 1] = (value, value // 2, 255 - value)

    indexed = to_indexed(pixels, 16)

    assert indexed is not None
    palette, indices = indexed
    assert len(palette) == 4
    assert np.array_equal(_rebuilt(palette, indices), pixels)


def test_too_many_colours_off_the_sample_return_none(small_chunks: None) -> None:
    pixels = np.zeros((60, 100, 3), dtype=np.uint8)
    flat = pixels.reshape(-1, 3)
    off_sample = np.flatnonzero(np.arange(len(flat)) % SAMPLE_STRIDE)[:300]
    flat[off_sample, 0] = np.arange(1, 301) % 256
    flat[off_sample, 1] = np.arange(1, 301) // 256

    assert to_indexed(pixels, 256) is None


def test_a_page_of_exactly_the_limit_is_indexed(small_chunks: None) -> None:
    rng = np.random.default_rng(4)
    table = rng.permutation(256 * 256)[:256]
    packed = table[rng.integers(0, 256, (60, 100))]
    pixels = np.dstack([packed & 0xFF, packed >> 8, np.zeros_like(packed)]).astype(
        np.uint8
    )
    assert len(np.unique(packed)) == 256

    indexed = to_indexed(pixels, 256)

    assert indexed is not None
    assert np.array_equal(_rebuilt(*indexed), pixels)
