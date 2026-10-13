"""Local binarization methods against a direct transcription of the C++ windows.

The C++ window of size s spans s // 2 pixels before and s - s // 2 after the
pixel, clipped to the image, so statistics at the border use fewer pixels.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc.binarize import (
    binarize_bradley,
    binarize_sauvola,
)

_SEED = 7


def _noisy_image(height: int = 23, width: int = 31) -> NDArray[np.uint8]:
    """Random gray image whose border windows differ from interior ones."""
    rng = np.random.default_rng(_SEED)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


def _window_mean_and_deviation(
    image: NDArray[np.uint8], y: int, x: int, size: int
) -> tuple[float, float]:
    height, width = image.shape
    before = size >> 1
    after = size - before
    window = image[
        max(0, y - before) : min(height, y + after),
        max(0, x - before) : min(width, x + after),
    ].astype(np.float64)
    return float(window.mean()), float(window.std())


def _reference(
    image: NDArray[np.uint8],
    size: int,
    threshold_of: Callable[[float, float], float],
) -> NDArray[np.uint8]:
    """Per-pixel transcription: ink where gray < threshold_of(mean, deviation)."""
    result = np.full(image.shape, 255, dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            mean, deviation = _window_mean_and_deviation(image, y, x, size)
            if int(image[y, x]) < threshold_of(mean, deviation):
                result[y, x] = 0
    return result


@pytest.mark.parametrize("size", [7, 8])
def test_sauvola_matches_clipped_window_reference(size: int) -> None:
    image = _noisy_image()
    k, delta = 0.3, 4.0

    expected = _reference(
        image,
        size,
        lambda mean, dev: mean * (1.0 - k * (1.0 - (dev / 128.0 + delta / 128.0))),
    )

    assert np.array_equal(binarize_sauvola(image, size, k, delta), expected)


@pytest.mark.parametrize("size", [7, 8])
def test_bradley_matches_clipped_window_reference(size: int) -> None:
    image = _noisy_image()
    k, delta = 0.15, -3.0

    expected = _reference(image, size, lambda mean, _dev: mean * (1.0 - k) + delta)

    assert np.array_equal(binarize_bradley(image, size, k, delta), expected)
