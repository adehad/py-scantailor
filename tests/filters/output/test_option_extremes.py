"""The output filter renders at both ends of every widened option range."""

import numpy as np
import pytest

from scantailor.filters.output.binarization import (
    MAX_COEFFICIENT,
    MAX_WINDOW_SIZE,
    MIN_COEFFICIENT,
    MIN_WINDOW_SIZE,
    BinarizationMethod,
    BinarizationOptions,
)
from scantailor.filters.output.generator import _binarize
from scantailor.imageproc.filters import wiener_filter


def _small_page() -> np.ndarray:
    rng = np.random.default_rng(7)
    page = np.full((40, 30), 220, dtype=np.uint8)
    page[10:14, 4:26] = 30
    page[20:24, 4:20] = 40
    return np.clip(page + rng.integers(-10, 10, page.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("method", list(BinarizationMethod))
@pytest.mark.parametrize("window", [MIN_WINDOW_SIZE, MAX_WINDOW_SIZE])
@pytest.mark.parametrize("coefficient", [MIN_COEFFICIENT, MAX_COEFFICIENT])
def test_every_method_binarizes_a_page_smaller_than_the_window_at_the_range_ends(
    method: BinarizationMethod, window: int, coefficient: float
) -> None:
    options = BinarizationOptions(
        method=method,
        window_size=window,
        sauvola_coef=coefficient,
        wolf_coef=coefficient,
        bradley_coef=coefficient,
        edge_div_kep=coefficient,
        edge_div_kbd=coefficient,
    )

    binary = _binarize(_small_page(), options)

    assert binary.shape == (40, 30)
    assert set(np.unique(binary)) <= {0, 255}


@pytest.mark.parametrize("window", [MIN_WINDOW_SIZE, MAX_WINDOW_SIZE])
def test_the_wiener_filter_keeps_the_page_size_at_the_window_range_ends(
    window: int,
) -> None:
    filtered = wiener_filter(_small_page(), window, 255.0)

    assert filtered.shape == (40, 30)
    assert filtered.dtype == np.uint8
