"""EdgePlus and BlurDiv run the edge-div binarization with one of its two terms."""

import numpy as np
import pytest

from scantailor.filters.output.binarization import (
    BinarizationMethod,
    BinarizationOptions,
)
from scantailor.filters.output.generator import _binarize
from scantailor.imageproc.binarize import binarize_edge_div

WINDOW = 15


def _page() -> np.ndarray:
    """A noisy gradient page, so each term of the edge-div formula matters."""
    rng = np.random.default_rng(5)
    gradient = np.linspace(80, 220, 60)[None, :].repeat(50, axis=0)
    return np.clip(gradient + rng.normal(0, 25, gradient.shape), 0, 255).astype(
        np.uint8
    )


def _options(method: BinarizationMethod, **extra: float) -> BinarizationOptions:
    return BinarizationOptions(method=method, window_size=WINDOW, **extra)


class TestEdgePlus:
    def test_matches_edge_div_with_the_blur_term_switched_off(self) -> None:
        page = _page()
        options = _options(BinarizationMethod.EDGEPLUS, edge_div_kep=0.8)

        expected = binarize_edge_div(page, window_size=WINDOW, kep=0.8, kbd=0.0)

        assert np.array_equal(_binarize(page, options), expected)

    def test_ignores_the_blur_weight(self) -> None:
        page = _page()
        low = _options(BinarizationMethod.EDGEPLUS, edge_div_kbd=0.0)
        high = _options(BinarizationMethod.EDGEPLUS, edge_div_kbd=1.0)

        assert np.array_equal(_binarize(page, low), _binarize(page, high))

    def test_follows_the_edge_weight(self) -> None:
        page = _page()
        low = _options(BinarizationMethod.EDGEPLUS, edge_div_kep=0.1)
        high = _options(BinarizationMethod.EDGEPLUS, edge_div_kep=1.0)

        assert not np.array_equal(_binarize(page, low), _binarize(page, high))


class TestBlurDiv:
    def test_matches_edge_div_with_the_edge_term_switched_off(self) -> None:
        page = _page()
        options = _options(BinarizationMethod.BLURDIV, edge_div_kbd=0.7)

        expected = binarize_edge_div(page, window_size=WINDOW, kep=0.0, kbd=0.7)

        assert np.array_equal(_binarize(page, options), expected)

    def test_ignores_the_edge_weight(self) -> None:
        page = _page()
        low = _options(BinarizationMethod.BLURDIV, edge_div_kep=0.0)
        high = _options(BinarizationMethod.BLURDIV, edge_div_kep=1.0)

        assert np.array_equal(_binarize(page, low), _binarize(page, high))

    def test_follows_the_blur_weight(self) -> None:
        page = _page()
        low = _options(BinarizationMethod.BLURDIV, edge_div_kbd=0.1)
        high = _options(BinarizationMethod.BLURDIV, edge_div_kbd=1.0)

        assert not np.array_equal(_binarize(page, low), _binarize(page, high))


class TestEdgeDiv:
    def test_uses_both_weights(self) -> None:
        page = _page()
        options = _options(
            BinarizationMethod.EDGEDIV, edge_div_kep=0.6, edge_div_kbd=0.4
        )

        expected = binarize_edge_div(page, window_size=WINDOW, kep=0.6, kbd=0.4)

        assert np.array_equal(_binarize(page, options), expected)


@pytest.mark.parametrize(
    ("name", "method"),
    [
        ("edgeplus", BinarizationMethod.EDGEPLUS),
        ("blurdiv", BinarizationMethod.BLURDIV),
    ],
)
def test_the_method_round_trips_through_its_string_value(
    name: str, method: BinarizationMethod
) -> None:
    assert BinarizationMethod.from_string(name) is method
    assert method.value == name
