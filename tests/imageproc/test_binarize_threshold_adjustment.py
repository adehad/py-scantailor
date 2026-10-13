"""Threshold adjustment of the binarization methods.

Otsu shifts its global threshold and clamps it; the local methods take the
adjustment as a delta that enters each threshold the way ScanTailor Advanced
1.2.1 does it.
"""

import numpy as np

from scantailor.imageproc.binarize import (
    binarize_bradley,
    binarize_edge_div,
    binarize_otsu,
    binarize_sauvola,
    binarize_wolf,
)

INK = 0
BACKGROUND = 255


def _gray_ramp() -> np.ndarray:
    """Return a 4 x 256 image whose column ``x`` has gray level ``x``."""
    return np.tile(np.arange(256, dtype=np.uint8), (4, 1))


def _ink_columns(binary: np.ndarray) -> int:
    """Count the columns of a ramp result that are ink."""
    return int(np.count_nonzero(binary[0] == INK))


def _flat(level: int) -> np.ndarray:
    """Return a uniform 40 x 40 image of the given gray level."""
    return np.full((40, 40), level, dtype=np.uint8)


class TestOtsuAdjustment:
    """Otsu shifts its threshold by the adjustment and clamps the result."""

    def test_positive_adjustment_turns_more_pixels_black(self) -> None:
        ramp = _gray_ramp()

        base = _ink_columns(binarize_otsu(ramp))
        shifted = _ink_columns(binarize_otsu(ramp, adjustment=10))

        assert shifted == base + 10

    def test_negative_adjustment_turns_fewer_pixels_black(self) -> None:
        ramp = _gray_ramp()

        base = _ink_columns(binarize_otsu(ramp))
        shifted = _ink_columns(binarize_otsu(ramp, adjustment=-10))

        assert shifted == base - 10

    def test_threshold_is_clamped_to_the_bounds(self) -> None:
        ramp = _gray_ramp()

        darkest = binarize_otsu(ramp, adjustment=-1000, bounds=(30, 225))
        brightest = binarize_otsu(ramp, adjustment=1000, bounds=(30, 225))

        assert _ink_columns(darkest) == 30
        assert _ink_columns(brightest) == 225

    def test_default_bounds_do_not_clamp(self) -> None:
        ramp = _gray_ramp()

        assert _ink_columns(binarize_otsu(ramp, adjustment=-1000)) == 0
        assert _ink_columns(binarize_otsu(ramp, adjustment=1000)) == 256

    def test_adjustment_moves_the_threshold_not_the_pixels(self) -> None:
        """A shifted image would change the clamped result; the threshold must not."""
        image = np.full((20, 20), 40, dtype=np.uint8)
        image[:, 10:] = 200

        result = binarize_otsu(image, adjustment=-100, bounds=(30, 225))

        # The Otsu threshold is 120; minus 100 clamps to 30, below the dark level.
        assert np.all(result == BACKGROUND)

    def test_input_is_not_modified(self) -> None:
        ramp = _gray_ramp()
        before = ramp.copy()

        binarize_otsu(ramp, adjustment=25, bounds=(30, 225))

        assert np.array_equal(ramp, before)


class TestLocalMethodsTakeDelta:
    """On a flat page every local threshold is known, so delta is exact."""

    def test_bradley_delta_is_added_to_the_threshold(self) -> None:
        image = _flat(100)  # Threshold 100 * (1 - 0.2) = 80.

        assert np.all(binarize_bradley(image, k=0.2, delta=0.0) == BACKGROUND)
        assert np.all(binarize_bradley(image, k=0.2, delta=19.0) == BACKGROUND)
        assert np.all(binarize_bradley(image, k=0.2, delta=21.0) == INK)

    def test_bradley_delta_with_k_of_one_is_the_whole_threshold(self) -> None:
        image = _flat(5)

        assert np.all(binarize_bradley(image, k=1.0, delta=5.0) == BACKGROUND)
        assert np.all(binarize_bradley(image, k=1.0, delta=6.0) == INK)

    def test_sauvola_delta_adds_delta_over_128_to_the_contrast_term(self) -> None:
        image = _flat(
            100
        )  # Zero deviation: threshold 100 * (1 - 0.2 * (1 - delta / 128)).

        assert np.all(binarize_sauvola(image, k=0.2, delta=-5.0) == BACKGROUND)
        assert np.all(binarize_sauvola(image, k=0.2, delta=0.0) == BACKGROUND)
        assert np.all(binarize_sauvola(image, k=0.2, delta=127.0) == BACKGROUND)
        assert np.all(binarize_sauvola(image, k=0.2, delta=130.0) == INK)

    def test_wolf_delta_adds_delta_over_128_to_the_contrast_term(self) -> None:
        # A window over the whole page gives every pixel the same statistics,
        # a mean just under 100, and the page minimum is the single 0 pixel.
        image = _flat(100)
        image[0, 0] = 0

        background = binarize_wolf(
            image, window_size=101, k=0.3, delta=0.0, lower_bound=0
        )
        shifted = binarize_wolf(image, window_size=101, k=0.3, delta=1.0, lower_bound=0)

        assert np.all(background[image == 100] == BACKGROUND)
        assert np.all(shifted[image == 100] == INK)

    def test_wolf_lower_bound_forces_ink(self) -> None:
        image = _flat(100)

        result = binarize_wolf(image, k=0.3, delta=-50.0, lower_bound=101)

        assert np.all(result == INK)

    def test_wolf_upper_bound_forces_background(self) -> None:
        image = _flat(100)

        result = binarize_wolf(image, k=0.3, delta=50.0, upper_bound=99)

        assert np.all(result == BACKGROUND)

    def test_edge_div_delta_is_added_to_the_unclamped_otsu_threshold(self) -> None:
        image = np.full((40, 40), 10, dtype=np.uint8)
        image[:, 20:] = 220

        low = binarize_edge_div(image, delta=-300.0)
        high = binarize_edge_div(image, delta=300.0)

        assert np.all(low == BACKGROUND)
        assert np.all(high == INK)

    def test_edge_div_positive_delta_never_removes_ink(self) -> None:
        rng = np.random.default_rng(3)
        image = rng.integers(40, 220, (60, 60), dtype=np.uint8)

        base = binarize_edge_div(image) == INK
        more = binarize_edge_div(image, delta=15.0) == INK

        assert np.all(more[base])
        assert more.sum() >= base.sum()

    def test_local_methods_do_not_modify_the_input(self) -> None:
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, (50, 50), dtype=np.uint8)
        before = image.copy()

        binarize_sauvola(image, delta=10.0)
        binarize_wolf(image, delta=10.0)
        binarize_bradley(image, delta=10.0)
        binarize_edge_div(image, delta=10.0)

        assert np.array_equal(image, before)
