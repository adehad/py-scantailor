"""The output generator applies the threshold adjustment per binarization method."""

import numpy as np
import pytest

from scantailor.filters.output.binarization import (
    BinarizationMethod,
    BinarizationOptions,
)
from scantailor.filters.output.generator import _binarize
from scantailor.imageproc.binarize import (
    binarize_edge_div,
    binarize_fox,
    binarize_grad,
    binarize_sauvola,
    binarize_window,
)

INK = 0
BACKGROUND = 255


def _flat(level: int) -> np.ndarray:
    return np.full((40, 40), level, dtype=np.uint8)


def _options(
    method: BinarizationMethod, adjustment: int, **extra
) -> BinarizationOptions:
    extra.setdefault("window_size", 15)
    return BinarizationOptions(method=method, threshold_adjustment=adjustment, **extra)


class TestOtsu:
    def test_threshold_is_clamped_to_30_after_the_adjustment(self) -> None:
        image = np.full((20, 20), 40, dtype=np.uint8)
        image[:, 10:] = 200

        result = _binarize(image, _options(BinarizationMethod.OTSU, -100))

        # Otsu gives 120; 120 - 100 = 20 clamps to 30, which is below the dark level.
        assert np.all(result == BACKGROUND)

    def test_threshold_is_clamped_to_225_after_the_adjustment(self) -> None:
        image = np.full((20, 20), 220, dtype=np.uint8)
        image[:, 10:] = 250

        result = _binarize(image, _options(BinarizationMethod.OTSU, 100))

        # Otsu gives 236; 236 + 100 clamps to 225, which is above the darker half.
        assert np.all(result[:, :10] == INK)
        assert np.all(result[:, 10:] == BACKGROUND)

    def test_positive_adjustment_keeps_more_gray_as_ink(self) -> None:
        ramp = np.tile(np.arange(256, dtype=np.uint8), (4, 1))

        plain = _binarize(ramp, _options(BinarizationMethod.OTSU, 0))
        darker = _binarize(ramp, _options(BinarizationMethod.OTSU, 15))

        assert np.count_nonzero(darker == INK) == np.count_nonzero(plain == INK) + 60

    def test_input_is_not_modified(self) -> None:
        image = np.full((20, 20), 40, dtype=np.uint8)
        image[:, 10:] = 200
        before = image.copy()

        _binarize(image, _options(BinarizationMethod.OTSU, 30))

        assert np.array_equal(image, before)


class TestLocalMethods:
    def test_bradley_adjustment_is_a_threshold_delta(self) -> None:
        image = _flat(100)
        plain = _options(BinarizationMethod.BRADLEY, 0, bradley_coef=0.2)
        darker = _options(BinarizationMethod.BRADLEY, 21, bradley_coef=0.2)

        assert np.all(_binarize(image, plain) == BACKGROUND)
        assert np.all(_binarize(image, darker) == INK)

    def test_sauvola_adjustment_is_a_contrast_delta(self) -> None:
        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, (36, 36), dtype=np.uint8)
        options = _options(BinarizationMethod.SAUVOLA, 60, sauvola_coef=0.2)

        expected = binarize_sauvola(image, window_size=15, k=0.2, delta=60)

        assert np.array_equal(_binarize(image, options), expected)
        plain = _binarize(
            image, _options(BinarizationMethod.SAUVOLA, 0, sauvola_coef=0.2)
        )
        assert np.count_nonzero(expected == INK) > np.count_nonzero(plain == INK)

    def test_wolf_adjustment_is_a_contrast_delta(self) -> None:
        image = _flat(100)
        image[0, 0] = 0
        plain = _options(
            BinarizationMethod.WOLF, 0, window_size=101, wolf_lower_bound=0
        )
        darker = _options(
            BinarizationMethod.WOLF, 1, window_size=101, wolf_lower_bound=0
        )

        assert np.all(_binarize(image, plain)[image == 100] == BACKGROUND)
        assert np.all(_binarize(image, darker)[image == 100] == INK)

    def test_wolf_bounds_come_from_the_options(self) -> None:
        image = _flat(100)

        always_ink = _options(BinarizationMethod.WOLF, -50, wolf_lower_bound=101)
        never_ink = _options(BinarizationMethod.WOLF, 50, wolf_upper_bound=99)

        assert np.all(_binarize(image, always_ink) == INK)
        assert np.all(_binarize(image, never_ink) == BACKGROUND)

    @pytest.mark.parametrize(
        ("method", "function"),
        [
            (BinarizationMethod.FOX, binarize_fox),
            (BinarizationMethod.WINDOW, binarize_window),
            (BinarizationMethod.GRAD, binarize_grad),
        ],
    )
    def test_wolf_family_methods_use_the_wolf_coefficient_and_bounds(
        self, method: BinarizationMethod, function
    ) -> None:
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, (36, 36), dtype=np.uint8)
        options = _options(
            method,
            9,
            wolf_coef=0.42,
            wolf_lower_bound=20,
            wolf_upper_bound=230,
            sauvola_coef=0.9,
        )

        expected = function(
            image, window_size=15, k=0.42, delta=9, lower_bound=20, upper_bound=230
        )

        assert np.array_equal(_binarize(image, options), expected)

    def test_local_methods_are_not_clamped_like_otsu(self) -> None:
        """A flat dark page stays ink at adjustment 0 for Bradley; no 30..225 clamp."""
        image = _flat(10)
        options = _options(BinarizationMethod.BRADLEY, 0, bradley_coef=0.2)

        assert np.all(_binarize(image, options) == BACKGROUND)

    def test_edge_div_adjustment_shifts_the_unclamped_otsu_threshold(self) -> None:
        rng = np.random.default_rng(11)
        image = rng.integers(40, 220, (50, 50), dtype=np.uint8)
        options = _options(BinarizationMethod.EDGEDIV, 12)

        expected = binarize_edge_div(
            image,
            window_size=options.window_size,
            kep=options.edge_div_kep,
            kbd=options.edge_div_kbd,
            delta=12,
        )

        assert np.array_equal(_binarize(image, options), expected)
