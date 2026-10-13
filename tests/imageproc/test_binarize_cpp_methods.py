"""Fox, Window, Grad and the 1.2.1 delta of Sauvola and Wolf against C++ code.

The references follow ``Binarize.cpp`` of ScanTailor Advanced 1.2.1 line by line:
a summed-area table, ``windowSum * (1 / area)`` means, the same operation order in
every formula. They are slow and meant for small pages only.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc.binarize import (
    binarize_fox,
    binarize_grad,
    binarize_sauvola,
    binarize_window,
    binarize_wolf,
)

INK = 0
BACKGROUND = 255
_SEED = 11
_UPPER = 254


class _Windows:
    """Window sums of a gray image as the C++ integral image gives them."""

    def __init__(self, gray: NDArray[np.uint8], window_size: int) -> None:
        self.height, self.width = gray.shape
        self.before = window_size >> 1
        self.after = window_size - self.before
        pixels = gray.astype(np.int64)
        self._sum = self._integral(pixels)
        self._sqsum = self._integral(pixels * pixels)

    @staticmethod
    def _integral(values: NDArray[np.int64]) -> NDArray[np.int64]:
        table = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.int64)
        table[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
        return table

    def bounds(self, y: int, x: int) -> tuple[int, int, int, int]:
        top = max(0, y - self.before)
        bottom = min(self.height, y + self.after)
        left = max(0, x - self.before)
        right = min(self.width, x + self.after)
        return top, bottom, left, right

    def mean_and_sqmean(self, y: int, x: int) -> tuple[float, float]:
        top, bottom, left, right = self.bounds(y, x)
        area = (bottom - top) * (right - left)
        reciprocal = 1.0 / area

        def total(table: NDArray[np.int64]) -> float:
            return float(
                table[bottom, right]
                - table[top, right]
                + table[top, left]
                - table[bottom, left]
            )

        return total(self._sum) * reciprocal, total(self._sqsum) * reciprocal

    def mean_and_deviation(self, y: int, x: int) -> tuple[float, float]:
        mean, sqmean = self.mean_and_sqmean(y, x)
        return mean, float(np.sqrt(abs(sqmean - mean * mean)))


def _is_ink(level: int, lower: int, upper: int, threshold: float) -> bool:
    return level < lower or (level <= upper and float(level) < threshold)


def _picture(kind: str) -> NDArray[np.uint8]:
    """Small pages of different character: noise, text-like strokes, a gradient."""
    rng = np.random.default_rng(_SEED)
    if kind == "noise":
        return rng.integers(0, 256, size=(23, 31), dtype=np.uint8)
    if kind == "strokes":
        page = np.full((40, 57), 235, dtype=np.uint8)
        page[5:8, 4:50] = 30
        page[14:30:3, 10:40] = 60
        page[:, 22] = 90
        noise = rng.integers(-12, 13, size=page.shape)
        return np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ramp = np.tile(np.linspace(20, 240, 45), (28, 1))
    ramp[10:14, :] = 15
    return ramp.astype(np.uint8)


_PICTURES = ["noise", "strokes", "ramp"]
_WINDOW_SIZES = [7, 8, 61]


def _fox_reference(
    gray: NDArray[np.uint8],
    size: int,
    k: float,
    delta: float,
    lower: int,
    upper: int,
) -> NDArray[np.uint8]:
    windows = _Windows(gray, size)
    height, width = gray.shape
    darkest = float(gray.min())
    deviations = np.zeros(gray.shape)
    max_deviation = 0.0
    for y in range(height):
        for x in range(width):
            mean, _ = windows.mean_and_sqmean(y, x)
            di = float(gray[y, x]) - mean
            deviations[y, x] = di / (256.0 - di)
            max_deviation = max(max_deviation, deviations[y, x])
    max_deviation = max_deviation if max_deviation > 0.0 else 1.0

    frac_d = delta / 128.0
    result = np.full(gray.shape, BACKGROUND, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            mean, _ = windows.mean_and_sqmean(y, x)
            base = mean - darkest
            frac_sn = deviations[y, x] / max_deviation
            threshold = base * (1.0 - k * 0.5 * (1.0 - (frac_sn + frac_d))) + darkest
            if _is_ink(int(gray[y, x]), lower, upper, threshold):
                result[y, x] = INK
    return result


def _window_reference(
    gray: NDArray[np.uint8],
    size: int,
    k: float,
    delta: float,
    lower: int,
    upper: int,
) -> NDArray[np.uint8]:
    windows = _Windows(gray, size)
    height, width = gray.shape
    mean_full = float(int(gray.astype(np.uint64).sum()) // gray.size)
    deviations = np.zeros(gray.shape)
    for y in range(height):
        for x in range(width):
            deviations[y, x] = windows.mean_and_deviation(y, x)[1]
    deviation_max = 0.0
    deviation_min = 256.0
    for value in deviations.ravel():
        deviation_max = max(deviation_max, value)
        deviation_min = min(deviation_min, value)
    deviation_d = deviation_max - deviation_min

    coefw = k * 3.0
    result = np.full(gray.shape, BACKGROUND, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            mean, _ = windows.mean_and_sqmean(y, x)
            deviation = deviations[y, x]
            md = (mean + 1.0 - delta) / (mean_full + deviation + 1.0)
            kdm = (mean_full + mean_full + 1.0) / (deviation + 1.0)
            kds = (
                (deviation - deviation_min) / deviation_d if deviation_d > 0.0 else 1.0
            )
            kd = 1.0 + kdm * kds
            threshold = mean * (1.0 - coefw * md / kd)
            if _is_ink(int(gray[y, x]), lower, upper, threshold):
                result[y, x] = INK
    return result


def _grad_reference(
    gray: NDArray[np.uint8],
    size: int,
    k: float,
    delta: float,
    lower: int,
    upper: int,
) -> NDArray[np.uint8]:
    windows = _Windows(gray, size)
    height, width = gray.shape
    local = np.zeros(gray.shape, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            mean, _ = windows.mean_and_sqmean(y, x)
            mean = mean + 0.5 + delta
            local[y, x] = int(0.0 if mean < 0.0 else mean if mean < 255.0 else 255.0)

    sum_g = 0.0
    sum_gi = 0.0
    for y in range(height):
        for x in range(width):
            gi = float(gray[y, x])
            g = abs(float(local[y, x]) - gi)
            sum_g += g
            sum_gi += gi * g
    gvalue = sum_gi / sum_g if sum_g > 0.0 else 127.5
    mean_grad = gvalue * (1.0 - k)

    result = np.full(gray.shape, BACKGROUND, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            threshold = mean_grad + float(local[y, x]) * k
            if _is_ink(int(gray[y, x]), lower, upper, threshold):
                result[y, x] = INK
    return result


def _wolf_reference(
    gray: NDArray[np.uint8],
    size: int,
    k: float,
    delta: float,
    lower: int,
    upper: int,
) -> NDArray[np.uint8]:
    windows = _Windows(gray, size)
    height, width = gray.shape
    darkest = float(gray.min())
    means = np.zeros(gray.shape, dtype=np.float32)
    deviations = np.zeros(gray.shape, dtype=np.float32)
    max_deviation = 0.0
    for y in range(height):
        for x in range(width):
            mean, deviation = windows.mean_and_deviation(y, x)
            max_deviation = max(max_deviation, deviation)
            means[y, x] = np.float32(mean)
            deviations[y, x] = np.float32(deviation)
    max_deviation = max_deviation if max_deviation > 0.0 else 1.0

    frac_d = delta / 128.0
    result = np.full(gray.shape, BACKGROUND, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            base = float(means[y, x]) - darkest
            frac_sn = float(deviations[y, x]) / max_deviation
            threshold = base * (1.0 - k * (1.0 - (frac_sn + frac_d))) + darkest
            if _is_ink(int(gray[y, x]), lower, upper, threshold):
                result[y, x] = INK
    return result


def _sauvola_reference(
    gray: NDArray[np.uint8], size: int, k: float, delta: float
) -> NDArray[np.uint8]:
    windows = _Windows(gray, size)
    frac_d = delta / 128.0
    result = np.full(gray.shape, BACKGROUND, dtype=np.uint8)
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            mean, deviation = windows.mean_and_deviation(y, x)
            frac_s = deviation / 128.0
            threshold = mean * (1.0 - k * (1.0 - (frac_s + frac_d)))
            if float(gray[y, x]) < threshold:
                result[y, x] = INK
    return result


@pytest.mark.parametrize("picture", _PICTURES)
@pytest.mark.parametrize("size", _WINDOW_SIZES)
@pytest.mark.parametrize(("k", "delta"), [(0.3, 0.0), (0.45, 7.0), (0.2, -12.0)])
class TestAgainstTheCppTranscription:
    def test_fox_matches(self, picture: str, size: int, k: float, delta: float):
        gray = _picture(picture)

        actual = binarize_fox(gray, size, k, delta, 1, _UPPER)

        assert np.array_equal(actual, _fox_reference(gray, size, k, delta, 1, _UPPER))

    def test_window_matches(self, picture: str, size: int, k: float, delta: float):
        gray = _picture(picture)

        actual = binarize_window(gray, size, k, delta, 1, _UPPER)

        expected = _window_reference(gray, size, k, delta, 1, _UPPER)
        assert np.array_equal(actual, expected)

    def test_grad_matches(self, picture: str, size: int, k: float, delta: float):
        gray = _picture(picture)

        actual = binarize_grad(gray, size, k, delta, 1, _UPPER)

        assert np.array_equal(actual, _grad_reference(gray, size, k, delta, 1, _UPPER))

    def test_wolf_matches_the_1_2_1_adjustment(
        self, picture: str, size: int, k: float, delta: float
    ):
        gray = _picture(picture)

        actual = binarize_wolf(gray, size, k, delta, 1, _UPPER)

        assert np.array_equal(actual, _wolf_reference(gray, size, k, delta, 1, _UPPER))

    def test_sauvola_matches_the_1_2_1_adjustment(
        self, picture: str, size: int, k: float, delta: float
    ):
        gray = _picture(picture)

        actual = binarize_sauvola(gray, size, k, delta)

        assert np.array_equal(actual, _sauvola_reference(gray, size, k, delta))


@pytest.mark.parametrize(
    ("method", "reference"),
    [
        (binarize_fox, _fox_reference),
        (binarize_window, _window_reference),
        (binarize_grad, _grad_reference),
    ],
)
class TestBounds:
    def test_lower_and_upper_bound_override_the_threshold(self, method, reference):
        gray = _picture("strokes")

        actual = method(gray, 9, 0.3, 0.0, 70, 200)

        assert np.array_equal(actual, reference(gray, 9, 0.3, 0.0, 70, 200))
        assert np.all(actual[gray < 70] == INK)
        assert np.all(actual[gray > 200] == BACKGROUND)

    def test_uniform_page_is_background_without_bounds_in_the_way(
        self, method, reference
    ):
        gray = np.full((20, 20), 180, dtype=np.uint8)

        assert np.all(method(gray, 7) == BACKGROUND)

    def test_input_is_not_modified(self, method, reference):
        gray = _picture("noise")
        before = gray.copy()

        method(gray, 7, 0.3, 5.0)

        assert np.array_equal(gray, before)

    def test_result_is_bilevel_uint8_of_the_input_shape(self, method, reference):
        gray = _picture("strokes")

        result = method(gray, 15)

        assert result.dtype == np.uint8
        assert result.shape == gray.shape
        assert set(np.unique(result)) <= {INK, BACKGROUND}

    def test_window_of_zero_is_rejected(self, method, reference):
        with pytest.raises(ValueError, match="window_size"):
            method(_picture("noise"), 0)

    def test_empty_image_gives_an_empty_image(self, method, reference):
        empty = np.zeros((0, 5), dtype=np.uint8)

        assert method(empty, 5).shape == (0, 5)

    def test_single_pixel_image_is_handled(self, method, reference):
        pixel = np.array([[120]], dtype=np.uint8)

        assert method(pixel, 25).shape == (1, 1)


class TestMethodCharacter:
    """What each method does on a page where the answer is known by inspection."""

    @staticmethod
    def _text_on_shaded_paper() -> NDArray[np.uint8]:
        page = np.tile(np.linspace(170, 240, 80), (60, 1))
        page[25:29, 10:70] = 40
        return page.astype(np.uint8)

    @pytest.mark.parametrize("method", [binarize_fox, binarize_window, binarize_grad])
    def test_dark_stroke_on_shaded_paper_is_ink_and_paper_is_not(self, method):
        page = self._text_on_shaded_paper()

        result = method(page, 31)

        assert np.all(result[25:29, 10:70] == INK)
        assert np.all(result[:15] == BACKGROUND)

    @pytest.mark.parametrize("method", [binarize_fox, binarize_window, binarize_grad])
    def test_positive_delta_never_removes_ink(self, method):
        page = _picture("strokes")

        base = method(page, 15, 0.3, 0.0) == INK
        more = method(page, 15, 0.3, 20.0) == INK

        assert np.all(more[base])

    def test_window_meets_a_page_of_uniform_deviation(self):
        """With no spread in deviation C++ sets kds to 1 instead of dividing by 0."""
        gray = np.tile(np.array([[50, 200]], dtype=np.uint8), (16, 8))

        actual = binarize_window(gray, 1_000)

        assert np.array_equal(actual, _window_reference(gray, 1_000, 0.33, 0.0, 1, 254))

    def test_grad_on_a_flat_page_uses_the_fallback_level(self):
        """No pixel differs from its local mean, so the page level is 127.5."""
        gray = np.full((12, 12), 100, dtype=np.uint8)

        assert np.array_equal(
            binarize_grad(gray, 5, 0.3, 40.0),
            _grad_reference(gray, 5, 0.3, 40.0, 1, 254),
        )
