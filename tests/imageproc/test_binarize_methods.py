"""Binarization methods on pages whose ink is known.

Fox, Window, Grad, Wolf, Sauvola and Bradley compare each pixel with a level
taken from a window around it; Otsu, Peak, Mokji and EdgeDiv use one level for
the whole page.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.imageproc.binarize import (
    WindowSize,
    binarize_bradley,
    binarize_edge_div,
    binarize_fox,
    binarize_grad,
    binarize_mokji,
    binarize_otsu,
    binarize_peak,
    binarize_sauvola,
    binarize_window,
    binarize_wolf,
)

INK = 0
BACKGROUND = 255
_SEED = 11

Binarize = Callable[..., NDArray[np.uint8]]


def _picture(kind: str) -> NDArray[np.uint8]:
    """Small pages of different character: noise or text-like strokes.

    Args:
        kind (str): ``"noise"`` for a 31x23 page of uniform noise; anything else
            for a 57x40 page of dark strokes with 12 levels of noise.

    Returns:
        NDArray[np.uint8]: The page.
    """
    rng = np.random.default_rng(_SEED)
    if kind == "noise":
        return rng.integers(0, 256, size=(23, 31), dtype=np.uint8)
    page = np.full((40, 57), 235, dtype=np.uint8)
    page[5:8, 4:50] = 30
    page[14:30:3, 10:40] = 60
    page[:, 22] = 90
    noise = rng.integers(-12, 13, size=page.shape)
    return np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("method", [binarize_fox, binarize_window, binarize_grad])
class TestBounds:
    def test_lower_and_upper_bound_override_the_threshold(
        self, method: Binarize
    ) -> None:
        gray = _picture("strokes")

        actual = method(gray, 9, 0.3, 0.0, 70, 200)

        assert np.all(actual[gray < 70] == INK)
        assert np.all(actual[gray > 200] == BACKGROUND)

    def test_uniform_page_is_background_without_bounds_in_the_way(
        self, method: Binarize
    ) -> None:
        gray = np.full((20, 20), 180, dtype=np.uint8)

        assert np.all(method(gray, 7) == BACKGROUND)

    def test_input_is_not_modified(self, method: Binarize) -> None:
        gray = _picture("noise")
        before = gray.copy()

        method(gray, 7, 0.3, 5.0)

        assert np.array_equal(gray, before)

    def test_result_is_bilevel_uint8_of_the_input_shape(self, method: Binarize) -> None:
        gray = _picture("strokes")

        result = method(gray, 15)

        assert result.dtype == np.uint8
        assert result.shape == gray.shape
        assert set(np.unique(result)) <= {INK, BACKGROUND}

    def test_window_of_zero_is_rejected(self, method: Binarize) -> None:
        with pytest.raises(ValueError, match="window_size"):
            method(_picture("noise"), 0)

    def test_empty_image_gives_an_empty_image(self, method: Binarize) -> None:
        empty = np.zeros((0, 5), dtype=np.uint8)

        assert method(empty, 5).shape == (0, 5)

    def test_single_pixel_image_is_handled(self, method: Binarize) -> None:
        pixel = np.array([[120]], dtype=np.uint8)

        assert method(pixel, 25).shape == (1, 1)


class TestMethodCharacter:
    """What each method does on a page where the answer is known by inspection."""

    @staticmethod
    def _text_on_shaded_paper() -> NDArray[np.uint8]:
        """An 80x60 page shaded from 170 to 240 with a stroke of 40 at rows 25-28.

        Returns:
            NDArray[np.uint8]: The page.
        """
        page = np.tile(np.linspace(170, 240, 80), (60, 1))
        page[25:29, 10:70] = 40
        return page.astype(np.uint8)

    @pytest.mark.parametrize("method", [binarize_fox, binarize_window, binarize_grad])
    def test_dark_stroke_on_shaded_paper_is_ink_and_paper_is_not(
        self, method: Binarize
    ) -> None:
        page = self._text_on_shaded_paper()

        result = method(page, 31)

        assert np.all(result[25:29, 10:70] == INK)
        assert np.all(result[:15] == BACKGROUND)

    @pytest.mark.parametrize(
        "method",
        [binarize_fox, binarize_window, binarize_grad, binarize_wolf, binarize_sauvola],
    )
    def test_positive_delta_never_removes_ink(self, method: Binarize) -> None:
        page = _picture("strokes")

        base = method(page, 15, 0.3, 0.0) == INK
        more = method(page, 15, 0.3, 20.0) == INK

        assert np.all(more[base])

    def test_window_on_a_page_of_uniform_deviation_still_separates_the_tones(
        self,
    ) -> None:
        """Every window covers the whole page, so all deviations are equal and
        the spread term falls back to 1 instead of dividing by zero.

        Mean and deviation are 125 and 75 everywhere, so with k = 0.33,
        md = 126 / 201, kd = 1 + 251 / 76 and the threshold is
        125 * (1 - 0.99 * md / kd), about 107: the 50s are ink, the 200s paper.
        """
        gray = np.tile(np.array([[50, 200]], dtype=np.uint8), (16, 8))

        actual = binarize_window(gray, 1_000)

        assert np.array_equal(actual, np.where(gray == 50, INK, BACKGROUND))

    @pytest.mark.parametrize(
        ("level", "expected"), [(100, INK), (127, INK), (128, BACKGROUND)]
    )
    def test_grad_on_a_flat_page_uses_the_fallback_level(
        self, level: int, expected: int
    ) -> None:
        """No pixel differs from its local mean, so the page level is 127.5.

        The threshold is 127.5 * 0.7 + level * 0.3, which a flat page is below
        exactly when its level is below 127.5.
        """
        gray = np.full((12, 12), level, dtype=np.uint8)

        assert np.all(binarize_grad(gray, 5, 0.3, 0.0) == expected)

    def test_sauvola_scales_the_deviation_by_128(self) -> None:
        """Every window covers the whole checkerboard: mean 150, deviation 50.

        With k = 0.5 the threshold is 150 * (1 - 0.5 * (1 - 50 / 128)), about
        104, so the 100s are ink and the 200s paper. Without the deviation
        term it would be 75 and nothing would be ink.
        """
        checker = np.indices((10, 10)).sum(axis=0) % 2 == 0
        gray = np.where(checker, 100, 200).astype(np.uint8)

        actual = binarize_sauvola(gray, 1_000, 0.5, 0.0)

        assert np.array_equal(actual, np.where(checker, INK, BACKGROUND))


def _known_page(
    paper_left: float, paper_right: float, ink_contrast: float
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Rows of letter-like strokes on paper that brightens from left to right.

    Args:
        paper_left (float): Paper level at the left edge.
        paper_right (float): Paper level at the right edge.
        ink_contrast (float): Ink is the paper level times ``1 - ink_contrast``.

    Returns:
        tuple[NDArray[np.uint8], NDArray[np.bool_]]: The (120, 160) page, with
        Gaussian noise of 6 levels on every pixel, and where its ink is.
    """
    rng = np.random.default_rng(4)
    height, width = 120, 160
    paper = np.tile(np.linspace(paper_left, paper_right, width), (height, 1))
    ink = np.zeros((height, width), dtype=bool)
    for top in range(12, height - 12, 14):
        x = 8
        while x < width - 20:
            word = int(rng.integers(8, 20))
            for stem in range(x, x + word, 4):
                ink[top : top + 7, stem : stem + 2] = True
            ink[top + 6, x : x + word] = True
            x += word + int(rng.integers(5, 9))
    page = np.where(ink, paper * (1.0 - ink_contrast), paper)
    page += rng.normal(0.0, 6.0, page.shape)
    return np.clip(page, 0, 255).astype(np.uint8), ink


_LOCAL_METHODS = [
    binarize_fox,
    binarize_window,
    binarize_wolf,
    binarize_sauvola,
    binarize_bradley,
]
_GLOBAL_METHODS = [
    binarize_otsu,
    binarize_peak,
    binarize_mokji,
    binarize_edge_div,
    binarize_grad,
]

# One pixel in a hundred: a working method misses no stroke on these pages,
# while a broken one gets whole strokes or a band of paper wrong.
_ERROR_RATE = 0.01


def _error_rates(
    result: NDArray[np.uint8], ink: NDArray[np.bool_]
) -> tuple[float, float]:
    """Fraction of ink called paper, and of paper called ink.

    Args:
        result (NDArray[np.uint8]): A binarized page.
        ink (NDArray[np.bool_]): Where the page's ink really is.

    Returns:
        tuple[float, float]: Missed ink and false ink, each from 0 to 1.
    """
    called_ink = result == INK
    return float((~called_ink[ink]).mean()), float(called_ink[~ink].mean())


class TestSeparatesInkFromPaper:
    @pytest.mark.parametrize("method", _LOCAL_METHODS + _GLOBAL_METHODS)
    def test_every_method_separates_dark_text_on_even_paper(
        self, method: Binarize
    ) -> None:
        page, ink = _known_page(200.0, 200.0, 0.7)

        missed, false_ink = _error_rates(method(page), ink)

        assert missed <= _ERROR_RATE
        assert false_ink <= _ERROR_RATE

    @pytest.mark.parametrize("method", _LOCAL_METHODS)
    def test_local_methods_follow_paper_that_darkens_below_the_ink_elsewhere(
        self, method: Binarize
    ) -> None:
        """Ink on the bright side (about 96) is lighter than paper on the dark
        side (70), so no single level separates them; a 31 pixel window does.
        """
        page, ink = _known_page(70.0, 240.0, 0.6)

        missed, false_ink = _error_rates(method(page, 31), ink)

        assert missed <= _ERROR_RATE
        assert false_ink <= _ERROR_RATE

    @pytest.mark.parametrize("method", _LOCAL_METHODS)
    def test_a_window_wider_than_the_page_cannot_follow_the_shading(
        self, method: Binarize
    ) -> None:
        page, ink = _known_page(70.0, 240.0, 0.6)

        missed, false_ink = _error_rates(method(page, 400), ink)

        assert max(missed, false_ink) > _ERROR_RATE


def _flat_with(
    level: int, marks: dict[tuple[int, int], int], shape: tuple[int, int] = (6, 6)
) -> NDArray[np.uint8]:
    """A page of one ``level`` with the pixels in ``marks`` set to other values.

    Args:
        level (int): Gray level of the page.
        marks (dict[tuple[int, int], int]): Gray level of each (row, column)
            that differs.
        shape (tuple[int, int]): (height, width) of the page.

    Returns:
        NDArray[np.uint8]: The page.
    """
    page = np.full(shape, level, dtype=np.uint8)
    for position, value in marks.items():
        page[position] = value
    return page


def _fox_mean(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """``binarize_fox`` with k = 0, so the threshold is the window mean.

    Args:
        image (NDArray[np.uint8]): (H, W) gray page.
        size (int): Window size.

    Returns:
        NDArray[np.uint8]: The page binarized to ``INK`` and ``BACKGROUND``.
    """
    return binarize_fox(image, size, 0.0)


def _window_mean(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """``binarize_window`` with k = 0, so the threshold is the window mean.

    Args:
        image (NDArray[np.uint8]): (H, W) gray page.
        size (int): Window size.

    Returns:
        NDArray[np.uint8]: The page binarized to ``INK`` and ``BACKGROUND``.
    """
    return binarize_window(image, size, 0.0)


def _wolf_mean(image: NDArray[np.uint8], size: WindowSize) -> NDArray[np.uint8]:
    """``binarize_wolf`` with k = 0, so the threshold is the window mean.

    Args:
        image (NDArray[np.uint8]): (H, W) gray page.
        size (WindowSize): Window size.

    Returns:
        NDArray[np.uint8]: The page binarized to ``INK`` and ``BACKGROUND``.
    """
    return binarize_wolf(image, size, 0.0)


def _sauvola_mean(image: NDArray[np.uint8], size: WindowSize) -> NDArray[np.uint8]:
    """``binarize_sauvola`` with k = 0, so the threshold is the window mean.

    Args:
        image (NDArray[np.uint8]): (H, W) gray page.
        size (WindowSize): Window size.

    Returns:
        NDArray[np.uint8]: The page binarized to ``INK`` and ``BACKGROUND``.
    """
    return binarize_sauvola(image, size, 0.0)


def _bradley_mean(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """``binarize_bradley`` with k = 0, so the threshold is the window mean.

    Args:
        image (NDArray[np.uint8]): (H, W) gray page.
        size (int): Window size.

    Returns:
        NDArray[np.uint8]: The page binarized to ``INK`` and ``BACKGROUND``.
    """
    return binarize_bradley(image, size, 0.0)


# With k = 0 each of these compares the pixel with the plain window mean.
_MEAN_THRESHOLD = [_fox_mean, _window_mean, _wolf_mean, _sauvola_mean, _bradley_mean]


@pytest.mark.parametrize("method", _MEAN_THRESHOLD)
class TestWindow:
    def test_border_windows_are_clipped_to_the_image(self, method: Binarize) -> None:
        """Pixel (0, 1) is 170 beside a black corner on paper of 200.

        Its 5 pixel window clipped to the image holds 12 pixels with mean
        (0 + 170 + 10 * 200) / 12, about 181, so it is ink. Padding with zeros
        and dividing by 25 gives 87, and replicating the edge gives 148, either
        of which would make it paper.
        """
        page = _flat_with(200, {(0, 0): 0, (0, 1): 170})

        result = method(page, 5)

        assert result[0, 1] == INK

    def test_an_even_window_reaches_one_pixel_further_before_the_centre(
        self, method: Binarize
    ) -> None:
        """A window of 8 spans the 4 pixels before the centre and 3 after."""
        bright_before = _flat_with(100, {(0, 6): 255}, shape=(1, 21))
        bright_after = _flat_with(100, {(0, 14): 255}, shape=(1, 21))

        assert method(bright_before, 8)[0, 10] == INK
        assert method(bright_after, 8)[0, 10] == BACKGROUND


@pytest.mark.parametrize("method", [_sauvola_mean, _wolf_mean])
def test_a_rectangular_window_reaches_only_along_its_long_side(
    method: Binarize,
) -> None:
    """A window 1 wide and 5 tall sees two rows above and below, no columns."""
    bright_above = _flat_with(100, {(3, 5): 255}, shape=(11, 11))
    bright_beside = _flat_with(100, {(5, 4): 255}, shape=(11, 11))

    assert method(bright_above, (1, 5))[5, 5] == INK
    assert method(bright_beside, (1, 5))[5, 5] == BACKGROUND
