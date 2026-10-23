"""Tests for page background estimation and illumination normalisation."""

import math
import time

import numpy as np
import pytest

from scantailor.core.estimate_background import (
    estimate_background,
    normalize_illumination_gray,
    raise_above_background,
)

PAPER = 235
INK = 40


def _text_block(
    page: np.ndarray, top: int, bottom: int, left: int, right: int
) -> np.ndarray:
    """Draw rows of 3 px thick ink bars separated by 5 px gaps; returns the ink mask."""
    ink = np.zeros(page.shape, dtype=bool)
    for y in range(top, bottom, 8):
        ink[y : y + 3, left:right] = True
    return ink


def _lit_page(
    height: int = 400, width: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a page with a lighting gradient, a gutter shadow and text.

    Returns:
        The page, the ink mask and the illumination field (fraction of full brightness).
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    gradient = 1.0 - 0.25 * (ys / height) - 0.1 * (xs / width)
    gutter = 1.0 - 0.35 * np.exp(-(((xs - 8) / 25.0) ** 2))
    illumination = gradient * gutter
    page = np.full((height, width), PAPER, dtype=np.uint8)
    ink = _text_block(page, 60, height - 60, 50, width - 40)
    page[ink] = INK
    lit = np.clip(page * illumination, 0, 255).astype(np.uint8)
    return lit, ink, illumination


class TestEstimateBackground:
    """The background surface follows the paper, not the ink."""

    def test_flattens_gradient_and_gutter_shadow_to_uniform_paper(self):
        page, ink, _ = _lit_page()

        flat = normalize_illumination_gray(page)

        paper = flat[~ink]
        assert abs(float(np.median(paper)) - 255) <= 6
        assert np.percentile(paper, 5) >= 225

    def test_flattened_paper_is_more_uniform_than_the_input(self):
        page, ink, _ = _lit_page()

        flat = normalize_illumination_gray(page)

        assert flat[~ink].std() < page[~ink].std() / 3

    def test_text_remains_clearly_darker_than_paper_after_flattening(self):
        page, ink, _ = _lit_page()

        flat = normalize_illumination_gray(page)

        assert flat[ink].mean() < 0.4 * flat[~ink].mean()

    def test_dense_text_block_is_not_washed_out(self):
        page, ink, _ = _lit_page()
        threshold = 128

        flat = normalize_illumination_gray(page)

        ink_before = int(np.count_nonzero(page < threshold))
        ink_after = int(np.count_nonzero(flat < threshold))
        assert ink_after >= 0.9 * ink_before
        assert ink_after >= 0.9 * np.count_nonzero(ink)

    def test_page_dominated_by_ink_keeps_its_ink(self):
        page = np.full((300, 300), PAPER, dtype=np.uint8)
        ink = np.zeros(page.shape, dtype=bool)
        for y in range(20, 280, 4):
            ink[y : y + 2, 20:280] = True
        page[ink] = INK

        flat = normalize_illumination_gray(page)

        assert np.count_nonzero(flat < 128) >= 0.95 * np.count_nonzero(ink)

    def test_uniform_page_stays_white(self):
        page = np.full((120, 90), 200, dtype=np.uint8)

        flat = normalize_illumination_gray(page)

        assert int(flat.min()) >= 253

    def test_surface_renders_at_requested_size(self):
        page, _, _ = _lit_page(200, 150)

        background = estimate_background(page).render((60, 40))

        assert background.shape == (60, 40)
        assert background.dtype == np.uint8

    def test_area_to_consider_excludes_a_dark_border(self):
        page = np.full((200, 200), 220, dtype=np.uint8)
        page[:, :40] = 20
        interior = [(40.0, 0.0), (199.0, 0.0), (199.0, 199.0), (40.0, 199.0)]

        background = estimate_background(page, interior).render(page.shape)

        assert abs(int(background[100, 120]) - 220) <= 8

    def test_input_is_not_modified(self):
        page, _, _ = _lit_page(150, 120)
        original = page.copy()

        normalize_illumination_gray(page)

        np.testing.assert_array_equal(page, original)

    def test_result_has_input_shape_and_dtype(self):
        page, _, _ = _lit_page(150, 120)

        flat = normalize_illumination_gray(page)

        assert flat.shape == page.shape
        assert flat.dtype == np.uint8

    @pytest.mark.parametrize("shape", [(1, 1), (5, 400), (400, 5), (30, 30)])
    def test_tiny_and_degenerate_shapes_do_not_raise(self, shape):
        page = np.full(shape, 180, dtype=np.uint8)

        flat = normalize_illumination_gray(page)

        assert flat.shape == shape

    @pytest.mark.parametrize(
        "bad", [np.zeros((0, 10), np.uint8), np.zeros((4, 4, 3), np.uint8)]
    )
    def test_rejects_non_2d_or_empty_input(self, bad):
        with pytest.raises(ValueError):
            estimate_background(bad)


class TestAreaToConsiderMissingImage:
    """An area polygon that misses the image must not blank the page."""

    def test_polygon_outside_image_matches_whole_image_estimate(self):
        """A polygon entirely off the page gives the same surface as no polygon."""
        page, _, _ = _lit_page()
        off_page = [(1000.0, 1000.0), (1100.0, 1000.0), (1100.0, 1100.0)]

        fallback = estimate_background(page, off_page).render(page.shape)
        whole = estimate_background(page).render(page.shape)

        np.testing.assert_array_equal(fallback, whole)

    def test_polygon_outside_image_does_not_blank_page(self):
        """Normalising with an off-page polygon keeps the paper bright."""
        page, ink, _ = _lit_page()
        off_page = [(-500.0, -500.0), (-400.0, -500.0), (-400.0, -400.0)]

        result = normalize_illumination_gray(page, off_page)

        assert np.median(result[~ink]) > 200


class TestRaiseAboveBackground:
    """Pixel mapping that turns the background white."""

    def test_pixel_equal_to_background_becomes_white(self):
        gray = np.array([[100]], dtype=np.uint8)

        assert raise_above_background(gray, gray)[0, 0] == 255

    def test_half_brightness_pixel_becomes_mid_gray(self):
        gray = np.array([[50]], dtype=np.uint8)
        background = np.array([[100]], dtype=np.uint8)

        assert raise_above_background(gray, background)[0, 0] == 128

    def test_pixel_brighter_than_background_saturates(self):
        gray = np.array([[200]], dtype=np.uint8)
        background = np.array([[100]], dtype=np.uint8)

        assert raise_above_background(gray, background)[0, 0] == 255

    def test_zero_background_does_not_divide_by_zero(self):
        gray = np.array([[0]], dtype=np.uint8)

        assert raise_above_background(gray, gray)[0, 0] == 255


class TestWorkingSize:
    """The page is scaled so its longer side is 300 px, enlarging small pages."""

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [((100, 150), (200, 300)), ((600, 900), (200, 300)), ((300, 200), (300, 200))],
    )
    def test_longer_side_becomes_300_px(self, monkeypatch, shape, expected):
        from scantailor.core import estimate_background as module

        seen: list[tuple[int, ...]] = []
        original = module._suppress_text_and_shadow

        def spy(small, dbg=None):
            seen.append(small.shape)
            return original(small, dbg)

        monkeypatch.setattr(module, "_suppress_text_and_shadow", spy)

        estimate_background(np.full(shape, PAPER, dtype=np.uint8))

        assert seen == [expected]


def _raised_by_division(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Direct evaluation of ``round(gray * 255 / background)``, 255 at or above it."""
    src = gray.astype(np.uint32)
    bg = background.astype(np.uint32)
    scaled = np.minimum((src * 255 + bg // 2) // np.maximum(bg, 1), 255)
    return np.where(bg.astype(np.int32) - src.astype(np.int32) < 1, 255, scaled).astype(
        np.uint8
    )


def _normalized_through_a_full_render(gray: np.ndarray) -> np.ndarray:
    return _raised_by_division(gray, estimate_background(gray).render(gray.shape))


def _noisy_page(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    page = rng.integers(150, 235, size=(height, width)).astype(np.uint8)
    page[height // 4 : height // 2, width // 4 : width // 2] = INK
    return page


class TestMatchesTheDirectEvaluation:
    """Division and banded rendering change no pixel."""

    def test_every_gray_and_background_pair_gives_the_same_level(self):
        gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        background = np.repeat(np.arange(256, dtype=np.uint8), 256).reshape(256, 256)

        assert np.array_equal(
            raise_above_background(gray, background),
            _raised_by_division(gray, background),
        )

    @pytest.mark.parametrize("width", [1, 3, 16, 33, 256, 4096])
    def test_every_pair_gives_the_same_level_in_rows_of_any_width(self, width):
        # Shuffled so that every pair also lands in the unaligned tail of a row.
        shuffled = np.random.default_rng(width).permutation(256 * 256)
        pairs = np.resize(shuffled, math.ceil(shuffled.size / width) * width)
        gray = (pairs % 256).astype(np.uint8).reshape(-1, width)
        background = (pairs // 256).astype(np.uint8).reshape(-1, width)

        assert np.array_equal(
            raise_above_background(gray, background),
            _raised_by_division(gray, background),
        )

    def test_an_exact_half_rounds_up_rather_than_to_even(self):
        # 1 * 255 / 6 is exactly 42.5.
        gray = np.array([[1]], dtype=np.uint8)
        background = np.array([[6]], dtype=np.uint8)

        assert raise_above_background(gray, background)[0, 0] == 43

    @pytest.mark.parametrize(
        "shape", [(333, 257), (31, 1000), (1, 50), (70, 1), (65, 65), (500, 40)]
    )
    def test_normalised_page_is_identical_for_any_page_shape(self, shape):
        page = _noisy_page(*shape, seed=sum(shape))

        assert np.array_equal(
            normalize_illumination_gray(page), _normalized_through_a_full_render(page)
        )

    def test_normalised_page_is_identical_when_a_polygon_limits_the_fit(self):
        page = _noisy_page(400, 300, seed=5)
        polygon = [(20.0, 30.0), (280.0, 40.0), (270.0, 380.0), (30.0, 360.0)]

        expected = _raised_by_division(
            page, estimate_background(page, polygon).render(page.shape)
        )

        assert np.array_equal(normalize_illumination_gray(page, polygon), expected)

    def test_non_contiguous_input_is_normalised_like_a_copy(self):
        page = _noisy_page(200, 300, seed=2)
        view = page[::-1, ::2]

        assert np.array_equal(
            normalize_illumination_gray(view),
            normalize_illumination_gray(np.ascontiguousarray(view)),
        )

    def test_input_page_is_not_modified(self):
        page = _noisy_page(120, 90, seed=9)
        before = page.copy()

        normalize_illumination_gray(page)

        assert np.array_equal(page, before)


@pytest.mark.slow
class TestSpeed:
    def test_normalising_is_faster_than_rendering_the_whole_surface(self):
        page = _noisy_page(3500, 2500, seed=1)

        timings = {}
        for name, run in (
            ("banded", lambda: normalize_illumination_gray(page)),
            ("whole", lambda: _normalized_through_a_full_render(page)),
        ):
            best = float("inf")
            for _ in range(2):
                start = time.perf_counter()
                run()
                best = min(best, time.perf_counter() - start)
            timings[name] = best

        assert timings["banded"] < timings["whole"]
