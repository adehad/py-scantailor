"""Tests for page background estimation and illumination normalisation."""

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

        def spy(small):
            seen.append(small.shape)
            return original(small)

        monkeypatch.setattr(module, "_suppress_text_and_shadow", spy)

        estimate_background(np.full(shape, PAPER, dtype=np.uint8))

        assert seen == [expected]
