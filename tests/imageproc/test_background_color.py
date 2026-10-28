"""Tests for the dominant background colour calculation."""

import numpy as np
import pytest

from scantailor.imageproc import background_color
from scantailor.imageproc.background_color import (
    dominant_background_color,
    dominant_level,
    otsu_threshold,
)


def _histogram(counts: dict[int, int]) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for level, count in counts.items():
        hist[level] = count
    return hist


class TestDominantLevel:
    """The median level of the densest ten-level window."""

    def test_single_peak_is_returned(self):
        assert dominant_level(_histogram({200: 500})) == 200

    def test_densest_window_wins_over_a_taller_lone_bin(self):
        hist = _histogram({40: 300, 150: 200, 151: 200, 152: 200})

        assert dominant_level(hist) == 151

    def test_median_inside_the_window(self):
        hist = _histogram({100: 10, 101: 10, 102: 10, 103: 100, 104: 10})

        assert dominant_level(hist) == 103

    def test_peak_at_the_end_of_the_range(self):
        assert dominant_level(_histogram({255: 40})) == 255

    def test_empty_histogram_gives_a_valid_level(self):
        assert 0 <= dominant_level(_histogram({})) <= 255


class TestOtsuThreshold:
    """Threshold between the dark and light classes."""

    def test_bimodal_histogram_splits_between_the_modes(self):
        threshold = otsu_threshold(_histogram({50: 400, 200: 600}))

        assert 51 <= threshold <= 200

    def test_single_level_gives_zero(self):
        assert otsu_threshold(_histogram({128: 10})) == 0


class TestDominantBackgroundColor:
    """Most common colour among the light pixels."""

    def test_gray_page_with_text_gives_the_paper_level(self):
        page = np.full((100, 100), 232, dtype=np.uint8)
        page[20:80, 20:30] = 15

        assert dominant_background_color(page) == 232

    def test_page_that_is_mostly_ink_still_gives_the_paper(self):
        page = np.full((100, 100), 20, dtype=np.uint8)
        page[:30, :] = 225

        assert dominant_background_color(page) == 225

    def test_colour_page_gives_a_value_per_channel_in_bgr_order(self):
        page = np.empty((100, 100, 3), dtype=np.uint8)
        page[:] = (200, 220, 240)
        page[40:60, 40:60] = (10, 10, 10)

        assert dominant_background_color(page) == (200, 220, 240)

    def test_alpha_channel_is_reported_opaque(self):
        page = np.empty((50, 50, 4), dtype=np.uint8)
        page[:] = (200, 220, 240, 17)
        page[10:20, 10:20, :3] = 10

        assert dominant_background_color(page) == (200, 220, 240, 255)

    def test_mask_limits_where_the_colour_is_sampled(self):
        page = np.full((100, 100), 230, dtype=np.uint8)
        page[:, 50:] = 180
        page[40:45, 5:45] = 10
        mask = np.zeros(page.shape, dtype=bool)
        mask[:, 50:] = True

        assert dominant_background_color(page, mask) == 180

    def test_empty_mask_considers_the_whole_image(self):
        page = np.full((30, 30), 210, dtype=np.uint8)
        page[5:10, 5:10] = 20

        assert dominant_background_color(page, np.zeros(page.shape, dtype=bool)) == 210

    def test_flat_image_gives_its_level(self):
        assert dominant_background_color(np.full((10, 10), 77, dtype=np.uint8)) == 77

    def test_mask_of_other_size_is_rejected(self):
        with pytest.raises(ValueError, match="different sizes"):
            dominant_background_color(
                np.zeros((10, 10), dtype=np.uint8), np.ones((5, 5), dtype=bool)
            )

    def test_empty_image_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            dominant_background_color(np.zeros((0, 5), dtype=np.uint8))

    def test_input_is_not_modified(self):
        page = np.full((20, 20), 200, dtype=np.uint8)
        page[3:6, 3:6] = 0
        before = page.copy()

        dominant_background_color(page)

        assert np.array_equal(page, before)


def _reference_background(
    image: np.ndarray, mask: np.ndarray | None = None
) -> int | tuple[int, ...]:
    """Gathered pixels: each channel's dominant level above the gray Otsu split."""
    if mask is None or not mask.any():
        mask = np.ones(image.shape[:2], dtype=bool)
    if image.ndim == 2:
        gray = image
    else:
        wide = image.astype(np.int64)
        gray = (
            (11 * wide[..., 2] + 16 * wide[..., 1] + 5 * wide[..., 0]) // 32
        ).astype(np.uint8)
    background = mask & (gray >= otsu_threshold(np.bincount(gray[mask], minlength=256)))
    if not background.any():
        background = mask
    if image.ndim == 2:
        return dominant_level(np.bincount(image[background], minlength=256))
    levels = [
        dominant_level(np.bincount(image[..., c][background], minlength=256))
        for c in range(min(3, image.shape[2]))
    ]
    return (*levels, 255) if image.shape[2] == 4 else tuple(levels)


class TestMatchesTheGatherDefinition:
    """Counting with a masked histogram gives the gathered pixels' answer."""

    @pytest.mark.parametrize("channels", [None, 3, 4], ids=["gray", "bgr", "bgra"])
    @pytest.mark.parametrize("masked", [False, True], ids=["whole", "masked"])
    def test_random_pages_give_the_same_colour(
        self, channels: int | None, masked: bool
    ) -> None:
        """Noisy two-tone pages, with and without a mask, agree channel by channel."""
        rng = np.random.default_rng(17)
        shape = (90, 70) if channels is None else (90, 70, channels)
        ink = rng.random((90, 70)) < 0.3
        if channels is not None:
            ink = ink[..., None]
        image = np.clip(rng.normal(200, 25, shape) - 150 * ink, 0, 255).astype(np.uint8)
        mask = rng.random(shape[:2]) < 0.6 if masked else None

        assert dominant_background_color(image, mask) == _reference_background(
            image, mask
        )

    def test_counting_in_small_chunks_gives_the_same_colour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunk edges inside rows' worth of pixels drop or double nothing."""
        rng = np.random.default_rng(23)
        image = rng.integers(0, 256, (61, 47, 3), dtype=np.uint8)
        mask = rng.random((61, 47)) < 0.7
        whole = dominant_background_color(image, mask)

        monkeypatch.setattr(background_color, "_MAX_CHUNK_PIXELS", 47 * 5 + 3)

        assert dominant_background_color(image, mask) == whole
        assert whole == _reference_background(image, mask)

    def test_two_equally_common_light_levels_resolve_to_the_lower_one(self) -> None:
        """A tie between two dense background windows picks the first one."""
        image = np.full((40, 40), 50, dtype=np.uint8)
        image[:, 20:30] = 220
        image[:, 30:] = 240

        assert dominant_background_color(image) == 220
        assert _reference_background(image) == 220
