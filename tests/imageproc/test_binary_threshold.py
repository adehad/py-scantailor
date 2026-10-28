"""Gray conversion and Otsu threshold used by page split detection."""

import numpy as np
import pytest

from scantailor.imageproc.binary_threshold import otsu_threshold, to_gray


class TestToGray:
    def test_a_gray_image_is_returned_as_is(self) -> None:
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)

        assert to_gray(gray) is gray

    @pytest.mark.parametrize(
        ("bgr", "expected"),
        [
            ((0, 0, 0), 0),
            ((255, 255, 255), 255),
            ((0, 0, 255), 87),  # red: 255 * 11 // 32
            ((0, 255, 0), 127),  # green: 255 * 16 // 32
            ((255, 0, 0), 39),  # blue: 255 * 5 // 32
        ],
    )
    def test_weights_channels_like_qt(
        self, bgr: tuple[int, int, int], expected: int
    ) -> None:
        image = np.array([[bgr]], dtype=np.uint8)

        assert to_gray(image).tolist() == [[expected]]

    def test_ignores_alpha(self) -> None:
        opaque = np.array([[(10, 20, 30, 255)]], dtype=np.uint8)
        clear = np.array([[(10, 20, 30, 0)]], dtype=np.uint8)

        assert to_gray(opaque).tolist() == to_gray(clear).tolist()


class TestOtsuThreshold:
    @pytest.mark.parametrize("seed", range(5))
    def test_a_noisy_two_tone_page_splits_between_the_tones(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        ink = rng.random((80, 100)) < 0.2
        noise = rng.integers(-20, 20, (80, 100))
        page = (np.where(ink, 40, 220) + noise).clip(0, 255).astype(np.uint8)

        threshold = otsu_threshold(page)

        # The tones span 20..59 and 200..239, so every level that separates them
        # lies above the darkest tone's top and at or below the lightest's bottom.
        assert 59 < threshold <= 200
        np.testing.assert_array_equal(page < threshold, ink)

    def test_two_tones_split_at_the_middle_of_the_gap_between_them(self) -> None:
        page = np.array([[50, 200], [50, 200]], dtype=np.uint8)

        threshold = otsu_threshold(page)

        # Every threshold from 51 to 200 separates the tones equally well; the
        # middle of that range, rounded down, is (51 + 200) // 2.
        assert threshold == 125

    def test_the_split_follows_the_between_class_variance_not_the_level_midpoint(
        self,
    ) -> None:
        # Three tones: the dark mode is so heavy that the best split isolates it.
        page = np.repeat(np.array([10, 120, 130], dtype=np.uint8), [900, 50, 50])

        threshold = otsu_threshold(page.reshape(10, 100))

        assert 10 < threshold <= 120

    def test_a_single_tone_gives_zero(self) -> None:
        assert otsu_threshold(np.full((4, 4), 128, dtype=np.uint8)) == 0
