"""Gray conversion and Otsu threshold used by page split detection."""

import numpy as np
import pytest

from scantailor.imageproc.binary_threshold import otsu_threshold, to_gray
from tests.filters.page_split import cpp_reference as cpp


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
    def test_matches_upstream_on_noisy_pages(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        page = np.where(rng.random((80, 100)) < 0.2, 40, 220) + rng.integers(
            -20, 20, (80, 100)
        )
        page = page.clip(0, 255).astype(np.uint8)

        assert otsu_threshold(page) == cpp.otsu_threshold(page)

    def test_two_tones_split_between_them(self) -> None:
        page = np.array([[50, 200], [50, 200]], dtype=np.uint8)

        threshold = otsu_threshold(page)

        assert threshold == cpp.otsu_threshold(page)
        assert 50 < threshold <= 200

    def test_a_single_tone_gives_zero(self) -> None:
        assert otsu_threshold(np.full((4, 4), 128, dtype=np.uint8)) == 0
