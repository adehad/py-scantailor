"""Area-averaging scales match the fixed-point arithmetic of upstream."""

import numpy as np
import pytest

from scantailor.imageproc.area_scale import scale_to_gray, transform_scale_to_gray
from tests.filters.page_split import cpp_reference as cpp


def _random_gray(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8)


class TestTransformScaleToGray:
    @pytest.mark.parametrize(
        ("height", "width", "scale"),
        [
            (90, 120, 1 / 3),
            (91, 127, 1 / 3),
            (200, 301, 1 / 6),
            (150, 150, 0.5),
            (100, 140, 0.2),
            (60, 80, 100 / 72),
        ],
    )
    def test_matches_upstream_per_pixel(
        self, height: int, width: int, scale: float
    ) -> None:
        src = _random_gray(height, width, seed=height + width)
        size = (max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5)))

        got = transform_scale_to_gray(src, scale, size)

        inverse = (1.0 / scale) * 32.0
        want = cpp.transform_to_gray(src, inverse, inverse, size[0], size[1], 5.0, 5.0)
        np.testing.assert_array_equal(got, want)

    def test_destination_pixels_past_the_source_average_only_what_lies_inside(
        self,
    ) -> None:
        src = np.full((10, 10), 200, dtype=np.uint8)

        got = transform_scale_to_gray(src, 0.4, (5, 5))

        # A window that hangs over the edge still averages only real pixels,
        # so a flat image stays flat all the way to the border.
        assert (got == 200).all()

    def test_a_pixel_whose_window_misses_the_source_is_black(self) -> None:
        src = np.full((10, 10), 200, dtype=np.uint8)

        got = transform_scale_to_gray(src, 1.0, (20, 3))

        assert (got[:, 12:] == 0).all()
        assert (got[:, :9] == 200).all()

    def test_the_window_is_never_smaller_than_the_minimum(self) -> None:
        src = np.zeros((40, 40), dtype=np.uint8)
        src[:, 20] = 255

        got = transform_scale_to_gray(src, 1.0, (40, 40))

        # One bright column spreads over a 5 pixel window: 255 / 5 each.
        assert got[20, 18:23].tolist() == [51] * 5
        assert got[20, 17] == 0
        assert got[20, 23] == 0


class TestScaleToGray:
    @pytest.mark.parametrize(
        ("src_size", "dst_size"),
        [
            ((30, 20), (10, 10)),
            ((20, 20), (20, 20)),
            ((10, 7), (30, 21)),
            ((10, 12), (15, 20)),
            ((10, 12), (16, 25)),
            ((30, 20), (40, 10)),
            ((31, 17), (13, 9)),
            ((120, 90), (50, 37)),
            ((1, 1), (5, 5)),
        ],
    )
    def test_matches_upstream(
        self, src_size: tuple[int, int], dst_size: tuple[int, int]
    ) -> None:
        src = _random_gray(src_size[1], src_size[0], seed=sum(src_size))

        got = scale_to_gray(src, dst_size)

        np.testing.assert_array_equal(got, cpp.scale_to_gray(src, *dst_size))

    def test_enlarging_by_whole_numbers_repeats_pixels(self) -> None:
        src = np.array([[10, 20], [30, 40]], dtype=np.uint8)

        got = scale_to_gray(src, (4, 6))

        assert got.shape == (6, 4)
        assert got[0].tolist() == [10, 10, 20, 20]
        assert got[5].tolist() == [30, 30, 40, 40]

    def test_reducing_by_whole_numbers_rounds_the_block_mean(self) -> None:
        src = np.array([[0, 1], [1, 1]], dtype=np.uint8)

        assert scale_to_gray(src, (1, 1))[0, 0] == 1  # 0.75 rounds up
