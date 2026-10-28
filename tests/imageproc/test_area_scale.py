"""Area-averaging scales of gray images."""

import cv2
import numpy as np
import pytest

from scantailor.imageproc.area_scale import scale_to_gray, transform_scale_to_gray


def _random_gray(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8)


class TestTransformScaleToGray:
    @pytest.mark.parametrize("factor", [5, 6, 8])
    def test_a_reduction_by_a_whole_factor_of_at_least_five_is_the_block_mean(
        self, factor: int
    ) -> None:
        src = _random_gray(90, 120, seed=factor)
        height, width = 90 // factor, 120 // factor

        got = transform_scale_to_gray(src, 1 / factor, (width, height))

        blocks = src[: height * factor, : width * factor].reshape(
            height, factor, width, factor
        )
        # The block mean rounds half up.
        expected = np.floor(blocks.mean(axis=(1, 3)) + 0.5)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize(
        ("height", "width", "scale"),
        [(90, 120, 1 / 3), (91, 127, 1 / 3), (150, 150, 0.5), (60, 80, 100 / 72)],
    )
    def test_the_mean_gray_level_is_kept(
        self, height: int, width: int, scale: float
    ) -> None:
        src = _random_gray(height, width, seed=height + width)
        size = (max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5)))

        got = transform_scale_to_gray(src, scale, size)

        # Windows wider than a destination pixel overlap and the border ones are
        # clipped, so source pixels near the edges weigh a little differently;
        # on uniform noise that moves the mean by well under one level.
        assert got.mean() == pytest.approx(src.mean(), abs=1.0)

    def test_a_window_wider_than_the_minimum_smooths_a_mild_reduction(self) -> None:
        src = np.zeros((30, 30), dtype=np.uint8)
        src[:, 15] = 250

        got = transform_scale_to_gray(src, 0.5, (15, 15))

        # Destination column i centres on source x = 2i + 1, and the five pixel
        # window [2i - 1.5, 2i + 3.5) holds all of bright column 15 for i = 7
        # and 8 (250 / 5) and half of it for i = 6, where a two pixel box at
        # this scale would reach only i = 7.
        assert got[7].tolist() == [0] * 6 + [25, 50, 50] + [0] * 6

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
    def test_the_same_size_returns_the_source(self) -> None:
        src = _random_gray(20, 20)

        assert scale_to_gray(src, (20, 20)) is src

    @pytest.mark.parametrize(
        ("src_size", "dst_size"), [((30, 20), (10, 10)), ((30, 20), (15, 5))]
    )
    def test_reducing_by_whole_numbers_is_the_rounded_block_mean(
        self, src_size: tuple[int, int], dst_size: tuple[int, int]
    ) -> None:
        src = _random_gray(src_size[1], src_size[0], seed=sum(src_size))
        fx, fy = src_size[0] // dst_size[0], src_size[1] // dst_size[1]

        got = scale_to_gray(src, dst_size)

        blocks = src.reshape(dst_size[1], fy, dst_size[0], fx).mean(axis=(1, 3))
        np.testing.assert_array_equal(got, np.floor(blocks + 0.5))

    @pytest.mark.parametrize(
        ("src_size", "dst_size"),
        [((31, 17), (13, 9)), ((120, 90), (50, 37)), ((30, 20), (40, 10))],
    )
    def test_other_reductions_average_the_covered_source_area(
        self, src_size: tuple[int, int], dst_size: tuple[int, int]
    ) -> None:
        src = _random_gray(src_size[1], src_size[0], seed=sum(src_size))

        got = scale_to_gray(src, dst_size).astype(int)

        exact = cv2.resize(src, dst_size, interpolation=cv2.INTER_AREA).astype(int)
        # Window edges sit on a 1/32 pixel grid and the scale is nudged so the
        # last window ends inside the source, so edge pixels can carry a few
        # percent more or less weight than in the exact area mean.
        assert np.abs(got - exact).max() <= 3
        assert got.mean() == pytest.approx(src.mean(), abs=0.5)

    @pytest.mark.parametrize(
        ("src_size", "dst_size"), [((10, 12), (15, 20)), ((10, 12), (16, 25))]
    )
    def test_other_enlargements_interpolate_linearly(
        self, src_size: tuple[int, int], dst_size: tuple[int, int]
    ) -> None:
        ramp = np.tile(
            np.arange(0, 25 * src_size[0], 25, dtype=np.uint8), (src_size[1], 1)
        )

        got = scale_to_gray(ramp, dst_size).astype(int)

        # The first and last output pixels sit on the first and last source
        # pixels, so the ramp's 25 levels per source pixel spread over
        # (dst - 1) / (src - 1) output pixels.
        slope = 25 * (src_size[0] - 1) / (dst_size[0] - 1)
        expected = slope * np.arange(dst_size[0])
        # Positions are truncated to 1/32 pixel and results are whole levels.
        assert np.abs(got[0] - expected).max() <= 1.5
        assert (got == got[0]).all()

    def test_enlarging_a_single_pixel_repeats_it(self) -> None:
        src = np.array([[77]], dtype=np.uint8)

        assert (scale_to_gray(src, (5, 5)) == 77).all()

    def test_enlarging_by_whole_numbers_repeats_pixels(self) -> None:
        src = np.array([[10, 20], [30, 40]], dtype=np.uint8)

        got = scale_to_gray(src, (4, 6))

        assert got.shape == (6, 4)
        assert got[0].tolist() == [10, 10, 20, 20]
        assert got[5].tolist() == [30, 30, 40, 40]

    def test_reducing_by_whole_numbers_rounds_the_block_mean(self) -> None:
        src = np.array([[0, 1], [1, 1]], dtype=np.uint8)

        assert scale_to_gray(src, (1, 1))[0, 0] == 1  # 0.75 rounds up
