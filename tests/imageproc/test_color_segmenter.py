"""Tests for colour segmentation of binarised ink."""

from collections import deque

import cv2
import numpy as np
import pytest

from scantailor.imageproc import color_segmenter
from scantailor.imageproc.color_segmenter import segment_colors

_RED_INK = (30, 30, 200)  # BGR
_BLUE_INK = (200, 30, 30)  # BGR
_PAPER = (255, 255, 255)


def _page(height=120, width=200):
    image = np.full((height, width, 3), _PAPER, dtype=np.uint8)
    return image, np.zeros((height, width), dtype=bool)


def _draw(image, ink, rows, cols, color):
    image[rows, cols] = color
    ink[rows, cols] = True


class TestTwoInkPage:
    @pytest.fixture
    def segmented(self):
        image, ink = _page()
        _draw(image, ink, slice(10, 40), slice(10, 60), _RED_INK)
        _draw(image, ink, slice(10, 40), slice(100, 150), _BLUE_INK)
        return segment_colors(ink, image, (300, 300), 7), ink

    def test_each_ink_keeps_its_own_colour(self, segmented):
        result, _ = segmented
        assert tuple(result[20, 30]) == _RED_INK
        assert tuple(result[20, 120]) == _BLUE_INK

    def test_exactly_the_two_inks_and_white_remain(self, segmented):
        result, _ = segmented
        colours = {tuple(c) for c in np.unique(result.reshape(-1, 3), axis=0)}
        assert colours == {_RED_INK, _BLUE_INK, _PAPER}

    def test_paper_stays_white(self, segmented):
        result, ink = segmented
        assert (result[~ink] == 255).all()


def test_noisy_ink_colour_is_averaged_over_the_piece():
    image, ink = _page()
    _draw(image, ink, slice(10, 40), slice(10, 60), _RED_INK)
    image[10:40, 10:35] = (30, 30, 190)
    result = segment_colors(ink, image, (300, 300), 7)
    assert (result[10:40, 10:60] == result[20, 30]).all()
    assert result[20, 30, 2] == 195


def test_thin_stroke_of_another_colour_joins_the_neighbouring_piece():
    image, ink = _page()
    _draw(image, ink, slice(20, 60), slice(20, 60), _RED_INK)
    _draw(image, ink, slice(20, 60), slice(60, 61), _BLUE_INK)
    result = segment_colors(ink, image, (300, 300), 7)
    assert (result[20:60, 20:61] == result[40, 40]).all()


def test_noise_limit_grows_with_resolution():
    image, ink = _page()
    _draw(image, ink, slice(20, 60), slice(20, 60), _RED_INK)
    _draw(image, ink, slice(30, 35), slice(60, 65), _BLUE_INK)

    at_300 = segment_colors(ink, image, (300, 300), 7)
    at_600 = segment_colors(ink, image, (600, 600), 7)

    assert tuple(at_300[32, 62]) == _BLUE_INK
    assert (at_600[32, 62] == at_600[40, 40]).all()


def test_separate_ink_that_is_noise_still_gets_a_colour():
    image, ink = _page()
    _draw(image, ink, slice(20, 60), slice(20, 60), _RED_INK)
    _draw(image, ink, slice(100, 102), slice(150, 152), _BLUE_INK)
    result = segment_colors(ink, image, (300, 300), 7)
    assert tuple(result[100, 150]) == _BLUE_INK


def test_page_without_ink_is_all_white():
    image, ink = _page()
    result = segment_colors(ink, image, (300, 300), 7)
    assert (result == 255).all()


def test_alpha_channel_is_opaque_and_kept():
    image, ink = _page()
    _draw(image, ink, slice(10, 40), slice(10, 60), _RED_INK)
    rgba = np.dstack([image, np.full(ink.shape, 7, dtype=np.uint8)])
    result = segment_colors(ink, rgba, (300, 300), 7)
    assert result.shape == rgba.shape
    assert (result[..., 3] == 255).all()
    assert tuple(result[20, 30, :3]) == _RED_INK


def test_inputs_are_not_modified():
    image, ink = _page()
    _draw(image, ink, slice(10, 40), slice(10, 60), _RED_INK)
    image_before, ink_before = image.copy(), ink.copy()
    segment_colors(ink, image, (300, 300), 7)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(ink, ink_before)


class TestGrayPage:
    def test_pieces_take_their_mean_gray_and_paper_is_white(self):
        gray = np.full((60, 100), 255, dtype=np.uint8)
        ink = np.zeros(gray.shape, dtype=bool)
        gray[10:20, 10:30] = 40
        gray[30:40, 50:80] = 100
        ink[10:20, 10:30] = True
        ink[30:40, 50:80] = True
        result = segment_colors(ink, gray, (300, 300), 7)
        assert result.ndim == 2
        assert (result[10:20, 10:30] == 40).all()
        assert (result[30:40, 50:80] == 100).all()
        assert (result[~ink] == 255).all()

    def test_mean_rounds_half_up(self):
        gray = np.full((5, 5), 255, dtype=np.uint8)
        ink = np.zeros(gray.shape, dtype=bool)
        gray[2, 1], gray[2, 2] = 10, 11
        ink[2, 1:3] = True
        assert (segment_colors(ink, gray, (300, 300), 7)[2, 1:3] == 11).all()

    def test_diagonal_pixels_are_one_piece(self):
        gray = np.full((6, 6), 255, dtype=np.uint8)
        ink = np.zeros(gray.shape, dtype=bool)
        gray[1, 1], gray[2, 2] = 20, 40
        ink[1, 1] = ink[2, 2] = True
        assert (segment_colors(ink, gray, (300, 300), 7)[ink] == 30).all()


class TestRejectsBadInput:
    def test_size_mismatch(self):
        with pytest.raises(ValueError, match="same height and width"):
            segment_colors(
                np.zeros((4, 4), dtype=bool),
                np.zeros((5, 4, 3), dtype=np.uint8),
                (300, 300),
                7,
            )

    def test_two_channel_image(self):
        with pytest.raises(ValueError, match="3 or 4 channels"):
            segment_colors(
                np.zeros((4, 4), dtype=bool),
                np.zeros((4, 4, 2), dtype=np.uint8),
                (300, 300),
                7,
            )


class TestThresholds:
    def test_adjusted_threshold_stays_within_one_and_255(self):
        channel = np.array([[0, 1, 254, 255]], dtype=np.uint8)
        low = color_segmenter._dark_channel(channel, -1000)
        high = color_segmenter._dark_channel(channel, 1000)
        assert low.tolist() == [[True, False, False, False]]
        assert high.tolist() == [[True, True, True, False]]

    def test_raising_a_threshold_merges_touching_pieces_of_different_classes(self):
        image, ink = _page()
        _draw(image, ink, slice(10, 40), slice(10, 60), (30, 30, 30))
        _draw(image, ink, slice(10, 40), slice(60, 110), (30, 200, 30))

        plain = segment_colors(ink, image, (300, 300), 7)
        merged = segment_colors(ink, image, (300, 300), 7, (0, 255, 0))

        assert tuple(plain[20, 30]) != tuple(plain[20, 90])
        assert tuple(merged[20, 30]) == tuple(merged[20, 90])


def _spread_one_pixel_at_a_time(labels: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Reference: a plain FIFO queue, popping one pixel and offering in turn."""
    height, width = labels.shape
    sources = labels != 0
    result = labels.copy()
    reach = np.where(ink & ~sources, np.iinfo(np.int32).max, 0).astype(np.int64)
    vector = np.zeros((height, width, 2), dtype=np.int64)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    queue = deque(zip(*np.nonzero(sources), strict=True))
    while queue:
        y, x = queue.popleft()
        vx, vy = vector[y, x]
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            offer_x, offer_y = vx - dx, vy - dy
            distance = offer_x * offer_x + offer_y * offer_y
            if distance < reach[ny, nx]:
                result[ny, nx] = result[y, x]
                reach[ny, nx] = distance
                vector[ny, nx] = (offer_x, offer_y)
                queue.append((ny, nx))
    return result


class TestLabelSpreadMatchesOnePixelAtATime:
    """The batched spread takes the same pixels in the same order as a plain queue."""

    @pytest.mark.parametrize("seed", range(40))
    def test_random_blobs_with_few_sources_get_identical_labels(self, seed):
        rng = np.random.default_rng(seed)
        height, width = rng.integers(5, 70, 2)
        blur = rng.uniform(0.5, 3.0)
        noise = cv2.GaussianBlur(rng.random((height, width)), (0, 0), blur)
        ink = noise > rng.uniform(0.4, 0.6)
        labels = np.zeros(ink.shape, dtype=np.int32)
        ys, xs = np.nonzero(ink)
        if ys.size:
            picks = rng.choice(ys.size, min(int(rng.integers(1, 12)), ys.size), False)
            labels[ys[picks], xs[picks]] = rng.integers(1, 5, picks.size)

        np.testing.assert_array_equal(
            color_segmenter._extend_into_ink(labels, ink),
            _spread_one_pixel_at_a_time(labels, ink),
        )

    def test_equidistant_sources_resolve_by_queue_order(self):
        ink = np.ones((1, 5), dtype=bool)
        labels = np.array([[1, 0, 0, 0, 2]], dtype=np.int32)

        extended = color_segmenter._extend_into_ink(labels, ink)

        np.testing.assert_array_equal(extended, [[1, 1, 1, 2, 2]])
        np.testing.assert_array_equal(
            extended, _spread_one_pixel_at_a_time(labels, ink)
        )
