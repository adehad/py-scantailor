"""Hough line finder: votes, peaks and the lines they give."""

import numpy as np
import pytest

from scantailor.imageproc import hough_line_detector
from scantailor.imageproc.hough_line_detector import HoughLineDetector
from tests.filters.page_split import cpp_reference as cpp

THICKNESS = 5.0
START = -7.0
STEP = 0.25
ANGLES = 57


def _detect(width: int, height: int, xs, ys, weights, bound: int):
    detector = HoughLineDetector(width, height, THICKNESS, START, STEP, ANGLES)
    detector.process(
        np.asarray(xs, dtype=np.int64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(weights, dtype=np.uint32),
    )
    return detector.find_lines(bound)


def _vertical_line_points(x: int, height: int, tilt: float = 0.0):
    ys = np.arange(height)
    return np.round(x + tilt * ys).astype(np.int64), ys


@pytest.mark.parametrize(
    "lines",
    [[(60, 0.0)], [(40, 0.05), (130, -0.08)], [(25, 0.0), (28, 0.0), (90, 0.1)]],
)
def test_lines_match_upstream(lines: list[tuple[int, float]]) -> None:
    width, height = 160, 220
    xs, ys = [], []
    for x, tilt in lines:
        line_xs, line_ys = _vertical_line_points(x, height, tilt)
        xs.extend(line_xs)
        ys.extend(line_ys)
    rng = np.random.default_rng(3)
    xs.extend(rng.integers(0, width, 300))
    ys.extend(rng.integers(0, height, 300))
    weights = rng.integers(3, 10, len(xs))

    got = _detect(width, height, xs, ys, weights, bound=500)

    want = cpp.hough_find_lines(
        width, height, THICKNESS, START, STEP, ANGLES, xs, ys, weights, 500
    )
    assert len(got) == len(want) > 0
    for mine, theirs in zip(got, want, strict=True):
        assert mine.quality == theirs.quality
        assert mine.norm_x == pytest.approx(theirs.norm_uv[0], abs=1e-12)
        assert mine.norm_y == pytest.approx(theirs.norm_uv[1], abs=1e-12)
        assert mine.distance == pytest.approx(theirs.distance, abs=1e-9)


def test_a_vertical_line_is_found_where_it_is() -> None:
    xs, ys = _vertical_line_points(60, 200)

    lines = _detect(150, 200, xs, ys, np.full(200, 5), bound=500)

    assert len(lines) >= 1
    x_top, _ = lines[0].point_at_y(0.0)
    x_bottom, _ = lines[0].point_at_y(200.0)
    assert x_top == pytest.approx(60, abs=THICKNESS)
    assert x_bottom == pytest.approx(60, abs=THICKNESS)


def test_quality_is_the_weight_of_the_points_within_the_line_width() -> None:
    xs, ys = _vertical_line_points(60, 100)

    lines = _detect(150, 100, xs, ys, np.full(100, 4), bound=1)

    assert lines[0].quality == 400


def test_a_tilted_line_is_reported_with_its_tilt() -> None:
    xs, ys = _vertical_line_points(50, 300, tilt=0.1)

    lines = _detect(150, 300, xs, ys, np.full(300, 5), bound=500)

    top = lines[0].point_at_y(0.0)[0]
    bottom = lines[0].point_at_y(300.0)[0]
    assert (bottom - top) / 300 == pytest.approx(0.1, abs=0.01)


def test_nothing_is_found_below_the_quality_bound() -> None:
    xs, ys = _vertical_line_points(60, 100)

    assert _detect(150, 100, xs, ys, np.full(100, 4), bound=10_000) == []


def test_lines_come_best_first() -> None:
    xs1, ys1 = _vertical_line_points(30, 200)
    xs2, ys2 = _vertical_line_points(100, 200)
    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])
    weights = np.concatenate([np.full(200, 3), np.full(200, 8)])

    lines = _detect(150, 200, xs, ys, weights, bound=500)

    assert [line.quality for line in lines] == sorted(
        (line.quality for line in lines), reverse=True
    )
    assert lines[0].point_at_y(100.0)[0] == pytest.approx(100, abs=THICKNESS)


def _votes_one_point_at_a_time(
    detector: HoughLineDetector, xs: np.ndarray, ys: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Reference: every point adds its weight to one bin per angle."""
    histogram = np.zeros_like(detector._histogram)
    for x, y, weight in zip(xs.tolist(), ys.tolist(), weights.tolist(), strict=True):
        for angle, (cos, sin) in enumerate(
            zip(detector._cos.tolist(), detector._sin.tolist(), strict=True)
        ):
            distance = cos * x + sin * y
            index = int(
                (distance + detector._bias) * detector._reciprocal_resolution + 0.5
            )
            histogram[angle, index] += weight
    return histogram


@pytest.mark.parametrize("resolution", [1.0, 2.5, 5.0])
def test_votes_match_adding_one_point_at_a_time(resolution: float) -> None:
    rng = np.random.default_rng(int(resolution * 10))
    xs = rng.integers(0, 120, 400)
    ys = rng.integers(0, 90, 400)
    weights = rng.integers(1, 255, 400).astype(np.uint32)
    detector = HoughLineDetector(120, 90, resolution, -7.0, 0.25, 57)

    detector.process(xs, ys, weights)

    np.testing.assert_array_equal(
        detector._histogram, _votes_one_point_at_a_time(detector, xs, ys, weights)
    )


def test_votes_split_into_full_and_partial_angle_chunks_match_one_point_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(11)
    xs = rng.integers(0, 120, 400)
    ys = rng.integers(0, 90, 400)
    weights = rng.integers(1, 255, 400).astype(np.uint32)
    # Five angles per chunk: 57 angles make eleven full chunks and one of two.
    monkeypatch.setattr(hough_line_detector, "_VOTE_CHUNK_ELEMENTS", 5 * 400)
    detector = HoughLineDetector(120, 90, 1.0, -7.0, 0.25, 57)

    detector.process(xs, ys, weights)

    np.testing.assert_array_equal(
        detector._histogram, _votes_one_point_at_a_time(detector, xs, ys, weights)
    )


def test_votes_from_several_calls_add_up() -> None:
    rng = np.random.default_rng(3)
    xs = rng.integers(0, 50, 60)
    ys = rng.integers(0, 50, 60)
    weights = np.ones(60, dtype=np.uint32)
    once = HoughLineDetector(50, 50, 1.0, -5.0, 1.0, 11)
    twice = HoughLineDetector(50, 50, 1.0, -5.0, 1.0, 11)

    once.process(xs, ys, weights)
    twice.process(xs[:25], ys[:25], weights[:25])
    twice.process(xs[25:], ys[25:], weights[25:])
    twice.process(xs[:0], ys[:0], weights[:0])

    np.testing.assert_array_equal(once._histogram, twice._histogram)


def test_equal_peaks_come_in_raster_order_of_their_first_bin() -> None:
    detector = HoughLineDetector(50, 50, 1.0, 0.0, 1.0, 12)
    histogram = np.zeros_like(detector._histogram)
    # Peak A starts first in raster order but reaches further down than peak B.
    histogram[2:9, 30] = 7
    histogram[4, 5] = 7
    histogram[10, 15] = 9
    detector._histogram = histogram

    lines = detector.find_lines(1)

    assert [line.quality for line in lines] == [9, 7, 7]
    centre_columns = [round(line.distance + detector._bias - 0.5) for line in lines]
    assert centre_columns == [15, 30, 5]


def test_hough_space_picture_tints_the_strongest_peak_and_few_other_bins() -> None:
    height = 220
    xs, ys = _vertical_line_points(60, height)
    detector = HoughLineDetector(160, height, THICKNESS, START, STEP, ANGLES)
    detector.process(xs, ys, np.full(len(xs), 5, dtype=np.uint32))

    visual = detector.visualize_hough_space(500)

    tinted = visual[..., 2] > visual[..., 0]
    assert visual.shape[2] == 3
    assert 0 < np.count_nonzero(tinted) < 10
    strongest = np.unravel_index(np.argmax(detector._histogram), tinted.shape)
    assert tinted[strongest]
