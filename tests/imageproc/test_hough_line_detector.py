"""Hough line finder: votes, peaks and the lines they give."""

import numpy as np
import pytest

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
