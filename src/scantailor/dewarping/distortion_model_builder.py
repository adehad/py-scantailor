"""Build a distortion model from traced text lines.

Every traced line is extended to the vertical content bounds as a smooth curve.
The model's top and bottom curves are the pair of lines that, once dewarped,
leave all the other lines the straightest. Geometry is in the image's own
coordinates with "down" along +y, as in the output stage's working space.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.dewarping.distortion_model import Curve, DistortionModel
from scantailor.dewarping.text_lines import TextLine, VerticalBounds

# Degree of the polynomial that carries a text line across the page.
_MAX_POLY_DEGREE = 3
# Samples along each extended curve; enough for a smooth page edge.
_CURVE_SAMPLES = 30
# Points per line used to judge straightness after dewarping.
_SCORE_SAMPLES = 16
_MIN_LINE_POINTS = 3
_MIN_LINE_WIDTH = 1.0
# Depth perception only shapes the scoring dewarper; any sane value ranks alike.
_SCORING_DEPTH_PERCEPTION = 2.0
_EXTREME_PAIR_COUNT = 3
_RANDOM_PAIR_COUNT = 10
# Fixed seed: the same page must give the same model on every run.
_RANDOM_SEED = 0
_NON_HORIZONTAL_PENALTY = 1000.0
_BOUND_ITERATIONS = 3
_DEWARPER_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError)


@dataclass(frozen=True)
class _TracedCurve:
    """A text line extended to the vertical bounds.

    Attributes:
        trimmed: Points of the line that lie between the bounds, shape (N, 2),
            left to right.
        extended: Polyline from the left bound to the right bound, shape (M, 2).
        order: Mean y of the line; lower values are higher on the page.
    """

    trimmed: NDArray[np.float64]
    extended: NDArray[np.float64]
    order: float


def build_distortion_model(
    lines: Sequence[TextLine],
    bounds: VerticalBounds,
) -> DistortionModel | None:
    """Choose the page's top and bottom curves from traced text lines.

    Args:
        lines (Sequence[TextLine]): Traced lines in image coordinates.
        bounds (VerticalBounds): Left and right content bounds, each running
            from top to bottom.

    Returns:
        DistortionModel | None: A valid model, or None when fewer than two
            usable lines exist, a bound is degenerate, or no pair of lines
            yields a valid dewarper.
    """
    left = _as_line(bounds.left_top, bounds.left_bottom)
    right = _as_line(bounds.right_top, bounds.right_bottom)
    if left is None or right is None or len(lines) < 2:
        return None

    curves = [
        curve
        for line in lines
        if (curve := _extend_line(line, left, right)) is not None
    ]
    if len(curves) < 2:
        return None
    curves.sort(key=lambda curve: curve.order)

    sample_points, sample_line = _score_samples(curves)
    best_error = np.inf
    best_pair: tuple[_TracedCurve, _TracedCurve] | None = None
    for top, bottom in _candidate_pairs(curves):
        error = _straightness_error(top, bottom, sample_points, sample_line)
        if error is not None and error < best_error:
            best_error = error
            best_pair = (top, bottom)
    if best_pair is None:
        return None
    return DistortionModel(
        top_curve=Curve.from_polyline(best_pair[0].extended),
        bottom_curve=Curve.from_polyline(best_pair[1].extended),
    )


def _as_line(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return a bound as (origin, direction), or None when it is a point."""
    origin = np.asarray(start, dtype=np.float64)
    direction = np.asarray(end, dtype=np.float64) - origin
    if float(np.hypot(*direction)) < 1e-6:
        return None
    return origin, direction


def _bound_x_at(
    bound: tuple[NDArray[np.float64], NDArray[np.float64]], y: float
) -> float:
    """X of a bound at height ``y``; a horizontal bound is treated as vertical."""
    origin, direction = bound
    if abs(direction[1]) < 1e-9:
        return float(origin[0])
    return float(origin[0] + (y - origin[1]) * direction[0] / direction[1])


def _extend_line(
    line: TextLine,
    left: tuple[NDArray[np.float64], NDArray[np.float64]],
    right: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> _TracedCurve | None:
    """Fit a line with a smooth curve running from the left bound to the right.

    Args:
        line (TextLine): Traced points.
        left (tuple): Left bound as (origin, direction).
        right (tuple): Right bound as (origin, direction).

    Returns:
        _TracedCurve | None: The curve, or None if too few points lie between
            the bounds or the curve would have no width.
    """
    points = np.asarray(line.points, dtype=np.float64).reshape(-1, 2)
    if len(points) < _MIN_LINE_POINTS:
        return None
    points = points[np.argsort(points[:, 0], kind="stable")]

    left_x = np.array([_bound_x_at(left, y) for y in points[:, 1]])
    right_x = np.array([_bound_x_at(right, y) for y in points[:, 1]])
    trimmed = points[(points[:, 0] >= left_x) & (points[:, 0] <= right_x)]
    if len(trimmed) < _MIN_LINE_POINTS or np.ptp(trimmed[:, 0]) < _MIN_LINE_WIDTH:
        return None

    degree = min(_MAX_POLY_DEGREE, len(trimmed) - 1)
    poly = Polynomial.fit(trimmed[:, 0], trimmed[:, 1], degree)
    x_first, x_last = float(trimmed[0, 0]), float(trimmed[-1, 0])

    def height_at(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Polynomial inside the data, its tangent beyond either end."""
        x = np.asarray(x, dtype=np.float64)
        inside = np.clip(x, x_first, x_last)
        return poly(inside) + poly.deriv()(inside) * (x - inside)

    x_start, x_end = x_first, x_last
    for _ in range(_BOUND_ITERATIONS):
        x_start = _bound_x_at(left, float(height_at(x_start)))
        x_end = _bound_x_at(right, float(height_at(x_end)))
    if x_end - x_start < _MIN_LINE_WIDTH:
        return None

    xs = np.linspace(x_start, x_end, _CURVE_SAMPLES)
    extended = np.column_stack([xs, height_at(xs)])
    segment_mid = 0.5 * (trimmed[1:] + trimmed[:-1])
    weights = np.hypot(*np.diff(trimmed, axis=0).T)
    order = (
        float(np.average(segment_mid[:, 1], weights=weights))
        if weights.sum() > 1e-6
        else float(trimmed[:, 1].mean())
    )
    return _TracedCurve(trimmed=trimmed, extended=extended, order=order)


def _candidate_pairs(
    curves: Sequence[_TracedCurve],
) -> list[tuple[_TracedCurve, _TracedCurve]]:
    """Pairs worth trying: the outermost lines first, then some random ones."""
    count = len(curves)
    pairs = [
        (curves[i], curves[j])
        for i in range(min(_EXTREME_PAIR_COUNT, count))
        for j in range(max(0, count - _EXTREME_PAIR_COUNT), count)
        if i < j
    ]
    rng = np.random.default_rng(_RANDOM_SEED)
    for _ in range(_RANDOM_PAIR_COUNT):
        first, second = sorted(rng.integers(0, count, size=2))
        if first < second:
            pairs.append((curves[first], curves[second]))
    return pairs


def _straightness_error(
    top: _TracedCurve,
    bottom: _TracedCurve,
    sample_points: NDArray[np.float64],
    sample_line: NDArray[np.intp],
) -> float | None:
    """Sum over lines of their mean deviation from a straight line once dewarped.

    Args:
        top (_TracedCurve): Candidate top curve.
        bottom (_TracedCurve): Candidate bottom curve.
        sample_points (NDArray[np.float64]): Points of every line, shape (K, 2).
        sample_line (NDArray[np.intp]): Index of the line each point belongs to.

    Returns:
        float | None: The error, or None if the pair is not a valid model.
    """
    model = DistortionModel(
        top_curve=Curve.from_polyline(top.extended),
        bottom_curve=Curve.from_polyline(bottom.extended),
    )
    if not model.is_valid():
        return None
    try:
        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top.extended, bottom.extended, _SCORING_DEPTH_PERCEPTION
        )
        dewarped = dewarper.map_points_to_dewarped_space(sample_points)
    except _DEWARPER_ERRORS:
        return None
    return _total_deviation_from_lines(dewarped, sample_line)


def _total_deviation_from_lines(
    points: NDArray[np.float64], line_of_point: NDArray[np.intp]
) -> float:
    """Sum over lines of the mean absolute residual of a least-squares line y(x)."""
    count = line_of_point.max() + 1
    sizes = np.bincount(line_of_point, minlength=count)
    mean_x = np.bincount(line_of_point, weights=points[:, 0], minlength=count) / sizes
    mean_y = np.bincount(line_of_point, weights=points[:, 1], minlength=count) / sizes
    dx = points[:, 0] - mean_x[line_of_point]
    dy = points[:, 1] - mean_y[line_of_point]
    sxx = np.bincount(line_of_point, weights=dx * dx, minlength=count)
    sxy = np.bincount(line_of_point, weights=dx * dy, minlength=count)

    vertical = sxx < 1e-18
    slope = np.where(vertical, 0.0, sxy / np.where(vertical, 1.0, sxx))
    residual = np.abs(dy - slope[line_of_point] * dx)
    mean_residual = (
        np.bincount(line_of_point, weights=residual, minlength=count) / sizes
    )
    return float(np.where(vertical, _NON_HORIZONTAL_PENALTY, mean_residual).sum())


def _score_samples(
    curves: Sequence[_TracedCurve],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Evenly spaced points of every line, with the index of the line they came from."""
    points = []
    owners = []
    for index, curve in enumerate(curves):
        picks = np.unique(
            np.linspace(0, len(curve.trimmed) - 1, _SCORE_SAMPLES).round().astype(int)
        )
        points.append(curve.trimmed[picks])
        owners.append(np.full(len(picks), index, dtype=np.intp))
    return np.vstack(points), np.concatenate(owners)
