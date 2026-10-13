"""Build a distortion model from traced curves, like the C++ model builder.

Every added curve is trimmed to the vertical bounds and fitted with an X-spline
that is extended to them. The model's top and bottom curves are the pair of
extended curves that, once dewarped, leave all the traced curves the straightest.
Candidate pairs are the three topmost with the three bottommost curves, then ten
random pairs from a generator with a fixed seed.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.dewarping.distortion_model import Curve, DistortionModel
from scantailor.dewarping.line_geometry import (
    Line,
    Point,
    line_intersection,
    projection_dist,
    sides_of_line,
)
from scantailor.dewarping.qt_random import QtRandomGenerator
from scantailor.dewarping.spline_fit import fit_extended_spline, lu_solve
from scantailor.dewarping.text_line_tracer import TextLineResult, VerticalBounds

# Candidate pairs: this many outermost curves on each side are paired off.
_EXTREME_CURVE_COUNT = 3
_RANDOM_PAIR_COUNT = 10
# Fixed seed: the same page must give the same model on every run.
_RANDOM_SEED = 0
# Only shapes how pairs are ranked; any plausible value ranks alike.
_SCORING_DEPTH_PERCEPTION = 2.0
# Charged for a curve that dewarps to a vertical line, which has no fit.
_VERTICAL_LINE_ERROR = 1000.0
_DEWARPER_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError)

_AffineMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class TracedCurve:
    """A traced curve and what was fitted to it.

    Attributes:
        trimmed (NDArray[np.float64]): The traced points between the bounds,
            shape ``(n, 2)``, left to right.
        extended (NDArray[np.float64]): The fitted spline sampled from the left
            bound to the right one, shape ``(m, 2)``.
        order (float): Position along the down direction; smaller values lie
            higher on the page.
    """

    trimmed: NDArray[np.float64]
    extended: NDArray[np.float64]
    order: float


class DistortionModelBuilder:
    """Collects vertical bounds and horizontal curves, then builds a model."""

    def __init__(self, down_direction: tuple[float, float]) -> None:
        """Create a builder.

        Args:
            down_direction (tuple[float, float]): A vector pointing roughly down
                the page in terms of its content; not zero.
        """
        self._down = (float(down_direction[0]), float(down_direction[1]))
        self._right = (self._down[1], -self._down[0])
        self._bound1: Line = ((0.0, 0.0), (0.0, 0.0))
        self._bound2: Line = ((0.0, 0.0), (0.0, 0.0))
        # Left to right in terms of content.
        self._curves: list[NDArray[np.float64]] = []

    def set_vertical_bounds(self, bound1: Line, bound2: Line) -> None:
        """Set the vertical content bounds, as lines; their order does not matter."""
        self._bound1 = bound1
        self._bound2 = bound2

    def vertical_bounds(self) -> tuple[Line, Line]:
        """The current bounds, in the order they were set."""
        return self._bound1, self._bound2

    def add_horizontal_curve(self, polyline: NDArray[np.float64]) -> None:
        """Add a curve that should become straight and horizontal after dewarping.

        Long curves are better; curves with fewer than two points are ignored.

        Args:
            polyline (NDArray[np.float64]): Points, shape ``(n, 2)``, in either
                direction.
        """
        points = np.asarray(polyline, dtype=np.float64)
        if len(points) < 2:
            return
        span = points[-1] - points[0]
        if span[0] * self._right[0] + span[1] * self._right[1] > 0:
            self._curves.append(points)
        else:
            self._curves.append(points[::-1].copy())

    def transform(self, matrix: _AffineMatrix) -> None:
        """Apply an affine transformation to everything collected so far.

        Args:
            matrix (NDArray[np.float64]): ``(2, 3)`` or ``(3, 3)`` matrix acting
                on column vectors ``(x, y, 1)``.
        """
        affine = np.asarray(matrix, dtype=np.float64)[:2]
        linear, shift = affine[:, :2], affine[:, 2]
        down = linear @ np.array(self._down)
        right = linear @ np.array(self._right)
        self._down = (float(down[0]), float(down[1]))
        self._right = (float(right[0]), float(right[1]))

        def move(point: Point) -> Point:
            moved = linear @ np.array(point) + shift
            return (float(moved[0]), float(moved[1]))

        self._bound1 = (move(self._bound1[0]), move(self._bound1[1]))
        self._bound2 = (move(self._bound2[0]), move(self._bound2[1]))
        self._curves = [points @ linear.T + shift for points in self._curves]

    def try_build_model(self) -> DistortionModel:
        """Try to build a distortion model from what was collected.

        Returns:
            DistortionModel: A model that is invalid, and has no curves, when
                fewer than two curves were added, a bound is a point, or no pair
                of curves gives a workable dewarper.
        """
        if (
            len(self._curves) < 2
            or self._bound1[0] == self._bound1[1]
            or self._bound2[0] == self._bound2[1]
        ):
            return DistortionModel()

        curves = []
        for polyline in self._curves:
            curve = self._polyline_to_curve(polyline)
            if curve is not None:
                curves.append(curve)
        if not curves:
            return DistortionModel()
        curves.sort(key=lambda curve: curve.order)

        assessor = _RansacAssessor(curves)
        count = len(curves)
        for i in range(min(_EXTREME_CURVE_COUNT, count)):
            for j in range(max(0, count - _EXTREME_CURVE_COUNT), count):
                if i < j:
                    assessor.assess(i, j)
        generator = QtRandomGenerator(_RANDOM_SEED)
        for _ in range(_RANDOM_PAIR_COUNT):
            i = generator.generate() % count
            j = generator.generate() % count
            if i > j:
                i, j = j, i
            if i < j:
                assessor.assess(i, j)

        best = assessor.best
        if best is None:
            return DistortionModel()
        return DistortionModel(
            top_curve=Curve.from_polyline(curves[best[0]].extended),
            bottom_curve=Curve.from_polyline(curves[best[1]].extended),
        )

    def _polyline_to_curve(self, polyline: NDArray[np.float64]) -> TracedCurve | None:
        """Trim a polyline to the bounds and extend it with a fitted spline.

        Returns:
            TracedCurve | None: None when the spline folds back on itself.
        """
        bounds = self._front_back_bounds(polyline)
        trimmed = _maybe_trim_polyline(polyline, bounds)
        spline = fit_extended_spline(trimmed, bounds[0], bounds[1])
        if spline is None:
            return None
        centroid = _centroid(polyline)
        return TracedCurve(
            trimmed=trimmed,
            extended=spline.to_polyline(),
            order=centroid[0] * self._down[0] + centroid[1] * self._down[1],
        )

    def _front_back_bounds(self, polyline: NDArray[np.float64]) -> tuple[Line, Line]:
        """The bounds ordered to match the polyline, the first at its front."""
        front = (float(polyline[0, 0]), float(polyline[0, 1]))
        back = (float(polyline[-1, 0]), float(polyline[-1, 1]))
        direct = projection_dist(self._bound1, front) + projection_dist(
            self._bound2, back
        )
        swapped = projection_dist(self._bound1, back) + projection_dist(
            self._bound2, front
        )
        if direct < swapped:
            return self._bound1, self._bound2
        return self._bound2, self._bound1


class _RansacAssessor:
    """Scores candidate top and bottom curves by how straight they leave all curves."""

    def __init__(self, curves: Sequence[TracedCurve]) -> None:
        self._curves = curves
        self._best_error = math.inf
        self.best: tuple[int, int] | None = None

    def assess(self, top: int, bottom: int) -> None:
        """Score a pair and keep it when it beats the best so far."""
        top_curve = self._curves[top].extended
        bottom_curve = self._curves[bottom].extended
        model = DistortionModel(
            top_curve=Curve.from_polyline(top_curve),
            bottom_curve=Curve.from_polyline(bottom_curve),
        )
        if not model.is_valid():
            return
        try:
            dewarper = CylindricalSurfaceDewarper.from_directrices(
                top_curve, bottom_curve, _SCORING_DEPTH_PERCEPTION
            )
            error = 0.0
            for curve in self._curves:
                error += _straightness_error(
                    dewarper.map_points_to_dewarped_space(curve.trimmed)
                )
        except _DEWARPER_ERRORS:
            return
        if error < self._best_error:
            self._best_error = error
            self.best = (top, bottom)


def _straightness_error(points: NDArray[np.float64]) -> float:
    """Mean absolute residual of the least-squares line ``y = a x + b`` through points.

    A line that is vertical, so that the fit is singular, is charged a fixed
    penalty.
    """
    x = points[:, 0]
    y = points[:, 1]
    count = len(points)
    # Normal equations of the fit, solved with the C++ solver's singularity rule.
    gram = np.array([[np.dot(x, x), x.sum()], [x.sum(), float(count)]])
    rhs = np.array([np.dot(x, y), y.sum()])
    solution = lu_solve(gram, rhs)
    if solution is None:
        return _VERTICAL_LINE_ERROR
    residual = y - (solution[0] * x + solution[1])
    return float(np.abs(residual).sum() / count)


def _centroid(polyline: NDArray[np.float64]) -> Point:
    """Length-weighted centre of a polyline's segments."""
    if len(polyline) == 1:
        return (float(polyline[0, 0]), float(polyline[0, 1]))
    starts = polyline[:-1]
    ends = polyline[1:]
    weights = np.hypot(*(ends - starts).T)
    total = float(weights.sum())
    if total < 1e-6:
        return (float(polyline[0, 0]), float(polyline[0, 1]))
    centers = 0.5 * (starts + ends)
    accum = (centers * weights[:, None]).sum(axis=0)
    return (float(accum[0] / total), float(accum[1] / total))


def _maybe_trim_polyline(
    polyline: NDArray[np.float64], bounds: tuple[Line, Line]
) -> NDArray[np.float64]:
    """Cut a polyline back to the bounds where it sticks out of them."""
    points = [(float(x), float(y)) for x, y in polyline]
    _maybe_trim(points, bounds[0], front=True)
    _maybe_trim(points, bounds[1], front=False)
    return np.array(points, dtype=np.float64)


def _maybe_trim(points: list[Point], bound: Line, front: bool) -> None:
    """Drop the points beyond ``bound`` at one end and put the end on it."""
    if sides_of_line(bound, points[0], points[-1]) >= 0:
        return
    if front:
        while len(points) > 2 and sides_of_line(bound, points[0], points[1]) > 0:
            del points[0]
        crossing = line_intersection(bound, (points[0], points[1]))
        if crossing is not None:
            points[0] = crossing
    else:
        while len(points) > 2 and sides_of_line(bound, points[-2], points[-1]) > 0:
            del points[-1]
        crossing = line_intersection(bound, (points[-2], points[-1]))
        if crossing is not None:
            points[-1] = crossing


def build_distortion_model(
    traced: TextLineResult | Sequence[NDArray[np.float64]],
    bounds: VerticalBounds | None = None,
    down_direction: tuple[float, float] = (0.0, 1.0),
) -> DistortionModel | None:
    """Build a model from traced curves and bounds in one step.

    Args:
        traced (TextLineResult | Sequence[NDArray[np.float64]]): A tracing
            result, or traced curves of shape ``(n, 2)``.
        bounds (VerticalBounds | None): The content bounds. Not needed with a
            tracing result.
        down_direction (tuple[float, float]): Down the page in terms of content.

    Returns:
        DistortionModel | None: The model, or None when none could be built.
    """
    if isinstance(traced, TextLineResult):
        curves = traced.lines
        bounds = traced.vertical_bounds if bounds is None else bounds
    else:
        curves = traced
    if bounds is None:
        return None
    builder = DistortionModelBuilder(down_direction)
    builder.set_vertical_bounds(bounds.first, bounds.second)
    for curve in curves:
        builder.add_horizontal_curve(curve)
    model = builder.try_build_model()
    return model if model.is_valid() else None
