"""Fit an open X-spline to a traced polyline and extend it to two bounds.

This is the spline fitting of the C++ distortion model builder: an X-spline of
five control points is pulled towards the polyline by locally quadratic
distance approximants, while two linear constraints keep its ends on the
vertical bounds. Every step is a quadratic program in the control point
displacements, solved with the same LU routine as the C++ code.

Control points are an ``(n, 2)`` array with a tension per point; variable
``2 * i + k`` is coordinate ``k`` of control point ``i``. Forces are quadratic
functions ``x^T A x + b^T x + c`` of the control point displacements.

Sums over samples are accumulated one term at a time, in the order the C++
loops add them (``np.add.accumulate``, not ``np.sum``, which adds pairwise), so
the fit reproduces the C++ numbers rather than numbers close to them.
"""

import functools
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# Flags of a sample on the spline.
DEFAULT_SAMPLE = 0
HEAD_SAMPLE = 1
TAIL_SAMPLE = 2
JUNCTION_SAMPLE = 4

# Knot positions of a segment with its parameter running from 0 to 1.
_KNOTS = np.array([-1.0, 0.0, 1.0, 2.0])
_KNOT1 = 0.0
_KNOT2 = 1.0

_EPS = 2.220446049250313e-16
_SQRT_EPS = 1.4901161193847656e-08
# Samples are never closer than this to each other, so the subdivision ends.
_MIN_SAMPLE_SQDIST = 1e-6
_MAX_SAMPLES = 200_000
# The sampling grid gets about one point per this many pixels of the control
# polygon, then _GRID_EXTRA_DEPTH more halvings, within the depth limits.
_GRID_SPACING = 50.0
_GRID_EXTRA_DEPTH = 1
_GRID_MIN_DEPTH = 2
_GRID_MAX_DEPTH = 12

_INITIAL_CONTROL_POINTS = 5
_MAX_DIST_BETWEEN_SAMPLES = 10.0
_DEFAULT_MAX_DIST_FROM_SPLINE = 0.2
_MAX_ITERATIONS = 20
_INITIAL_FORCE_RATIO = 0.8
_TARGET_FORCE_RATIO = 0.1
_MIN_IMPROVEMENT_PERCENT = 0.5

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _g_blend(
    q: FloatArray, c3: FloatArray, c4: FloatArray, c5: FloatArray, u: FloatArray
) -> FloatArray:
    """Value of the G blending function (formula 20 of the X-spline paper)."""
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    return q * u + 2.0 * q * u2 + c3 * u3 + c4 * u4 + c5 * u5


def _g_blend_d1(
    q: FloatArray, c3: FloatArray, c4: FloatArray, c5: FloatArray, u: FloatArray
) -> FloatArray:
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    return q + 2.0 * (2.0 * q) * u + 3.0 * c3 * u2 + 4.0 * c4 * u3 + 5.0 * c5 * u4


def _g_blend_d2(
    q: FloatArray, c3: FloatArray, c4: FloatArray, c5: FloatArray, u: FloatArray
) -> FloatArray:
    u2 = u * u
    u3 = u2 * u
    return 2.0 * (2.0 * q) + 6.0 * c3 * u + 12.0 * c4 * u2 + 20.0 * c5 * u3


def _h_blend(q: FloatArray, u: FloatArray) -> FloatArray:
    """Value of the H blending function (formula 20 of the X-spline paper)."""
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    return q * u + 2.0 * q * u2 + (-2.0 * q) * u4 + (-q) * u5


def _h_blend_d1(q: FloatArray, u: FloatArray) -> FloatArray:
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    return q + 2.0 * (2.0 * q) * u + 4.0 * (-2.0 * q) * u3 + 5.0 * (-q) * u4


def _h_blend_d2(q: FloatArray, u: FloatArray) -> FloatArray:
    u2 = u * u
    u3 = u2 * u
    return 2.0 * (2.0 * q) + 12.0 * (-2.0 * q) * u2 + 20.0 * (-q) * u3


class _Segments(NamedTuple):
    """Blending constants of every segment of a spline with given tensions.

    The four blend functions of a segment belong to the control points before,
    at the start of, at the end of and after the segment.

    Attributes:
        indexes (numpy.ndarray of numpy.int64): The four control points of each
            segment, shape ``(segments, 4)``; the first two coincide on the
            first segment and the last two on the last one.
        start (numpy.ndarray of numpy.float64): Knot each blend function starts
            from, shape ``(segments, 4)``.
        span (numpy.ndarray of numpy.float64): From that knot to the blend
            function's own knot.
        q (numpy.ndarray of numpy.float64): q of each blend function.
        c3 (numpy.ndarray of numpy.float64): Cubic coefficient of the G form.
        c4 (numpy.ndarray of numpy.float64): Quartic coefficient of the G form.
        c5 (numpy.ndarray of numpy.float64): Quintic coefficient of the G form.
    """

    indexes: IntArray
    start: FloatArray
    span: FloatArray
    q: FloatArray
    c3: FloatArray
    c4: FloatArray
    c5: FloatArray


@functools.lru_cache(maxsize=64)
def _segments_of(tension_bytes: bytes) -> _Segments:
    tension = np.frombuffer(tension_bytes, dtype=np.float64)
    count = tension.shape[0]
    segment = np.arange(count - 1)
    indexes = np.stack(
        [
            np.maximum(segment - 1, 0),
            segment,
            segment + 1,
            np.minimum(segment + 2, count - 1),
        ],
        axis=1,
    )
    tension1 = tension[:-1]
    tension2 = tension[1:]
    s1 = np.maximum(tension1, 0.0)
    s2 = np.maximum(tension2, 0.0)
    q_a = -0.5 * np.minimum(tension1, 0.0)
    q_b = -0.5 * np.minimum(tension2, 0.0)
    start = np.stack([_KNOT1 + s1, _KNOT2 + s2, _KNOT1 - s1, _KNOT2 - s2], axis=1)
    p = 2.0 * (_KNOTS - start) * (_KNOTS - start)
    q = np.stack([q_a, q_b, q_a, q_b], axis=1)
    return _Segments(
        indexes=indexes,
        start=start,
        span=_KNOTS - start,
        q=q,
        c3=10.0 - 12.0 * q - p,
        c4=2.0 * p + 14.0 * q - 15.0,
        c5=6.0 - 5.0 * q - p,
    )


def _segments(tension: FloatArray) -> _Segments:
    """Blending constants of the spline's segments, cached by tensions."""
    return _segments_of(np.ascontiguousarray(tension, dtype=np.float64).tobytes())


def _locate(num_segments: int, t: FloatArray) -> tuple[IntArray, FloatArray]:
    """Segment of each spline parameter and the parameter within it, ``(m, 1)``."""
    scaled = t * num_segments
    segment = np.floor(scaled)
    at_end = (t == 1.0) | (segment >= num_segments)
    local_t = np.where(at_end, 1.0, scaled - segment)
    segment = np.where(at_end, num_segments - 1, segment).astype(np.int64)
    return segment, local_t[:, None]


def _sum_columns(values: FloatArray) -> FloatArray:
    """Sum of the four columns, added left to right."""
    return values[:, 0] + values[:, 1] + values[:, 2] + values[:, 3]


class _Combination(NamedTuple):
    """Each spline point as a weighted sum of control points.

    Attributes:
        indexes (numpy.ndarray of numpy.int64): Control point of each of four
            slots, shape ``(m, 4)``.
        coefficients (numpy.ndarray of numpy.float64): Weight of each slot. Where
            two slots name the same control point, the first carries the sum of
            their weights and the second is zero.
    """

    indexes: IntArray
    coefficients: FloatArray


def _linear_combinations(tension: FloatArray, t: FloatArray) -> _Combination:
    """Control points and normalised weights of the spline points at ``t``."""
    segments = _segments(tension)
    num_segments = segments.indexes.shape[0]
    segment, local_t = _locate(num_segments, t)
    start = segments.start[segment]
    q = segments.q[segment]
    u = (local_t - start) / segments.span[segment]
    weights = _g_blend(
        q, segments.c3[segment], segments.c4[segment], segments.c5[segment], u
    )
    # The outer two blend functions switch to the H form inside the segment.
    h_head = local_t[:, 0] > start[:, 0]
    weights[h_head, 0] = _h_blend(q[h_head, 0], u[h_head, 0])
    h_tail = local_t[:, 0] < start[:, 3]
    weights[h_tail, 3] = _h_blend(q[h_tail, 3], u[h_tail, 3])
    weights /= _sum_columns(weights)[:, None]

    first = segment == 0
    weights[first, 0] += weights[first, 1]
    weights[first, 1] = 0.0
    last = segment == num_segments - 1
    weights[last, 2] += weights[last, 3]
    weights[last, 3] = 0.0
    return _Combination(segments.indexes[segment], weights)


def _dense_weights(combination: _Combination, count: int) -> FloatArray:
    """The combinations as rows of weights over all control points, ``(m, count)``."""
    rows = np.arange(combination.indexes.shape[0])
    dense = np.zeros((rows.shape[0], count))
    for slot in range(4):
        dense[rows, combination.indexes[:, slot]] += combination.coefficients[:, slot]
    return dense


def _combine(
    pos: FloatArray, indexes: IntArray, coefficients: FloatArray
) -> FloatArray:
    """Weighted sums of control points, added slot by slot, shape ``(m, 2)``."""
    out = pos[indexes[:, 0]] * coefficients[:, 0, None]
    for slot in range(1, indexes.shape[1]):
        out += pos[indexes[:, slot]] * coefficients[:, slot, None]
    return out


def _points_at(pos: FloatArray, tension: FloatArray, t: FloatArray) -> FloatArray:
    """Spline points at the parameters ``t``, shape ``(m, 2)``."""
    combination = _linear_combinations(tension, t)
    return _combine(pos, combination.indexes, combination.coefficients)


def _points_and_derivs(
    pos: FloatArray, tension: FloatArray, t: FloatArray
) -> FloatArray:
    """Points and first and second derivatives with respect to the spline parameter.

    Args:
        pos (numpy.ndarray of numpy.float64): Control points, shape ``(n, 2)``.
        tension (numpy.ndarray of numpy.float64): Tension of each control point.
        t (numpy.ndarray of numpy.float64): Parameters in [0, 1], shape ``(m,)``.

    Returns:
        numpy.ndarray of numpy.float64: Shape ``(m, 6)``: ``x, y, dx, dy, ddx, ddy``.
    """
    segments = _segments(tension)
    segment, local_t = _locate(segments.indexes.shape[0], t)
    start = segments.start[segment]
    q = segments.q[segment]
    c3, c4, c5 = segments.c3[segment], segments.c4[segment], segments.c5[segment]
    ta = 1.0 / segments.span[segment]
    u = ta * local_t + (-start * ta)
    scale = ta * float(pos.shape[0] - 1)
    use_h = np.zeros(u.shape, dtype=bool)
    use_h[:, 0] = local_t[:, 0] > start[:, 0]
    use_h[:, 3] = local_t[:, 0] < start[:, 3]
    a = np.where(use_h, _h_blend(q, u), _g_blend(q, c3, c4, c5, u))
    da = np.where(use_h, _h_blend_d1(q, u), _g_blend_d1(q, c3, c4, c5, u)) * scale
    dda = np.where(use_h, _h_blend_d2(q, u), _g_blend_d2(q, c3, c4, c5, u))
    dda = dda * scale * scale

    total = _sum_columns(a)[:, None]
    total2 = total * total
    total4 = total2 * total2
    d_total = _sum_columns(da)[:, None]
    dd_total = _sum_columns(dda)[:, None]
    d1 = da * total - a * d_total
    dd1 = dda * total + da * d_total
    dd2 = da * d_total + a * dd_total
    c2 = ((dd1 - dd2) * total2 - d1 * (2 * total * d_total)) / total4
    indexes = segments.indexes[segment]
    return np.concatenate(
        [
            _combine(pos, indexes, a / total),
            _combine(pos, indexes, d1 / total2),
            _combine(pos, indexes, c2),
        ],
        axis=1,
    )


def _sample(
    pos: FloatArray,
    tension: FloatArray,
    max_dist_from_spline: float,
    max_dist_between_samples: float,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Adaptive samples along the whole spline, in order.

    The C++ code halves an interval recursively until its chord is close enough
    to the spline and short enough. Whether an interval is split depends only
    on its own ends, so all the intervals of one depth are tested together, and
    the result is the same samples in the same order.

    Args:
        pos (numpy.ndarray of numpy.float64): Control points, shape ``(n, 2)``.
        tension (numpy.ndarray of numpy.float64): Tension of each control point.
        max_dist_from_spline (float): Largest distance from a chord to the
            spline.
        max_dist_between_samples (float): Largest chord length.

    Returns:
        tuple: ``(t, points, flags)`` with the first sample flagged HEAD, the last
            TAIL and the samples on segment boundaries JUNCTION.
    """
    if pos.shape[0] == 1:
        head = _points_at(pos, tension, np.zeros(1))
        return np.zeros(1), head, np.array([HEAD_SAMPLE], dtype=np.int64)
    return _sample_on(
        _SampleGrid(pos, tension, max_dist_between_samples),
        max_dist_from_spline,
        max_dist_between_samples,
    )


def _sample_on(
    grid: _SampleGrid, max_dist_from_spline: float, max_dist_between_samples: float
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Adaptive samples of the spline behind ``grid``; see ``_sample``."""
    max_sq_to_spline = max_dist_from_spline * max_dist_from_spline
    max_sq_between = max_dist_between_samples * max_dist_between_samples
    dyadic = _sample_dyadic(grid, max_sq_to_spline, max_sq_between)
    if dyadic is not None:
        return dyadic
    num_segments = grid.num_segments
    # Columns: t, x, y, flag.
    samples = np.column_stack(
        [[0.0, 1.0], grid.points_at(np.array([0.0, 1.0])), [HEAD_SAMPLE, TAIL_SAMPLE]]
    )
    pending = np.array([0])
    while pending.size and samples.shape[0] < _MAX_SAMPLES:
        prev = samples[pending]
        after = samples[pending + 1]
        prev_t = prev[:, 0]
        next_t = after[:, 0]
        mid_t = 0.5 * (prev_t + next_t)
        nearby = np.floor(mid_t * num_segments + 0.5) * (1.0 / num_segments)
        # A segment boundary strictly inside the interval takes the middle's place.
        junction = (nearby > prev_t) & (nearby < next_t)
        mid_t[junction] = nearby[junction]

        new_samples = np.empty((mid_t.shape[0], 4))
        new_samples[:, 0] = mid_t
        new_samples[:, 1:3] = grid.points_at(mid_t)
        new_samples[:, 3] = np.where(junction, JUNCTION_SAMPLE, DEFAULT_SAMPLE)
        split = _splits(
            prev[:, 1:3],
            after[:, 1:3],
            new_samples[:, 1:3],
            junction,
            max_sq_to_spline,
            max_sq_between,
        )
        if not split.any():
            break
        new_samples = new_samples[split]
        before = pending[split] + 1
        samples = np.insert(samples, before, new_samples, axis=0)
        # Both halves of a split interval are tested in the next round.
        inserted = before + np.arange(before.size)
        pending = np.stack([inserted - 1, inserted], axis=1).ravel()
    return samples[:, 0].copy(), samples[:, 1:3].copy(), samples[:, 3].astype(np.int64)


def _sample_dyadic(
    grid: _SampleGrid, max_sq_to_spline: float, max_sq_between: float
) -> tuple[FloatArray, FloatArray, IntArray] | None:
    """The adaptive samples, with the tests of every depth run at once.

    When the segment count is a power of two, every segment boundary inside an
    interval is that interval's middle, so the intervals the subdivision visits
    are the halves, quarters and so on of [0, 1], all with their ends and
    middles on the grid. An interval is visited when all the intervals that
    contain it were split.

    Returns:
        tuple | None: As ``_sample``, or None when the segment count is not a
            power of two or a visited interval is too short to halve on the grid.
    """
    num_segments = grid.num_segments
    if num_segments & (num_segments - 1):
        return None
    scale = grid.scale
    depth = scale.bit_length() - 2
    # Intervals of all depths, the whole [0, 1] first, then its halves and so on.
    lengths = scale >> np.arange(depth + 1)
    starts = np.concatenate([np.arange(0, scale, length) for length in lengths])
    ends = starts + np.repeat(lengths, 1 << np.arange(depth + 1))
    middles = (starts + ends) >> 1
    start_t = starts / scale
    end_t = ends / scale
    mid_t = 0.5 * (start_t + end_t)
    nearby = np.floor(mid_t * num_segments + 0.5) * (1.0 / num_segments)
    junction = (nearby > start_t) & (nearby < end_t)
    points = grid.points
    split = _splits(
        points[starts],
        points[ends],
        points[middles],
        junction,
        max_sq_to_spline,
        max_sq_between,
    )

    visited = np.ones(1, dtype=bool)
    first = 0
    chosen = np.zeros(split.shape[0], dtype=bool)
    for level in range(depth + 1):
        count = 1 << level
        chosen[first : first + count] = split[first : first + count] & visited
        visited = np.repeat(chosen[first : first + count], 2)
        first += count
    if visited.any():
        return None
    order = np.argsort(middles[chosen])
    indexes = np.concatenate([[0], middles[chosen][order], [scale]])
    flags = np.where(junction[chosen][order], JUNCTION_SAMPLE, DEFAULT_SAMPLE)
    flags = np.concatenate([[HEAD_SAMPLE], flags, [TAIL_SAMPLE]]).astype(np.int64)
    return indexes / scale, points[indexes], flags


def _splits(
    prev: FloatArray,
    after: FloatArray,
    middle: FloatArray,
    junction: NDArray[np.bool_],
    max_sq_to_spline: float,
    max_sq_between: float,
) -> NDArray[np.bool_]:
    """Which intervals the subdivision halves.

    An interval is split at a segment boundary, or when its chord is too long
    or too far from the spline's middle point, unless its ends are too close
    for the subdivision to go on.

    Args:
        prev (numpy.ndarray of numpy.float64): Start of each interval, ``(m, 2)``.
        after (numpy.ndarray of numpy.float64): End of each interval.
        middle (numpy.ndarray of numpy.float64): Spline point in between.
        junction (numpy.ndarray of numpy.bool): Whether that point is a segment
            boundary.
        max_sq_to_spline (float): Squared largest distance from chord to spline.
        max_sq_between (float): Squared largest chord length.

    Returns:
        numpy.ndarray of numpy.bool: True for the intervals to split.
    """
    chord = after - prev
    sqdist = chord[:, 0] ** 2 + chord[:, 1] ** 2
    off_chord = _sq_dist_from_chord(prev, chord, middle)
    return (sqdist >= _MIN_SAMPLE_SQDIST) & (
        junction | (sqdist > max_sq_between) | (off_chord > max_sq_to_spline)
    )


def _sq_dist_from_chord(
    start: FloatArray, chord: FloatArray, point: FloatArray
) -> FloatArray:
    """Squared distance of each point from its projection on the chord's line."""
    sqnorm = chord[:, 0] * chord[:, 0] + chord[:, 1] * chord[:, 1]
    usable = np.abs(sqnorm) > _EPS
    safe = np.where(usable, sqnorm, 1.0)
    offset = point - start
    scalar = (
        np.where(usable, chord[:, 0] / safe, 0.0) * offset[:, 0]
        + np.where(usable, chord[:, 1] / safe, 0.0) * offset[:, 1]
    )
    foot = start + chord * scalar[:, None]
    return (point[:, 0] - foot[:, 0]) ** 2 + (point[:, 1] - foot[:, 1]) ** 2


@functools.lru_cache(maxsize=64)
def _grid_weights(tension_bytes: bytes, depth: int) -> tuple[_Combination, FloatArray]:
    """Linear combinations at the parameters ``k / 2**depth``, sparse and dense.

    They depend on the tensions only, which stay the same through a whole fit.
    """
    tension = np.frombuffer(tension_bytes, dtype=np.float64)
    scale = 2**depth
    combination = _linear_combinations(tension, np.arange(scale + 1) / scale)
    dense = _dense_weights(combination, tension.shape[0])
    for array in (*combination, dense):
        array.flags.writeable = False
    return combination, dense


class _SampleGrid:
    """A spline evaluated at the parameters ``k / 2**depth``.

    Halving intervals that start at 0 and 1 only ever asks for such parameters,
    as does snapping to a junction when the segment count is a power of two, so
    the sampling looks points up instead of evaluating a few at a time. The
    depth follows the length of the control polygon, deep enough for the
    sample spacing; anything finer is evaluated when asked for.
    """

    def __init__(
        self, pos: FloatArray, tension: FloatArray, max_between: float
    ) -> None:
        self._pos = pos
        self._tension = np.ascontiguousarray(tension, dtype=np.float64)
        self.num_segments = pos.shape[0] - 1
        polygon = float(np.hypot(*np.diff(pos, axis=0).T).sum())
        spacing = min(max_between, _GRID_SPACING)
        depth = math.ceil(math.log2(max(polygon / spacing, 1.0))) + _GRID_EXTRA_DEPTH
        depth = min(max(depth, _GRID_MIN_DEPTH), _GRID_MAX_DEPTH)
        self.scale = 2**depth
        self._combination, self._dense = _grid_weights(self._tension.tobytes(), depth)
        self.points = _combine(pos, *self._combination)

    def _grid_index(self, t: FloatArray) -> tuple[IntArray, NDArray[np.bool_]]:
        scaled = t * self.scale
        on_grid = scaled == np.floor(scaled)
        return np.where(on_grid, scaled, 0).astype(np.int64), on_grid

    def points_at(self, t: FloatArray) -> FloatArray:
        """Spline points at ``t``, shape ``(m, 2)``."""
        index, on_grid = self._grid_index(t)
        points = self.points[index]
        if not on_grid.all():
            points[~on_grid] = _points_at(self._pos, self._tension, t[~on_grid])
        return points

    def dense_weights_at(self, t: FloatArray) -> FloatArray:
        """Weights of every control point in the spline points at ``t``."""
        index, on_grid = self._grid_index(t)
        dense = self._dense[index]
        if not on_grid.all():
            off_grid = _linear_combinations(self._tension, t[~on_grid])
            dense[~on_grid] = _dense_weights(off_grid, self._pos.shape[0])
        return dense


def _control_points_force(pos: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Squared distances between adjacent control points, for displacements."""
    count = pos.shape[0]
    num_vars = 2 * count
    diff = (pos[1:] - pos[:-1]).ravel()
    matrix = np.zeros((num_vars, num_vars))
    later = np.arange(2, num_vars)
    earlier = later - 2
    np.add.at(matrix, (later, later), 1.0)
    np.add.at(matrix, (earlier, earlier), 1.0)
    matrix[later, earlier] -= 1.0
    matrix[earlier, later] -= 1.0
    linear = np.zeros(num_vars)
    linear[2:] += 2.0 * diff
    linear[:-2] -= 2.0 * diff
    return matrix, linear, _sequential_sum(diff * diff)


def _sequential_sum(values: FloatArray, start: float = 0.0) -> float:
    """``start`` plus the values, added one at a time in order."""
    if values.size == 0:
        return start
    return float(np.add.accumulate(np.concatenate(([start], values.ravel())))[-1])


def _junction_points_force(
    pos: FloatArray, tension: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    """Squared distances between adjacent junction points, for displacements."""
    count = pos.shape[0]
    num_vars = 2 * count
    matrix = np.zeros((num_vars, num_vars))
    linear = np.zeros(num_vars)
    constant = 0.0
    junctions = _dense_weights(
        _linear_combinations(tension, np.arange(count) / (count - 1.0)), count
    )
    for diff in np.diff(junctions, axis=0):
        for k in range(2):
            residual = _sequential_sum(diff * pos[:, k])
            constant += residual * residual
            linear[k::2] += 2.0 * residual * diff
            matrix[k::2, k::2] += diff[:, None] * diff[None, :]
    return matrix, linear, constant


def _lu_solve(matrix: FloatArray, rhs: FloatArray) -> tuple[FloatArray, bool]:
    """Solve a square system with the pivoting and zero rules of the C++ solver.

    Entries no larger than the square root of the machine epsilon count as zero,
    and a pivot that small makes the matrix singular. The systems are at most
    17 by 17, where plain Python lists beat NumPy's per-call overhead.

    Returns:
        ``(solution, ok)``; the solution is only meaningful when ``ok``.
    """
    size = matrix.shape[0]
    lu = matrix.tolist()
    perm = list(range(size))
    for i in range(size):
        virt_pivot = i
        largest = abs(lu[perm[i]][i])
        for j in range(i + 1, size):
            candidate = abs(lu[perm[j]][i])
            if candidate > largest:
                largest = candidate
                virt_pivot = j
        if largest <= _SQRT_EPS:
            return np.zeros(size), False
        phys_pivot = perm[virt_pivot]
        perm[virt_pivot] = perm[i]
        perm[i] = phys_pivot

        pivot_row = lu[phys_pivot]
        r_pivot = 1.0 / pivot_row[i]
        for j in range(i + 1, size):
            row = lu[perm[j]]
            if abs(row[i]) <= _SQRT_EPS:
                row[i] = 0.0
                continue
            factor = row[i] * r_pivot
            row[i] = factor
            for col in range(i + 1, size):
                row[col] -= pivot_row[col] * factor

    rhs_values = rhs.tolist()
    y = [0.0] * size
    for v in range(size):
        row = lu[perm[v]]
        right = rhs_values[perm[v]]
        for c in range(v):
            right -= row[c] * y[c]
        y[v] = right
    x = [0.0] * size
    for v in range(size - 1, -1, -1):
        row = lu[perm[v]]
        right = y[v]
        for c in range(size - 1, v, -1):
            right -= row[c] * x[c]
        x[v] = right / row[v]
    return np.array(x), True


def _constraints_to_lines(
    pos: FloatArray, tension: FloatArray, ts: list[float], lines: list[FloatArray]
) -> tuple[FloatArray, FloatArray]:
    """Linear constraints ``a . d + b = 0`` keeping spline points on lines.

    Args:
        pos (numpy.ndarray of numpy.float64): Control points.
        tension (numpy.ndarray of numpy.float64): Tension of each control point.
        ts (list[float]): Position on the spline of each constrained point.
        lines (list[numpy.ndarray of numpy.float64]): ``[[x1, y1], [x2, y2]]``
            of the line each point must stay on.

    Returns:
        tuple[numpy.ndarray of numpy.float64, numpy.ndarray of numpy.float64]:
            The rows ``a``, shape ``(len(ts), 2 * count)``, and the constants ``b``.
    """
    combination = _linear_combinations(tension, np.array(ts, dtype=np.float64))
    rows = np.zeros((len(ts), 2 * pos.shape[0]))
    constants = np.zeros(len(ts))
    for c, line in enumerate(lines):
        p1x, p1y = float(line[0, 0]), float(line[0, 1])
        dx = float(line[1, 0]) - p1x
        dy = float(line[1, 1]) - p1y
        constant = p1y * dx - p1x * dy
        for index, coeff in _merged_terms(combination, c):
            rows[c, index * 2] = coeff * dy
            rows[c, index * 2 + 1] = -coeff * dx
            constant += (
                float(pos[index, 0]) * coeff * dy - float(pos[index, 1]) * coeff * dx
            )
        constants[c] = constant
    return rows, constants


def _merged_terms(combination: _Combination, row: int) -> list[tuple[int, float]]:
    """``(control point, weight)`` of one spline point, one entry per point."""
    indexes = combination.indexes[row].tolist()
    coefficients = combination.coefficients[row].tolist()
    return [
        (indexes[slot], coefficients[slot])
        for slot in range(4)
        if slot in (0, 2) or indexes[slot] != indexes[slot - 1]
    ]


def _has_loops(pos: FloatArray, tension: FloatArray) -> bool:
    """True if a control point step runs against the spline's overall direction."""
    ends, _ = _grid_weights(
        np.ascontiguousarray(tension, dtype=np.float64).tobytes(), 0
    )
    head, tail = _combine(pos, *ends)
    chord = tail - head
    steps = np.diff(pos, axis=0)
    return bool(np.any(steps[:, 0] * chord[0] + steps[:, 1] * chord[1] < 0))


def _model_vertices(polyline: FloatArray) -> FloatArray:
    """Point, tangent and curvature at each polyline vertex.

    An interpolating X-spline through the vertices gives the derivatives.

    Returns:
        numpy.ndarray of numpy.float64: Shape ``(n, 5)``: ``x, y, dx, dy,
            signed curvature``.
    """
    count = polyline.shape[0]
    tension = np.full(count, -1.0)
    t = np.minimum(np.arange(count) * (1.0 / (count - 1)), 1.0)
    pd = _points_and_derivs(polyline, tension, t)
    cross = pd[:, 2] * pd[:, 5] - pd[:, 3] * pd[:, 4]
    tlen = np.sqrt(pd[:, 2] * pd[:, 2] + pd[:, 3] * pd[:, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = cross / (tlen * tlen * tlen)
    return np.column_stack([pd[:, :4], curvature])


def _sq_dist_approximants(
    origin: FloatArray,
    u: FloatArray,
    v: FloatArray,
    m: FloatArray,
    n: FloatArray,
) -> FloatArray:
    """Quadratics ``m (u.(x-o))^2 + n (v.(x-o))^2`` of points ``x``.

    Each row holds ``[A00, A01, A11, b0, b1, c]`` of ``x^T A x + b^T x + c``.

    Args:
        origin (numpy.ndarray of numpy.float64): Origins, shape ``(m, 2)``.
        u (numpy.ndarray of numpy.float64): First unit directions, ``(m, 2)``.
        v (numpy.ndarray of numpy.float64): Second unit directions, ``(m, 2)``.
        m (numpy.ndarray of numpy.float64): Weights along ``u``.
        n (numpy.ndarray of numpy.float64): Weights along ``v``.

    Returns:
        numpy.ndarray of numpy.float64: Coefficients, shape ``(m, 6)``.
    """
    ux, uy, vx, vy = u[:, 0], u[:, 1], v[:, 0], v[:, 1]
    ox, oy = origin[:, 0], origin[:, 1]
    tx = -(ux * ox + uy * oy)
    ty = -(vx * ox + vy * oy)
    return np.column_stack(
        [
            m * ux * ux + n * vx * vx,
            m * ux * uy + n * vx * vy,
            m * uy * uy + n * vy * vy,
            2 * (m * tx * ux + n * ty * vx),
            2 * (m * tx * uy + n * ty * vy),
            m * tx * tx + n * ty * ty,
        ]
    )


def _curve_distances(
    ref: FloatArray, origin: FloatArray, tangent: FloatArray, curvature: FloatArray
) -> FloatArray:
    """Local squared distances to curves with the given curvature at ``origin``.

    Args:
        ref (numpy.ndarray of numpy.float64): The points measured, ``(m, 2)``.
        origin (numpy.ndarray of numpy.float64): Nearest curve points, ``(m, 2)``.
        tangent (numpy.ndarray of numpy.float64): Curve tangents there, ``(m, 2)``.
        curvature (numpy.ndarray of numpy.float64): Signed curvatures there.

    Returns:
        numpy.ndarray of numpy.float64: Approximant coefficients, ``(m, 6)``.
    """
    sqlen = tangent[:, 0] * tangent[:, 0] + tangent[:, 1] * tangent[:, 1]
    usable = sqlen > 1e-6
    norm = np.sqrt(np.where(usable, sqlen, 1.0))[:, None]
    unit = np.where(usable[:, None], tangent / norm, 0.0)
    # The normal points to the right of the tangent when y points down.
    normal = np.column_stack([unit[:, 1], -unit[:, 0]])
    abs_curvature = np.abs(curvature)
    curved = abs_curvature > _EPS
    offset = ref - origin
    d = np.abs(normal[:, 0] * offset[:, 0] + normal[:, 1] * offset[:, 1])
    radius = 1.0 / np.where(curved, abs_curvature, 1.0)
    m = np.where(curved, d / (d + radius), 0.0)
    return _sq_dist_approximants(origin, unit, normal, m, np.ones_like(m))


def _local_approximants(
    vertices: FloatArray, points: FloatArray, flags: IntArray
) -> FloatArray:
    """Squared distance of each spline sample to the polyline, approximated locally.

    The nearest point of the polyline is the foot on the nearest segment, unless
    a vertex is strictly nearer; ties go to the earlier segment or vertex. Near
    the ends of the polyline only junction samples are pulled in, to the end
    point itself; elsewhere the distance to the curve is used, discounted by the
    curvature of the polyline.

    Args:
        vertices (numpy.ndarray of numpy.float64): ``_model_vertices`` of the
            polyline, shape ``(v, 5)``.
        points (numpy.ndarray of numpy.float64): Spline samples, shape ``(m, 2)``.
        flags (numpy.ndarray of numpy.int64): Sample flags.

    Returns:
        numpy.ndarray of numpy.float64: Approximant coefficients, ``(m, 6)``.
    """
    count = vertices.shape[0]
    rows = np.arange(points.shape[0])
    px = points[:, 0, None]
    py = points[:, 1, None]
    origin_x = vertices[:-1, 0]
    origin_y = vertices[:-1, 1]
    seg_x = vertices[1:, 0] - origin_x
    seg_y = vertices[1:, 1] - origin_y
    norm = seg_x * seg_x + seg_y * seg_y
    usable = np.abs(norm) > _EPS
    safe_norm = np.where(usable, norm, 1.0)
    unit_x = np.where(usable, seg_x / safe_norm, 0.0)
    unit_y = np.where(usable, seg_y / safe_norm, 0.0)
    s = unit_x * (px - origin_x) + unit_y * (py - origin_y)
    foot_x = origin_x + seg_x * s
    foot_y = origin_y + seg_y * s
    seg_sqdist = np.where(
        (s > 0) & (s < 1), (px - foot_x) ** 2 + (py - foot_y) ** 2, np.inf
    )
    vertex_sqdist = (px - vertices[:, 0]) ** 2 + (py - vertices[:, 1]) ** 2

    best_segment = np.argmin(seg_sqdist, axis=1)
    best_vertex = np.argmin(vertex_sqdist, axis=1)
    on_segment = seg_sqdist[rows, best_segment] <= vertex_sqdist[rows, best_vertex]
    on_segment &= np.isfinite(seg_sqdist[rows, best_segment])

    seg = best_segment
    seg_t = s[rows, seg]
    k1 = vertices[seg, 4]
    k2 = vertices[seg + 1, 4]
    segment_approx = _curve_distances(
        points,
        np.column_stack([foot_x[rows, seg], foot_y[rows, seg]]),
        np.column_stack([seg_x[seg], seg_y[seg]]),
        k1 + seg_t * (k2 - k1),
    )

    vertex = vertices[best_vertex]
    vertex_approx = _curve_distances(
        points, vertex[:, :2], vertex[:, 2:4], vertex[:, 4]
    )
    at_end = (best_vertex == 0) | (best_vertex == count - 1)
    pulled_in = (flags & JUNCTION_SAMPLE) != 0
    ones = np.ones(points.shape[0])
    end_approx = _sq_dist_approximants(
        vertex[:, :2],
        np.column_stack([ones, np.zeros_like(ones)]),
        np.column_stack([np.zeros_like(ones), ones]),
        ones,
        ones,
    )
    end_approx = np.where(pulled_in[:, None], end_approx, 0.0)
    vertex_approx = np.where(at_end[:, None], end_approx, vertex_approx)
    return np.where(on_segment[:, None], segment_approx, vertex_approx)


def _add_attraction_forces(
    pos: FloatArray,
    tension: FloatArray,
    vertices: FloatArray,
    ext_matrix: FloatArray,
    ext_linear: FloatArray,
) -> float:
    """Add the pull of the polyline on every spline sample to the external force.

    The squared distance approximant ``Q`` of a sample is a function of the
    point ``L = sum(c_m * P_m)``; as a function of displacements ``d`` of the
    control points it is ``Q(L) + (2 A L + b) . w + w^T A w`` with
    ``w = sum(c_m * d_m)``.

    Returns:
        float: The constant term added to the external force.
    """
    count = pos.shape[0]
    grid = _SampleGrid(pos, tension, _MAX_DIST_BETWEEN_SAMPLES)
    ts, points, flags = _sample_on(
        grid, _DEFAULT_MAX_DIST_FROM_SPLINE, _MAX_DIST_BETWEEN_SAMPLES
    )
    approx = _local_approximants(vertices, points, flags)
    a00, a01, a11, b0, b1, c = approx.T
    weights = grid.dense_weights_at(ts)
    lx = weights[:, 0] * pos[0, 0]
    ly = weights[:, 0] * pos[0, 1]
    for m in range(1, count):
        lx = lx + weights[:, m] * pos[m, 0]
        ly = ly + weights[:, m] * pos[m, 1]
    values = lx * lx * a00 + 2 * lx * ly * a01 + ly * ly * a11 + b0 * lx + b1 * ly + c
    gx = 2 * (a00 * lx + a01 * ly) + b0
    gy = 2 * (a01 * lx + a11 * ly) + b1

    num_samples = ts.shape[0]
    linear = np.empty((num_samples, count, 2))
    linear[:, :, 0] = gx[:, None] * weights
    linear[:, :, 1] = gy[:, None] * weights
    pair = weights[:, :, None] * weights[:, None, :]
    blocks = np.empty((num_samples, count, 2, count, 2))
    blocks[:, :, 0, :, 0] = a00[:, None, None] * pair
    blocks[:, :, 0, :, 1] = a01[:, None, None] * pair
    blocks[:, :, 1, :, 0] = a01[:, None, None] * pair
    blocks[:, :, 1, :, 1] = a11[:, None, None] * pair
    num_vars = 2 * count
    ext_linear += np.add.accumulate(linear.reshape(num_samples, num_vars))[-1]
    ext_matrix += np.add.accumulate(blocks.reshape(num_samples, num_vars, num_vars))[-1]
    return _sequential_sum(values)


def _optimize(
    internal: tuple[FloatArray, FloatArray, float],
    external: tuple[FloatArray, FloatArray, float],
    weight: float,
    cons_a: FloatArray,
    cons_b: FloatArray,
) -> tuple[FloatArray, float, float]:
    """Minimise the weighted sum of both forces under the linear constraints.

    Args:
        internal (tuple): Internal force as ``(matrix, linear, constant)``.
        external (tuple): External force as ``(matrix, linear, constant)``.
        weight (float): Weight of the internal force.
        cons_a (numpy.ndarray of numpy.float64): Constraint rows.
        cons_b (numpy.ndarray of numpy.float64): Constraint constants.

    Returns:
        tuple: ``(displacements, force_before, force_after)``. When the system
            cannot be solved the displacements are zero and the two forces are
            equal.
    """
    int_matrix, int_linear, int_constant = internal
    ext_matrix, ext_linear, ext_constant = external
    num_vars = int_matrix.shape[0]
    total_matrix = weight * int_matrix + ext_matrix
    total_linear = weight * int_linear + ext_linear
    total_constant = weight * int_constant + ext_constant

    num_cons = cons_a.shape[0]
    size = num_vars + num_cons
    system = np.zeros((size, size))
    system[:num_vars, :num_vars] = total_matrix + total_matrix.T
    system[num_vars:, :num_vars] = cons_a
    system[:num_vars, num_vars:] = cons_a.T
    rhs = np.concatenate([-total_linear, -cons_b])
    solution, ok = _lu_solve(system, rhs)
    if not ok:
        return np.zeros(num_vars), total_constant, total_constant
    delta = solution[:num_vars]
    # Each variable adds its linear term, then its row of the quadratic term.
    terms = np.column_stack(
        [total_linear * delta, delta[:, None] * delta[None, :] * total_matrix]
    )
    return delta, total_constant, _sequential_sum(terms, total_constant)


def _fit_extended_spline(
    polyline: FloatArray, bound1: FloatArray, bound2: FloatArray
) -> FloatArray | None:
    """Run the fit; see :func:`fit_extended_spline`.

    Returns:
        numpy.ndarray of numpy.float64 | None: The control points, or None when
            the first round folds the spline back on itself.
    """
    count = _INITIAL_CONTROL_POINTS
    num_vars = 2 * count
    chord_start = polyline[0]
    chord_end = polyline[-1]
    fraction = (np.arange(count) / (count - 1.0))[:, None]
    pos = chord_start + (chord_end - chord_start) * fraction
    tension = np.ones(count)
    vertices = _model_vertices(polyline)
    chord = np.stack([chord_start, chord_end])

    # Initial stage: spread the junction points evenly along the chord.
    cons_a, cons_b = _constraints_to_lines(
        pos,
        tension,
        [0.0, 1.0] + [i / (count - 1.0) for i in range(count)],
        [bound1, bound2] + [chord] * count,
    )
    no_force = (np.zeros((num_vars, num_vars)), np.zeros(num_vars), 0.0)
    delta, _, _ = _optimize(
        _junction_points_force(pos, tension), no_force, 1.0, cons_a, cons_b
    )
    pos = pos + delta.reshape(count, 2)

    # Main stage: only the two bounds stay constrained.
    cons_a, cons_b = _constraints_to_lines(pos, tension, [0.0, 1.0], [bound1, bound2])
    return _pull_onto_polyline(pos, tension, vertices, cons_a, cons_b)


def _pull_onto_polyline(
    pos: FloatArray,
    tension: FloatArray,
    vertices: FloatArray,
    cons_a: FloatArray,
    cons_b: FloatArray,
) -> FloatArray | None:
    """Pull a spline onto a polyline over up to ``_MAX_ITERATIONS`` rounds.

    Each round balances the pull of the polyline against an internal force that
    keeps the control points evenly spread, and that force weakens from
    ``_INITIAL_FORCE_RATIO`` to ``_TARGET_FORCE_RATIO`` of the pull. The fit
    stops early when a round improves the total force by less than
    ``_MIN_IMPROVEMENT_PERCENT``, and steps back when a round folds the spline.

    Args:
        pos (numpy.ndarray of numpy.float64): Starting control points, ``(n, 2)``.
        tension (numpy.ndarray of numpy.float64): Tension of each control point.
        vertices (numpy.ndarray of numpy.float64): The polyline as
            :func:`_model_vertices` describes it.
        cons_a (numpy.ndarray of numpy.float64): Constraint rows, see
            :func:`_constraints_to_lines`.
        cons_b (numpy.ndarray of numpy.float64): Constraint constants.

    Returns:
        numpy.ndarray of numpy.float64 | None: The control points, or None when
            the first round folds the spline back on itself.
    """
    count = pos.shape[0]
    num_vars = 2 * count
    ratio = _INITIAL_FORCE_RATIO
    iterations_to_target = _MAX_ITERATIONS - 1
    rate = (_TARGET_FORCE_RATIO - ratio) / iterations_to_target
    for iteration in range(_MAX_ITERATIONS):
        ext_matrix = np.zeros((num_vars, num_vars))
        ext_linear = np.zeros(num_vars)
        ext_constant = _add_attraction_forces(
            pos, tension, vertices, ext_matrix, ext_linear
        )
        internal = _control_points_force(pos)
        weight = 0.0
        if abs(internal[2]) > 1e-6:
            weight = ratio * ext_constant / internal[2]
        delta, before, after = _optimize(
            internal,
            (ext_matrix, ext_linear, ext_constant),
            weight,
            cons_a,
            cons_b,
        )
        moved = pos + delta.reshape(count, 2)
        cons_b = cons_b + np.add.accumulate(cons_a * delta, axis=1)[:, -1]

        if _has_loops(moved, tension):
            if iteration == 0:
                return None
            # Undone by subtraction, as the C++ code does, rather than by
            # keeping the old points: the round trip can change the last bit.
            pos = moved - delta.reshape(count, 2)
            break
        pos = moved

        before = max(before, 0.0)
        after = max(after, 0.0)
        improvement = (before - after) / (before + _EPS) * 100.0
        if improvement < _MIN_IMPROVEMENT_PERCENT:
            break
        if iterations_to_target > 0:
            iterations_to_target -= 1
            ratio += rate
    return pos


@dataclass(frozen=True)
class FittedSpline:
    """An X-spline as the fit leaves it.

    Attributes:
        positions (numpy.ndarray of numpy.float64): Control points, shape ``(n, 2)``.
        tensions (numpy.ndarray of numpy.float64): Tension of each control point.
    """

    positions: FloatArray
    tensions: FloatArray

    def to_polyline(
        self,
        max_dist_from_spline: float = _DEFAULT_MAX_DIST_FROM_SPLINE,
        max_dist_between_samples: float = math.inf,
    ) -> FloatArray:
        """Sample the spline into a polyline.

        Args:
            max_dist_from_spline (float): Largest distance from any sample's
                chord to the spline.
            max_dist_between_samples (float): Largest distance between two
                neighbouring samples.

        Returns:
            numpy.ndarray of numpy.float64: Points, shape ``(m, 2)``, head to tail.
        """
        return sample_spline(
            self.positions,
            self.tensions,
            max_dist_from_spline,
            max_dist_between_samples,
        )[1]


def sample_spline(
    positions: FloatArray,
    tensions: FloatArray,
    max_dist_from_spline: float = _DEFAULT_MAX_DIST_FROM_SPLINE,
    max_dist_between_samples: float = math.inf,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Adaptive samples along an X-spline.

    Args:
        positions (numpy.ndarray of numpy.float64):
            Control points, shape ``(n, 2)``, n >= 2.
        tensions (numpy.ndarray of numpy.float64):
            Tension of each control point, in [-1, 1].
        max_dist_from_spline (float): Largest distance from the chord between
            two samples to the spline between them.
        max_dist_between_samples (float): Largest distance between samples.

    Returns:
        tuple: Parameters ``t``, points of shape ``(m, 2)`` and sample flags
            (HEAD_SAMPLE, TAIL_SAMPLE, JUNCTION_SAMPLE or DEFAULT_SAMPLE).
    """
    return _sample(
        np.asarray(positions, dtype=np.float64),
        np.asarray(tensions, dtype=np.float64),
        float(max_dist_from_spline),
        float(max_dist_between_samples),
    )


def spline_point_at(
    positions: FloatArray, tensions: FloatArray, t: float
) -> tuple[float, float]:
    """Point of an X-spline at parameter ``t`` in [0, 1]."""
    point = _points_at(
        np.asarray(positions, dtype=np.float64),
        np.asarray(tensions, dtype=np.float64),
        np.array([float(t)]),
    )[0]
    return float(point[0]), float(point[1])


def linear_combination_at(tensions: FloatArray, t: float) -> list[tuple[int, float]]:
    """Control points and weights that make up the spline point at ``t``.

    Args:
        tensions (numpy.ndarray of numpy.float64): Tension of each control point.
        t (float): Position on the spline, 0 to 1.

    Returns:
        list[tuple[int, float]]: ``(control point index, weight)`` pairs.
    """
    combination = _linear_combinations(
        np.asarray(tensions, dtype=np.float64), np.array([float(t)])
    )
    return _merged_terms(combination, 0)


def lu_solve(matrix: FloatArray, rhs: FloatArray) -> FloatArray | None:
    """Solve ``matrix @ x = rhs`` with the C++ solver's pivoting and singularity rule.

    Args:
        matrix (numpy.ndarray of numpy.float64): Square matrix.
        rhs (numpy.ndarray of numpy.float64): Right-hand side vector.

    Returns:
        numpy.ndarray of numpy.float64 | None: The solution, or None when the matrix is
            singular by the solver's tolerance.
    """
    solution, ok = _lu_solve(
        np.asarray(matrix, dtype=np.float64), np.asarray(rhs, dtype=np.float64)
    )
    return solution if ok else None


def control_points_attraction_force(
    positions: FloatArray,
) -> tuple[FloatArray, FloatArray, float]:
    """Quadratic function of displacements for the spread of adjacent control points."""
    return _control_points_force(np.asarray(positions, dtype=np.float64))


def junction_points_attraction_force(
    positions: FloatArray, tensions: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    """Quadratic function of displacements for the spread of junction points."""
    return _junction_points_force(
        np.asarray(positions, dtype=np.float64),
        np.asarray(tensions, dtype=np.float64),
    )


def fit_extended_spline(
    polyline: FloatArray,
    bound_first: tuple[tuple[float, float], tuple[float, float]],
    bound_second: tuple[tuple[float, float], tuple[float, float]],
) -> FittedSpline | None:
    """Fit a spline to a polyline, with its ends on two bound lines.

    The spline starts as five control points evenly spaced on the chord of the
    polyline, then is pulled onto the polyline over up to twenty rounds while
    an internal force keeps the control points evenly spread, weaker each round.

    Args:
        polyline (numpy.ndarray of numpy.float64): Trimmed polyline, shape ``(n, 2)``,
            n >= 2, running from the first bound to the second.
        bound_first (tuple): Line the spline's start must lie on.
        bound_second (tuple): Line the spline's end must lie on.

    Returns:
        FittedSpline | None: The spline, or None when the polyline's ends
            coincide or the first round already folds the spline back on itself.
    """
    points = np.asarray(polyline, dtype=np.float64)
    if np.array_equal(points[0], points[-1]):
        return None
    positions = _fit_extended_spline(
        points,
        np.asarray(bound_first, dtype=np.float64),
        np.asarray(bound_second, dtype=np.float64),
    )
    if positions is None:
        return None
    return FittedSpline(positions=positions, tensions=np.ones(len(positions)))


def fit_spline_between_ends(polyline: FloatArray, tensions: FloatArray) -> FittedSpline:
    """Fit a spline to a polyline, with its ends pinned to the polyline's ends.

    The control points start evenly spaced on the chord and are pulled onto the
    polyline as :func:`fit_extended_spline` pulls them, without its first stage.
    When the first round already folds the spline, the chord is returned.

    Args:
        polyline (numpy.ndarray of numpy.float64): Points, shape ``(n, 2)``,
            n >= 2, whose ends differ.
        tensions (numpy.ndarray of numpy.float64): Tension of each control
            point; their number sets how many control points the spline gets.

    Returns:
        FittedSpline: The spline, starting at ``polyline[0]`` and ending at
            ``polyline[-1]``.
    """
    points = np.asarray(polyline, dtype=np.float64)
    tension = np.asarray(tensions, dtype=np.float64)
    count = tension.shape[0]
    fraction = (np.arange(count) / (count - 1.0))[:, None]
    chord = points[0] + (points[-1] - points[0]) * fraction
    pins = [
        np.stack([end, end + offset])
        for end in (points[0], points[-1])
        for offset in (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    ]
    cons_a, cons_b = _constraints_to_lines(chord, tension, [0.0, 0.0, 1.0, 1.0], pins)
    fitted = _pull_onto_polyline(
        chord, tension, _model_vertices(points), cons_a, cons_b
    )
    return FittedSpline(positions=chord if fitted is None else fitted, tensions=tension)
