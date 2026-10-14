"""Fit an open X-spline to a traced polyline and extend it to two bounds.

This is the spline fitting of the C++ distortion model builder: an X-spline of
five control points is pulled towards the polyline by locally quadratic
distance approximants, while two linear constraints keep its ends on the
vertical bounds. Every step is a quadratic program in the control point
displacements, solved with the same LU routine as the C++ code.

Control points are an ``(n, 2)`` array with a tension per point; variable
``2 * i + k`` is coordinate ``k`` of control point ``i``. Forces are quadratic
functions ``x^T A x + b^T x + c`` of the control point displacements.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

# Flags of a sample on the spline.
DEFAULT_SAMPLE = 0
HEAD_SAMPLE = 1
TAIL_SAMPLE = 2
JUNCTION_SAMPLE = 4

# Knot positions of a segment with its parameter running from 0 to 1.
_KNOT0 = -1.0
_KNOT1 = 0.0
_KNOT2 = 1.0
_KNOT3 = 2.0

_EPS = 2.220446049250313e-16
_SQRT_EPS = 1.4901161193847656e-08
# Samples are never closer than this to each other, so the recursion ends.
_MIN_SAMPLE_SQDIST = 1e-6
_MAX_SAMPLES = 200_000
_STACK_SIZE = 4096

_INITIAL_CONTROL_POINTS = 5
_MAX_DIST_BETWEEN_SAMPLES = 10.0
_DEFAULT_MAX_DIST_FROM_SPLINE = 0.2
_MAX_ITERATIONS = 20
_INITIAL_FORCE_RATIO = 0.8
_TARGET_FORCE_RATIO = 0.1
_MIN_IMPROVEMENT_PERCENT = 0.5

STATUS_OK = 0
STATUS_BAD_CURVE = 1


@njit(cache=True)
def _g_blend(q: float, p: float, u: float) -> float:
    """Value of the G blending function (formula 20 of the X-spline paper)."""
    c3 = 10.0 - 12.0 * q - p
    c4 = 2.0 * p + 14.0 * q - 15.0
    c5 = 6.0 - 5.0 * q - p
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    return q * u + 2.0 * q * u2 + c3 * u3 + c4 * u4 + c5 * u5


@njit(cache=True)
def _g_blend_d1(q: float, p: float, u: float) -> float:
    c3 = 10.0 - 12.0 * q - p
    c4 = 2.0 * p + 14.0 * q - 15.0
    c5 = 6.0 - 5.0 * q - p
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    return q + 2.0 * (2.0 * q) * u + 3.0 * c3 * u2 + 4.0 * c4 * u3 + 5.0 * c5 * u4


@njit(cache=True)
def _g_blend_d2(q: float, p: float, u: float) -> float:
    c3 = 10.0 - 12.0 * q - p
    c4 = 2.0 * p + 14.0 * q - 15.0
    c5 = 6.0 - 5.0 * q - p
    u2 = u * u
    u3 = u2 * u
    return 2.0 * (2.0 * q) + 6.0 * c3 * u + 12.0 * c4 * u2 + 20.0 * c5 * u3


@njit(cache=True)
def _h_blend(q: float, u: float) -> float:
    """Value of the H blending function (formula 20 of the X-spline paper)."""
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    return q * u + 2.0 * q * u2 + (-2.0 * q) * u4 + (-q) * u5


@njit(cache=True)
def _h_blend_d1(q: float, u: float) -> float:
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    return q + 2.0 * (2.0 * q) * u + 4.0 * (-2.0 * q) * u3 + 5.0 * (-q) * u4


@njit(cache=True)
def _h_blend_d2(q: float, u: float) -> float:
    u2 = u * u
    u3 = u2 * u
    return 2.0 * (2.0 * q) + 12.0 * (-2.0 * q) * u2 + 20.0 * (-q) * u3


@njit(cache=True)
def _tension_params(tension1: float, tension2: float) -> NDArray[np.float64]:
    """Knots and blending parameters of a segment between two tensions.

    Returns:
        Array ``[T0p, T1p, T2m, T3m, q_a, q_b, p0, p1, p2, p3]``; the q of
        blend functions 0 and 2 is ``q_a``, that of 1 and 3 is ``q_b``.
    """
    s1 = max(tension1, 0.0)
    s2 = max(tension2, 0.0)
    out = np.empty(10)
    out[0] = _KNOT1 + s1
    out[1] = _KNOT2 + s2
    out[2] = _KNOT1 - s1
    out[3] = _KNOT2 - s2
    out[4] = -0.5 * min(tension1, 0.0)
    out[5] = -0.5 * min(tension2, 0.0)
    out[6] = 2.0 * (_KNOT0 - out[0]) * (_KNOT0 - out[0])
    out[7] = 2.0 * (_KNOT1 - out[1]) * (_KNOT1 - out[1])
    out[8] = 2.0 * (_KNOT2 - out[2]) * (_KNOT2 - out[2])
    out[9] = 2.0 * (_KNOT3 - out[3]) * (_KNOT3 - out[3])
    return out


@njit(cache=True)
def _segment_and_t(t: float, num_segments: int) -> tuple[int, float]:
    """Split a spline parameter into a segment index and a parameter within it."""
    if t == 1.0:
        return num_segments - 1, 1.0
    t2 = t * num_segments
    segment = math.floor(t2)
    return int(segment), t2 - segment


@njit(cache=True)
def _segment_control_points(segment: int, count: int) -> NDArray[np.int64]:
    idxs = np.empty(4, dtype=np.int64)
    idxs[0] = max(0, segment - 1)
    idxs[1] = segment
    idxs[2] = segment + 1
    idxs[3] = min(segment + 2, count - 1)
    return idxs


@njit(cache=True)
def _blend_weights(tp: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    """Unnormalised blending weights of the four control points around a segment."""
    weights = np.empty(4)
    u0 = (t - tp[0]) / (_KNOT0 - tp[0])
    if t <= tp[0]:
        weights[0] = _g_blend(tp[4], tp[6], u0)
    else:
        weights[0] = _h_blend(tp[4], u0)
    weights[1] = _g_blend(tp[5], tp[7], (t - tp[1]) / (_KNOT1 - tp[1]))
    weights[2] = _g_blend(tp[4], tp[8], (t - tp[2]) / (_KNOT2 - tp[2]))
    u3 = (t - tp[3]) / (_KNOT3 - tp[3])
    if t >= tp[3]:
        weights[3] = _g_blend(tp[5], tp[9], u3)
    else:
        weights[3] = _h_blend(tp[5], u3)
    return weights


@njit(cache=True)
def _linear_combination_for(
    tension: NDArray[np.float64], segment: int, t: float
) -> tuple[int, NDArray[np.int64], NDArray[np.float64]]:
    """Control points and weights of the spline point at ``t`` in a segment.

    Returns:
        ``(count, indexes, coefficients)``; at most one entry per control
        point, entries beyond ``count`` are unused.
    """
    count = tension.shape[0]
    idxs = _segment_control_points(segment, count)
    tp = _tension_params(tension[idxs[1]], tension[idxs[2]])
    weights = _blend_weights(tp, t)
    total = weights[0] + weights[1] + weights[2] + weights[3]
    weights = weights / total

    out_idx = np.empty(4, dtype=np.int64)
    out_coeff = np.empty(4)
    n = 0
    if idxs[0] == idxs[1]:
        out_idx[n] = idxs[0]
        out_coeff[n] = weights[0] + weights[1]
        n += 1
    else:
        out_idx[n] = idxs[0]
        out_coeff[n] = weights[0]
        n += 1
        out_idx[n] = idxs[1]
        out_coeff[n] = weights[1]
        n += 1
    if idxs[2] == idxs[3]:
        out_idx[n] = idxs[2]
        out_coeff[n] = weights[2] + weights[3]
        n += 1
    else:
        out_idx[n] = idxs[2]
        out_coeff[n] = weights[2]
        n += 1
        out_idx[n] = idxs[3]
        out_coeff[n] = weights[3]
        n += 1
    return n, out_idx, out_coeff


@njit(cache=True)
def _linear_combination_at(
    tension: NDArray[np.float64], t: float
) -> tuple[int, NDArray[np.int64], NDArray[np.float64]]:
    segment, local_t = _segment_and_t(t, tension.shape[0] - 1)
    return _linear_combination_for(tension, segment, local_t)


@njit(cache=True)
def _point_at_impl(
    pos: NDArray[np.float64], tension: NDArray[np.float64], segment: int, t: float
) -> tuple[float, float]:
    count, idxs, coeffs = _linear_combination_for(tension, segment, t)
    x = 0.0
    y = 0.0
    for i in range(count):
        x += pos[idxs[i], 0] * coeffs[i]
        y += pos[idxs[i], 1] * coeffs[i]
    return x, y


@njit(cache=True)
def _point_at(
    pos: NDArray[np.float64], tension: NDArray[np.float64], t: float
) -> tuple[float, float]:
    segment, local_t = _segment_and_t(t, pos.shape[0] - 1)
    return _point_at_impl(pos, tension, segment, local_t)


@njit(cache=True)
def _point_and_derivs(
    pos: NDArray[np.float64], tension: NDArray[np.float64], t: float
) -> NDArray[np.float64]:
    """Point and first and second derivatives with respect to the spline parameter.

    Returns:
        Array ``[x, y, dx, dy, ddx, ddy]``.
    """
    num_segments = pos.shape[0] - 1
    segment, local_t = _segment_and_t(t, num_segments)
    idxs = _segment_control_points(segment, pos.shape[0])
    tp = _tension_params(tension[idxs[1]], tension[idxs[2]])
    dtdT = float(num_segments)

    a = np.zeros(4)
    da = np.zeros(4)
    dda = np.zeros(4)
    knots_start = np.array([tp[0], tp[1], tp[2], tp[3]])
    knots_span = np.array(
        [_KNOT0 - tp[0], _KNOT1 - tp[1], _KNOT2 - tp[2], _KNOT3 - tp[3]]
    )
    for i in range(4):
        ta = 1.0 / knots_span[i]
        u = ta * local_t + (-knots_start[i] * ta)
        scale = ta * dtdT
        if i == 0:
            use_g = local_t <= tp[0]
            q = tp[4]
            p = tp[6]
        elif i == 1:
            use_g = True
            q = tp[5]
            p = tp[7]
        elif i == 2:
            use_g = True
            q = tp[4]
            p = tp[8]
        else:
            use_g = local_t >= tp[3]
            q = tp[5]
            p = tp[9]
        if use_g:
            a[i] = _g_blend(q, p, u)
            da[i] = _g_blend_d1(q, p, u) * scale
            dda[i] = _g_blend_d2(q, p, u) * scale * scale
        else:
            a[i] = _h_blend(q, u)
            da[i] = _h_blend_d1(q, u) * scale
            dda[i] = _h_blend_d2(q, u) * scale * scale

    total = a[0] + a[1] + a[2] + a[3]
    total2 = total * total
    total4 = total2 * total2
    d_total = da[0] + da[1] + da[2] + da[3]
    dd_total = dda[0] + dda[1] + dda[2] + dda[3]
    out = np.zeros(6)
    for i in range(4):
        c0 = a[i] / total
        d1 = da[i] * total - a[i] * d_total
        c1 = d1 / total2
        dd1 = dda[i] * total + da[i] * d_total
        dd2 = da[i] * d_total + a[i] * dd_total
        c2 = ((dd1 - dd2) * total2 - d1 * (2 * total * d_total)) / total4
        for k in range(2):
            value = pos[idxs[i], k]
            out[k] += value * c0
            out[2 + k] += value * c1
            out[4 + k] += value * c2
    return out


@njit(cache=True)
def _project_scalar(
    ox: float, oy: float, vx: float, vy: float, px: float, py: float
) -> float:
    norm = vx * vx + vy * vy
    if abs(norm) > _EPS:
        mx = vx / norm
        my = vy / norm
    else:
        mx = 0.0
        my = 0.0
    return mx * (px - ox) + my * (py - oy)


@njit(cache=True)
def _sample(
    pos: NDArray[np.float64],
    tension: NDArray[np.float64],
    max_dist_from_spline: float,
    max_dist_between_samples: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Adaptive samples along the whole spline, in order.

    Returns:
        ``(t, points, flags)`` with the first sample flagged HEAD, the last
        TAIL and the samples on segment boundaries JUNCTION.
    """
    num_segments = pos.shape[0] - 1
    max_sq_to_spline = max_dist_from_spline * max_dist_from_spline
    max_sq_between = max_dist_between_samples * max_dist_between_samples

    ts = np.empty(256)
    pts = np.empty((256, 2))
    flags = np.empty(256, dtype=np.int64)
    count = 0

    from_x, from_y = _point_at(pos, tension, 0.0)
    to_x, to_y = _point_at(pos, tension, 1.0)
    ts[0] = 0.0
    pts[0, 0] = from_x
    pts[0, 1] = from_y
    flags[0] = HEAD_SAMPLE
    count = 1
    if num_segments == 0:
        return ts[:count], pts[:count], flags[:count]

    r_num_segments = 1.0 / num_segments
    # Each frame is a segment still to be refined (kind 0) or a sample ready to
    # be emitted (kind 1); frames are popped in the order the C++ recursion
    # visits them.
    kinds = np.empty(_STACK_SIZE, dtype=np.int64)
    frames = np.empty((_STACK_SIZE, 6))
    top = 0
    kinds[0] = 0
    frames[0, 0] = 0.0
    frames[0, 1] = from_x
    frames[0, 2] = from_y
    frames[0, 3] = 1.0
    frames[0, 4] = to_x
    frames[0, 5] = to_y
    top = 1
    while top > 0:
        top -= 1
        kind = kinds[top]
        if kind == 1:
            if count >= ts.shape[0]:
                ts = np.concatenate((ts, np.empty(ts.shape[0])))
                pts = np.concatenate((pts, np.empty((pts.shape[0], 2))))
                flags = np.concatenate((flags, np.empty(flags.shape[0], np.int64)))
            ts[count] = frames[top, 0]
            pts[count, 0] = frames[top, 1]
            pts[count, 1] = frames[top, 2]
            flags[count] = int(frames[top, 3])
            count += 1
            continue

        prev_t = frames[top, 0]
        prev_x = frames[top, 1]
        prev_y = frames[top, 2]
        next_t = frames[top, 3]
        next_x = frames[top, 4]
        next_y = frames[top, 5]
        sqdist = (next_x - prev_x) ** 2 + (next_y - prev_y) ** 2
        if sqdist < _MIN_SAMPLE_SQDIST:
            continue

        flag = DEFAULT_SAMPLE
        mid_t = 0.5 * (prev_t + next_t)
        nearby = math.floor(mid_t * num_segments + 0.5) * r_num_segments
        if (nearby - prev_t) * (next_t - prev_t) > 0 and (nearby - next_t) * (
            prev_t - next_t
        ) > 0:
            mid_t = nearby
            flag = JUNCTION_SAMPLE
        mid_x, mid_y = _point_at(pos, tension, mid_t)

        if flag != JUNCTION_SAMPLE:
            scalar = _project_scalar(
                prev_x, prev_y, next_x - prev_x, next_y - prev_y, mid_x, mid_y
            )
            foot_x = prev_x + (next_x - prev_x) * scalar
            foot_y = prev_y + (next_y - prev_y) * scalar
            if sqdist <= max_sq_between:
                if (mid_x - foot_x) ** 2 + (mid_y - foot_y) ** 2 <= max_sq_to_spline:
                    continue
        if count + (top + 1) * 2 >= _MAX_SAMPLES or top + 3 >= _STACK_SIZE:
            continue

        # Pushed in reverse: refine [prev, mid], emit mid, refine [mid, next].
        kinds[top] = 0
        frames[top, 0] = mid_t
        frames[top, 1] = mid_x
        frames[top, 2] = mid_y
        frames[top, 3] = next_t
        frames[top, 4] = next_x
        frames[top, 5] = next_y
        kinds[top + 1] = 1
        frames[top + 1, 0] = mid_t
        frames[top + 1, 1] = mid_x
        frames[top + 1, 2] = mid_y
        frames[top + 1, 3] = float(flag)
        kinds[top + 2] = 0
        frames[top + 2, 0] = prev_t
        frames[top + 2, 1] = prev_x
        frames[top + 2, 2] = prev_y
        frames[top + 2, 3] = mid_t
        frames[top + 2, 4] = mid_x
        frames[top + 2, 5] = mid_y
        top += 3

    if count >= ts.shape[0]:
        ts = np.concatenate((ts, np.empty(ts.shape[0])))
        pts = np.concatenate((pts, np.empty((pts.shape[0], 2))))
        flags = np.concatenate((flags, np.empty(flags.shape[0], np.int64)))
    ts[count] = 1.0
    pts[count, 0] = to_x
    pts[count, 1] = to_y
    flags[count] = TAIL_SAMPLE
    count += 1
    return ts[:count], pts[:count], flags[:count]


@njit(cache=True)
def _control_points_force(
    pos: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Squared distances between adjacent control points, for displacements."""
    count = pos.shape[0]
    num_vars = 2 * count
    matrix = np.zeros((num_vars, num_vars))
    linear = np.zeros(num_vars)
    constant = 0.0
    for i in range(1, count):
        for k in range(2):
            a = 2 * i + k
            b = 2 * (i - 1) + k
            diff = pos[i, k] - pos[i - 1, k]
            constant += diff * diff
            linear[a] += 2.0 * diff
            linear[b] -= 2.0 * diff
            matrix[a, a] += 1.0
            matrix[b, b] += 1.0
            matrix[a, b] -= 1.0
            matrix[b, a] -= 1.0
    return matrix, linear, constant


@njit(cache=True)
def _junction_points_force(
    pos: NDArray[np.float64], tension: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Squared distances between adjacent junction points, for displacements."""
    count = pos.shape[0]
    num_vars = 2 * count
    matrix = np.zeros((num_vars, num_vars))
    linear = np.zeros(num_vars)
    constant = 0.0

    previous = np.zeros(count)
    current = np.zeros(count)
    for i in range(count):
        current[:] = 0.0
        n, idxs, coeffs = _linear_combination_at(tension, i / (count - 1.0))
        for j in range(n):
            current[idxs[j]] += coeffs[j]
        if i > 0:
            diff = current - previous
            for k in range(2):
                residual = 0.0
                for m in range(count):
                    residual += diff[m] * pos[m, k]
                constant += residual * residual
                for m in range(count):
                    linear[2 * m + k] += 2.0 * residual * diff[m]
                    for n2 in range(count):
                        matrix[2 * m + k, 2 * n2 + k] += diff[m] * diff[n2]
        previous[:] = current
    return matrix, linear, constant


@njit(cache=True)
def _lu_solve(
    matrix: NDArray[np.float64], rhs: NDArray[np.float64]
) -> tuple[NDArray[np.float64], bool]:
    """Solve a square system with the pivoting and zero rules of the C++ solver.

    Entries no larger than the square root of the machine epsilon count as zero,
    and a pivot that small makes the matrix singular.

    Returns:
        ``(solution, ok)``; the solution is only meaningful when ``ok``.
    """
    size = matrix.shape[0]
    lu = matrix.copy()
    perm = np.arange(size)
    for i in range(size):
        virt_pivot = i
        largest = abs(lu[perm[i], i])
        for j in range(i + 1, size):
            candidate = abs(lu[perm[j], i])
            if candidate > largest:
                largest = candidate
                virt_pivot = j
        if largest <= _SQRT_EPS:
            return np.zeros(size), False
        phys_pivot = perm[virt_pivot]
        perm[virt_pivot] = perm[i]
        perm[i] = phys_pivot

        r_pivot = 1.0 / lu[phys_pivot, i]
        for j in range(i + 1, size):
            row = perm[j]
            if abs(lu[row, i]) <= _SQRT_EPS:
                lu[row, i] = 0.0
                continue
            factor = lu[row, i] * r_pivot
            lu[row, i] = factor
            for col in range(i + 1, size):
                lu[row, col] -= lu[phys_pivot, col] * factor

    y = np.zeros(size)
    for v in range(size):
        right = rhs[perm[v]]
        for c in range(v):
            right -= lu[perm[v], c] * y[c]
        y[v] = right
    x = np.zeros(size)
    for v in range(size - 1, -1, -1):
        right = y[v]
        for c in range(size - 1, v, -1):
            right -= lu[perm[v], c] * x[c]
        x[v] = right / lu[perm[v], v]
    return x, True


@njit(cache=True)
def _constrain_spline_point_to_line(
    pos: NDArray[np.float64],
    tension: NDArray[np.float64],
    t: float,
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    out_a: NDArray[np.float64],
) -> float:
    """Linear constraint ``a . d + b = 0`` keeping the spline point at ``t`` on a line.

    Args:
        pos (numpy.ndarray of numpy.float64): Control points.
        tension (numpy.ndarray of numpy.float64): Tension of each control point.
        t (float): Position on the spline.
        p1x (float): X of the line's first point.
        p1y (float): Y of the line's first point.
        p2x (float): X of the line's second point.
        p2y (float): Y of the line's second point.
        out_a (numpy.ndarray of numpy.float64):
            Row of length ``2 * count`` that receives ``a``.

    Returns:
        float: The constant ``b``.
    """
    out_a[:] = 0.0
    n, idxs, coeffs = _linear_combination_at(tension, t)
    dx = p2x - p1x
    dy = p2y - p1y
    constant = p1y * dx - p1x * dy
    for i in range(n):
        idx = idxs[i]
        out_a[idx * 2] = coeffs[i] * dy
        out_a[idx * 2 + 1] = -coeffs[i] * dx
        constant += pos[idx, 0] * coeffs[i] * dy - pos[idx, 1] * coeffs[i] * dx
    return constant


@njit(cache=True)
def _has_loops(pos: NDArray[np.float64], tension: NDArray[np.float64]) -> bool:
    """True if a control point step runs against the spline's overall direction."""
    x0, y0 = _point_at(pos, tension, 0.0)
    x1, y1 = _point_at(pos, tension, 1.0)
    dx = x1 - x0
    dy = y1 - y0
    for i in range(1, pos.shape[0]):
        step_x = pos[i, 0] - pos[i - 1, 0]
        step_y = pos[i, 1] - pos[i - 1, 1]
        if step_x * dx + step_y * dy < 0:
            return True
    return False


@njit(cache=True)
def _model_vertices(polyline: NDArray[np.float64]) -> NDArray[np.float64]:
    """Point, tangent and curvature at each polyline vertex.

    An interpolating X-spline through the vertices gives the derivatives.

    Returns:
        Array of shape ``(n, 5)``: ``x, y, dx, dy, signed curvature``.
    """
    count = polyline.shape[0]
    tension = np.full(count, -1.0)
    scale = 1.0 / (count - 1)
    out = np.empty((count, 5))
    for i in range(count):
        t = min(i * scale, 1.0)
        pd = _point_and_derivs(polyline, tension, t)
        cross = pd[2] * pd[5] - pd[3] * pd[4]
        tlen = math.sqrt(pd[2] * pd[2] + pd[3] * pd[3])
        out[i, 0] = pd[0]
        out[i, 1] = pd[1]
        out[i, 2] = pd[2]
        out[i, 3] = pd[3]
        out[i, 4] = cross / (tlen * tlen * tlen)
    return out


@njit(cache=True)
def _sq_dist_approximant(
    origin_x: float,
    origin_y: float,
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    m: float,
    n: float,
) -> NDArray[np.float64]:
    """Quadratic ``m (u.(x-o))^2 + n (v.(x-o))^2`` as ``[A00, A01, A11, b0, b1, c]``."""
    tx = -(ux * origin_x + uy * origin_y)
    ty = -(vx * origin_x + vy * origin_y)
    out = np.empty(6)
    out[0] = m * ux * ux + n * vx * vx
    out[1] = m * ux * uy + n * vx * vy
    out[2] = m * uy * uy + n * vy * vy
    out[3] = 2 * (m * tx * ux + n * ty * vx)
    out[4] = 2 * (m * tx * uy + n * ty * vy)
    out[5] = m * tx * tx + n * ty * ty
    return out


@njit(cache=True)
def _frenet_unit(tx: float, ty: float) -> tuple[float, float, float, float]:
    """Unit tangent and the normal pointing to the right of it when y points down."""
    sqlen = tx * tx + ty * ty
    if sqlen > 1e-6:
        norm = math.sqrt(sqlen)
        ux = tx / norm
        uy = ty / norm
        return ux, uy, uy, -ux
    return 0.0, 0.0, 0.0, 0.0


@njit(cache=True)
def _curve_distance(
    ref_x: float,
    ref_y: float,
    origin_x: float,
    origin_y: float,
    tx: float,
    ty: float,
    curvature: float,
) -> NDArray[np.float64]:
    """Local squared distance to a curve with the given curvature at ``origin``."""
    ux, uy, nx, ny = _frenet_unit(tx, ty)
    abs_curvature = abs(curvature)
    m = 0.0
    if abs_curvature > _EPS:
        d = abs(nx * (ref_x - origin_x) + ny * (ref_y - origin_y))
        p = 1.0 / abs_curvature
        m = d / (d + p)
    return _sq_dist_approximant(origin_x, origin_y, ux, uy, nx, ny, m, 1.0)


@njit(cache=True)
def _local_approximant(
    vertices: NDArray[np.float64],
    pt_x: float,
    pt_y: float,
    sample_flags: int,
) -> NDArray[np.float64]:
    """Squared distance of a spline sample to the traced polyline, approximated locally.

    Near the ends of the polyline only junction samples are pulled in, to the
    end point itself; elsewhere the distance to the curve is used, discounted
    by the curvature of the polyline.
    """
    count = vertices.shape[0]
    best_sqdist = np.inf
    best_x = 0.0
    best_y = 0.0
    segment_idx = -1
    segment_t = -1.0
    vertex_idx = -1
    for i in range(count - 1):
        s = _project_scalar(
            vertices[i, 0],
            vertices[i, 1],
            vertices[i + 1, 0] - vertices[i, 0],
            vertices[i + 1, 1] - vertices[i, 1],
            pt_x,
            pt_y,
        )
        if s > 0 and s < 1:
            foot_x = vertices[i, 0] + (vertices[i + 1, 0] - vertices[i, 0]) * s
            foot_y = vertices[i, 1] + (vertices[i + 1, 1] - vertices[i, 1]) * s
            sqdist = (pt_x - foot_x) ** 2 + (pt_y - foot_y) ** 2
            if sqdist < best_sqdist:
                best_sqdist = sqdist
                best_x = foot_x
                best_y = foot_y
                segment_idx = i
                segment_t = s
                vertex_idx = -1
    for i in range(count):
        sqdist = (pt_x - vertices[i, 0]) ** 2 + (pt_y - vertices[i, 1]) ** 2
        if sqdist < best_sqdist:
            best_sqdist = sqdist
            best_x = vertices[i, 0]
            best_y = vertices[i, 1]
            vertex_idx = i
            segment_idx = -1

    if segment_idx != -1:
        k1 = vertices[segment_idx, 4]
        k2 = vertices[segment_idx + 1, 4]
        curvature = k1 + segment_t * (k2 - k1)
        return _curve_distance(
            pt_x,
            pt_y,
            best_x,
            best_y,
            vertices[segment_idx + 1, 0] - vertices[segment_idx, 0],
            vertices[segment_idx + 1, 1] - vertices[segment_idx, 1],
            curvature,
        )

    at_end = vertex_idx == 0 or vertex_idx == count - 1
    if at_end:
        if sample_flags & JUNCTION_SAMPLE:
            return _sq_dist_approximant(best_x, best_y, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        return np.zeros(6)
    return _curve_distance(
        pt_x,
        pt_y,
        best_x,
        best_y,
        vertices[vertex_idx, 2],
        vertices[vertex_idx, 3],
        vertices[vertex_idx, 4],
    )


@njit(cache=True)
def _add_attraction_forces(
    pos: NDArray[np.float64],
    tension: NDArray[np.float64],
    vertices: NDArray[np.float64],
    ext_matrix: NDArray[np.float64],
    ext_linear: NDArray[np.float64],
) -> float:
    """Add the pull of the polyline on every spline sample to the external force.

    The squared distance approximant ``Q`` of a sample is a function of the
    point ``L = sum(c_m * P_m)``; as a function of displacements ``d`` of the
    control points it is ``Q(L) + (2 A L + b) . w + w^T A w`` with
    ``w = sum(c_m * d_m)``.

    Returns:
        float: The constant term added to the external force.
    """
    ts, pts, flags = _sample(
        pos, tension, _DEFAULT_MAX_DIST_FROM_SPLINE, _MAX_DIST_BETWEEN_SAMPLES
    )
    constant = 0.0
    for s in range(ts.shape[0]):
        approx = _local_approximant(vertices, pts[s, 0], pts[s, 1], flags[s])
        n, idxs, coeffs = _linear_combination_at(tension, ts[s])
        lx = 0.0
        ly = 0.0
        for m in range(n):
            lx += coeffs[m] * pos[idxs[m], 0]
            ly += coeffs[m] * pos[idxs[m], 1]
        a00 = approx[0]
        a01 = approx[1]
        a11 = approx[2]
        constant += (
            lx * lx * a00
            + 2 * lx * ly * a01
            + ly * ly * a11
            + approx[3] * lx
            + approx[4] * ly
            + approx[5]
        )
        gx = 2 * (a00 * lx + a01 * ly) + approx[3]
        gy = 2 * (a01 * lx + a11 * ly) + approx[4]
        for m in range(n):
            vm = idxs[m] * 2
            ext_linear[vm] += gx * coeffs[m]
            ext_linear[vm + 1] += gy * coeffs[m]
            for k in range(n):
                vk = idxs[k] * 2
                weight = coeffs[m] * coeffs[k]
                ext_matrix[vm, vk] += a00 * weight
                ext_matrix[vm, vk + 1] += a01 * weight
                ext_matrix[vm + 1, vk] += a01 * weight
                ext_matrix[vm + 1, vk + 1] += a11 * weight
    return constant


@njit(cache=True)
def _optimize(
    int_matrix: NDArray[np.float64],
    int_linear: NDArray[np.float64],
    int_constant: float,
    ext_matrix: NDArray[np.float64],
    ext_linear: NDArray[np.float64],
    ext_constant: float,
    weight: float,
    cons_a: NDArray[np.float64],
    cons_b: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """Minimise the weighted sum of both forces under the linear constraints.

    Returns:
        ``(displacements, force_before, force_after)``. When the system cannot
        be solved the displacements are zero and the two forces are equal.
    """
    num_vars = int_matrix.shape[0]
    num_cons = cons_a.shape[0]
    total_matrix = weight * int_matrix + ext_matrix
    total_linear = weight * int_linear + ext_linear
    total_constant = weight * int_constant + ext_constant

    size = num_vars + num_cons
    system = np.zeros((size, size))
    rhs = np.zeros(size)
    for i in range(num_vars):
        rhs[i] = -total_linear[i]
        for j in range(num_vars):
            system[i, j] = total_matrix[i, j] + total_matrix[j, i]
    for c in range(num_cons):
        rhs[num_vars + c] = -cons_b[c]
        for j in range(num_vars):
            system[num_vars + c, j] = cons_a[c, j]
            system[j, num_vars + c] = cons_a[c, j]
    solution, ok = _lu_solve(system, rhs)
    if not ok:
        return np.zeros(num_vars), total_constant, total_constant
    delta = solution[:num_vars].copy()
    after = total_constant
    for i in range(num_vars):
        after += total_linear[i] * delta[i]
        for j in range(num_vars):
            after += delta[i] * delta[j] * total_matrix[i, j]
    return delta, total_constant, after


@njit(cache=True)
def _fit_extended_spline(
    polyline: NDArray[np.float64],
    bound1: NDArray[np.float64],
    bound2: NDArray[np.float64],
) -> tuple[int, NDArray[np.float64]]:
    """Run the fit; see :func:`fit_extended_spline`.

    Returns:
        ``(status, control_points)``.
    """
    count = _INITIAL_CONTROL_POINTS
    num_vars = 2 * count
    chord_start = polyline[0]
    chord_end = polyline[polyline.shape[0] - 1]
    pos = np.empty((count, 2))
    for i in range(count):
        fraction = i / (count - 1.0)
        pos[i, 0] = chord_start[0] + (chord_end[0] - chord_start[0]) * fraction
        pos[i, 1] = chord_start[1] + (chord_end[1] - chord_start[1]) * fraction
    tension = np.ones(count)
    vertices = _model_vertices(polyline)

    # Initial stage: spread the junction points evenly along the chord.
    cons_a = np.zeros((count + 2, num_vars))
    cons_b = np.zeros(count + 2)
    cons_b[0] = _constrain_spline_point_to_line(
        pos,
        tension,
        0.0,
        bound1[0, 0],
        bound1[0, 1],
        bound1[1, 0],
        bound1[1, 1],
        cons_a[0],
    )
    cons_b[1] = _constrain_spline_point_to_line(
        pos,
        tension,
        1.0,
        bound2[0, 0],
        bound2[0, 1],
        bound2[1, 0],
        bound2[1, 1],
        cons_a[1],
    )
    for i in range(count):
        cons_b[2 + i] = _constrain_spline_point_to_line(
            pos,
            tension,
            i / (count - 1.0),
            chord_start[0],
            chord_start[1],
            chord_end[0],
            chord_end[1],
            cons_a[2 + i],
        )
    int_matrix, int_linear, int_constant = _junction_points_force(pos, tension)
    delta, _, _ = _optimize(
        int_matrix,
        int_linear,
        int_constant,
        np.zeros((num_vars, num_vars)),
        np.zeros(num_vars),
        0.0,
        1.0,
        cons_a,
        cons_b,
    )
    for i in range(count):
        pos[i, 0] += delta[2 * i]
        pos[i, 1] += delta[2 * i + 1]

    # Main stage: only the two bounds stay constrained.
    cons_a = np.zeros((2, num_vars))
    cons_b = np.zeros(2)
    cons_b[0] = _constrain_spline_point_to_line(
        pos,
        tension,
        0.0,
        bound1[0, 0],
        bound1[0, 1],
        bound1[1, 0],
        bound1[1, 1],
        cons_a[0],
    )
    cons_b[1] = _constrain_spline_point_to_line(
        pos,
        tension,
        1.0,
        bound2[0, 0],
        bound2[0, 1],
        bound2[1, 0],
        bound2[1, 1],
        cons_a[1],
    )

    ratio = _INITIAL_FORCE_RATIO
    iterations_to_target = _MAX_ITERATIONS - 1
    rate = (_TARGET_FORCE_RATIO - ratio) / iterations_to_target
    last_delta = np.zeros(num_vars)
    for iteration in range(_MAX_ITERATIONS):
        ext_matrix = np.zeros((num_vars, num_vars))
        ext_linear = np.zeros(num_vars)
        ext_constant = _add_attraction_forces(
            pos, tension, vertices, ext_matrix, ext_linear
        )
        int_matrix, int_linear, int_constant = _control_points_force(pos)
        weight = 0.0
        if abs(int_constant) > 1e-6:
            weight = ratio * ext_constant / int_constant
        delta, before, after = _optimize(
            int_matrix,
            int_linear,
            int_constant,
            ext_matrix,
            ext_linear,
            ext_constant,
            weight,
            cons_a,
            cons_b,
        )
        for i in range(count):
            pos[i, 0] += delta[2 * i]
            pos[i, 1] += delta[2 * i + 1]
        for c in range(cons_a.shape[0]):
            shift = 0.0
            for j in range(num_vars):
                shift += cons_a[c, j] * delta[j]
            cons_b[c] += shift
        last_delta = delta

        if _has_loops(pos, tension):
            if iteration == 0:
                return STATUS_BAD_CURVE, pos
            for i in range(count):
                pos[i, 0] -= last_delta[2 * i]
                pos[i, 1] -= last_delta[2 * i + 1]
            break

        before = max(before, 0.0)
        after = max(after, 0.0)
        improvement = (before - after) / (before + _EPS) * 100.0
        if improvement < _MIN_IMPROVEMENT_PERCENT:
            break
        if iterations_to_target > 0:
            iterations_to_target -= 1
            ratio += rate
    return STATUS_OK, pos


@dataclass(frozen=True)
class FittedSpline:
    """An X-spline as the fit leaves it.

    Attributes:
        positions (numpy.ndarray of numpy.float64): Control points, shape ``(n, 2)``.
        tensions (numpy.ndarray of numpy.float64): Tension of each control point.
    """

    positions: NDArray[np.float64]
    tensions: NDArray[np.float64]

    def to_polyline(
        self,
        max_dist_from_spline: float = _DEFAULT_MAX_DIST_FROM_SPLINE,
        max_dist_between_samples: float = math.inf,
    ) -> NDArray[np.float64]:
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
    positions: NDArray[np.float64],
    tensions: NDArray[np.float64],
    max_dist_from_spline: float = _DEFAULT_MAX_DIST_FROM_SPLINE,
    max_dist_between_samples: float = math.inf,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
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
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(tensions, dtype=np.float64),
        float(max_dist_from_spline),
        float(max_dist_between_samples),
    )


def spline_point_at(
    positions: NDArray[np.float64], tensions: NDArray[np.float64], t: float
) -> tuple[float, float]:
    """Point of an X-spline at parameter ``t`` in [0, 1]."""
    return _point_at(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(tensions, dtype=np.float64),
        float(t),
    )


def linear_combination_at(
    tensions: NDArray[np.float64], t: float
) -> list[tuple[int, float]]:
    """Control points and weights that make up the spline point at ``t``.

    Args:
        tensions (numpy.ndarray of numpy.float64): Tension of each control point.
        t (float): Position on the spline, 0 to 1.

    Returns:
        list[tuple[int, float]]: ``(control point index, weight)`` pairs.
    """
    count, idxs, coeffs = _linear_combination_at(
        np.ascontiguousarray(tensions, dtype=np.float64), float(t)
    )
    return [(int(idxs[i]), float(coeffs[i])) for i in range(count)]


def lu_solve(
    matrix: NDArray[np.float64], rhs: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Solve ``matrix @ x = rhs`` with the C++ solver's pivoting and singularity rule.

    Args:
        matrix (numpy.ndarray of numpy.float64): Square matrix.
        rhs (numpy.ndarray of numpy.float64): Right-hand side vector.

    Returns:
        numpy.ndarray of numpy.float64 | None: The solution, or None when the matrix is
            singular by the solver's tolerance.
    """
    solution, ok = _lu_solve(
        np.ascontiguousarray(matrix, dtype=np.float64),
        np.ascontiguousarray(rhs, dtype=np.float64),
    )
    return solution if ok else None


def control_points_attraction_force(
    positions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Quadratic function of displacements for the spread of adjacent control points."""
    return _control_points_force(np.ascontiguousarray(positions, dtype=np.float64))


def junction_points_attraction_force(
    positions: NDArray[np.float64], tensions: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Quadratic function of displacements for the spread of junction points."""
    return _junction_points_force(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(tensions, dtype=np.float64),
    )


def fit_extended_spline(
    polyline: NDArray[np.float64],
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
    points = np.ascontiguousarray(polyline, dtype=np.float64)
    if np.array_equal(points[0], points[-1]):
        return None
    first = np.asarray(bound_first, dtype=np.float64)
    second = np.asarray(bound_second, dtype=np.float64)
    status, positions = _fit_extended_spline(points, first, second)
    if status != STATUS_OK:
        return None
    return FittedSpline(positions=positions, tensions=np.ones(len(positions)))
