"""Refine traced text lines with active contours ("snakes").

A traced line is replaced by a chain of nodes, each with a rib across the line
that reaches for the top and the bottom edge of the text. The chain is moved
along its tangents and normals and the ribs are resized until the top rib ends
sit on dark-to-light transitions and the bottom ends on light-to-dark ones, with
penalties for uneven spacing and bending. Every move is chosen by dynamic
programming along the chain.
"""

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

from scantailor.dewarping.grid_filters import directional_gradient, gauss_blur

# Blur of the first stage, in pixels of a 200 dpi page; the second stage halves it.
_FIRST_BLUR_SIGMA = 4.0
_BLUR_SIGMA_RATIO = 0.5
_NODE_SPACING = 20.0
_INITIAL_RIB_HALF_LENGTH = 4.0
_ELASTICITY_WEIGHT = 0.2
_BENDING_WEIGHT = 1.8
_TOP_EXTERNAL_WEIGHT = 1.0
_BOTTOM_EXTERNAL_WEIGHT = 1.0
# Charged for nodes closer than a pixel; makes such moves never worth it.
_TOO_CLOSE_PENALTY = 1000.0
_MIN_NODE_DISTANCE = 1.0
_FLOAT_EPS = 1.1920929e-07
_FLOAT_MAX = 3.4028235e38


def refine_text_lines(
    gray: NDArray[np.uint8],
    polylines: list[NDArray[np.float64]],
    unit_down: tuple[float, float],
    iterations: int = 100,
) -> list[NDArray[np.float64]]:
    """Move traced lines onto the middle of the text they follow.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page at about 200 dpi, dark text on
            a light background.
        polylines (list[NDArray[np.float64]]): Traced lines, each of shape
            ``(n, 2)`` with x, y in pixels.
        unit_down (tuple[float, float]): Unit vector pointing down the page
            along the text columns.
        iterations (int): Evolution steps granted to each snake over both blur
            stages.

    Returns:
        list[NDArray[np.float64]]: One refined line per input line, the nodes
            of its snake. A line too short to carry two nodes comes back as one
            node.
    """
    if not polylines:
        return []
    snakes = [_make_snake(polyline) for polyline in polylines]
    remaining = [iterations] * len(snakes)

    sigma = _FIRST_BLUR_SIGMA
    gradient = _blurred_gradient(gray, unit_down, sigma)
    for index, (cx, cy, rib) in enumerate(snakes):
        remaining[index] = _evolve_snake(
            cx, cy, rib, gradient, unit_down[0], unit_down[1], remaining[index], True
        )

    gradient = _blurred_gradient(gray, unit_down, sigma * _BLUR_SIGMA_RATIO)
    for index, (cx, cy, rib) in enumerate(snakes):
        remaining[index] = _evolve_snake(
            cx, cy, rib, gradient, unit_down[0], unit_down[1], remaining[index], False
        )
    return [np.column_stack([cx, cy]) for cx, cy, _ in snakes]


def _blurred_gradient(
    gray: NDArray[np.uint8], unit_down: tuple[float, float], sigma: float
) -> NDArray[np.float32]:
    """Blurred derivative of the image along the down vector.

    Positive values mark the bottom edges of lines of text.
    """
    return gauss_blur(directional_gradient(gray, unit_down), sigma, sigma)


def _make_snake(
    polyline: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Spread nodes evenly along a polyline, one per ``_NODE_SPACING`` pixels.

    Returns:
        tuple: Node x, node y and rib half length arrays. The polyline's first
            and last points are always nodes; a polyline shorter than two
            spacings gets a single node at its start.
    """
    deltas = np.diff(polyline, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    total = float(lengths.sum())
    nodes_wanted = int(total / _NODE_SPACING)
    if nodes_wanted < 2:
        return (
            polyline[:1, 0].copy(),
            polyline[:1, 1].copy(),
            np.full(1, _INITIAL_RIB_HALF_LENGTH),
        )
    # Arc length at each polyline point; nodes sit at equal fractions of the total.
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = total * np.arange(nodes_wanted) / (nodes_wanted - 1)
    targets[-1] = arc[-1]
    cx = np.asarray(np.interp(targets, arc, polyline[:, 0]), dtype=np.float64)
    cy = np.asarray(np.interp(targets, arc, polyline[:, 1]), dtype=np.float64)
    return cx, cy, np.full(nodes_wanted, _INITIAL_RIB_HALF_LENGTH, dtype=np.float64)


@njit(cache=True)
def _external_energy_at(grid: NDArray[np.float32], px: float, py: float) -> float:
    """Bilinear sample of the gradient; zero outside the grid."""
    x_base = math.floor(px)
    y_base = math.floor(py)
    xi = int(x_base)
    yi = int(y_base)
    if xi < 0 or yi < 0 or xi + 1 >= grid.shape[1] or yi + 1 >= grid.shape[0]:
        return 0.0
    x = px - x_base
    y = py - y_base
    x1 = 1.0 - x
    y1 = 1.0 - y
    return (
        grid[yi, xi] * x1 * y1
        + grid[yi, xi + 1] * x * y1
        + grid[yi + 1, xi] * x1 * y
        + grid[yi + 1, xi + 1] * x * y
    )


@njit(cache=True)
def _node_energy(
    grid: NDArray[np.float32],
    cx: float,
    cy: float,
    rib: float,
    nx: float,
    ny: float,
) -> float:
    """External energy of a node: its rib ends want the right gradient signs.

    Opposite-signed gradient is not penalised, because an edge sometimes has to
    cross the wrong ridge on its way to the right one.
    """
    top = _external_energy_at(grid, cx - rib * nx, cy - rib * ny)
    bottom = _external_energy_at(grid, cx + rib * nx, cy + rib * ny)
    top_energy = _TOP_EXTERNAL_WEIGHT * min(top, 0.0)
    bottom_energy = _BOTTOM_EXTERNAL_WEIGHT * max(bottom, 0.0)
    return top_energy - bottom_energy


@njit(cache=True)
def _elasticity_energy(
    ax: float, ay: float, bx: float, by: float, avg_dist: float
) -> float:
    length = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    if length < _MIN_NODE_DISTANCE:
        return _TOO_CLOSE_PENALTY
    return _ELASTICITY_WEIGHT * (abs(avg_dist - length) / avg_dist)


@njit(cache=True)
def _bending_energy(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    """Bending at node ``b`` between its neighbours ``c`` (before) and ``a``."""
    vx = ax - bx
    vy = ay - by
    length = math.sqrt(vx * vx + vy * vy)
    if length < _MIN_NODE_DISTANCE:
        return _TOO_CLOSE_PENALTY
    pvx = bx - cx
    pvy = by - cy
    prev_length = math.sqrt(pvx * pvx + pvy * pvy)
    if prev_length < _MIN_NODE_DISTANCE:
        return _TOO_CLOSE_PENALTY
    bend_x = vx / length - pvx / prev_length
    bend_y = vy / length - pvy / prev_length
    return _BENDING_WEIGHT * (bend_x * bend_x + bend_y * bend_y)


@njit(cache=True)
def _frames(
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
    arc: NDArray[np.float64],
    down_x: float,
    down_y: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit tangent and downward unit normal at every node.

    Returns:
        tuple: Arrays of shape ``(n, 2)``: tangents and down-pointing normals.
    """
    n = cx.shape[0]
    tangent = np.zeros((n, 2))
    normal = np.zeros((n, 2))
    if n < 2:
        return tangent, normal

    first_x = cx[1] - cx[0]
    first_y = cy[1] - cy[0]
    first_len = arc[1]
    if first_len > _FLOAT_EPS:
        first_x /= first_len
        first_y /= first_len
        tangent[0, 0] = first_x
        tangent[0, 1] = first_y
    prev_x = first_x
    prev_y = first_y
    for i in range(1, n - 1):
        next_x = cx[i + 1] - cx[i]
        next_y = cy[i + 1] - cy[i]
        next_len = arc[i + 1] - arc[i]
        if next_len > _FLOAT_EPS:
            next_x /= next_len
            next_y /= next_len
        tx = 0.5 * (prev_x + next_x)
        ty = 0.5 * (prev_y + next_y)
        length = math.sqrt(tx * tx + ty * ty)
        if length > _FLOAT_EPS:
            tx /= length
            ty /= length
        tangent[i, 0] = tx
        tangent[i, 1] = ty
        prev_x = next_x
        prev_y = next_y

    last_x = cx[n - 1] - cx[n - 2]
    last_y = cy[n - 1] - cy[n - 2]
    last_len = arc[n - 1] - arc[n - 2]
    if last_len > _FLOAT_EPS:
        last_x /= last_len
        last_y /= last_len
        tangent[n - 1, 0] = last_x
        tangent[n - 1, 1] = last_y

    for i in range(n):
        nx = tangent[i, 1]
        ny = -tangent[i, 0]
        if nx * down_x + ny * down_y < 0:
            nx = -nx
            ny = -ny
        normal[i, 0] = nx
        normal[i, 1] = ny
    return tangent, normal


@njit(cache=True)
def _arc_lengths(
    cx: NDArray[np.float64], cy: NDArray[np.float64]
) -> NDArray[np.float64]:
    n = cx.shape[0]
    arc = np.zeros(n)
    total = 0.0
    for i in range(1, n):
        total += math.sqrt((cx[i] - cx[i - 1]) ** 2 + (cy[i] - cy[i - 1]) ** 2)
        arc[i] = total
    return arc


@njit(cache=True)
def _thickness_adjustment(
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
    rib: NDArray[np.float64],
    normal: NDArray[np.float64],
    fraction: NDArray[np.float64],
    grid: NDArray[np.float32],
    factor: float,
) -> bool:
    """Try wider and narrower end ribs, with the inner ribs interpolated."""
    n = cx.shape[0]
    adjustments = np.array([0.0, 0.5 * factor, -0.5 * factor])
    best_i = 0
    best_j = 0
    best_cost = _FLOAT_MAX
    for i in range(3):
        head = rib[0] + adjustments[i]
        if head <= _FLOAT_EPS:
            continue
        for j in range(3):
            tail = rib[n - 1] + adjustments[j]
            if tail <= _FLOAT_EPS:
                continue
            cost = 0.0
            for k in range(n):
                r = head + fraction[k] * (tail - head)
                cost += _node_energy(grid, cx[k], cy[k], r, normal[k, 0], normal[k, 1])
            if cost < best_cost:
                best_cost = cost
                best_i = i
                best_j = j
    head = rib[0] + adjustments[best_i]
    tail = rib[n - 1] + adjustments[best_j]
    # Inner ribs are recomputed even when the end ribs stay: nodes have moved
    # along the line since they were last interpolated.
    for k in range(n):
        rib[k] = head + fraction[k] * (tail - head)
    return adjustments[best_i] != 0 or adjustments[best_j] != 0


@njit(cache=True)
def _tangent_movement(
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
    rib: NDArray[np.float64],
    tangent: NDArray[np.float64],
    normal: NDArray[np.float64],
    avg_segment: float,
    grid: NDArray[np.float32],
    factor: float,
) -> bool:
    """Slide inner nodes along the line; the end nodes stay put."""
    n = cx.shape[0]
    if n < 3:
        return False
    movements = np.array([0.0, factor, -factor])
    capacity = 1 + 3 * (n - 2)
    step_x = np.zeros(capacity)
    step_y = np.zeros(capacity)
    step_prev = np.full(capacity, -1, dtype=np.int64)
    step_cost = np.zeros(capacity)
    count = 1
    step_x[0] = cx[0]
    step_y[0] = cy[0]
    paths = np.zeros(3, dtype=np.int64)
    num_paths = 1
    paths[0] = 0

    for node in range(1, n - 1):
        new_paths = np.zeros(3, dtype=np.int64)
        for m in range(3):
            x = cx[node] + movements[m] * tangent[node, 0]
            y = cy[node] + movements[m] * tangent[node, 1]
            base = _node_energy(grid, x, y, rib[node], normal[node, 0], normal[node, 1])
            if node == n - 2:
                base += _elasticity_energy(x, y, cx[n - 1], cy[n - 1], avg_segment)
            best = _FLOAT_MAX
            best_prev = -1
            for p in range(num_paths):
                prev = paths[p]
                cost = (
                    base
                    + step_cost[prev]
                    + _elasticity_energy(x, y, step_x[prev], step_y[prev], avg_segment)
                )
                if cost < best:
                    best = cost
                    best_prev = prev
            step_x[count] = x
            step_y[count] = y
            step_prev[count] = best_prev
            step_cost[count] = best
            new_paths[m] = count
            count += 1
        paths = new_paths
        num_paths = 3

    best_path = -1
    best_cost = _FLOAT_MAX
    for p in range(num_paths):
        if step_cost[paths[p]] < best_cost:
            best_cost = step_cost[paths[p]]
            best_path = paths[p]
    max_sqdist = 0.0
    step = best_path
    for node in range(n - 2, 0, -1):
        sqdist = (cx[node] - step_x[step]) ** 2 + (cy[node] - step_y[step]) ** 2
        max_sqdist = max(max_sqdist, sqdist)
        cx[node] = step_x[step]
        cy[node] = step_y[step]
        step = step_prev[step]
    return max_sqdist > _FLOAT_EPS


@njit(cache=True)
def _normal_movement(
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
    rib: NDArray[np.float64],
    normal: NDArray[np.float64],
    grid: NDArray[np.float32],
    factor: float,
) -> bool:
    """Move every node across the line, charging for bending."""
    n = cx.shape[0]
    if n < 3:
        return False
    movements = np.array([0.0, factor, -factor])
    capacity = 12 + 3 * (n - 2)
    step_x = np.zeros(capacity)
    step_y = np.zeros(capacity)
    step_prev = np.full(capacity, -1, dtype=np.int64)
    step_cost = np.zeros(capacity)
    count = 0
    paths = np.zeros(9, dtype=np.int64)
    num_paths = 0

    # The first two nodes lack two predecessors, so every combination of their
    # moves starts a path of its own.
    for a in range(3):
        first = count
        step_x[count] = cx[0] + movements[a] * normal[0, 0]
        step_y[count] = cy[0] + movements[a] * normal[0, 1]
        step_cost[count] = _node_energy(
            grid, step_x[count], step_y[count], rib[0], normal[0, 0], normal[0, 1]
        )
        count += 1
        for b in range(3):
            step_x[count] = cx[1] + movements[b] * normal[1, 0]
            step_y[count] = cy[1] + movements[b] * normal[1, 1]
            step_prev[count] = first
            step_cost[count] = step_cost[first] + _node_energy(
                grid, step_x[count], step_y[count], rib[1], normal[1, 0], normal[1, 1]
            )
            paths[num_paths] = count
            num_paths += 1
            count += 1

    for node in range(2, n):
        new_paths = np.zeros(3, dtype=np.int64)
        for m in range(3):
            x = cx[node] + movements[m] * normal[node, 0]
            y = cy[node] + movements[m] * normal[node, 1]
            base = _node_energy(grid, x, y, rib[node], normal[node, 0], normal[node, 1])
            best = _FLOAT_MAX
            best_prev = -1
            for p in range(num_paths):
                prev = paths[p]
                before = step_prev[prev]
                cost = (
                    base
                    + step_cost[prev]
                    + _bending_energy(
                        x, y, step_x[prev], step_y[prev], step_x[before], step_y[before]
                    )
                )
                if cost < best:
                    best = cost
                    best_prev = prev
            step_x[count] = x
            step_y[count] = y
            step_prev[count] = best_prev
            step_cost[count] = best
            new_paths[m] = count
            count += 1
        paths = new_paths
        num_paths = 3

    best_path = -1
    best_cost = _FLOAT_MAX
    for p in range(num_paths):
        if step_cost[paths[p]] < best_cost:
            best_cost = step_cost[paths[p]]
            best_path = paths[p]
    max_sqdist = 0.0
    step = best_path
    for node in range(n - 1, -1, -1):
        sqdist = (cx[node] - step_x[step]) ** 2 + (cy[node] - step_y[step]) ** 2
        max_sqdist = max(max_sqdist, sqdist)
        cx[node] = step_x[step]
        cy[node] = step_y[step]
        step = step_prev[step]
    return max_sqdist > _FLOAT_EPS


@njit(cache=True)
def _evolve_snake(
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
    rib: NDArray[np.float64],
    grid: NDArray[np.float32],
    down_x: float,
    down_y: float,
    iterations_remaining: int,
    stop_on_convergence: bool,
) -> int:
    """Evolve one snake in place until it converges or its iterations run out.

    A snake that has converged either stops or, when ``stop_on_convergence`` is
    false, halves the step size and carries on.

    Returns:
        int: Iterations left.
    """
    factor = 1.0
    n = cx.shape[0]
    while iterations_remaining > 0:
        iterations_remaining -= 1
        arc = _arc_lengths(cx, cy)
        total = arc[n - 1]
        avg_segment = total / (n - 1) if n > 1 else 0.0
        fraction = np.zeros(n)
        if total > _FLOAT_EPS:
            fraction = arc / total
        tangent, normal = _frames(cx, cy, arc, down_x, down_y)

        changed = False
        changed |= _thickness_adjustment(cx, cy, rib, normal, fraction, grid, factor)
        changed |= _tangent_movement(
            cx, cy, rib, tangent, normal, avg_segment, grid, factor
        )
        changed |= _normal_movement(cx, cy, rib, normal, grid, factor)
        if not changed:
            if stop_on_convergence:
                break
            factor *= 0.5
    return iterations_remaining
