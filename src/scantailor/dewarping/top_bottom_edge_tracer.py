"""Trace the top and bottom edges of the page between the vertical content bounds.

A minimax shortest path search over the directional derivative of the image
(taken along the bounds) finds paths from one bound to the other that cross as
little low-contrast ground as possible. Each of the two best paths is turned
into a snake that is first slid downhill into the blurred gradient's valley,
then uphill onto its ridge, which is where the edge lies.

A straight edge that hugs the first or the last row of the image is the scan's
own border, not the paper's edge, and is left out. Upstream keeps it, and then
often prefers it to the last text line, which flattens the bottom of the model.
"""

import math

import cv2
import numpy as np
from numba import njit
from numpy.typing import NDArray

from scantailor.dewarping.grid_filters import (
    SOBEL_GRAY_SCALE,
    gauss_blur,
    sobel_xy,
)
from scantailor.dewarping.line_geometry import (
    Line,
    grid_line_cells,
    line_bounded_by_rect,
    projection_point,
)

# Images up to this size are used as they are; larger ones are shrunk.
_NO_DOWNSCALE_MAX_SIDE = 1500
_DOWNSCALED_SIDE = 1000
_NUM_BEST_PATHS = 2
# Two path ends closer than this belong to the same path.
_MIN_ENDPOINT_SQDIST = 100 * 100
# A path whose worst step crosses less contrast than this is not an edge.
_MAX_ENDPOINT_COST = 0.95
_MAX_KNOT_DIST = 15
_BLUR_SIGMA = 2.0
_SNAKE_ITERATIONS = 40
_TOO_CLOSE_PENALTY = 1000.0
_OUTSIDE_ENERGY = 1000.0
_MIN_SEGMENT_DIST = 1.0

_DOWNHILL_ELASTICITY = 0.6
_DOWNHILL_BENDING = 8.0
_DOWNHILL_EXTERNAL = 0.4
_UPHILL_ELASTICITY = 0.6
_UPHILL_BENDING = 3.0
_UPHILL_EXTERNAL = 2.0
_UPHILL_STAY_PENALTY_LEVEL = -0.02
_STAY_PENALTY = 100.0
_FLAT_ENERGY_LEVEL = 0.01

# The scanner's border is a band of a few dozen rows on a 3000-row scan; a
# paper edge that close to the image border is kept only when it bends, as a
# bent page's edge does. Distances are in pixels of the traced (shrunk) image.
_BORDER_BAND_FRACTION = 0.015
_MIN_BORDER_BAND = 3.0
_MAX_BORDER_BEND = 2.0

_NEIGHBOUR_VECTORS = np.array(
    [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]],
    dtype=np.int64,
)


def trace_top_bottom_edges(
    gray: NDArray[np.uint8], bounds: tuple[Line, Line]
) -> list[NDArray[np.float64]]:
    """Find the curves that bound the page at its top and bottom.

    Args:
        gray (numpy.ndarray of numpy.uint8): 2-D gray image.
        bounds (tuple[tuple, tuple]): The two vertical content bounds, as lines
            in the pixels of ``gray``.

    Returns:
        list[numpy.ndarray of numpy.float64]:
            Up to two curves, each of shape ``(n, 2)``,
            in the pixels of ``gray``. They run from the second bound towards
            the first. Empty when the bounds are degenerate or miss the image.
            A straight curve along the first or last row of the image is not
            returned; see ``is_image_border``.
    """
    first, second = bounds
    if first[0] == first[1] or second[0] == second[1]:
        return []

    height, width = gray.shape
    if max(width, height) < _NO_DOWNSCALE_MAX_SIDE:
        scaled = gray
        factor_x = factor_y = 1.0
    else:
        scaled_width, scaled_height = _fit_size(width, height, _DOWNSCALED_SIDE)
        scaled = np.asarray(
            cv2.resize(
                gray, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
            ),
            dtype=np.uint8,
        )
        factor_x = scaled_width / width
        factor_y = scaled_height / height
        first = _scale_line(first, factor_x, factor_y)
        second = _scale_line(second, factor_x, factor_y)
    scaled_height, scaled_width = scaled.shape

    # One pixel less on the right and bottom keeps rounded end points inside.
    rect = (0.0, 0.0, scaled_width - 1.0, scaled_height - 1.0)
    first_clipped = line_bounded_by_rect(first, rect)
    second_clipped = line_bounded_by_rect(second, rect)
    if first_clipped is None or second_clipped is None:
        return []
    first, second = first_clipped, _same_direction(first_clipped, second_clipped)

    avg_dir = _average_unit_vector(first, second)
    deriv = _directional_derivative(scaled, avg_dir)

    starts = grid_line_cells(first)
    direction = _direction_from_point_to_line(
        ((first[0][0] + first[1][0]) * 0.5, (first[0][1] + first[1][1]) * 0.5), second
    )
    cost, previous = _propagate_shortest_paths(deriv, starts, direction)
    endpoints = _locate_best_path_endpoints(cost, grid_line_cells(second))

    absolute = np.abs(deriv)
    blurred = gauss_blur(absolute, _BLUR_SIGMA, _BLUR_SIGMA)
    snakes = []
    for endpoint in endpoints:
        snake = _path_to_snake(previous, endpoint)
        down = _down_the_hill_direction(scaled_width, scaled_height, snake, avg_dir)
        snakes.append(_run_snake(snake, blurred, down, uphill=False))

    result = []
    for snake in snakes:
        down = _down_the_hill_direction(scaled_width, scaled_height, snake, avg_dir)
        refined = _run_snake(snake, absolute, (-down[0], -down[1]), uphill=True)
        if is_image_border(refined, scaled_height):
            continue
        result.append(refined * np.array([1.0 / factor_x, 1.0 / factor_y]))
    return result


def is_image_border(curve: NDArray[np.float64], height: int) -> bool:
    """Tell a straight curve along the first or last row of the image.

    Args:
        curve (numpy.ndarray of numpy.float64):
            Knots of a traced edge, shape ``(n, 2)``,
            in the pixels of the traced image.
        height (int): Rows of the traced image.

    Returns:
        bool: True when every knot lies within 1.5 percent of the image height
            (at least 3 pixels) of the first or the last row and the knots
            deviate from their best-fit line by at most 2 pixels.
    """
    if len(curve) == 0:
        return False
    band = max(_MIN_BORDER_BAND, _BORDER_BAND_FRACTION * height)
    ys = curve[:, 1]
    hugs_top = float(ys.max()) < band
    hugs_bottom = float(ys.min()) > height - 1 - band
    if not (hugs_top or hugs_bottom):
        return False
    return _bend(curve) <= _MAX_BORDER_BEND


def _bend(curve: NDArray[np.float64]) -> float:
    """Largest distance of a knot from the least-squares line through the knots."""
    xs = curve[:, 0]
    ys = curve[:, 1]
    if len(curve) < 3 or float(np.ptp(xs)) < 1.0:
        return float(np.ptp(ys))
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(np.abs(ys - (slope * xs + intercept)).max())


def _fit_size(width: int, height: int, side: int) -> tuple[int, int]:
    """Shrink a size to fit a square, keeping its aspect ratio."""
    rounded_width = side * width // height
    if rounded_width <= side:
        return max(1, rounded_width), side
    return side, max(1, side * height // width)


def _scale_line(line: Line, factor_x: float, factor_y: float) -> Line:
    (x1, y1), (x2, y2) = line
    return ((x1 * factor_x, y1 * factor_y), (x2 * factor_x, y2 * factor_y))


def _same_direction(first: Line, second: Line) -> Line:
    """Flip ``second`` when it runs against ``first``."""
    v1 = (first[1][0] - first[0][0], first[1][1] - first[0][1])
    v2 = (second[1][0] - second[0][0], second[1][1] - second[0][1])
    if v1[0] * v2[0] + v1[1] * v2[1] < 0:
        return (second[1], second[0])
    return second


def _average_unit_vector(first: Line, second: Line) -> tuple[float, float]:
    units = []
    for (x1, y1), (x2, y2) in (first, second):
        norm = math.hypot(x2 - x1, y2 - y1)
        units.append(((x2 - x1) / norm, (y2 - y1) / norm))
    sx = units[0][0] + units[1][0]
    sy = units[0][1] + units[1][1]
    norm = math.hypot(sx, sy)
    return sx / norm, sy / norm


def _direction_from_point_to_line(
    point: tuple[float, float], line: Line
) -> tuple[float, float]:
    foot = projection_point(line, point)
    vx, vy = foot[0] - point[0], foot[1] - point[1]
    sqlen = vx * vx + vy * vy
    if sqlen > 1e-5:
        norm = math.sqrt(sqlen)
        vx /= norm
        vy /= norm
    return vx, vy


def _directional_derivative(
    gray: NDArray[np.uint8], direction: tuple[float, float]
) -> NDArray[np.float32]:
    """Derivative of the image along ``direction``, within [-1, 1]; edges replicated."""
    scaled = gray.astype(np.float32) * np.float32(SOBEL_GRAY_SCALE)
    grad_x, grad_y = sobel_xy(scaled)
    return grad_x * np.float32(direction[0]) + grad_y * np.float32(direction[1])


def _propagate_shortest_paths(
    deriv: NDArray[np.float32],
    starts: list[tuple[int, int]],
    direction: tuple[float, float],
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    """Minimax path costs from the start cells, towards ``direction``.

    The cost of a path is the largest value of ``1 - |derivative|`` on it, so a
    path is cheap when it keeps to strong edges. Padding of one cell surrounds
    the returned arrays.

    Returns:
        tuple: Path cost, shape ``(h + 2, w + 2)``, and for every cell the index
            (0 to 7) of the neighbour the path came from, -1 at a path start or
            on unreached cells.
    """
    height, width = deriv.shape
    padded = np.zeros((height + 2, width + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = deriv
    start_array = np.array(
        [(y + 1) * (width + 2) + (x + 1) for x, y in starts], dtype=np.int64
    )
    direction32 = (np.float32(direction[0]), np.float32(direction[1]))
    return _shortest_paths(
        padded, start_array, float(direction32[0]), float(direction32[1])
    )


@njit(cache=True)
def _shortest_paths(
    padded: NDArray[np.float32],
    starts: NDArray[np.int64],
    direction_x: float,
    direction_y: float,
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    stride = padded.shape[1]
    total = padded.shape[0] * stride
    values = padded.reshape(total)
    cost = np.full(total, np.float32(3.4028235e38), dtype=np.float32)
    prev_nbh = np.full(total, -1, dtype=np.int8)
    heap_pos = np.full(total, -1, dtype=np.int64)
    heap = np.empty(total, dtype=np.int64)
    heap_size = 0

    # Padding cells are never entered: nothing is cheaper than their cost.
    for x in range(stride):
        cost[x] = -1.0
        cost[total - stride + x] = -1.0
    for y in range(padded.shape[0]):
        cost[y * stride] = -1.0
        cost[y * stride + stride - 1] = -1.0

    offsets = np.array(
        [-stride - 1, -stride, -stride + 1, -1, 1, stride - 1, stride, stride + 1]
    )
    vectors = np.array(
        [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    )
    opposite = np.array([7, 6, 5, 4, 3, 2, 1, 0])
    num_nb = 0
    nb_offset = np.zeros(8, dtype=np.int64)
    nb_prev = np.zeros(8, dtype=np.int64)
    for i in range(8):
        if vectors[i, 0] * direction_x + vectors[i, 1] * direction_y > 0:
            nb_offset[num_nb] = offsets[i]
            nb_prev[num_nb] = opposite[i]
            num_nb += 1

    for s in range(starts.shape[0]):
        node = starts[s]
        cost[node] = 0.0
        heap[heap_size] = node
        heap_pos[node] = heap_size
        heap_size += 1
        _bubble_up(heap, heap_pos, cost, heap_size - 1)

    while heap_size > 0:
        node = heap[0]
        # Pop: the last element takes the front and sinks.
        heap[0] = heap[heap_size - 1]
        heap_pos[heap[0]] = 0
        heap_size -= 1
        heap_pos[node] = -1
        if heap_size > 0:
            _bubble_down(heap, heap_pos, cost, heap_size, 0)

        step_cost = np.float32(1.0) - abs(values[node])
        new_cost = cost[node] if cost[node] > step_cost else step_cost
        for i in range(num_nb):
            nb = node + nb_offset[i]
            if new_cost < cost[nb]:
                cost[nb] = new_cost
                prev_nbh[nb] = nb_prev[i]
                if heap_pos[nb] == -1:
                    heap[heap_size] = nb
                    heap_pos[nb] = heap_size
                    heap_size += 1
                    _bubble_up(heap, heap_pos, cost, heap_size - 1)
                else:
                    position = _bubble_down(
                        heap, heap_pos, cost, heap_size, heap_pos[nb]
                    )
                    _bubble_up(heap, heap_pos, cost, position)
    return cost.reshape(padded.shape), prev_nbh.reshape(padded.shape)


@njit(cache=True)
def _bubble_up(
    heap: NDArray[np.int64],
    heap_pos: NDArray[np.int64],
    cost: NDArray[np.float32],
    index: int,
) -> int:
    """Move an element towards the root while it costs less than its parent."""
    while index > 0:
        parent = (index - 1) // 2
        if not cost[heap[index]] < cost[heap[parent]]:
            break
        swapped = heap[index]
        heap[index] = heap[parent]
        heap[parent] = swapped
        heap_pos[heap[index]] = index
        heap_pos[heap[parent]] = parent
        index = parent
    return index


@njit(cache=True)
def _bubble_down(
    heap: NDArray[np.int64],
    heap_pos: NDArray[np.int64],
    cost: NDArray[np.float32],
    size: int,
    index: int,
) -> int:
    """Move an element away from the root while a child costs less."""
    while True:
        left = index * 2 + 1
        right = index * 2 + 2
        if right < size:
            best = left if cost[heap[left]] < cost[heap[right]] else right
        elif left < size:
            best = left
        else:
            break
        if cost[heap[best]] < cost[heap[index]]:
            swapped = heap[index]
            heap[index] = heap[best]
            heap[best] = swapped
            heap_pos[heap[index]] = index
            heap_pos[heap[best]] = best
            index = best
        else:
            break
    return index


def _locate_best_path_endpoints(
    cost: NDArray[np.float32], cells: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Ends of the cheapest paths along the target bound, at least 100 pixels apart."""
    best: list[list] = []
    for x, y in cells:
        point_cost = cost[y + 1, x + 1]
        closest = None
        closest_sqdist = np.iinfo(np.int64).max
        for entry in best:
            sqdist = (entry[0][0] - x) ** 2 + (entry[0][1] - y) ** 2
            if sqdist < closest_sqdist:
                closest = entry
                closest_sqdist = sqdist
        if closest is not None and closest_sqdist < _MIN_ENDPOINT_SQDIST:
            if point_cost < closest[1]:
                closest[0] = (x, y)
                closest[1] = point_cost
            continue
        if len(best) < _NUM_BEST_PATHS:
            best.append([(x, y), point_cost])
        else:
            for entry in best:
                if point_cost < entry[1]:
                    entry[0] = (x, y)
                    entry[1] = point_cost
                    break
    return [entry[0] for entry in best if entry[1] < np.float32(_MAX_ENDPOINT_COST)]


def _path_to_snake(
    previous: NDArray[np.int8], endpoint: tuple[int, int]
) -> NDArray[np.float64]:
    """Follow a path back from its end and keep a knot about every 15 pixels.

    Args:
        previous (numpy.ndarray of numpy.int8):
            Neighbour index each cell was reached from,
            -1 at a path start, with one cell of padding.
        endpoint (tuple[int, int]): Where the path ends.

    Returns:
        numpy.ndarray of numpy.float64: Knots from the end point back to the path start,
            shape ``(n, 2)``.
    """
    max_dist_sq = _MAX_KNOT_DIST * _MAX_KNOT_DIST
    half_dist_sq = (_MAX_KNOT_DIST // 2) ** 2
    knots = [endpoint]
    tail = endpoint
    x, y = endpoint
    while True:
        sqdist = (x - tail[0]) ** 2 + (y - tail[1]) ** 2
        step = int(previous[y + 1, x + 1])
        if step < 0:
            if sqdist >= half_dist_sq:
                knots.append((x, y))
            break
        if sqdist >= max_dist_sq:
            knots.append((x, y))
            tail = (x, y)
        dx, dy = _NEIGHBOUR_VECTORS[step]
        x += int(dx)
        y += int(dy)
    return np.array(knots, dtype=np.float64)


def _down_the_hill_direction(
    width: int,
    height: int,
    snake: NDArray[np.float64],
    bounds_dir: tuple[float, float],
) -> tuple[float, float]:
    """Direction from the snake's centroid towards the larger part of the page."""
    centroid = (float(snake[:, 0].mean()), float(snake[:, 1].mean()))
    line = (centroid, (centroid[0] + bounds_dir[0], centroid[1] + bounds_dir[1]))
    clipped = line_bounded_by_rect(line, (0.0, 0.0, float(width), float(height)))
    if clipped is not None:
        line = clipped
    v1 = (line[0][0] - centroid[0], line[0][1] - centroid[1])
    v2 = (line[1][0] - centroid[0], line[1][1] - centroid[1])
    if v1[0] ** 2 + v1[1] ** 2 > v2[0] ** 2 + v2[1] ** 2:
        return v1
    return v2


def _run_snake(
    snake: NDArray[np.float64],
    grid: NDArray[np.float32],
    direction: tuple[float, float],
    uphill: bool,
) -> NDArray[np.float64]:
    """Move a snake into the gradient valley, or onto the gradient ridge.

    Args:
        snake (numpy.ndarray of numpy.float64): Knots, shape ``(n, 2)``.
        grid (numpy.ndarray of numpy.float32):
            Blurred gradient magnitude for the downhill
            pass, the gradient magnitude for the uphill one.
        direction (tuple[float, float]): Knots only move to neighbours that lie
            this way.
        uphill (bool): True for the uphill pass.

    Returns:
        numpy.ndarray of numpy.float64: The moved knots in the same order.
    """
    if len(snake) <= 1:
        return snake
    displacements = _displacement_vectors(direction)
    if uphill:
        displacements = displacements * 0.5
        elasticity, bending, external = (
            _UPHILL_ELASTICITY,
            _UPHILL_BENDING,
            _UPHILL_EXTERNAL,
        )
    else:
        elasticity, bending, external = (
            _DOWNHILL_ELASTICITY,
            _DOWNHILL_BENDING,
            _DOWNHILL_EXTERNAL,
        )
    return _snake_search(
        np.ascontiguousarray(snake),
        np.ascontiguousarray(grid, dtype=np.float32),
        displacements,
        uphill,
        elasticity,
        bending,
        external,
    )


def _displacement_vectors(direction: tuple[float, float]) -> NDArray[np.float64]:
    """Standing still plus every neighbour step that points the given way."""
    steps = [(0.0, 0.0)]
    for dx, dy in _NEIGHBOUR_VECTORS:
        if dx * direction[0] + dy * direction[1] > 0:
            steps.append((float(dx), float(dy)))
    return np.array(steps, dtype=np.float64)


@njit(cache=True)
def _energy_at(grid: NDArray[np.float32], px: float, py: float, uphill: bool) -> float:
    """Energy of a position: the bilinear grid value, negated going uphill.

    Positions outside the grid cost a large positive energy, whichever way the
    snake is moving.
    """
    x_base = math.floor(px)
    y_base = math.floor(py)
    xi = int(x_base)
    yi = int(y_base)
    if xi < 0 or yi < 0 or xi + 1 >= grid.shape[1] or yi + 1 >= grid.shape[0]:
        return _OUTSIDE_ENERGY
    x = px - x_base
    y = py - y_base
    x1 = 1.0 - x
    y1 = 1.0 - y
    value = (
        grid[yi, xi] * x1 * y1
        + grid[yi, xi + 1] * x * y1
        + grid[yi + 1, xi] * x1 * y
        + grid[yi + 1, xi + 1] * x * y
    )
    return -value if uphill else value


@njit(cache=True)
def _snake_search(
    snake: NDArray[np.float64],
    grid: NDArray[np.float32],
    displacements: NDArray[np.float64],
    uphill: bool,
    elasticity_weight: float,
    bending_weight: float,
    external_weight: float,
) -> NDArray[np.float64]:
    """Forty rounds of dynamic programming over the knots' displacements.

    Each round returns the knots in reverse order, as the best path is read off
    from its last knot. With an even number of rounds the order comes out as it
    went in.
    """
    num_nodes = snake.shape[0]
    num_disp = displacements.shape[0]
    avg_dist = 0.0
    for i in range(1, num_nodes):
        avg_dist += math.sqrt(
            (snake[i, 0] - snake[i - 1, 0]) ** 2 + (snake[i, 1] - snake[i - 1, 1]) ** 2
        )
    avg_dist /= num_nodes - 1

    capacity = num_nodes * num_disp
    step_x = np.zeros(capacity)
    step_y = np.zeros(capacity)
    step_prev = np.zeros(capacity, dtype=np.int64)
    step_cost = np.zeros(capacity)
    paths = np.zeros(num_disp, dtype=np.int64)
    new_paths = np.zeros(num_disp, dtype=np.int64)
    current = snake.copy()

    for _ in range(_SNAKE_ITERATIONS):
        count = 0
        num_paths = 0
        for node in range(num_nodes):
            px = current[node, 0]
            py = current[node, 1]
            cur_energy = _energy_at(grid, px, py, uphill)
            num_new = 0
            for d in range(num_disp):
                sx = px + displacements[d, 0]
                sy = py + displacements[d, 1]
                path_cost = 0.0
                adjusted = _energy_at(grid, sx, sy, uphill)
                if uphill:
                    # Standing still is discouraged where the gradient is weak.
                    if d == 0 and adjusted > _UPHILL_STAY_PENALTY_LEVEL:
                        path_cost += _STAY_PENALTY
                elif d == 0:
                    path_cost += _STAY_PENALTY
                elif cur_energy < _FLAT_ENERGY_LEVEL:
                    if cur_energy - adjusted < _FLAT_ENERGY_LEVEL:
                        continue
                path_cost += external_weight * adjusted

                best_cost = 3.4028235e38
                best_prev = -1
                for p in range(num_paths):
                    prev = paths[p]
                    cost = path_cost + step_cost[prev]
                    vx = sx - step_x[prev]
                    vy = sy - step_y[prev]
                    vec_len = math.sqrt(vx * vx + vy * vy)
                    if vec_len < _MIN_SEGMENT_DIST:
                        cost += _TOO_CLOSE_PENALTY
                    cost += elasticity_weight * (abs(avg_dist - vec_len) / avg_dist)
                    before = step_prev[prev]
                    if before != -1 and vec_len >= _MIN_SEGMENT_DIST:
                        nx = -(step_y[prev] - step_y[before])
                        ny = step_x[prev] - step_x[before]
                        normal_len = math.sqrt(nx * nx + ny * ny)
                        if normal_len < _MIN_SEGMENT_DIST:
                            cost += _TOO_CLOSE_PENALTY
                        else:
                            cos = (vx * nx + vy * ny) / (vec_len * normal_len)
                            cost += bending_weight * cos * cos
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = prev
                step_x[count] = sx
                step_y[count] = sy
                step_prev[count] = best_prev
                step_cost[count] = best_cost if best_prev != -1 else path_cost
                new_paths[num_new] = count
                num_new += 1
                count += 1
            for p in range(num_new):
                paths[p] = new_paths[p]
            num_paths = num_new

        best_path = -1
        best_cost = 3.4028235e38
        for p in range(num_paths):
            if step_cost[paths[p]] < best_cost:
                best_cost = step_cost[paths[p]]
                best_path = paths[p]
        out = 0
        step = best_path
        while step != -1:
            current[out, 0] = step_x[step]
            current[out, 1] = step_y[step]
            out += 1
            step = step_prev[step]
    return current
