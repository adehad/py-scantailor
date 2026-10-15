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
from typing import NamedTuple

import cv2
import numpy as np
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

_FLOAT32_MAX = 3.4028235e38
# Which step a path is taken to have arrived by when several offer the same
# cost, as (forward, across) in the sweep frame of ``_SweepFrame``.
_STEP_PREFERENCE = ((1, 0), (1, 1), (1, -1), (0, 1))
_RELAXATION_ROUNDS = 4
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

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
    snakes = [_path_to_snake(previous, endpoint) for endpoint in endpoints]
    downs = [
        _down_the_hill_direction(scaled_width, scaled_height, snake, avg_dir)
        for snake in snakes
    ]
    snakes = _run_snakes(snakes, blurred, downs, uphill=False)

    ups = []
    for snake in snakes:
        down = _down_the_hill_direction(scaled_width, scaled_height, snake, avg_dir)
        ups.append((-down[0], -down[1]))
    result = []
    for refined in _run_snakes(snakes, absolute, ups, uphill=True):
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
    path is cheap when it keeps to strong edges. A path only steps to the
    neighbours that lie ``direction``'s way.

    Every such step moves forward along ``direction``, so no path comes back to
    a cell and the costs can be swept in one pass: line by line across the
    dominant axis of ``direction``, and within a line along its other axis.
    The costs are those of a shortest path search. Where two neighbours offer
    a path of the same cost, the one whose own cost is lower is taken as the
    one the path came from, as a search that settles cheaper cells first
    would. Remaining ties go to the straight step forward, then the diagonal
    ones, then the step down the line; upstream's search breaks them in the
    order of its heap, so on flat stretches of cost the paths, though equally
    cheap, can run differently.

    Returns:
        tuple: Path cost, shape ``(h + 2, w + 2)``, and for every cell the index
            (0 to 7) of the neighbour the path came from, -1 at a path start or
            on unreached cells. Padding of one cell surrounds both arrays, with
            cost -1.
    """
    height, width = deriv.shape
    direction32 = (float(np.float32(direction[0])), float(np.float32(direction[1])))
    cost = np.full((height + 2, width + 2), -1.0, dtype=np.float32)
    previous = np.full((height + 2, width + 2), -1, dtype=np.int8)
    frame = _SweepFrame(direction32)
    step = np.float32(1.0) - np.abs(deriv)
    start_mask = np.zeros((height, width), dtype=bool)
    for x, y in starts:
        start_mask[y, x] = True
    interior = (slice(1, -1), slice(1, -1))
    sweep_cost, sweep_previous = _sweep(
        frame.to_sweep(step), frame.to_sweep(start_mask), frame.moves
    )
    cost[interior] = frame.from_sweep(sweep_cost)
    previous[interior] = frame.from_sweep(frame.neighbour_index[sweep_previous])
    return cost, previous


class _SweepFrame:
    """The grid turned so that every allowed step goes forward or down.

    In the sweep frame, axis 0 runs along the dominant axis of the direction
    and points the direction's way; axis 1 runs along the other axis, pointing
    the direction's way too. Allowed steps are then a subset of one line
    forward (and one cell either way across) or one cell down the line.

    Attributes:
        moves (list[tuple[int, int]]): Allowed steps as (forward, across), in
            the order of ``_STEP_PREFERENCE``.
        neighbour_index (numpy.ndarray of numpy.int8): For each allowed step,
            in ``moves`` order, the index in ``_NEIGHBOUR_VECTORS`` of the
            neighbour the step comes from, with -1 appended for "none".
    """

    def __init__(self, direction: tuple[float, float]) -> None:
        dx, dy = direction
        self._transposed = abs(dx) >= abs(dy)
        along, across = (dx, dy) if self._transposed else (dy, dx)
        self._flip_along = along < 0
        self._flip_across = across < 0
        allowed = []
        for index, (vx, vy) in enumerate(_NEIGHBOUR_VECTORS.tolist()):
            if vx * dx + vy * dy > 0:
                forward, side = (vx, vy) if self._transposed else (vy, vx)
                forward = -forward if self._flip_along else forward
                side = -side if self._flip_across else side
                allowed.append(((forward, side), 7 - index))
        allowed.sort(key=lambda entry: _STEP_PREFERENCE.index(entry[0]))
        self.moves = [move for move, _ in allowed]
        self.neighbour_index = np.array(
            [source for _, source in allowed] + [-1], dtype=np.int8
        )

    def to_sweep(self, grid: NDArray) -> NDArray:
        """The grid in the sweep frame, as a contiguous copy."""
        view = grid.T if self._transposed else grid
        if self._flip_along:
            view = view[::-1]
        if self._flip_across:
            view = view[:, ::-1]
        return np.ascontiguousarray(view)

    def from_sweep(self, grid: NDArray) -> NDArray:
        """A sweep frame grid back in the image frame."""
        view = grid
        if self._flip_across:
            view = view[:, ::-1]
        if self._flip_along:
            view = view[::-1]
        return view.T if self._transposed else view


def _sweep(
    step: NDArray[np.float32],
    starts: NDArray[np.bool_],
    moves: list[tuple[int, int]],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Minimax path costs in the sweep frame, line by line.

    Args:
        step (numpy.ndarray of numpy.float32): Cost of stepping off each cell.
        starts (numpy.ndarray of numpy.bool): Cells paths start from, at cost 0.
        moves (list[tuple[int, int]]): Allowed steps as (forward, across), each
            either ``(1, -1)``, ``(1, 0)``, ``(1, 1)`` or ``(0, 1)``.

    Returns:
        tuple: Costs, unreached cells at the float32 maximum, and for each
            cell the index in ``moves`` of the step that reached it, or
            ``len(moves)`` at starts and unreached cells.
    """
    lines, length = step.shape
    unreached = np.float32(_FLOAT32_MAX)
    forward = [side for along, side in moves if along == 1]
    down_the_line = (0, 1) in moves
    cost = np.empty((lines, length), dtype=np.float32)
    # Cost of a path leaving each cell of the previous line, one cell of
    # padding at each end.
    leaving = np.full(length + 2, unreached, dtype=np.float32)
    arriving = np.empty(length, dtype=np.float32)
    for line in range(lines):
        arriving.fill(unreached)
        for side in forward:
            np.minimum(arriving, leaving[1 - side : 1 - side + length], out=arriving)
        arriving[starts[line]] = 0.0
        if down_the_line:
            arriving = _down_the_line(arriving, step[line])
        cost[line] = arriving
        # Unreached cells stay unreached: no step costs more than the maximum.
        np.maximum(arriving, step[line], out=leaving[1:-1])
    return cost, _came_from(cost, step, starts, moves)


def _down_the_line(
    arriving: NDArray[np.float32], step: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Costs within a line where a path may also step down the line.

    ``cost[i] = min(arriving[i], max(cost[i - 1], step[i - 1]))``. A few rounds
    of relaxing every cell from the one before it settle almost every line;
    the rare line with a long run of such steps is finished by a prefix scan.
    """
    cost = arriving.copy()
    for _ in range(_RELAXATION_ROUNDS):
        through = np.maximum(cost[:-1], step[:-1])
        if not (through < cost[1:]).any():
            return cost
        np.minimum(cost[1:], through, out=cost[1:])
    return _down_the_line_scan(cost, step)


def _down_the_line_scan(
    arriving: NDArray[np.float32], step: NDArray[np.float32]
) -> NDArray[np.float32]:
    """``_down_the_line`` in a logarithmic number of whole-line operations.

    Each cell applies ``x -> min(high, max(x, low))`` to the cost of the one
    before it, and such clamps compose into clamps, so a prefix scan composes
    them for all cells at once.
    """
    high = arriving.copy()
    low = np.empty_like(arriving)
    low[0] = _FLOAT32_MAX
    low[1:] = step[:-1]
    span = 1
    while span < high.shape[0]:
        high[span:] = np.minimum(high[span:], np.maximum(high[:-span], low[span:]))
        low[span:] = np.maximum(low[:-span], low[span:])
        span *= 2
    return high


def _came_from(
    cost: NDArray[np.float32],
    step: NDArray[np.float32],
    starts: NDArray[np.bool_],
    moves: list[tuple[int, int]],
) -> NDArray[np.int64]:
    """For each cell, the step in ``moves`` its cheapest path arrived by.

    Of the neighbours that offer the cell's cost, the one with the lowest cost
    of its own wins; remaining ties go to the first in ``moves``.
    """
    lines, length = cost.shape
    unreached = np.float32(_FLOAT32_MAX)
    leaving = np.maximum(cost, step)
    padded_leaving = np.full((lines + 2, length + 2), unreached, dtype=np.float32)
    padded_leaving[1:-1, 1:-1] = leaving
    padded_cost = np.full((lines + 2, length + 2), unreached, dtype=np.float32)
    padded_cost[1:-1, 1:-1] = cost
    best_source = np.full((lines, length), np.inf, dtype=np.float32)
    came_from = np.full((lines, length), len(moves), dtype=np.int64)
    for index, (along, side) in enumerate(moves):
        source = (
            slice(1 - along, 1 - along + lines),
            slice(1 - side, 1 - side + length),
        )
        offers = padded_leaving[source] == cost
        better = offers & (padded_cost[source] < best_source)
        best_source[better] = padded_cost[source][better]
        came_from[better] = index
    came_from[starts | (cost == unreached)] = len(moves)
    return came_from


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
    snake: FloatArray,
    grid: NDArray[np.float32],
    direction: tuple[float, float],
    uphill: bool,
) -> FloatArray:
    """Move one snake; see ``_run_snakes``."""
    return _run_snakes([snake], grid, [direction], uphill)[0]


def _run_snakes(
    snakes: list[FloatArray],
    grid: NDArray[np.float32],
    directions: list[tuple[float, float]],
    uphill: bool,
) -> list[FloatArray]:
    """Move snakes into the gradient valley, or onto the gradient ridge.

    Args:
        snakes (list[numpy.ndarray of numpy.float64]): Knots of each snake, shape
            ``(n, 2)``.
        grid (numpy.ndarray of numpy.float32):
            Blurred gradient magnitude for the downhill
            pass, the gradient magnitude for the uphill one.
        directions (list[tuple[float, float]]): For each snake, the way its
            knots may move to neighbours.
        uphill (bool): True for the uphill pass.

    Returns:
        list[numpy.ndarray of numpy.float64]: The moved knots of each snake, in
            the same order.
    """
    result = list(snakes)
    moving = [index for index, snake in enumerate(snakes) if len(snake) > 1]
    if not moving:
        return result
    displacements = [_displacement_vectors(directions[index]) for index in moving]
    if uphill:
        displacements = [steps * 0.5 for steps in displacements]
        weights = (_UPHILL_ELASTICITY, _UPHILL_BENDING, _UPHILL_EXTERNAL)
    else:
        weights = (_DOWNHILL_ELASTICITY, _DOWNHILL_BENDING, _DOWNHILL_EXTERNAL)
    moved = _snake_search(
        [np.asarray(snakes[index], dtype=np.float64) for index in moving],
        np.ascontiguousarray(grid, dtype=np.float32),
        displacements,
        uphill,
        weights,
    )
    for index, knots in zip(moving, moved, strict=True):
        result[index] = knots
    return result


def _displacement_vectors(direction: tuple[float, float]) -> FloatArray:
    """Standing still plus every neighbour step that points the given way."""
    steps = [(0.0, 0.0)]
    for dx, dy in _NEIGHBOUR_VECTORS:
        if dx * direction[0] + dy * direction[1] > 0:
            steps.append((float(dx), float(dy)))
    return np.array(steps, dtype=np.float64)


def _energy_at(
    grid: NDArray[np.float32], px: FloatArray, py: FloatArray, uphill: bool
) -> FloatArray:
    """Energy of positions: the bilinear grid value, negated going uphill.

    Positions whose 2x2 cell leaves the grid cost a large positive energy,
    whichever way the snake is moving. The float32 grid values are weighted in
    double precision.
    """
    height, width = grid.shape
    flat = grid.ravel()
    x_base = np.floor(px)
    y_base = np.floor(py)
    xi = x_base.astype(np.int64)
    yi = y_base.astype(np.int64)
    # Negative indexes wrap round to huge unsigned ones, so one comparison per
    # axis tells whether the 2x2 cell lies inside the grid.
    inside = (xi.view(np.uint64) < width - 1) & (yi.view(np.uint64) < height - 1)
    index = np.where(inside, yi * width + xi, 0)
    x = px - x_base
    y = py - y_base
    x1 = 1.0 - x
    y1 = 1.0 - y
    value = (
        flat[index] * x1 * y1
        + flat[index + 1] * x * y1
        + flat[index + width] * x1 * y
        + flat[index + width + 1] * x * y
    )
    return np.where(inside, -value if uphill else value, _OUTSIDE_ENERGY)


class _SnakeBatch(NamedTuple):
    """Snakes padded to common knot and displacement counts.

    Padding knots repeat a snake's last knot and padding displacements stand
    still; neither is ever used: padded displacements are ruled out, and each
    snake's path is read back from its own last knot.

    Attributes:
        knots (numpy.ndarray of numpy.float64): Shape ``(s, n, 2)``.
        counts (numpy.ndarray of numpy.int64): Knots of each snake.
        displacements (numpy.ndarray of numpy.float64): Shape ``(s, d, 2)``.
        real_displacement (numpy.ndarray of numpy.bool): Shape ``(s, d)``.
        avg_dist (numpy.ndarray of numpy.float64): Mean knot spacing of each
            snake as it started.
    """

    knots: FloatArray
    counts: IntArray
    displacements: FloatArray
    real_displacement: NDArray[np.bool_]
    avg_dist: FloatArray


def _snake_batch(
    snakes: list[FloatArray], displacements: list[FloatArray]
) -> _SnakeBatch:
    counts = np.array([len(snake) for snake in snakes], dtype=np.int64)
    num_disp = max(len(steps) for steps in displacements)
    knots = np.empty((len(snakes), int(counts.max()), 2))
    padded_steps = np.zeros((len(snakes), num_disp, 2))
    real = np.zeros((len(snakes), num_disp), dtype=bool)
    avg_dist = np.empty(len(snakes))
    for row, (snake, steps) in enumerate(zip(snakes, displacements, strict=True)):
        knots[row, : len(snake)] = snake
        knots[row, len(snake) :] = snake[-1]
        padded_steps[row, : len(steps)] = steps
        real[row, : len(steps)] = True
        lengths = np.sqrt(np.diff(snake[:, 0]) ** 2 + np.diff(snake[:, 1]) ** 2)
        # Added one at a time, as upstream does.
        avg_dist[row] = float(np.add.accumulate(lengths)[-1]) / (len(snake) - 1)
    return _SnakeBatch(knots, counts, padded_steps, real, avg_dist)


def _snake_search(
    snakes: list[FloatArray],
    grid: NDArray[np.float32],
    displacements: list[FloatArray],
    uphill: bool,
    weights: tuple[float, float, float],
) -> list[FloatArray]:
    """Forty rounds of dynamic programming over the knots' displacements.

    Each round returns the knots in reverse order, as the best path is read off
    from its last knot. With an even number of rounds the order comes out as it
    went in. A knot whose every move is ruled out cuts the path: the knots
    after it start afresh, and only they are rewritten that round.

    Args:
        snakes (list[numpy.ndarray of numpy.float64]): Knots of each snake, shape
            ``(n, 2)``, n >= 2.
        grid (numpy.ndarray of numpy.float32): Energy grid.
        displacements (list[numpy.ndarray of numpy.float64]): Moves each snake's
            knots may make, shape ``(d, 2)``, standing still first.
        uphill (bool): True for the uphill pass.
        weights (tuple[float, float, float]): Elasticity, bending and external
            weights.

    Returns:
        list[numpy.ndarray of numpy.float64]: The moved knots of each snake.
    """
    batch = _snake_batch(snakes, displacements)
    current = batch.knots.copy()
    for _ in range(_SNAKE_ITERATIONS):
        _snake_round(current, grid, batch, uphill, weights)
    return [current[row, :count].copy() for row, count in enumerate(batch.counts)]


def _snake_round(
    current: FloatArray,
    grid: NDArray[np.float32],
    batch: _SnakeBatch,
    uphill: bool,
    weights: tuple[float, float, float],
) -> None:
    """One round of the snake search, rewriting ``current`` in place.

    Args:
        current (numpy.ndarray of numpy.float64): Knots, shape ``(s, n, 2)``.
        grid (numpy.ndarray of numpy.float32): Energy grid.
        batch (_SnakeBatch): The snakes' counts, moves and spacing.
        uphill (bool): True for the uphill pass.
        weights (tuple[float, float, float]): Elasticity, bending and external
            weights.
    """
    elasticity_weight, bending_weight, external_weight = weights
    num_snakes, num_knots, _ = current.shape
    num_disp = batch.displacements.shape[1]
    sx = current[:, :, 0, None] + batch.displacements[:, None, :, 0]
    sy = current[:, :, 1, None] + batch.displacements[:, None, :, 1]
    adjusted = _energy_at(grid, sx, sy, uphill)
    # The first displacement stands still.
    cur_energy = adjusted[:, :, :1]
    stay = np.zeros(sx.shape, dtype=bool)
    usable = np.broadcast_to(batch.real_displacement[:, None, :], sx.shape).copy()
    if uphill:
        # Standing still is discouraged where the gradient is weak.
        stay[:, :, 0] = adjusted[:, :, 0] > _UPHILL_STAY_PENALTY_LEVEL
    else:
        stay[:, :, 0] = True
        # On flat ground a knot moves only where the energy drops enough.
        usable[:, :, 1:] &= ~(
            (cur_energy < _FLAT_ENERGY_LEVEL)
            & (cur_energy - adjusted[:, :, 1:] < _FLAT_ENERGY_LEVEL)
        )
    path_cost = np.where(stay, _STAY_PENALTY, 0.0) + external_weight * adjusted
    # A candidate that may not be used costs infinity, so no path joins it.
    own_cost = np.where(usable, path_cost, np.inf)

    # Terms of a move to candidate d of knot k after candidate p of knot k - 1,
    # at [:, k - 1, d, p].
    vx = sx[:, 1:, :, None] - sx[:, :-1, None, :]
    vy = sy[:, 1:, :, None] - sy[:, :-1, None, :]
    vec_len = np.sqrt(vx * vx + vy * vy)
    close = vec_len < _MIN_SEGMENT_DIST
    too_close = np.where(close, _TOO_CLOSE_PENALTY, 0.0)
    avg = batch.avg_dist[:, None, None, None]
    elastic = elasticity_weight * (np.abs(avg - vec_len) / avg)
    bending = _snake_bending(sx, sy, vx, vy, vec_len, close, bending_weight)

    fresh = num_disp
    # With these, bending[snake, k, d, p, b] has shape (s, d, d) and holds the
    # bending of every snake for every d and p, b varying with snake and p.
    snake = np.arange(num_snakes)[:, None, None]
    d = np.arange(num_disp)[None, :, None]
    p = np.arange(num_disp)[None, None, :]
    step_costs = np.empty(sx.shape)
    step_costs[:, 0] = own_cost[:, 0]
    came_from = np.full(sx.shape, fresh, dtype=np.int64)
    for k in range(1, num_knots):
        cost = own_cost[:, k, :, None] + step_costs[:, k - 1, None, :]
        cost += too_close[:, k - 1]
        cost += elastic[:, k - 1]
        cost += bending[snake, k - 1, d, p, came_from[:, k - 1, None, :]]
        best_cost = np.minimum.reduce(cost, axis=2)
        joined = best_cost < _FLOAT32_MAX
        came_from[:, k] = np.where(joined, cost.argmin(axis=2), fresh)
        step_costs[:, k] = np.where(joined, best_cost, own_cost[:, k])
    final_cost = step_costs[np.arange(num_snakes), batch.counts - 1]
    for row in range(num_snakes):
        _read_back(
            current[row],
            sx[row],
            sy[row],
            came_from[row],
            final_cost[row],
            int(batch.counts[row]),
        )


def _read_back(
    current: FloatArray,
    sx: FloatArray,
    sy: FloatArray,
    came_from: IntArray,
    final_cost: FloatArray,
    count: int,
) -> None:
    """Write one snake's best path into its knots, last knot first.

    The path is read back until it reaches a knot that started it afresh, which
    ``came_from`` marks with the displacement count; the knots beyond that keep
    their positions. Nothing changes when no candidate of the last knot is
    usable.
    """
    if not (final_cost < _FLOAT32_MAX).any():
        return
    fresh = came_from.shape[1]
    pick = int(np.argmin(final_cost))
    picks = came_from.tolist()
    out = 0
    for k in range(count - 1, -1, -1):
        current[out] = (sx[k, pick], sy[k, pick])
        out += 1
        pick = picks[k][pick]
        if pick == fresh:
            break


def _snake_bending(
    sx: FloatArray,
    sy: FloatArray,
    vx: FloatArray,
    vy: FloatArray,
    vec_len: FloatArray,
    close: NDArray[np.bool_],
    bending_weight: float,
) -> FloatArray:
    """Bending of every move against the segment before it.

    Returns:
        numpy.ndarray of numpy.float64: At ``[:, k - 1, d, p, b]``, the bending of
            a move to candidate d of knot k after candidate p of knot k - 1,
            which followed candidate b of knot k - 2; the extra last slot of
            ``b`` is a path that starts afresh at knot k - 1 and does not bend.
            Zero for moves whose segment is too short.
    """
    num_snakes, num_knots, num_disp = sx.shape
    bending = np.zeros((num_snakes, num_knots - 1, num_disp, num_disp, num_disp + 1))
    # Normal of the segment from candidate b of knot k - 2 to candidate p of
    # knot k - 1, at [:, k - 2, p, b].
    nx = -(sy[:, 1:-1, :, None] - sy[:, :-2, None, :])
    ny = sx[:, 1:-1, :, None] - sx[:, :-2, None, :]
    normal_len = np.sqrt(nx * nx + ny * ny)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (
            vx[:, 1:, :, :, None] * nx[:, :, None]
            + vy[:, 1:, :, :, None] * ny[:, :, None]
        ) / (vec_len[:, 1:, :, :, None] * normal_len[:, :, None])
    terms = np.where(
        normal_len[:, :, None] < _MIN_SEGMENT_DIST,
        _TOO_CLOSE_PENALTY,
        bending_weight * cos * cos,
    )
    bending[:, 1:, :, :, :num_disp] = np.where(close[:, 1:, :, :, None], 0.0, terms)
    return bending
