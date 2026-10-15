"""Refine traced text lines with active contours ("snakes").

A traced line is replaced by a chain of nodes, each with a rib across the line
that reaches for the top and the bottom edge of the text. The chain is moved
along its tangents and normals and the ribs are resized until the top rib ends
sit on dark-to-light transitions and the bottom ends on light-to-dark ones, with
penalties for uneven spacing and bending. Every move is chosen by dynamic
programming along the chain.

All the snakes of a page evolve together: their nodes are padded to a common
count, every candidate position of every node is scored in one go, and the
dynamic programmes step along the node index for all snakes at once.
"""

import numpy as np
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
# Candidate moves of a node, in units of the current step: stay, forward, back.
_MOVES = np.array([0.0, 1.0, -1.0])

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def refine_text_lines(
    gray: NDArray[np.uint8],
    polylines: list[FloatArray],
    unit_down: tuple[float, float],
    iterations: int = 100,
) -> list[FloatArray]:
    """Move traced lines onto the middle of the text they follow.

    Args:
        gray (numpy.ndarray of numpy.uint8):
            2-D gray page at about 200 dpi, dark text on
            a light background.
        polylines (list[numpy.ndarray of numpy.float64]): Traced lines, each of shape
            ``(n, 2)`` with x, y in pixels.
        unit_down (tuple[float, float]): Unit vector pointing down the page
            along the text columns.
        iterations (int): Evolution steps granted to each snake over both blur
            stages.

    Returns:
        list[numpy.ndarray of numpy.float64]: One refined line per input line, the nodes
            of its snake. A line too short to carry two nodes comes back as one
            node.
    """
    if not polylines:
        return []
    snakes = [_make_snake(polyline) for polyline in polylines]
    remaining = [iterations] * len(snakes)

    sigma = _FIRST_BLUR_SIGMA
    gradient = _blurred_gradient(gray, unit_down, sigma)
    remaining = _evolve_snakes(snakes, gradient, unit_down, remaining, True)
    gradient = _blurred_gradient(gray, unit_down, sigma * _BLUR_SIGMA_RATIO)
    _evolve_snakes(snakes, gradient, unit_down, remaining, False)
    return [np.column_stack([cx, cy]) for cx, cy, _ in snakes]


def _blurred_gradient(
    gray: NDArray[np.uint8], unit_down: tuple[float, float], sigma: float
) -> NDArray[np.float32]:
    """Blurred derivative of the image along the down vector.

    Positive values mark the bottom edges of lines of text.
    """
    return gauss_blur(directional_gradient(gray, unit_down), sigma, sigma)


def _make_snake(polyline: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
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


def _evolve_snake(
    cx: FloatArray,
    cy: FloatArray,
    rib: FloatArray,
    grid: NDArray[np.float32],
    down_x: float,
    down_y: float,
    iterations_remaining: int,
    stop_on_convergence: bool,
) -> int:
    """Evolve one snake in place; see ``_evolve_snakes``.

    Returns:
        int: Iterations left.
    """
    return _evolve_snakes(
        [(cx, cy, rib)],
        grid,
        (down_x, down_y),
        [iterations_remaining],
        stop_on_convergence,
    )[0]


def _evolve_snakes(
    snakes: list[tuple[FloatArray, FloatArray, FloatArray]],
    grid: NDArray[np.float32],
    down: tuple[float, float],
    remaining: list[int],
    stop_on_convergence: bool,
) -> list[int]:
    """Evolve snakes in place until each converges or its iterations run out.

    A snake that has converged either stops or, when ``stop_on_convergence`` is
    false, halves its step size and carries on.

    Args:
        snakes (list[tuple]): Node x, node y and rib half length of each snake,
            updated in place.
        grid (numpy.ndarray of numpy.float32): Blurred gradient along ``down``.
        down (tuple[float, float]): Unit vector down the page.
        remaining (list[int]): Iterations each snake may still take.
        stop_on_convergence (bool): Whether a converged snake stops.

    Returns:
        list[int]: Iterations each snake has left.
    """
    batch = _SnakeBatch(snakes, grid, down)
    left = np.array(remaining, dtype=np.int64)
    factor = np.ones(len(snakes))
    running = left > 0
    while running.any():
        rows = np.flatnonzero(running)
        left[rows] -= 1
        changed = batch.step(rows, factor[rows])
        settled = rows[~changed]
        if stop_on_convergence:
            running[settled] = False
        else:
            factor[settled] *= 0.5
        running &= left > 0
    batch.write_back(snakes)
    return left.tolist()


class _SnakeBatch:
    """Snakes padded to a common node count, with what scoring them needs.

    Padding nodes repeat a snake's last node, so they add nothing to its arc
    length. Every result is read at or before each snake's own last node, so
    what the padding nodes do later does not matter.
    """

    def __init__(
        self,
        snakes: list[tuple[FloatArray, FloatArray, FloatArray]],
        grid: NDArray[np.float32],
        down: tuple[float, float],
    ) -> None:
        self.counts = np.array([len(cx) for cx, _, _ in snakes], dtype=np.int64)
        width = int(self.counts.max())
        self.cx = np.empty((len(snakes), width))
        self.cy = np.empty((len(snakes), width))
        self.rib = np.empty((len(snakes), width))
        for row, (cx, cy, rib) in enumerate(snakes):
            for target, source in ((self.cx, cx), (self.cy, cy), (self.rib, rib)):
                target[row, : len(source)] = source
                target[row, len(source) :] = source[-1]
        self.grid = _Grid(grid)
        self.down = down

    def write_back(
        self, snakes: list[tuple[FloatArray, FloatArray, FloatArray]]
    ) -> None:
        """Copy the nodes and ribs into the caller's arrays."""
        for row, (cx, cy, rib) in enumerate(snakes):
            count = len(cx)
            cx[:] = self.cx[row, :count]
            cy[:] = self.cy[row, :count]
            rib[:] = self.rib[row, :count]

    def step(self, rows: IntArray, factor: FloatArray) -> NDArray[np.bool_]:
        """One evolution step of the given snakes.

        Args:
            rows (numpy.ndarray of numpy.int64): Snakes to move.
            factor (numpy.ndarray of numpy.float64): Their step sizes.

        Returns:
            numpy.ndarray of numpy.bool: Whether each snake moved or resized.
        """
        counts = self.counts[rows]
        width = int(counts.max())
        cx = self.cx[rows, :width]
        cy = self.cy[rows, :width]
        rib = self.rib[rows, :width]
        last = counts - 1
        arc = _arc_lengths(cx, cy)
        total = arc[np.arange(rows.size), last]
        avg_segment = np.where(last > 0, total / np.maximum(last, 1), 0.0)
        long_enough = total > _FLOAT_EPS
        fraction = np.where(
            long_enough[:, None], arc / np.where(long_enough, total, 1.0)[:, None], 0.0
        )
        tangent, normal = _frames(cx, cy, arc, counts, self.down)

        changed = _thickness_adjustment(
            cx, cy, rib, normal, fraction, counts, self.grid, factor
        )
        changed |= _tangent_movement(
            cx, cy, rib, tangent, normal, avg_segment, counts, self.grid, factor
        )
        changed |= _normal_movement(cx, cy, rib, normal, counts, self.grid, factor)
        self.cx[rows, :width] = cx
        self.cy[rows, :width] = cy
        self.rib[rows, :width] = rib
        return changed


class _Grid:
    """The blurred gradient, sampled bilinearly; zero outside the grid."""

    def __init__(self, grid: NDArray[np.float32]) -> None:
        self.height, self.width = grid.shape
        self.flat = np.ascontiguousarray(grid, dtype=np.float32).ravel()

    def sample(self, px: FloatArray, py: FloatArray) -> FloatArray:
        """Bilinear values at the points; 0 where the 2x2 cell leaves the grid.

        The float32 grid values are weighted in double precision.
        """
        x_base = np.floor(px)
        y_base = np.floor(py)
        xi = x_base.astype(np.int64)
        yi = y_base.astype(np.int64)
        # Negative indexes wrap round to huge unsigned ones, so one comparison
        # per axis tells whether the 2x2 cell lies inside the grid.
        inside = (xi.view(np.uint64) < self.width - 1) & (
            yi.view(np.uint64) < self.height - 1
        )
        index = np.where(inside, yi * self.width + xi, 0)
        x = px - x_base
        y = py - y_base
        x1 = 1.0 - x
        y1 = 1.0 - y
        value = (
            self.flat[index] * x1 * y1
            + self.flat[index + 1] * x * y1
            + self.flat[index + self.width] * x1 * y
            + self.flat[index + self.width + 1] * x * y
        )
        return np.where(inside, value, 0.0)


def _node_energy(
    grid: _Grid,
    cx: FloatArray,
    cy: FloatArray,
    rib: FloatArray,
    nx: FloatArray,
    ny: FloatArray,
) -> FloatArray:
    """External energy of nodes: their rib ends want the right gradient signs.

    Opposite-signed gradient is not penalised, because an edge sometimes has to
    cross the wrong ridge on its way to the right one.
    """
    rib_x = rib * nx
    rib_y = rib * ny
    top = grid.sample(cx - rib_x, cy - rib_y)
    bottom = grid.sample(cx + rib_x, cy + rib_y)
    top_energy = _TOP_EXTERNAL_WEIGHT * np.minimum(top, 0.0)
    bottom_energy = _BOTTOM_EXTERNAL_WEIGHT * np.maximum(bottom, 0.0)
    return top_energy - bottom_energy


def _elasticity_energy(
    ax: FloatArray, ay: FloatArray, bx: FloatArray, by: FloatArray, avg_dist: FloatArray
) -> FloatArray:
    length = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        stretch = _ELASTICITY_WEIGHT * (np.abs(avg_dist - length) / avg_dist)
    return np.where(length < _MIN_NODE_DISTANCE, _TOO_CLOSE_PENALTY, stretch)


def _bending_energies(xs: FloatArray, ys: FloatArray) -> FloatArray:
    """Bending at every node for every three candidates around it.

    Args:
        xs (numpy.ndarray of numpy.float64): Candidate x of every node, ``(s, n, 3)``.
        ys (numpy.ndarray of numpy.float64): Candidate y of every node.

    Returns:
        numpy.ndarray of numpy.float64: At ``[:, k, m, p, b]``, the bending at node
            ``k - 1`` between candidate ``b`` of node ``k - 2`` and candidate ``m``
            of node ``k``, with candidate ``p`` of node ``k - 1``, shape
            ``(s, n, 3, 3, 3)``; zero for the first two nodes.
    """
    # Direction and length from candidate p of node k - 1 to candidate m of node k.
    vx = xs[:, 1:, :, None] - xs[:, :-1, None, :]
    vy = ys[:, 1:, :, None] - ys[:, :-1, None, :]
    length = np.sqrt(vx * vx + vy * vy)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = vx / length
        uy = vy / length
    too_close = length < _MIN_NODE_DISTANCE
    bend_x = ux[:, 1:, :, :, None] - ux[:, :-1, None, :, :]
    bend_y = uy[:, 1:, :, :, None] - uy[:, :-1, None, :, :]
    bending = np.zeros((*xs.shape, 3, 3))
    bending[:, 2:] = np.where(
        too_close[:, 1:, :, :, None] | too_close[:, :-1, None, :, :],
        _TOO_CLOSE_PENALTY,
        _BENDING_WEIGHT * (bend_x * bend_x + bend_y * bend_y),
    )
    return bending


def _arc_lengths(cx: FloatArray, cy: FloatArray) -> FloatArray:
    """Arc length at every node, accumulated from the first, shape ``(s, n)``."""
    steps = np.sqrt(np.diff(cx, axis=1) ** 2 + np.diff(cy, axis=1) ** 2)
    arc = np.zeros(cx.shape)
    np.cumsum(steps, axis=1, out=arc[:, 1:])
    return arc


def _frames(
    cx: FloatArray,
    cy: FloatArray,
    arc: FloatArray,
    counts: IntArray,
    down: tuple[float, float],
) -> tuple[FloatArray, FloatArray]:
    """Unit tangent and downward unit normal at every node.

    A node's tangent is the mean of the directions of its two segments; the end
    nodes take their one segment's direction. Directions of segments shorter
    than the float epsilon are not normalised, and end nodes on such segments
    get no tangent at all.

    Returns:
        tuple: Arrays of shape ``(s, n, 2)``: tangents and down-pointing normals.
    """
    tangent = np.zeros((*cx.shape, 2))
    if cx.shape[1] < 2:
        return tangent, tangent.copy()
    rows = np.arange(cx.shape[0])
    node = np.arange(cx.shape[1])
    seg = np.stack([np.diff(cx, axis=1), np.diff(cy, axis=1)], axis=2)
    seg_len = np.diff(arc, axis=1)[:, :, None]
    normalised = seg_len > _FLOAT_EPS
    unit = np.where(normalised, seg / np.where(normalised, seg_len, 1.0), seg)

    mean = 0.5 * (unit[:, :-1] + unit[:, 1:])
    mean_len = np.sqrt(mean[:, :, 0] * mean[:, :, 0] + mean[:, :, 1] * mean[:, :, 1])
    mean_len = mean_len[:, :, None]
    unit_mean = np.where(
        mean_len > _FLOAT_EPS,
        mean / np.where(mean_len > _FLOAT_EPS, mean_len, 1.0),
        mean,
    )

    tangent[:, 1:-1] = unit_mean
    tangent[:, 0] = np.where(normalised[:, 0], unit[:, 0], 0.0)
    last = counts - 1
    last_seg = np.maximum(last - 1, 0)
    tangent[rows, last] = np.where(
        normalised[rows, last_seg], unit[rows, last_seg], 0.0
    )
    tangent[counts < 2] = 0.0
    tangent[node[None, :] > last[:, None]] = 0.0

    normal = np.stack([tangent[:, :, 1], -tangent[:, :, 0]], axis=2)
    upward = normal[:, :, 0] * down[0] + normal[:, :, 1] * down[1] < 0
    normal[upward] = -normal[upward]
    return tangent, normal


def _thickness_adjustment(
    cx: FloatArray,
    cy: FloatArray,
    rib: FloatArray,
    normal: FloatArray,
    fraction: FloatArray,
    counts: IntArray,
    grid: _Grid,
    factor: FloatArray,
) -> NDArray[np.bool_]:
    """Try wider and narrower end ribs, with the inner ribs interpolated.

    The nine combinations of keeping, widening or narrowing the two end ribs
    are scored by the sum of the node energies, added node by node; the first
    cheapest wins. Inner ribs are recomputed even when the end ribs stay: nodes
    have moved along the line since they were last interpolated.

    Returns:
        numpy.ndarray of numpy.bool: Whether each snake's end ribs changed.
    """
    rows = np.arange(cx.shape[0])
    last = counts - 1
    adjustments = _MOVES[None, :] * (0.5 * factor)[:, None]
    # Combination 3 * i + j changes the head rib by adjustment i, the tail by j.
    head = np.repeat(rib[:, :1] + adjustments, 3, axis=1)
    tail = np.tile(rib[rows, last][:, None] + adjustments, 3)
    ribs = head[:, :, None] + fraction[:, None, :] * (tail - head)[:, :, None]
    energy = _node_energy(
        grid,
        cx[:, None, :],
        cy[:, None, :],
        ribs,
        normal[:, None, :, 0],
        normal[:, None, :, 1],
    )
    cost = np.cumsum(energy, axis=2)[rows, :, last]
    usable = (head > _FLOAT_EPS) & (tail > _FLOAT_EPS) & (cost < _FLOAT_MAX)
    best = np.argmin(np.where(usable, cost, np.inf), axis=1)
    best = np.where(usable.any(axis=1), best, 0)
    rib[:] = ribs[rows, best]
    return (adjustments[rows, best // 3] != 0) | (adjustments[rows, best % 3] != 0)


def _tangent_movement(
    cx: FloatArray,
    cy: FloatArray,
    rib: FloatArray,
    tangent: FloatArray,
    normal: FloatArray,
    avg_segment: FloatArray,
    counts: IntArray,
    grid: _Grid,
    factor: FloatArray,
) -> NDArray[np.bool_]:
    """Slide inner nodes along the line; the end nodes stay put.

    Each inner node may stay or move a step either way along its tangent. A
    move costs the node energy plus the elasticity of the segment to the
    previous node; the last inner node also pays for its segment to the fixed
    last node.

    Returns:
        numpy.ndarray of numpy.bool: Whether each snake's nodes moved.
    """
    moving = counts >= 3
    if not moving.any():
        return np.zeros(cx.shape[0], dtype=bool)
    rows = np.arange(cx.shape[0])
    final = np.maximum(counts - 2, 1)
    steps = _MOVES * factor[:, None, None]
    xs = cx[:, :, None] + steps * tangent[:, :, None, 0]
    ys = cy[:, :, None] + steps * tangent[:, :, None, 1]
    avg = avg_segment[:, None]
    base = _node_energy(
        grid, xs, ys, rib[:, :, None], normal[:, :, None, 0], normal[:, :, None, 1]
    )
    last = counts - 1
    base[rows, final] += _elasticity_energy(
        xs[rows, final],
        ys[rows, final],
        cx[rows, last, None],
        cy[rows, last, None],
        avg,
    )
    # Elasticity of the segment from candidate p of node k - 1 to candidate m
    # of node k, at [:, k - 1, m, p].
    elasticity = _elasticity_energy(
        xs[:, 1:, :, None],
        ys[:, 1:, :, None],
        xs[:, :-1, None, :],
        ys[:, :-1, None, :],
        avg[:, :, None, None],
    )
    # The first inner node's only predecessor is the fixed first node.
    first_cost = base[:, 1] + _elasticity_energy(
        xs[:, 1], ys[:, 1], cx[:, :1], cy[:, :1], avg
    )
    programme = _Programme(first_cost, 1, xs.shape)
    for k in range(2, xs.shape[1]):
        programme.extend(base[:, k], elasticity[:, k - 1])
    path = programme.best_paths(final)
    return _move_along_path(cx, cy, xs, ys, path, 1, final, moving)


class _Programme:
    """Cheapest paths through the three candidates of successive nodes.

    All snakes advance one node at a time. Nodes past a snake's own final node
    are scored too, but its path is read back from its final node.
    """

    def __init__(self, cost: FloatArray, node: int, shape: tuple[int, ...]) -> None:
        """Start the paths at one node.

        Args:
            cost (numpy.ndarray of numpy.float64): Cost of each candidate of
                the starting node, shape ``(s, 3)``.
            node (int): The starting node.
            shape (tuple[int, ...]): Snakes, nodes and candidates, ``(s, n, 3)``.
        """
        snakes, nodes, candidates = shape
        self.first = node
        self.node = node
        # Node first, so that each step writes one contiguous block.
        self.costs = np.zeros((nodes, snakes, candidates))
        self.costs[node] = cost
        # choices[k, :, m] is the candidate of node k - 1 on the cheapest path
        # to candidate m of node k.
        self.choices = np.zeros((nodes, snakes, candidates), dtype=np.int64)

    def extend(self, base: FloatArray, transition: FloatArray) -> None:
        """Add the next node; the first of equally cheap predecessors wins.

        Args:
            base (numpy.ndarray of numpy.float64): Cost of each candidate m of
                the new node, shape ``(s, 3)``.
            transition (numpy.ndarray of numpy.float64): Extra cost of
                candidate m after candidate p of the node before, at
                ``[:, m, p]``.
        """
        total = base[:, :, None] + self.costs[self.node, :, None, :]
        total += transition
        self.node += 1
        total.argmin(axis=2, out=self.choices[self.node])
        np.minimum.reduce(total, axis=2, out=self.costs[self.node])

    def best_paths(self, final: IntArray) -> IntArray:
        """Candidate of every node on each snake's cheapest path to its final node.

        Args:
            final (numpy.ndarray of numpy.int64): Each snake's final node.

        Returns:
            numpy.ndarray of numpy.int64: Candidate per node, shape ``(s, n)``;
                zero outside the first to the final node.
        """
        rows = np.arange(len(final))
        picks = self.costs[final, rows].argmin(axis=1).tolist()
        choices = self.choices.tolist()
        paths = np.zeros((len(final), self.costs.shape[0]), dtype=np.int64)
        for row, (pick, end) in enumerate(zip(picks, final.tolist(), strict=True)):
            path = [0] * (end + 1)
            for k in range(end, self.first - 1, -1):
                path[k] = pick
                pick = choices[k][row][pick]
            paths[row, : end + 1] = path
        return paths


def _move_along_path(
    cx: FloatArray,
    cy: FloatArray,
    xs: FloatArray,
    ys: FloatArray,
    path: IntArray,
    first: int,
    final: IntArray,
    moving: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Move the nodes from ``first`` to ``final`` to their path's candidates.

    Returns:
        numpy.ndarray of numpy.bool: Whether any node of a snake moved further
            than the float epsilon, squared.
    """
    node = np.arange(cx.shape[1])
    on_path = (node >= first) & (node <= final[:, None]) & moving[:, None]
    new_x = np.take_along_axis(xs, path[:, :, None], 2)[:, :, 0]
    new_y = np.take_along_axis(ys, path[:, :, None], 2)[:, :, 0]
    sqdist = (cx - new_x) ** 2 + (cy - new_y) ** 2
    changed = (on_path & (sqdist > _FLOAT_EPS)).any(axis=1)
    cx[on_path] = new_x[on_path]
    cy[on_path] = new_y[on_path]
    return changed


def _normal_movement(
    cx: FloatArray,
    cy: FloatArray,
    rib: FloatArray,
    normal: FloatArray,
    counts: IntArray,
    grid: _Grid,
    factor: FloatArray,
) -> NDArray[np.bool_]:
    """Move every node across the line, charging for bending.

    The bending at a node depends on the next node's candidate, its own and the
    one the best path to its own chose before it. The first two nodes have no
    such history, so each of their nine combinations starts a path.

    Returns:
        numpy.ndarray of numpy.bool: Whether each snake's nodes moved.
    """
    moving = counts >= 3
    if not moving.any():
        return np.zeros(cx.shape[0], dtype=bool)
    rows = np.arange(cx.shape[0])
    last = counts - 1
    steps = _MOVES * factor[:, None, None]
    xs = cx[:, :, None] + steps * normal[:, :, None, 0]
    ys = cy[:, :, None] + steps * normal[:, :, None, 1]
    base = _node_energy(
        grid, xs, ys, rib[:, :, None], normal[:, :, None, 0], normal[:, :, None, 1]
    )
    # Bending at node k - 1 for candidates m of node k, p of node k - 1 and b of
    # node k - 2, at [:, k, m, p, b].
    bending = _bending_energies(xs, ys)
    # Paths through the first two nodes, numbered 3 * a + b for candidates a, b.
    pair_cost = (base[:, 0, :, None] + base[:, 1, None, :]).reshape(-1, 1, 9)
    pair_bending = bending[:, 2].transpose(0, 1, 3, 2).reshape(-1, 3, 9)
    candidates = base[:, 2, :, None] + pair_cost + pair_bending
    pair_choice = np.argmin(candidates, axis=2)
    programme = _Programme(candidates.min(axis=2), 2, xs.shape)
    programme.choices[2] = pair_choice % 3
    # With these, bending[snake, k, m, p, b] has shape (s, 3, 3) and holds the
    # bending of every snake for every m and p, b varying with snake and p.
    snake = rows[:, None, None]
    m = np.arange(3)[None, :, None]
    p = np.arange(3)[None, None, :]
    for k in range(3, cx.shape[1]):
        b = programme.choices[k - 1, :, None, :]
        programme.extend(base[:, k], bending[snake, k, m, p, b])

    path = programme.best_paths(last)
    pair = pair_choice[rows, path[:, 2]]
    path[:, 0] = pair // 3
    path[:, 1] = pair % 3
    return _move_along_path(cx, cy, xs, ys, path, 0, last, moving)
