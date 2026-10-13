"""Despeckling of bilevel images that keeps small marks lying close to text.

A connected component is kept when it is big, or when it is near enough to a
kept component that is not much smaller than itself. Dots over letters, full
stops and diacritics therefore survive, while isolated specks are removed.

Distances use the metric ``dx^2 + (2 * dy)^2`` when assigning pixels to their
nearest component, so horizontal proximity counts for more than vertical.

A second Voronoi pass looks again at components whose only close neighbours are
too small to anchor them. The ink of components that are already kept counts as
its own source there, so a speck beside a kept component can attach to it even
when a nearer small neighbour claimed the territory in between. The original C++
code measures those pixels against a sentinel coordinate far outside the image,
which makes its second pass unable to attach anything to a kept component. The
extra pass is deliberate: it keeps the dots of Arabic letters, the dots of i and
j and full stops, which the stricter C++ result removes.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numba import njit
from numpy.typing import NDArray

# Vertical offsets count double so that marks beside a word are preferred over
# marks above or below it.
_VERTICAL_SCALE = 2

# Background kept around the ink when cells are assigned, in attach radii. Cells
# of marks near the page edge extend into it.
_BORDER_RADII = 2

# Initial capacity of the border pair buffers, in pairs.
_INITIAL_PAIR_CAPACITY = 1 << 16

# Most threads that share the cell assignment of one image.
_MAX_BAND_THREADS = 8
# A band of rows is only worth a thread when it is this many times taller than
# the rows it shares with its neighbours.
_MIN_BAND_HEIGHT_IN_BORDERS = 2

_REFERENCE_DPI = 300.0
_DEFAULT_DPI = 300


class _Thresholds:
    """Size and distance limits for one despeckle level at one resolution.

    Attributes:
        min_relative_parent_weight (float): A component may attach to a target
            whose pixel count is at least this multiple of its own.
        pixels_to_sq_dist (int): Pixel count multiplied by this gives the
            largest squared distance over which a component may attach.
        big_object_size (int): A component with width or height at least this
            many pixels is always kept.
    """

    def __init__(self, level: float, dpi: int) -> None:
        dpi_factor = dpi / _REFERENCE_DPI
        self.min_relative_parent_weight = (0.05 * level + 0.075) * dpi_factor
        self.pixels_to_sq_dist = int((0.25 * level**2 - 4.25 * level + 14) ** 2)
        self.big_object_size = int(np.floor((5 * level + 2) * dpi_factor + 0.5))

    @property
    def attach_radius(self) -> int:
        """Longest distance in pixels over which any small component may attach."""
        return int(np.ceil(self.big_object_size * np.sqrt(self.pixels_to_sq_dist)))


def despeckle(
    binary: NDArray[np.uint8],
    dpi: int,
    level: float,
) -> NDArray[np.uint8]:
    """Remove small black components that are not near substantial ones.

    ``level`` is continuous: 1.0, 2.0 and 3.0 are the cautious, normal and
    aggressive presets. Thresholds scale linearly with ``dpi / 300``.

    Args:
        binary (NDArray[np.uint8]): 2-D bilevel image, 0 is ink and any other
            value is background. Not modified.
        dpi (int): Resolution of the image; use the smaller of the horizontal
            and vertical values. Non-positive values are treated as 300.
        level (float): Despeckle strength; values of 0 or below return a copy.

    Returns:
        NDArray[np.uint8]: New image of the same shape where removed ink is 255
            and everything else is unchanged. An image with ink but no
            component reaching the big-object size loses all of it.
    """
    if level <= 0:
        return binary.copy()
    thresholds = _Thresholds(level, dpi if dpi > 0 else _DEFAULT_DPI)

    ink = binary == 0
    left, top, width, height = cv2.boundingRect(ink.view(np.uint8))
    if width == 0:
        return binary.copy()
    # Everything but the Voronoi borders is decided inside the ink's bounding box.
    box = (slice(top, top + height), slice(left, left + width))
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(ink[box]).view(np.uint8), connectivity=8
    )
    labels: NDArray[np.int32] = np.asarray(component_map, dtype=np.int32)

    # Every big component becomes node 0; small ones get nodes 1..m.
    component_width = stats[:, cv2.CC_STAT_WIDTH]
    component_height = stats[:, cv2.CC_STAT_HEIGHT]
    is_big = (component_width >= thresholds.big_object_size) | (
        component_height >= thresholds.big_object_size
    )
    is_big[0] = False
    small_labels = np.flatnonzero(~is_big)[1:]
    node_of_label = np.zeros(count, dtype=np.int32)
    node_of_label[small_labels] = np.arange(1, small_labels.size + 1, dtype=np.int32)
    node_of_label[0] = -1
    cropped = binary[box]
    if not is_big.any():
        keep = np.zeros(small_labels.size + 1, dtype=bool)
        return _paste(binary, box, _remove_unkept(cropped, node_of_label, labels, keep))

    node_pixels = np.empty(small_labels.size + 1, dtype=np.float64)
    node_pixels[0] = binary.size  # Never too small to be a target.
    node_pixels[1:] = stats[small_labels, cv2.CC_STAT_AREA]

    nodes = _pad_with_background(
        node_of_label[labels],
        binary.shape,
        box,
        _BORDER_RADII * thresholds.attach_radius,
    )
    border = _BORDER_RADII * thresholds.attach_radius
    everyone = np.ones(node_pixels.size, dtype=bool)
    lo, hi, sq_dist = _nearest_neighbour_distances(nodes, everyone, False, border)
    near_small, near_big = _anchor_tags(
        lo, hi, sq_dist, node_pixels, thresholds, node_pixels.size
    )
    only_small = near_small & ~near_big
    only_small[0] = False
    if only_small.any():
        # Voronoi cells of nearer components can hide a bigger neighbour, so
        # components with only small neighbours look again with the others
        # fixed in place.
        second = _nearest_neighbour_distances(nodes, only_small, True, border)
        lo, hi, sq_dist = _min_per_pair(
            np.concatenate([lo, second[0]]),
            np.concatenate([hi, second[1]]),
            np.concatenate([sq_dist, second[2]]),
            node_pixels.size,
        )

    keep = _reachable_from_big(lo, hi, sq_dist, node_pixels, thresholds)
    return _paste(binary, box, _remove_unkept(cropped, node_of_label, labels, keep))


def _paste(
    binary: NDArray[np.uint8],
    box: tuple[slice, slice],
    cropped_result: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Return a copy of ``binary`` with ``cropped_result`` in place of ``box``."""
    result = binary.copy()
    result[box] = cropped_result
    return result


def _pad_with_background(
    nodes: NDArray[np.int32],
    image_shape: tuple[int, ...],
    box: tuple[slice, slice],
    border: int,
) -> NDArray[np.int32]:
    """Surround the nodes of ``box`` with the background the image has there.

    Args:
        nodes (NDArray[np.int32]): Node ids of the pixels of ``box``.
        image_shape (tuple[int, ...]): Shape of the whole image.
        box (tuple[slice, slice]): Rows and columns of the image that ``nodes`` cover.
        border (int): Most background pixels to add on each side; fewer where the
            image ends sooner.

    Returns:
        NDArray[np.int32]: Node ids with -1 for background in the padding.
    """
    rows, cols = box
    pad = (
        (min(border, rows.start), min(border, image_shape[0] - rows.stop)),
        (min(border, cols.start), min(border, image_shape[1] - cols.stop)),
    )
    return np.pad(nodes, pad, constant_values=-1)


def _remove_unkept(
    binary: NDArray[np.uint8],
    node_of_label: NDArray[np.int32],
    labels: NDArray[np.int32],
    keep: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Return a copy of ``binary`` with the ink of unkept nodes painted white."""
    # Background has node -1, which indexes the False appended at the end;
    # background pixels are then marked kept so only ink is repainted.
    keep_by_label = np.append(keep, False)[node_of_label]
    keep_by_label[0] = True
    result = binary.copy()
    result[~keep_by_label[labels]] = 255
    return result


def _nearest_neighbour_distances(
    nodes: NDArray[np.int32],
    claims: NDArray[np.bool_],
    rest_is_fixed: bool,
    halo: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Find the closest approach between components that share a Voronoi border.

    Every pixel is assigned to the nearest source pixel under the metric with
    doubled vertical distances. Wherever two adjacent pixels belong to cells of
    different components, the squared plain Euclidean distance between their
    two source pixels is a candidate for the distance between those components.

    The rows are cut into bands that threads work on side by side. Each band
    sees ``halo`` rows beyond its own, which settles the cells of every pixel
    that has a source within twice that many pixels; borders farther from their
    sources than that are not seen.

    Args:
        nodes (NDArray[np.int32]): Node id per pixel, -1 for background.
        claims (NDArray[np.bool_]): Per node, whether its ink claims territory.
        rest_is_fixed (bool): Whether the ink of nodes that do not claim keeps
            its own node as an unclaiming obstacle, for example because those
            components were already resolved.
        halo (int): Rows of context around a band.

    Returns:
        tuple: ``(lo, hi, sq_dist)`` with one entry per component pair, node
            ids ``lo < hi`` and the minimum squared distance found.
    """
    rows = nodes.shape[0]
    workers = min(cv2.getNumThreads(), _MAX_BAND_THREADS)
    band_count = max(1, min(workers, rows // (_MIN_BAND_HEIGHT_IN_BORDERS * halo)))
    edges = np.linspace(0, rows, band_count + 1).astype(int)

    def band(index: int) -> tuple[NDArray[np.int64], ...]:
        return _band_pairs(
            nodes, claims, rest_is_fixed, halo, edges[index], edges[index + 1]
        )

    if band_count == 1:
        parts = [band(0)]
    else:
        with ThreadPoolExecutor(max_workers=band_count) as pool:
            parts = list(pool.map(band, range(band_count)))
    return _min_per_pair(
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
        claims.size,
    )


def _band_pairs(
    nodes: NDArray[np.int32],
    claims: NDArray[np.bool_],
    rest_is_fixed: bool,
    halo: int,
    first_row: int,
    stop_row: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return the border pairs of the pixel pairs that start in a band of rows."""
    scale = _VERTICAL_SCALE
    top = max(0, first_row - halo)
    window = nodes[top : min(nodes.shape[0], stop_row + halo + 1)]
    ink = window >= 0
    sources = ink & claims[window]
    if not sources.any():
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    # Stretching rows makes the Euclidean transform use the scaled metric. Row
    # r of the stretched grid shows row r // scale of the image.
    stretched = np.repeat(np.where(sources, 0, 255).astype(np.uint8), scale, 0)
    _, label_image = cv2.distanceTransformWithLabels(
        stretched, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    cells: NDArray[np.int32] = np.asarray(label_image, dtype=np.int32)

    source_y, source_x = np.nonzero(sources)
    stretched_y = (source_y * scale)[:, None] + np.arange(scale)
    cell_ids = cells[stretched_y, source_x[:, None]].reshape(-1)
    table_x = np.zeros(cell_ids.max() + 1, dtype=np.int32)
    table_y = np.zeros_like(table_x)
    table_node = np.full_like(table_x, -1)
    table_x[cell_ids] = np.repeat(source_x, scale)
    table_y[cell_ids] = np.repeat(source_y, scale)
    table_node[cell_ids] = np.repeat(window[source_y, source_x], scale)

    own = ink & ~sources if rest_is_fixed else np.zeros(window.shape, dtype=bool)
    lo, hi, sq_dist = _border_pairs(
        cells,
        scale,
        table_x,
        table_y,
        table_node,
        window,
        own,
        first_row - top,
        stop_row - top,
    )
    return _min_per_pair(lo, hi, sq_dist, claims.size)


@njit(cache=True, nogil=True)
def _border_pairs(
    cells: NDArray[np.int32],
    scale: int,
    table_x: NDArray[np.int32],
    table_y: NDArray[np.int32],
    table_node: NDArray[np.int32],
    nodes: NDArray[np.int32],
    own: NDArray[np.bool_],
    row_start: int,
    row_stop: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """List the squared source distance at every border between two components.

    Only the first of the stretched rows that show an image row is looked at, so
    the neighbours compared are the 4-connected pixels of the image. A pixel in
    ``own`` belongs to its own node and is its own source; any other pixel
    belongs to the source of its cell. A pixel is compared with its left and its
    upper neighbour, for the rows ``row_start`` to ``row_stop`` exclusive.

    Returns:
        tuple: Parallel arrays of the lower node, the higher node and the squared
            distance, one entry per border pixel pair between components.
    """
    width = nodes.shape[1]
    capacity = _INITIAL_PAIR_CAPACITY
    lows = np.empty(capacity, dtype=np.int64)
    highs = np.empty(capacity, dtype=np.int64)
    dists = np.empty(capacity, dtype=np.int64)
    count = 0
    above_node = np.full(width, -1, dtype=np.int32)
    if row_start > 0:
        for col in range(width):
            if own[row_start - 1, col]:
                above_node[col] = nodes[row_start - 1, col]
            else:
                above_node[col] = table_node[cells[(row_start - 1) * scale, col]]
    for row in range(row_start, row_stop):
        stretched_row = row * scale
        left_node = np.int32(-1)
        for col in range(width):
            if own[row, col]:
                node = nodes[row, col]
            else:
                node = table_node[cells[stretched_row, col]]
            for neighbour in range(2):
                if neighbour == 0:
                    if col == 0:
                        continue
                    other = left_node
                    other_row, other_col = row, col - 1
                else:
                    if row == 0:
                        continue
                    other = above_node[col]
                    other_row, other_col = row - 1, col
                if node == other or node < 0 or other < 0:
                    continue
                if own[row, col]:
                    x_a, y_a = col, row
                else:
                    cell = cells[stretched_row, col]
                    x_a, y_a = table_x[cell], table_y[cell]
                if own[other_row, other_col]:
                    x_b, y_b = other_col, other_row
                else:
                    cell = cells[other_row * scale, other_col]
                    x_b, y_b = table_x[cell], table_y[cell]
                if count == capacity:
                    capacity *= 2
                    lows = _grown(lows, capacity)
                    highs = _grown(highs, capacity)
                    dists = _grown(dists, capacity)
                lows[count] = min(node, other)
                highs[count] = max(node, other)
                dists[count] = np.int64(x_a - x_b) ** 2 + np.int64(y_a - y_b) ** 2
                count += 1
            left_node = node
            above_node[col] = node
    return lows[:count], highs[:count], dists[:count]


@njit(cache=True, nogil=True)
def _grown(array: NDArray[np.int64], capacity: int) -> NDArray[np.int64]:
    bigger = np.empty(capacity, dtype=np.int64)
    bigger[: array.size] = array
    return bigger


def _min_per_pair(
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    sq_dist: NDArray[np.int64],
    node_count: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Collapse repeated ``(lo, hi)`` pairs to their smallest squared distance."""
    if lo.size == 0:
        return lo, hi, sq_dist
    pair = lo * node_count + hi
    dist_bits = int(sq_dist.max()).bit_length()
    if (node_count * node_count).bit_length() + dist_bits > 62:
        order = np.lexsort((sq_dist, pair))
        pair, sq_dist = pair[order], sq_dist[order]
    else:
        # One sort of packed (pair, distance) values beats a two-key lexsort.
        packed = np.sort((pair << dist_bits) | sq_dist)
        pair = packed >> dist_bits
        sq_dist = packed & ((1 << dist_bits) - 1)
    first = np.concatenate([[True], pair[1:] != pair[:-1]])
    pair, sq_dist = pair[first], sq_dist[first]
    return pair // node_count, pair % node_count, sq_dist


def _anchor_tags(
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    sq_dist: NDArray[np.int64],
    node_pixels: NDArray[np.float64],
    thresholds: _Thresholds,
    node_count: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Flag nodes that have a close neighbour, split by neighbour weight.

    Returns:
        tuple: ``(near_small, near_big)``. ``near_big`` is set when some close
            neighbour is heavy enough to anchor the node; ``near_small`` when
            some close neighbour is too light.
    """
    near_small = np.zeros(node_count, dtype=bool)
    near_big = np.zeros(node_count, dtype=bool)
    for source, target in ((lo, hi), (hi, lo)):
        close = sq_dist <= node_pixels[source] * thresholds.pixels_to_sq_dist
        heavy = (
            node_pixels[target]
            >= thresholds.min_relative_parent_weight * node_pixels[source]
        )
        near_big[source[close & heavy]] = True
        near_small[source[close & ~heavy]] = True
    return near_small, near_big


def _reachable_from_big(
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    sq_dist: NDArray[np.int64],
    node_pixels: NDArray[np.float64],
    thresholds: _Thresholds,
) -> NDArray[np.bool_]:
    """Return the nodes that chain back to the big component through attachments.

    A node attaches to a neighbour when it is close enough and the neighbour is
    not much lighter; attachment is followed transitively.
    """
    attach_source = []
    attach_target = []
    for source, target in ((lo, hi), (hi, lo)):
        can_attach = (sq_dist <= node_pixels[source] * thresholds.pixels_to_sq_dist) & (
            node_pixels[target]
            >= node_pixels[source] * thresholds.min_relative_parent_weight
        )
        attach_source.append(source[can_attach])
        attach_target.append(target[can_attach])
    source = np.concatenate(attach_source)
    target = np.concatenate(attach_target)

    keep = np.zeros(node_pixels.size, dtype=bool)
    keep[0] = True
    while True:
        newly = keep[target] & ~keep[source]
        if not newly.any():
            return keep
        keep[source[newly]] = True
