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

import cv2
import numpy as np
from numpy.typing import NDArray

# Vertical offsets count double so that marks beside a word are preferred over
# marks above or below it.
_VERTICAL_SCALE = 2

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
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.view(np.uint8), connectivity=8
    )
    if count == 1:
        return binary.copy()

    # Every big component becomes node 0; small ones get nodes 1..m.
    width = stats[:, cv2.CC_STAT_WIDTH]
    height = stats[:, cv2.CC_STAT_HEIGHT]
    is_big = (width >= thresholds.big_object_size) | (
        height >= thresholds.big_object_size
    )
    is_big[0] = False
    small_labels = np.flatnonzero(~is_big)[1:]
    node_of_label = np.zeros(count, dtype=np.int32)
    node_of_label[small_labels] = np.arange(1, small_labels.size + 1, dtype=np.int32)
    node_of_label[0] = -1
    if not is_big.any():
        keep = np.zeros(small_labels.size + 1, dtype=bool)
        return _remove_unkept(binary, node_of_label, labels, keep)

    nodes = node_of_label[labels]
    node_pixels = np.empty(small_labels.size + 1, dtype=np.float64)
    node_pixels[0] = binary.size  # Never too small to be a target.
    node_pixels[1:] = stats[small_labels, cv2.CC_STAT_AREA]

    lo, hi, sq_dist = _nearest_neighbour_distances(nodes, ink)
    near_small, near_big = _anchor_tags(
        lo, hi, sq_dist, node_pixels, thresholds, node_pixels.size
    )
    only_small = near_small & ~near_big
    only_small[0] = False
    if only_small.any():
        # Voronoi cells of nearer components can hide a bigger neighbour, so
        # components with only small neighbours look again with the others
        # fixed in place.
        second = _nearest_neighbour_distances(
            nodes, ink & only_small[nodes], fixed=ink & ~only_small[nodes]
        )
        lo, hi, sq_dist = _min_per_pair(
            np.concatenate([lo, second[0]]),
            np.concatenate([hi, second[1]]),
            np.concatenate([sq_dist, second[2]]),
            node_pixels.size,
        )

    keep = _reachable_from_big(lo, hi, sq_dist, node_pixels, thresholds)
    return _remove_unkept(binary, node_of_label, labels, keep)


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
    sources: NDArray[np.bool_],
    fixed: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Find the closest approach between components that share a Voronoi border.

    Every pixel is assigned to the nearest source pixel under the metric with
    doubled vertical distances. Wherever two adjacent pixels belong to cells of
    different components, the squared plain Euclidean distance between their
    two source pixels is a candidate for the distance between those components.

    Args:
        nodes (NDArray[np.int32]): Node id per pixel, -1 for background.
        sources (NDArray[np.bool_]): Pixels that claim territory.
        fixed (NDArray[np.bool_] | None): Pixels that keep their own node and
            never claim territory, for example ink of components that were
            already resolved.

    Returns:
        tuple: ``(lo, hi, sq_dist)`` with one entry per component pair, node
            ids ``lo < hi`` and the minimum squared distance found.
    """
    scale = _VERTICAL_SCALE
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
    table_node[cell_ids] = np.repeat(nodes[source_y, source_x], scale)

    cell_node = table_node[cells]
    if fixed is not None:
        fixed_stretched = np.repeat(fixed, scale, 0)
        cell_node[fixed_stretched] = np.repeat(nodes, scale, 0)[fixed_stretched]

    # Only the first of the stretched rows that show an image row is compared, so
    # the neighbours are the 4-connected pixels of the original image.
    parts_a, parts_b, parts_d = [], [], []
    comparisons = (
        (cell_node[::scale, :-1], cell_node[::scale, 1:], (scale, 1), (0, 1)),
        (cell_node[:-scale:scale], cell_node[scale::scale], (scale, 1), (scale, 0)),
    )
    for first, second, step, offset in comparisons:
        i, j = np.nonzero(first != second)
        row_a, col_a = i * step[0], j * step[1]
        row_b, col_b = row_a + offset[0], col_a + offset[1]
        x_a, y_a = _source_points(cells, table_x, table_y, fixed, row_a, col_a)
        x_b, y_b = _source_points(cells, table_x, table_y, fixed, row_b, col_b)
        parts_a.append(first[i, j].astype(np.int64))
        parts_b.append(second[i, j].astype(np.int64))
        parts_d.append((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

    node_a = np.concatenate(parts_a)
    node_b = np.concatenate(parts_b)
    sq_dist = np.concatenate(parts_d)
    valid = (node_a >= 0) & (node_b >= 0)
    return _min_per_pair(
        np.minimum(node_a, node_b)[valid],
        np.maximum(node_a, node_b)[valid],
        sq_dist[valid],
        int(nodes.max()) + 1,
    )


def _source_points(
    cells: NDArray[np.int32],
    table_x: NDArray[np.int32],
    table_y: NDArray[np.int32],
    fixed: NDArray[np.bool_] | None,
    rows: NDArray[np.intp],
    cols: NDArray[np.intp],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Image coordinates of the source pixel that owns each stretched position."""
    cell = cells[rows, cols]
    x = table_x[cell].astype(np.int64)
    y = table_y[cell].astype(np.int64)
    if fixed is not None:
        own = fixed[rows // _VERTICAL_SCALE, cols]
        x = np.where(own, cols, x)
        y = np.where(own, rows // _VERTICAL_SCALE, y)
    return x, y


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
