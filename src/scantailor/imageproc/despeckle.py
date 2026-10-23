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
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages

# Vertical offsets count double so that marks beside a word are preferred over
# marks above or below it.
_VERTICAL_SCALE = 2

# Background kept around the ink when cells are assigned, in attach radii. Cells
# of marks near the page edge extend into it.
_BORDER_RADII = 2

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
    dbg: DebugImages | None = None,
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
        dbg (DebugImages | None): Receives the components with the big ones
            merged into one, then the Voronoi cells of the first pass and, when
            it runs, of the second; each component in its own colour.

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
    if dbg is not None:
        dbg.add(
            _visualized(_paste_nodes(binary.shape, box, node_of_label[labels], 0)),
            "big_components_unified",
        )
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
    if dbg is not None:
        cells = _voronoi_cells(nodes, everyone, False)
        dbg.add(_visualized(_paste_nodes(binary.shape, box, cells, border)), "voronoi")
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
        if dbg is not None:
            cells = _voronoi_cells(nodes, only_small, True)
            dbg.add(
                _visualized(_paste_nodes(binary.shape, box, cells, border)),
                "voronoi_special",
            )
        lo, hi, sq_dist = _min_per_pair(
            np.concatenate([lo, second[0]]),
            np.concatenate([hi, second[1]]),
            np.concatenate([sq_dist, second[2]]),
            node_pixels.size,
        )

    keep = _reachable_from_big(lo, hi, sq_dist, node_pixels, thresholds)
    return _paste(binary, box, _remove_unkept(cropped, node_of_label, labels, keep))


def _voronoi_cells(
    nodes: NDArray[np.int32], claims: NDArray[np.bool_], rest_is_fixed: bool
) -> NDArray[np.int32]:
    """Assign every pixel to the node of its nearest claiming ink pixel.

    The whole array is transformed at once, unlike the banded transform of
    ``_nearest_neighbour_distances``, so it costs more memory and is only
    made for debug images.

    Args:
        nodes (NDArray[np.int32]): Node id per pixel, -1 for background.
        claims (NDArray[np.bool_]): Per node, whether its ink claims territory.
        rest_is_fixed (bool): Whether unclaiming ink keeps its own node.

    Returns:
        NDArray[np.int32]: Node id per pixel; -1 everywhere when no ink claims.
    """
    ink = nodes >= 0
    sources = ink & claims[np.where(ink, nodes, 0)]
    if not sources.any():
        return np.full(nodes.shape, -1, dtype=np.int32)
    scale = _VERTICAL_SCALE
    stretched = np.repeat(np.where(sources, np.uint8(0), np.uint8(255)), scale, 0)
    _, label_image = cv2.distanceTransformWithLabels(
        stretched, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    _, source_node = _sources_by_label(nodes, sources, scale)
    cells = source_node[np.asarray(label_image, dtype=np.int32)[::scale]]
    if rest_is_fixed:
        np.copyto(cells, nodes, where=ink & ~sources)
    return cells


def _paste_nodes(
    shape: tuple[int, ...],
    box: tuple[slice, slice],
    nodes: NDArray[np.int32],
    border: int,
) -> NDArray[np.int32]:
    """Place node ids of ``box`` padded as ``_pad_with_background`` does on a page."""
    rows, cols = box
    top = max(rows.start - border, 0)
    left = max(cols.start - border, 0)
    page = np.full(shape[:2], -1, dtype=np.int32)
    page[top : top + nodes.shape[0], left : left + nodes.shape[1]] = nodes
    return page


def _visualized(nodes: NDArray[np.int32]) -> NDArray[np.uint8]:
    """Draw each node in its own hue on white, as upstream's connectivity maps.

    Hues come from the bit-reversed node number, so neighbouring numbers get
    far apart colours.

    Args:
        nodes (NDArray[np.int32]): Node id per pixel, -1 for background.

    Returns:
        NDArray[np.uint8]: (H, W, 3) BGR picture.
    """
    labels = np.arange(1, int(nodes.max(initial=-1)) + 2, dtype=np.int64)
    bits = np.maximum(1, np.ceil(np.log2(labels + 1)).astype(np.int64))
    reversed_bits = np.zeros_like(labels)
    for bit in range(int(bits.max(initial=1))):
        reversed_bits |= ((labels >> bit) & 1) << np.maximum(bits - 1 - bit, 0)
    hue = np.round(0.99 * 179 * reversed_bits / ((1 << bits) - 1)).astype(np.uint8)
    hsv = np.full((labels.size, 1, 3), 255, dtype=np.uint8)
    hsv[:, 0, 0] = hue
    palette = np.vstack(
        [[[255, 255, 255]], cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]]
    ).astype(np.uint8)
    return palette[nodes + 1]


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
    sources = ink if claims.all() else ink & claims[window]
    if not sources.any():
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    # Stretching rows makes the Euclidean transform use the scaled metric. Row
    # r of the stretched grid shows row r // scale of the image.
    stretched = np.repeat(np.where(sources, np.uint8(0), np.uint8(255)), scale, 0)
    _, label_image = cv2.distanceTransformWithLabels(
        stretched, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    # Only the first stretched copy of each image row is looked at, so the
    # neighbours compared are the 4-connected pixels of the image.
    cells: NDArray[np.int32] = np.asarray(label_image, dtype=np.int32)[::scale]
    source_flat, source_node = _sources_by_label(window, sources, scale)

    lo, hi, sq_dist = _border_pairs(
        cells,
        source_flat,
        source_node,
        window,
        ink & ~sources if rest_is_fixed else None,
        first_row - top,
        stop_row - top,
    )
    return _min_per_pair(lo, hi, sq_dist, claims.size)


def _sources_by_label(
    nodes: NDArray[np.int32],
    sources: NDArray[np.bool_],
    scale: int,
) -> tuple[NDArray[np.int64], NDArray[np.int32]]:
    """List the source pixel behind every label of the stretched transform.

    OpenCV numbers the zero pixels of a ``DIST_LABEL_PIXEL`` transform from 1 in
    row-major order. The stretched image repeats each image row ``scale`` times,
    so in label order the sources of each row come ``scale`` times in a row.

    Args:
        nodes (NDArray[np.int32]): Node id per pixel of the unstretched image.
        sources (NDArray[np.bool_]): Pixels that were zero before stretching.
        scale (int): Number of stretched rows per image row.

    Returns:
        tuple: ``(flat, node)`` indexed by label: the flat index into ``nodes``
            of the label's source pixel and that pixel's node. Entry 0 is unused.
    """
    flat = np.flatnonzero(sources)
    node = nodes.reshape(-1)[flat]
    row_counts = np.count_nonzero(sources, axis=1)
    row_ends = np.cumsum(row_counts)
    row_starts = row_ends - row_counts
    rows_in_label_order = [
        slice(start, end)
        for start, end in zip(row_starts, row_ends, strict=True)
        for _ in range(scale)
    ]
    return (
        np.concatenate([[0], *(flat[row] for row in rows_in_label_order)]),
        np.concatenate([[-1], *(node[row] for row in rows_in_label_order)]),
    )


def _border_pairs(
    cells: NDArray[np.int32],
    source_flat: NDArray[np.int64],
    source_node: NDArray[np.int32],
    nodes: NDArray[np.int32],
    own: NDArray[np.bool_] | None,
    row_start: int,
    row_stop: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """List the squared source distance at every border between two components.

    A pixel in ``own`` belongs to its own node and is its own source; any other
    pixel belongs to the source of its cell. A pixel is compared with its left
    and its upper neighbour, for the rows ``row_start`` to ``row_stop``
    exclusive. Every pixel has a node, because every cell has a source.

    Args:
        cells (NDArray[np.int32]): Distance label of every pixel.
        source_flat (NDArray[np.int64]): Flat index into ``nodes`` of the
            source of each label.
        source_node (NDArray[np.int32]): Node of the source of each label.
        nodes (NDArray[np.int32]): Node id per pixel, -1 for background.
        own (NDArray[np.bool_] | None): Pixels that are their own source, or
            None for none.
        row_start (int): First row whose pixels are compared.
        row_stop (int): Row after the last row whose pixels are compared.

    Returns:
        tuple: Parallel arrays of the lower node, the higher node and the squared
            distance, one entry per border pixel pair between components, in no
            particular order.
    """
    first = max(row_start - 1, 0)
    cell = cells[first:row_stop]
    node = source_node[cell]
    own_rows = None if own is None else own[first:row_stop]
    if own_rows is not None:
        np.copyto(node, nodes[first:row_stop], where=own_rows)
    width = nodes.shape[1]

    def source_of(
        pixels: tuple[NDArray[np.intp], NDArray[np.intp]],
    ) -> NDArray[np.int64]:
        flat = source_flat[cell[pixels]]
        if own_rows is None:
            return flat
        rows, cols = pixels
        return np.where(own_rows[pixels], (rows + first) * width + cols, flat)

    def pairs_between(
        a: tuple[NDArray[np.intp], NDArray[np.intp]],
        b: tuple[NDArray[np.intp], NDArray[np.intp]],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        node_a = node[a].astype(np.int64)
        node_b = node[b].astype(np.int64)
        y_a, x_a = np.divmod(source_of(a), width)
        y_b, x_b = np.divmod(source_of(b), width)
        dx = x_a - x_b
        dy = y_a - y_b
        return np.minimum(node_a, node_b), np.maximum(node_a, node_b), dx * dx + dy * dy

    # The upper row is context only when the band does not start at the top.
    own_first = row_start - first
    rows, cols = _true_pixels(node[own_first:, 1:] != node[own_first:, :-1])
    rows += own_first
    left = pairs_between((rows, cols + 1), (rows, cols))
    rows, cols = _true_pixels(node[1:] != node[:-1])
    up = pairs_between((rows + 1, cols), (rows, cols))
    return (
        np.concatenate([left[0], up[0]]),
        np.concatenate([left[1], up[1]]),
        np.concatenate([left[2], up[2]]),
    )


def _true_pixels(mask: NDArray[np.bool_]) -> tuple[NDArray[np.intp], ...]:
    """Return ``np.nonzero(mask)``, faster for a sparse 2-D mask."""
    # np.nonzero on a 2-D mask costs about ten times more than on a flat one.
    return np.unravel_index(np.flatnonzero(mask), mask.shape)


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
