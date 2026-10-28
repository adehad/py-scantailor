"""Colour segmentation of binarised ink.

The ink of a bilevel page is split into a few colour classes using the original
page, and every connected piece of ink is painted with the average colour of
the original under it. The paper stays white.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .background_color import otsu_threshold
from .filters import gray_histogram

_LEVELS = 256
_COLOR_CHANNELS = 3
_OPAQUE = 255
_WHITE = 255
# Thresholds stay inside 1..255 so that every channel has a dark and a light range.
_MIN_THRESHOLD = 1
_MAX_THRESHOLD = 255
# Reference resolution of the noise parameters.
_REFERENCE_DPI = 300.0
_MIN_AVERAGE_WIDTH_AT_REFERENCE = 1.5
_UNREACHED = np.iinfo(np.int32).max
# Queue entries offered together in one vectorised step of the label spread.
_BATCH_ENTRIES = 1024
# Neighbour offsets in the order ties between equidistant labels are resolved.
_NEIGHBOUR_DY = np.array([-1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int32)
_NEIGHBOUR_DX = np.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=np.int32)


def segment_colors(
    ink: NDArray[np.bool_],
    image: NDArray[np.uint8],
    dpi: tuple[int, int],
    noise_threshold: int,
    threshold_adjustments: tuple[int, int, int] = (0, 0, 0),
) -> NDArray[np.uint8]:
    """Paint each connected piece of ink with one colour taken from the original.

    For colour input the ink is first sorted into black, yellow, magenta, cyan,
    red, green and blue classes by thresholding every channel at its Otsu level.
    Pieces of a class that are small or thin (strokes about a pixel wide) are
    dropped as noise, and the ink they covered is handed to the nearest
    surviving piece. Each remaining piece takes the rounded mean colour of
    ``image`` under it. Gray input is not split into classes; every connected
    piece of ink takes the rounded mean gray.

    Args:
        ink (NDArray[np.bool_]): True where binarisation found ink, shape (H, W).
        image (NDArray[np.uint8]): Original page, 2-D gray or (H, W, 3|4) BGR(A).
            Not modified.
        dpi (tuple[int, int]): Horizontal and vertical resolution; scales the
            noise limits.
        noise_threshold (int): Strength of noise removal; a piece of ink whose
            pixel count is at most ``noise_threshold ** sqrt(2)`` at 300 dpi is noise.
        threshold_adjustments (tuple[int, int, int]): Shift of the red, green and
            blue thresholds; each threshold ends up between 1 and 255. Ignored
            for gray input.

    Returns:
        NDArray[np.uint8]: New image of the same shape as ``image`` with white
            paper. Colour output has an opaque alpha channel when the input has one.

    Raises:
        ValueError: If ``ink`` and ``image`` differ in height or width, or
            ``image`` is not gray or three or four channels.
    """
    if image.shape[:2] != ink.shape:
        raise ValueError("ink and image must have the same height and width")
    if image.ndim == 2:
        labels, count = _label_pieces(ink)
        return _paint_pieces(labels, count, image)
    if image.ndim != 3 or image.shape[2] not in (_COLOR_CHANNELS, _COLOR_CHANNELS + 1):
        raise ValueError("image must be 2-D gray or have 3 or 4 channels")

    labels, count = _label_color_classes(
        ink, image[..., :_COLOR_CHANNELS], dpi, noise_threshold, threshold_adjustments
    )
    labels = _extend_into_ink(labels, ink)
    leftover, leftover_count = _label_pieces(ink & (labels == 0))
    labels = np.where(leftover > 0, leftover + count, labels)
    painted = _paint_pieces(labels, count + leftover_count, image)
    if painted.shape[2] > _COLOR_CHANNELS:
        painted[..., _COLOR_CHANNELS:] = _OPAQUE
    return painted


def _label_pieces(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], int]:
    """Label 8-connected pieces of a mask 1..n; 0 elsewhere."""
    count, labels = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    return np.asarray(labels, dtype=np.int32), count - 1


def _label_color_classes(
    ink: NDArray[np.bool_],
    color: NDArray[np.uint8],
    dpi: tuple[int, int],
    noise_threshold: int,
    adjustments: tuple[int, int, int],
) -> tuple[NDArray[np.int32], int]:
    """Label the pieces of every colour class and drop the noisy ones.

    Returns:
        tuple[NDArray[np.int32], int]: Labels 1..n over the ink that belongs to
            a surviving piece and 0 elsewhere, and n.
    """
    blue, green, red = (
        _dark_channel(np.where(ink, color[..., index], _WHITE), adjustment)
        for index, adjustment in zip((0, 1, 2), reversed(adjustments), strict=True)
    )
    yellow = red & green
    magenta = red & blue
    cyan = green & blue
    black = blue & yellow

    red &= ~black & ~yellow & ~magenta
    green &= ~black & ~yellow & ~cyan
    blue &= ~black & ~magenta & ~cyan
    yellow &= ~black
    magenta &= ~black
    cyan &= ~black

    labels = np.zeros(ink.shape, dtype=np.int32)
    areas: list[NDArray[np.int64]] = []
    boxes: list[NDArray[np.int64]] = []
    total = 0
    for mask in (black, yellow, magenta, cyan, red, green, blue):
        if not mask.any():
            continue
        count, piece_labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )
        labels[mask] = piece_labels[mask] + total
        areas.append(stats[1:, cv2.CC_STAT_AREA].astype(np.int64))
        boxes.append(
            stats[1:, cv2.CC_STAT_WIDTH : cv2.CC_STAT_HEIGHT + 1].astype(np.int64)
        )
        total += count - 1
    if total == 0:
        return labels, 0

    noisy = _is_noise(
        np.concatenate(areas),
        np.concatenate(boxes).astype(np.int64),
        dpi,
        noise_threshold,
    )
    renumbered = np.concatenate(([0], np.cumsum(~noisy))).astype(np.int32)
    renumbered[1:][noisy] = 0
    return renumbered[labels], int((~noisy).sum())


def _dark_channel(channel: NDArray[np.uint8], adjustment: int) -> NDArray[np.bool_]:
    """True where a channel is below its adjusted Otsu threshold."""
    histogram = gray_histogram(channel)
    threshold = otsu_threshold(histogram) + adjustment
    return channel < min(max(threshold, _MIN_THRESHOLD), _MAX_THRESHOLD)


def _is_noise(
    areas: NDArray[np.int64],
    sizes: NDArray[np.int64],
    dpi: tuple[int, int],
    noise_threshold: int,
) -> NDArray[np.bool_]:
    """Decide which pieces are noise from pixel counts and (width, height) boxes."""
    scale = ((dpi[0] + dpi[1]) // 2) / _REFERENCE_DPI
    big_piece = np.floor(noise_threshold ** np.sqrt(2.0) * scale + 0.5)
    min_average_width = _MIN_AVERAGE_WIDTH_AT_REFERENCE * scale

    box_area = sizes[:, 0] * sizes[:, 1]
    average_width = sizes.min(axis=1) * (areas / box_area)
    return (areas <= big_piece) | (average_width <= min_average_width)


def _extend_into_ink(
    labels: NDArray[np.int32], ink: NDArray[np.bool_]
) -> NDArray[np.int32]:
    """Give unlabelled ink the label of the nearest labelled pixel it connects to.

    Labels spread breadth first through ink, so ink that no piece reaches stays
    0. Each pixel carries the vector to its source, so distances are exact
    Euclidean distances to the closest source found along the way, and a pixel
    is taken over again whenever a strictly nearer source arrives. Of equally
    near offers the first one wins, so the queue order decides ties.

    Args:
        labels (NDArray[np.int32]): Piece labels, 0 for none. Not modified.
        ink (NDArray[np.bool_]): Where labels may spread.

    Returns:
        NDArray[np.int32]: New label array.
    """
    sources = labels != 0
    open_ink = ink & ~sources
    if not sources.any() or not open_ink.any():
        return labels.copy()

    # A frame of closed pixels keeps every neighbour index inside the arrays.
    height, width = labels.shape[0] + 2, labels.shape[1] + 2
    spread = _Spread(
        label=np.pad(labels, 1).ravel(),
        reach=np.where(np.pad(open_ink, 1), np.int32(_UNREACHED), np.int32(0)).ravel(),
        vector_x=np.zeros(height * width, dtype=np.int32),
        vector_y=np.zeros(height * width, dtype=np.int32),
        offsets=_NEIGHBOUR_DY.astype(np.int64) * width + _NEIGHBOUR_DX,
    )
    # Sources with no open neighbour cannot take anything, so they are left out.
    next_to_open = cv2.dilate(open_ink.view(np.uint8), np.ones((3, 3), np.uint8))
    queue = np.flatnonzero(np.pad(sources & next_to_open.view(np.bool_), 1))
    while queue.size:
        queue = _spread_generation(queue, spread)
    return spread.label.reshape(height, width)[1:-1, 1:-1].copy()


@dataclass(frozen=True)
class _Spread:
    """Per-pixel state of the label spread over the framed, flattened page.

    Attributes:
        label (NDArray[np.int32]): Current label, 0 for none.
        reach (NDArray[np.int32]): Squared distance to the source of the label;
            ``_UNREACHED`` where still open, 0 where the pixel never changes.
        vector_x (NDArray[np.int32]): Column offset from the pixel to its source.
        vector_y (NDArray[np.int32]): Row offset from the pixel to its source.
        offsets (NDArray[np.int64]): Flat index step to each neighbour, in the
            order of ``_NEIGHBOUR_DY`` and ``_NEIGHBOUR_DX``.
    """

    label: NDArray[np.int32]
    reach: NDArray[np.int32]
    vector_x: NDArray[np.int32]
    vector_y: NDArray[np.int32]
    offsets: NDArray[np.int64]


def _spread_generation(queue: NDArray[np.int64], spread: _Spread) -> NDArray[np.int64]:
    """Take every queue entry in order and let it offer its label to its neighbours.

    Entries are handled a batch at a time with whole-array operations, which
    gives the same result as one at a time as long as no entry of the batch is
    changed by an earlier one, because that entry has to offer its new label.
    A batch therefore ends just before the first such entry.

    Args:
        queue (NDArray[np.int64]): Flat pixel indices of one generation, in
            queue order; a pixel may appear more than once.
        spread (_Spread): State updated in place.

    Returns:
        NDArray[np.int64]: The pixels taken over, in the order they were taken
            over, which is the next generation.
    """
    taken: list[NDArray[np.int64]] = []
    start = 0
    while start < queue.size:
        batch = queue[start : start + _BATCH_ENTRIES]
        target = (batch[:, None] + spread.offsets).ravel()
        offer_x = (spread.vector_x[batch][:, None] - _NEIGHBOUR_DX).ravel()
        offer_y = (spread.vector_y[batch][:, None] - _NEIGHBOUR_DY).ravel()
        distance = offer_x * offer_x + offer_y * offer_y
        accepted = _accepted_offers(target, distance, spread.reach)
        entry = accepted // _NEIGHBOUR_DX.size
        stop = _first_entry_changed_earlier(batch, entry, target[accepted])
        accepted = accepted[entry < stop]

        nearest = _last_offer_per_target(target[accepted], accepted)
        cells = target[nearest]
        spread.label[cells] = spread.label[batch[nearest // _NEIGHBOUR_DX.size]]
        spread.reach[cells] = distance[nearest]
        spread.vector_x[cells] = offer_x[nearest]
        spread.vector_y[cells] = offer_y[nearest]
        taken.append(target[accepted])
        start += stop
    return np.concatenate(taken) if taken else np.empty(0, dtype=np.int64)


def _accepted_offers(
    target: NDArray[np.int64],
    distance: NDArray[np.int32],
    reach: NDArray[np.int32],
) -> NDArray[np.int64]:
    """Offers nearer than what their target holds and than every earlier offer to it.

    Args:
        target (NDArray[np.int64]): Pixel each offer is for, in offer order.
        distance (NDArray[np.int32]): Squared distance of each offer.
        reach (NDArray[np.int32]): Squared distance each pixel holds.

    Returns:
        NDArray[np.int64]: Indices of the accepted offers, in offer order.
    """
    candidate = np.flatnonzero(distance < reach[target])
    if candidate.size == 0:
        return candidate
    by_target = candidate[np.argsort(target[candidate], kind="stable")]
    sorted_target = target[by_target]
    new_target = np.ones(by_target.size, dtype=bool)
    new_target[1:] = sorted_target[1:] != sorted_target[:-1]
    group = np.cumsum(new_target)
    # Lifting the distances of each target above those of every later target
    # makes a single running minimum start afresh at each target.
    lifted = distance[by_target] + (group[-1] - group) * (np.int64(1) << 32)
    earlier_best = np.empty_like(lifted)
    earlier_best[1:] = np.minimum.accumulate(lifted)[:-1]
    earlier_best[new_target] = np.iinfo(np.int64).max
    return np.sort(by_target[lifted < earlier_best])


def _first_entry_changed_earlier(
    batch: NDArray[np.int64],
    entry: NDArray[np.int64],
    changed: NDArray[np.int64],
) -> int:
    """Position of the first batch entry whose pixel an earlier entry changes.

    Args:
        batch (NDArray[np.int64]): Pixels of the batch entries, in queue order.
        entry (NDArray[np.int64]): Batch position of the entry behind each change.
        changed (NDArray[np.int64]): Pixel each change is to.

    Returns:
        int: The position, or ``batch.size`` when no entry is changed earlier.
    """
    if changed.size == 0:
        return batch.size
    size = np.int64(batch.size)
    # Keys sort by pixel, then by position; the next key after (pixel, entry)
    # is the first later entry of the same pixel, if there is one.
    keys = np.sort(batch * size + np.arange(batch.size))
    following = np.searchsorted(keys, changed * size + entry, side="right")
    inside = following < keys.size
    later = keys[following[inside]]
    same_pixel = later // size == changed[inside]
    if not same_pixel.any():
        return batch.size
    return int((later[same_pixel] % size).min())


def _last_offer_per_target(
    target: NDArray[np.int64], offer: NDArray[np.int64]
) -> NDArray[np.int64]:
    """The last accepted offer to each target, which is also its nearest.

    Args:
        target (NDArray[np.int64]): Target of each accepted offer, in offer order.
        offer (NDArray[np.int64]): Index of each accepted offer.

    Returns:
        NDArray[np.int64]: One offer index per target.
    """
    _, last_from_end = np.unique(target[::-1], return_index=True)
    return offer[::-1][last_from_end]


def _paint_pieces(
    labels: NDArray[np.int32], count: int, image: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """Fill every labelled piece with the rounded mean of ``image`` under it."""
    painted = np.full_like(image, _WHITE)
    inked = labels != 0
    flat_labels = labels[inked]
    sizes = np.bincount(flat_labels, minlength=count + 1)
    sizes = np.maximum(sizes, 1)
    if image.ndim == 2:
        sums = np.bincount(flat_labels, weights=image[inked], minlength=count + 1)
        painted[inked] = _rounded_mean(sums, sizes)[flat_labels]
        return painted
    for channel in range(min(image.shape[2], _COLOR_CHANNELS)):
        sums = np.bincount(
            flat_labels, weights=image[..., channel][inked], minlength=count + 1
        )
        painted[inked, channel] = _rounded_mean(sums, sizes)[flat_labels]
    return painted


def _rounded_mean(sums: NDArray, sizes: NDArray[np.int64]) -> NDArray[np.uint8]:
    """Mean per piece, halves rounded up; exact because the sums are integers."""
    # npd: ok one value per piece, not per pixel
    return ((2 * sums.astype(np.int64) + sizes) // (2 * sizes)).astype(np.uint8)
