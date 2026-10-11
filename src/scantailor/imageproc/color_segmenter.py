"""Colour segmentation of binarised ink.

The ink of a bilevel page is split into a few colour classes using the original
page, and every connected piece of ink is painted with the average colour of
the original under it. The paper stays white.
"""

import cv2
import numpy as np
from numba import njit
from numpy.typing import NDArray

from .background_color import otsu_threshold

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
# Neighbour offsets (dy, dx) in the order ties between equidistant labels are resolved.
_NEIGHBOURS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)],
    dtype=np.int64,
)


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
    histogram = np.bincount(channel.ravel(), minlength=_LEVELS).astype(np.int64)
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


@njit(cache=True)
def _propagate(labels, reach, vector_x, vector_y, frontier, neighbours):
    """Spread labels breadth first through unreached cells, nearest source wins.

    ``reach`` holds the squared distance to the source and is ``_UNREACHED`` in
    cells that may still be taken; cells holding 0 and no label never change.
    Distances come from per-cell vectors to the source, so they are exact
    Euclidean distances to the closest source pixel found along the way.
    """
    height, width = labels.shape
    queue = frontier
    count = frontier.shape[0]
    head = 0
    while head < count:
        flat = queue[head]
        head += 1
        y = flat // width
        x = flat - y * width
        vx = vector_x[y, x]
        vy = vector_y[y, x]
        label = labels[y, x]
        for k in range(neighbours.shape[0]):
            dy = neighbours[k, 0]
            dx = neighbours[k, 1]
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            new_x = vx - dx
            new_y = vy - dy
            distance = new_x * new_x + new_y * new_y
            if distance < reach[ny, nx]:
                labels[ny, nx] = label
                reach[ny, nx] = distance
                vector_x[ny, nx] = new_x
                vector_y[ny, nx] = new_y
                if count == queue.shape[0]:
                    grown = np.empty(queue.shape[0] * 2, dtype=queue.dtype)
                    grown[:count] = queue[:count]
                    queue = grown
                queue[count] = ny * width + nx
                count += 1


def _extend_into_ink(
    labels: NDArray[np.int32], ink: NDArray[np.bool_]
) -> NDArray[np.int32]:
    """Give unlabelled ink the label of the nearest labelled pixel it connects to.

    Labels spread only through ink, so ink that no piece reaches stays 0.

    Args:
        labels (NDArray[np.int32]): Piece labels, 0 for none. Not modified.
        ink (NDArray[np.bool_]): Where labels may spread.

    Returns:
        NDArray[np.int32]: New label array.
    """
    sources = labels != 0
    if not sources.any():
        return labels.copy()
    extended = labels.copy()
    reach = np.where(ink & ~sources, _UNREACHED, 0).astype(np.int32)
    _propagate(
        extended,
        reach,
        np.zeros(labels.shape, dtype=np.int32),
        np.zeros(labels.shape, dtype=np.int32),
        np.flatnonzero(sources).astype(np.int32),
        _NEIGHBOURS,
    )
    return extended


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
    return ((2 * sums.astype(np.int64) + sizes) // (2 * sizes)).astype(np.uint8)
