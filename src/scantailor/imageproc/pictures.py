"""Automatic detection of picture regions on a page."""

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages

from .filters import stretch_gray_range
from .reconstruction import reconstruction

# Resolution the detector is tuned for.
DETECTION_DPI = 300
# Tails of the gray range clipped before detection so that text and paper are
# equally far from mid-gray; otherwise large type would read as a picture.
_RANGE_CLIP = 0.01
_GRADIENT_WINDOW = 3
# Window over which texture must persist; text strokes are narrower, so only
# wide textured areas survive the reconstruction.
_MARKER_WINDOW = 35
_HIGHER_SENSITIVITY_BLACK_CLIP = 5.0
# Gray level at which the detected map turns into a picture mask.
PICTURE_THRESHOLD = 48
# Distance in pixels at 300 dpi under which neighbouring picture runs are one picture.
_JOIN_DISTANCE = 16


def detect_pictures(
    gray: NDArray[np.uint8],
    *,
    higher_search_sensitivity: bool = False,
    dbg: DebugImages | None = None,
) -> NDArray[np.uint8]:
    """Map how picture-like each pixel of a 300 dpi gray page is.

    Pictures show as regions with strong local gray-level variation that is
    wider than a text stroke; the map is light there and dark elsewhere.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page at about 300 dpi. Not modified.
        higher_search_sensitivity (bool): Also stretch weak responses so that
            faint pictures reach the picture threshold.
        dbg (DebugImages | None): Receives each step. The gradient, marker and
            reconstruction are dark where upstream's are light, because this
            implementation reconstructs by erosion where upstream seed-fills.

    Returns:
        NDArray[np.uint8]: Map of the same shape; values from ``PICTURE_THRESHOLD``
            up are picture.
    """
    stretched = stretch_gray_range(gray, _RANGE_CLIP, _RANGE_CLIP)
    window = np.ones((_GRADIENT_WINDOW, _GRADIENT_WINDOW), dtype=np.uint8)
    lightest = cv2.dilate(
        stretched, window, borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    darkest = cv2.erode(
        stretched, window, borderType=cv2.BORDER_CONSTANT, borderValue=255
    )
    # Dark where a neighbourhood holds both very dark and very light pixels.
    gradient = (
        255 - (255 - darkest.astype(np.uint32)) * lightest.astype(np.uint32) // 255
    ).astype(np.uint8)
    if dbg is not None:
        dbg.add(stretched, "stretched")
        dbg.add(darkest, "eroded")
        dbg.add(lightest, "dilated")
        dbg.add(gradient, "grayGradient")

    marker = np.asarray(
        cv2.dilate(
            gradient,
            np.ones((_MARKER_WINDOW, _MARKER_WINDOW), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
        dtype=np.uint8,
    )
    reconstructed = _reconstruct_by_erosion(marker, gradient)
    picture_like = 255 - reconstructed
    if dbg is not None:
        dbg.add(marker, "marker")
        dbg.add(reconstructed, "reconstructed")
        dbg.add(picture_like, "reconstructed_inverted")

    frame = np.full_like(picture_like, 255)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = 0
    holes_filled = _reconstruct_by_erosion(frame, picture_like)
    if dbg is not None:
        dbg.add(holes_filled, "holesFilled")

    if higher_search_sensitivity:
        stretched2 = stretch_gray_range(
            holes_filled, _HIGHER_SENSITIVITY_BLACK_CLIP, _RANGE_CLIP
        )
        if dbg is not None:
            dbg.add(stretched2, "stretched2")
        return stretched2
    return holes_filled


def _reconstruct_by_erosion(
    seed: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Spread dark values from the seed as far as the mask lets them.

    The seed is raised to the mask first; the result is nowhere darker than the
    mask and is lighter wherever no dark path leads to a dark seed pixel
    (8-connectivity).
    """
    return reconstruction(np.maximum(seed, mask), mask, "erosion", 8)


def estimate_picture_mask(
    gray: NDArray[np.uint8],
    dpi: tuple[int, int],
    *,
    higher_search_sensitivity: bool = False,
    dbg: DebugImages | None = None,
) -> NDArray[np.bool_]:
    """Find picture regions of a gray page of any resolution.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page. Not modified.
        dpi (tuple[int, int]): Horizontal and vertical resolution of ``gray``.
        higher_search_sensitivity (bool): See ``detect_pictures``.
        dbg (DebugImages | None): Receives the steps of ``detect_pictures``.

    Returns:
        NDArray[np.bool_]: True where there is a picture, same shape as ``gray``.
    """
    height, width = gray.shape
    small_size = (
        max(1, int(width * DETECTION_DPI / dpi[0] + 0.5)),
        max(1, int(height * DETECTION_DPI / dpi[1] + 0.5)),
    )
    shrinking = small_size[0] * small_size[1] < width * height
    small = np.asarray(
        cv2.resize(
            gray,
            small_size,
            interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
        ),
        dtype=np.uint8,
    )
    detected = detect_pictures(
        small, higher_search_sensitivity=higher_search_sensitivity, dbg=dbg
    )
    full = cv2.resize(detected, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(full) >= PICTURE_THRESHOLD


def find_picture_rectangles(
    picture: NDArray[np.bool_],
    sensitivity: int = 100,
) -> list[tuple[int, int, int, int]]:
    """Bounding rectangles of picture clusters.

    Picture pixels closer than about 16 pixels join into one cluster. With a
    sensitivity below 100, each rectangle is then trimmed on every side to the
    first line where more than that share of its pixels is picture.

    Args:
        picture (NDArray[np.bool_]): True where there is a picture.
        sensitivity (int): Percentage from 0 to 100 of picture pixels a border
            line must contain to stay inside its rectangle.

    Returns:
        list[tuple[int, int, int, int]]: Rectangles as (left, top, right, bottom),
            all inclusive pixel coordinates.
    """
    if not picture.any():
        return []
    size = 2 * _JOIN_DISTANCE + 1
    joined = cv2.dilate(picture.astype(np.uint8), np.ones((size, size), dtype=np.uint8))
    count, cluster_map, stats, _ = cv2.connectedComponentsWithStats(
        joined, connectivity=8, ltype=cv2.CV_32S
    )
    clusters = np.asarray(cluster_map, dtype=np.int32)

    rectangles: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        rect = _picture_box(picture, clusters, label, stats[label])
        rectangles.append(
            _trim_rectangle(picture, rect, sensitivity / 100.0)
            if sensitivity < 100
            else rect
        )
    return rectangles


def _picture_box(
    picture: NDArray[np.bool_],
    clusters: NDArray[np.int32],
    label: int,
    cluster_stats: NDArray[np.int32],
) -> tuple[int, int, int, int]:
    """Bounding box (left, top, right, bottom) of the picture pixels of one cluster.

    The cluster grew out of its picture pixels, so they lie inside its box and
    only that box is searched.
    """
    left = int(cluster_stats[cv2.CC_STAT_LEFT])
    top = int(cluster_stats[cv2.CC_STAT_TOP])
    width = int(cluster_stats[cv2.CC_STAT_WIDTH])
    height = int(cluster_stats[cv2.CC_STAT_HEIGHT])
    box = (slice(top, top + height), slice(left, left + width))
    own = picture[box] & (clusters[box] == label)
    rows = np.flatnonzero(own.any(axis=1))
    cols = np.flatnonzero(own.any(axis=0))
    return (
        left + int(cols[0]),
        top + int(rows[0]),
        left + int(cols[-1]),
        top + int(rows[-1]),
    )


def _trim_rectangle(
    picture: NDArray[np.bool_],
    rect: tuple[int, int, int, int],
    share: float,
) -> tuple[int, int, int, int]:
    """Shrink a rectangle to the outermost lines that are mostly picture."""
    left, top, right, bottom = rect
    block = picture[top : bottom + 1, left : right + 1]
    rows = block.mean(axis=1) > share
    cols = block.mean(axis=0) > share
    if not rows.any() or not cols.any():
        return rect
    return (
        left + int(np.argmax(cols)),
        top + int(np.argmax(rows)),
        left + len(cols) - 1 - int(np.argmax(cols[::-1])),
        top + len(rows) - 1 - int(np.argmax(rows[::-1])),
    )
