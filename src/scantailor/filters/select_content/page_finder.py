"""Page box detection: where the sheet of paper lies in the scan.

The page is found as the white area left once the dark scanner background is
peeled off each side. The image is reduced to 150 dpi and binarised. From each
edge, lines are skipped while at least 95 % of their middle stretch is black;
the page starts at the last such line, once more than 10 lines in a row fall
short. An edge that the scanner trimmed off has no background to skip, so the
page simply reaches the image border there.

With an expected page size, five binarisations are tried and, separately for
the width and the height, the one whose result is closest to the expected size
wins; otherwise Sauvola's alone is used.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.imageproc.binarize import (
    binarize_mokji,
    binarize_otsu,
    binarize_sauvola,
    binarize_wolf,
    peak_threshold,
)

from .content_box import ContentBox
from .content_finder import paint_outside_darkest

_PROCESSING_DPI = 150.0
_MM_PER_INCH = 25.4
# Share of a line's middle stretch that must be black for it to be background.
_BACKGROUND_FRACTION = 0.95
# Lines short of background in a row that end the background.
_MAX_EDGE_GAP = 10
_SAUVOLA_K = 0.34
_WOLF_K = 0.3


@dataclass(frozen=True)
class _Rect:
    """Inclusive pixel bounds at 150 dpi."""

    left: int
    top: int
    right: int
    bottom: int


def find_page_box(
    image: NDArray[np.uint8],
    dpi: float,
    expected_size_mm: tuple[float, float] = (0.0, 0.0),
    tolerance: float = 0.1,
    fine_tune: bool = False,
    outside: NDArray[np.bool_] | None = None,
) -> ContentBox | None:
    """Find the page in a scan that shows the scanner background around it.

    Args:
        image (numpy.ndarray of numpy.uint8): Gray or BGR image, dark text on
            a light page. Not modified.
        dpi (float): Resolution of ``image``.
        expected_size_mm (tuple[float, float]): Width and height the page is
            known to have, in millimetres; ``(0, 0)`` when unknown.
        tolerance (float): How far from the expected size, as a fraction of
            it, fine tuning may shrink the box. Only used with an expected size.
        fine_tune (bool): Move each corner inwards along the diagonal while it
            is still on background, for a page that is slightly rotated.
        outside (numpy.ndarray of numpy.bool | None): Pixels with no source
            pixel behind them, such as corners exposed by a rotation; they
            count as background.

    Returns:
        ContentBox | None: The page in pixels of ``image``; None when no
        binarisation gave a box of the expected size.
    """
    gray = np.asarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image,
        dtype=np.uint8,
    )
    if outside is not None and outside.any() and not outside.all():
        gray = paint_outside_darkest(gray, outside)
    height, width = gray.shape
    scaled_width = max(1, round(width * _PROCESSING_DPI / dpi))
    scaled_height = max(1, round(height * _PROCESSING_DPI / dpi))
    interpolation = cv2.INTER_AREA if scaled_width < width else cv2.INTER_LINEAR
    gray150 = np.asarray(
        cv2.resize(gray, (scaled_width, scaled_height), interpolation=interpolation),
        dtype=np.uint8,
    )

    expected_width = int(_PROCESSING_DPI / _MM_PER_INCH * expected_size_mm[0])
    expected_height = int(_PROCESSING_DPI / _MM_PER_INCH * expected_size_mm[1])
    if expected_width <= 0 or expected_height <= 0:
        black = _sauvola(gray150)
        rect = _detect_borders(black)
        if fine_tune:
            rect = _fine_tune_corners(black, rect, (0, 0), 1.0)
    else:
        rect = _closest_to_size(
            gray150, (expected_width, expected_height), tolerance, fine_tune
        )
        if rect is None:
            return None

    x_scale = width / scaled_width
    y_scale = height / scaled_height
    return ContentBox(
        x=rect.left * x_scale,
        y=rect.top * y_scale,
        width=(rect.right - rect.left + 1) * x_scale,
        height=(rect.bottom - rect.top + 1) * y_scale,
    )


def _sauvola(gray: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Black pixels by Sauvola's threshold over a window the size of the image."""
    height, width = gray.shape
    return binarize_sauvola(gray, (width, height), k=_SAUVOLA_K) == 0


def _binarisations(gray: NDArray[np.uint8]) -> list[NDArray[np.bool_]]:
    """Black pixels of ``gray`` by each method tried against an expected size."""
    height, width = gray.shape
    window = (width, height)
    return [
        gray < peak_threshold(gray),
        binarize_otsu(gray) == 0,
        binarize_mokji(gray) == 0,
        _sauvola(gray),
        binarize_wolf(gray, window, k=_WOLF_K) == 0,
    ]


def _closest_to_size(
    gray: NDArray[np.uint8],
    expected: tuple[int, int],
    tolerance: float,
    fine_tune: bool,
) -> _Rect | None:
    """Take the width and the height each from the binarisation nearest to them."""
    expected_width, expected_height = expected
    best_width_error = best_height_error = 1.0
    horizontal: tuple[int, int] | None = None
    vertical: tuple[int, int] | None = None
    for black in _binarisations(gray):
        rect = _detect_borders(black)
        if fine_tune:
            rect = _fine_tune_corners(black, rect, expected, tolerance)
        width_error = (
            abs(expected_width - (rect.right - rect.left + 1)) / expected_width
        )
        height_error = (
            abs(expected_height - (rect.bottom - rect.top + 1)) / expected_height
        )
        if width_error < best_width_error:
            horizontal, best_width_error = (rect.left, rect.right), width_error
        if height_error < best_height_error:
            vertical, best_height_error = (rect.top, rect.bottom), height_error
    if horizontal is None or vertical is None:
        return None
    return _Rect(horizontal[0], vertical[0], horizontal[1], vertical[1])


def _detect_borders(black: NDArray[np.bool_]) -> _Rect:
    """Skip the background from the left, top, right and bottom in that order.

    Only the middle stretch of each line is looked at: twice the distance from
    its start to the centre of the image, which leaves out the far end.
    """
    height, width = black.shape
    rows_used = 2 * ((height - 1) // 2)
    columns_used = 2 * ((width - 1) // 2)
    column_black = black[:rows_used].sum(axis=0)
    row_black = black[:, :columns_used].sum(axis=1)
    column_minimum = int(rows_used * _BACKGROUND_FRACTION)
    row_minimum = int(columns_used * _BACKGROUND_FRACTION)

    left = _skip_background(column_black, column_minimum, 0, width - 1, 1)
    top = _skip_background(row_black, row_minimum, 0, height - 1, 1)
    right = _skip_background(column_black, column_minimum, width - 1, 0, -1)
    bottom = _skip_background(row_black, row_minimum, height - 1, top, -1)
    return _Rect(left, top, right, bottom)


def _skip_background(
    black_counts: NDArray[np.int64], minimum: int, start: int, end: int, step: int
) -> int:
    """Index of the last background line met walking from ``start`` towards ``end``.

    Args:
        black_counts (numpy.ndarray of numpy.int64): Black pixels per line.
        minimum (int): Black pixels that make a line background.
        start (int): First line looked at.
        end (int): Line where the walk stops, not looked at.
        step (int): 1 or -1.

    Returns:
        int: The last background line before the gap, or ``start`` if none.
    """
    edge = start
    gap = 0
    for index in range(start, end, step):
        if black_counts[index] < minimum:
            gap += 1
            if gap > _MAX_EDGE_GAP:
                break
        else:
            gap = 0
            edge = index
    return edge


def _fine_tune_corners(
    black: NDArray[np.bool_],
    rect: _Rect,
    expected: tuple[int, int],
    tolerance: float,
) -> _Rect:
    """Move the corners diagonally inwards while they sit on background.

    The four corners take one step each per round; a corner stops when it
    leaves the background, reaches the image border or, with an expected size,
    would make the box smaller than that size less the tolerance.
    """
    left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
    while True:
        moved = False
        step, left, top = _step_corner(
            black, left, top, right, bottom, 1, 1, expected, tolerance
        )
        moved |= step
        step, right, top = _step_corner(
            black, right, top, left, bottom, -1, 1, expected, tolerance
        )
        moved |= step
        step, left, bottom = _step_corner(
            black, left, bottom, right, top, 1, -1, expected, tolerance
        )
        moved |= step
        step, right, bottom = _step_corner(
            black, right, bottom, left, top, -1, -1, expected, tolerance
        )
        if not (moved or step):
            return _Rect(left, top, right, bottom)


def _step_corner(
    black: NDArray[np.bool_],
    x: int,
    y: int,
    opposite_x: int,
    opposite_y: int,
    step_x: int,
    step_y: int,
    expected: tuple[int, int],
    tolerance: float,
) -> tuple[bool, int, int]:
    """Move one corner a step along the diagonal if it may move.

    Returns:
        tuple[bool, int, int]: Whether it moved, and its new position.
    """
    height, width = black.shape
    expected_width, expected_height = expected
    if expected_width > 0 and expected_height > 0:
        smallest_width = int(expected_width * (1.0 - tolerance))
        smallest_height = int(expected_height * (1.0 - tolerance))
        if (
            abs(opposite_x - x) < smallest_width
            or abs(opposite_y - y) < smallest_height
        ):
            return False, x, y
    new_x, new_y = x + step_x, y + step_y
    if not black[y, x] or not (0 <= new_x < width and 0 <= new_y < height):
        return False, x, y
    return True, new_x, new_y
