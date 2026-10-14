"""Skew from the top page edge, for scans with a dark background around the page."""

import math

import numpy as np
from numpy.typing import NDArray

from scantailor.imageproc.analysis import (
    DEFAULT_MAX_ANGLE,
    DEFAULT_MIN_ANGLE,
    SkewResult,
)

_MIN_SIDE_PX = 10
_MAX_SEARCH_ROWS = 200
_SAMPLE_COLUMNS = 80
_WINDOW_ROWS = 3
# A window with this many ink pixels or fewer is light; any other is dark.
_LIGHT_WINDOW_MAX_INK = 1
_MIN_EDGE_POINTS = 10


def find_skew_from_top_edge(
    binary: NDArray[np.uint8],
    max_angle: float = DEFAULT_MAX_ANGLE,
    min_angle: float = DEFAULT_MIN_ANGLE,
) -> SkewResult:
    """Estimate the skew from where the dark surround ends at the top of the page.

    For sampled columns, the edge is the first row whose preceding three-row
    window holds at most one ink pixel, in a column whose first three rows
    hold more: the surround above the page must be ink, and it must reach the
    top of the image. A column that is light at the top has no edge; upstream
    takes its second row as the edge, so a page without a dark surround reads
    as straight with good confidence there, while here it has too few edge
    points. A line fitted through the edge points gives the angle. A found
    angle always has good confidence; a page with too few edge points gives
    none.

    Args:
        binary (numpy.ndarray of numpy.uint8): 2-D image with 0 for ink and 255 for
            background.
        max_angle (float): The fitted angle is clamped to this many degrees.
        min_angle (float): Angles smaller than this are reported as 0.

    Returns:
        SkewResult: Angle with the sign :func:`scantailor.imageproc.analysis.find_skew`
            uses, so the correction is its negation. That is the slope of the
            edge in degrees (y pointing down), which is the opposite of upstream's
            ``findSkewFromTopEdge``; upstream turns the page the wrong way.
    """
    height, width = binary.shape
    if width < _MIN_SIDE_PX or height < _MIN_SIDE_PX:
        return SkewResult(angle=0.0, confidence=0.0)

    search_rows = min(height // 3, _MAX_SEARCH_ROWS)
    step = max(1, width // _SAMPLE_COLUMNS)
    xs: list[float] = []
    ys: list[float] = []
    for column in range(step, width - step, step):
        edge_row = _edge_row(binary[:search_rows, column] == 0)
        if edge_row is not None:
            xs.append(float(column))
            ys.append(float(edge_row))
    if len(xs) < _MIN_EDGE_POINTS:
        return SkewResult(angle=0.0, confidence=0.0)

    x = np.asarray(xs)
    y = np.asarray(ys)
    count = float(len(xs))
    denominator = count * float(np.sum(x * x)) - float(np.sum(x)) ** 2
    if abs(denominator) < 1e-10:
        return SkewResult(angle=0.0, confidence=0.0)
    slope = (count * float(np.sum(x * y)) - float(np.sum(x)) * float(np.sum(y))) / (
        denominator
    )
    angle = math.degrees(math.atan(slope))
    angle = max(-max_angle, min(max_angle, angle))
    if abs(angle) < min_angle:
        return SkewResult(angle=0.0, confidence=SkewResult.GOOD_CONFIDENCE)
    return SkewResult(angle=angle, confidence=SkewResult.GOOD_CONFIDENCE)


def _edge_row(ink: NDArray[np.bool_]) -> int | None:
    """Return the last row of the first light window of a column that starts dark.

    Args:
        ink (numpy.ndarray of numpy.bool): One column of the search band, True for ink.

    Returns:
        int | None: The row, or None when the column is light at the top or
            stays dark to the end.
    """
    if len(ink) < _WINDOW_ROWS:
        return None
    counts = np.concatenate(([0], np.cumsum(ink, dtype=np.int64)))
    # Element i is the ink count of the window ending at row i + 2.
    window_ink = counts[_WINDOW_ROWS:] - counts[:-_WINDOW_ROWS]
    light = window_ink <= _LIGHT_WINDOW_MAX_INK
    if light[0]:
        return None
    light_windows = np.flatnonzero(light)
    if light_windows.size == 0:
        return None
    return int(light_windows[0] + _WINDOW_ROWS - 1)
