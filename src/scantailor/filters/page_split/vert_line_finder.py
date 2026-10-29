"""Find long thin dark vertical lines, such as a book's folding line.

The finder averages the page down to 100 dpi, with each working pixel covering
at least 0.42 mm of the page at any scan resolution. A gray-level detector then
keeps dark features that are thin horizontally but tall. A Hough transform over
a few degrees either side of vertical votes for lines, and lines whose
horizontal extents overlap are merged into one.

The detector responds where a column is darker than its neighbours a few pixels
away, which on a wide shadow, such as the one round a book's spine, is its
steepest flank rather than its darkest column. Each line is therefore moved
sideways down the brightness slope to the bottom of the valley it lies on,
when that valley is a clear dip with brighter page on both sides.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.imageproc import debug_drawing as draw
from scantailor.imageproc.area_scale import transform_scale_to_gray
from scantailor.imageproc.hough_line_detector import HoughLine, HoughLineDetector

Line = tuple[tuple[float, float], tuple[float, float]]

_WORKING_DPI = 100
_LINE_THICKNESS = 5.0
_MAX_ANGLE = 7.0
_ANGLE_STEP = 0.25
_MARGIN_MM = 3.5
_MM_PER_INCH = 25.4
# Smallest averaging box when reducing to 100 dpi: 5 pixels at 300 dpi. A box
# of 5 source pixels at any resolution would blur a 100 dpi scan over 5 working
# pixels, which joins letters stacked down a column across the gaps between
# lines into one long stroke.
_SMOOTHING_MM = 5.0 / 300.0 * _MM_PER_INCH
_WIDE_WINDOW = np.ones((1, 11), np.uint8)
_TALL_WINDOW = np.ones((11, 1), np.uint8)
_GAP_CLOSING_HEIGHT = 19
_MIN_QUALITY_FACTOR = 1.8
_MIN_WEIGHTED_LEVEL = 1
# Farthest the middle of a line moves while it descends into its valley, and
# again while it is centred on the valley's floor, so up to twice this in all;
# more than half the width of a dark spine shadow. Its ends may move up to
# twice as far again while the line turns.
_VALLEY_REACH_MM = 15.0
# Gray levels the column means must rise on both sides of a valley bottom for
# it to count as a dip; a flat scanner bed varies by well under one level.
_MIN_VALLEY_DEPTH = 8.0
# Column means within this many gray levels of the lowest form the bottom.
_VALLEY_FLOOR_TOLERANCE = 0.5
# Width in pixels of upstream's red lines in the debug image.
_VECTOR_LINE_WIDTH = 3.0


@dataclass(frozen=True)
class _QualityLine:
    """A candidate line with its top and bottom points ordered left to right."""

    left: tuple[float, float]
    right: tuple[float, float]
    quality: int

    @classmethod
    def of(
        cls, top: tuple[float, float], bottom: tuple[float, float], quality: int
    ) -> _QualityLine:
        if top[0] < bottom[0]:
            return cls(top, bottom, quality)
        return cls(bottom, top, quality)


class _LineGroup:
    """Candidate lines whose horizontal extents overlap; the best one leads."""

    def __init__(self, line: _QualityLine) -> None:
        self.leader = line
        self.left = line.left[0]
        self.right = line.right[0]

    def belongs_here(self, line: _QualityLine) -> bool:
        return not (self.left > line.right[0] or self.right < line.left[0])

    def add(self, line: _QualityLine) -> None:
        self.left = min(self.left, line.left[0])
        self.right = max(self.right, line.right[0])
        if line.quality > self.leader.quality:
            self.leader = line

    def merge(self, other: _LineGroup) -> None:
        self.left = min(self.left, other.left)
        self.right = max(self.right, other.right)
        if other.leader.quality > self.leader.quality:
            self.leader = other.leader


def find_lines(
    gray: NDArray[np.uint8],
    dpi: float,
    max_lines: int = 8,
    dbg: DebugImages | None = None,
) -> list[Line]:
    """Find nearly vertical dark lines in a page.

    Args:
        gray (numpy.ndarray of numpy.uint8): (H, W) gray page.
        dpi (float): Resolution of ``gray``.
        max_lines (int): How many lines to return at most; the rest are
            dropped in order of the quality of their best candidate.
        dbg (DebugImages | None): Receives the 100 dpi page, the strokes before
            and after short ones are removed, the Hough space and the lines
            found.

    Returns:
        list[tuple[tuple[float, float], tuple[float, float]]]:
            Lines as ``((x1, y1), (x2, y2))`` in ``gray`` pixels, each
        running from top to bottom of the page, the point with the smaller x first.
        Two lines that settle in the same valley are reported once.
    """
    gray100 = _reduce_to_working_dpi(gray, dpi)
    if dbg is not None:
        dbg.add(gray100, "gray100")
    raster_lines = _vertical_raster_lines(gray100, dbg)
    height, width = raster_lines.shape

    detector = HoughLineDetector(
        width,
        height,
        _LINE_THICKNESS,
        -_ANGLE_STEPS_TO_MAX * _ANGLE_STEP,
        _ANGLE_STEP,
        _TOTAL_ANGLE_STEPS,
    )
    margin = math.floor(0.5 + _MARGIN_MM * (1.0 / _MM_PER_INCH) * _WORKING_DPI)
    ys, xs = np.nonzero(raster_lines[:, margin : width - margin] > _MIN_WEIGHTED_LEVEL)
    xs = xs + margin
    weights = _WEIGHT_TABLE[raster_lines[ys, xs]]
    detector.process(xs.astype(np.int64), ys.astype(np.int64), weights)

    min_quality = int(height * _LINE_THICKNESS * _MIN_QUALITY_FACTOR) + 1
    if dbg is not None:
        dbg.add(detector.visualize_hough_space(min_quality), "hough_space")
    groups = _group_lines(detector.find_lines(min_quality), height)
    if dbg is not None:
        leaders = [(g.leader.left, g.leader.right) for g in groups[:max_lines]]
        canvas = draw.to_bgr(gray100)
        draw.draw_lines(canvas, leaders, draw.RED, _VECTOR_LINE_WIDTH)
        dbg.add(canvas, "vector_lines")

    scale = dpi / _WORKING_DPI
    reach = math.floor(0.5 + _VALLEY_REACH_MM / _MM_PER_INCH * _WORKING_DPI)
    lines: list[Line] = []
    centers: list[float] = []
    for group in groups[:max_lines]:
        (x1, y1), (x2, y2) = _to_valley_bottom(
            gray100, group.leader.left, group.leader.right, reach
        )
        center = 0.5 * (x1 + x2)
        if any(abs(center - other) < _LINE_THICKNESS for other in centers):
            continue
        centers.append(center)
        lines.append(((x1 * scale, y1 * scale), (x2 * scale, y2 * scale)))
    return lines


def _to_valley_bottom(
    gray100: NDArray[np.uint8],
    first: tuple[float, float],
    second: tuple[float, float],
    reach: int,
) -> Line:
    """Move a line sideways to the darkest part of the valley it lies on.

    Each end of the line moves a whole pixel at a time, alone or with the
    other, while that lowers the mean brightness along the line, so the line
    can turn to follow the valley. ``reach`` limits how far the middle of the
    line moves and how much its ends move apart; it does not limit each end,
    because the detector can find a line on the flank of a wide shadow at its
    steepest angle, and an end of it then lies more than ``reach`` from the
    bottom. The bottom of the valley is then the middle
    of the parallel shifts whose means lie within ``_VALLEY_FLOOR_TOLERANCE``
    of the lowest. A line on the edge of the scanner bed, where the dark side
    runs flat to the image border or past ``reach``, is not in a valley and
    stays where it is.

    Args:
        gray100 (numpy.ndarray of numpy.uint8): (H, W) page at 100 dpi.
        first (tuple[float, float]): One end of the line, in ``gray100`` pixels.
        second (tuple[float, float]): The other end, at a different y.
        reach (int): Farthest the middle of the line may move while it
            descends, and again while it is centred on the floor, in pixels.

    Returns:
        tuple[tuple[float, float], tuple[float, float]]: The moved line with
        its ends at the same y as before; the line unchanged when the valley
        is shallower than ``_MIN_VALLEY_DEPTH``.
    """
    rows = np.arange(gray100.shape[0])

    def mean_along(shift1: int, shift2: int) -> float:
        return _line_means(gray100, rows, first, second, np.array([[shift1, shift2]]))[
            0
        ]

    shift1, shift2 = _descend(mean_along, reach)
    parallel = np.arange(-reach, reach + 1)
    profile = _line_means(
        gray100,
        rows,
        first,
        second,
        np.stack([parallel + shift1, parallel + shift2], axis=1),
    )
    center = reach
    if not np.isfinite(profile[center]):
        return first, second
    floor = profile[center] + _VALLEY_FLOOR_TOLERANCE
    low, high = center, center
    while low > 0 and profile[low - 1] <= floor:
        low -= 1
    while high < len(profile) - 1 and profile[high + 1] <= floor:
        high += 1
    on_image = np.where(np.isfinite(profile), profile, -np.inf)
    rise_before = np.max(on_image[:low], initial=-np.inf) - profile[center]
    rise_after = np.max(on_image[high + 1 :], initial=-np.inf) - profile[center]
    if min(rise_before, rise_after) < _MIN_VALLEY_DEPTH:
        return first, second
    offset = 0.5 * (low + high) - center
    return (
        (first[0] + shift1 + offset, first[1]),
        (second[0] + shift2 + offset, second[1]),
    )


def _descend(mean_along: Callable[[int, int], float], reach: int) -> tuple[int, int]:
    """Shifts of the two line ends at the local minimum reached from no shift.

    Args:
        mean_along (Callable[[int, int], float]): Mean brightness of the line
            with its ends shifted by the given pixels; infinite off the image.
        reach (int): Largest shift of the line's middle, and half the largest
            difference between the shifts of its ends.

    Returns:
        tuple[int, int]: The shifts of the first and second end.
    """
    means: dict[tuple[int, int], float] = {}

    def mean(shifts: tuple[int, int]) -> float:
        if shifts not in means:
            means[shifts] = mean_along(*shifts)
        return means[shifts]

    current = (0, 0)
    while True:
        neighbours = [
            (current[0] + step1, current[1] + step2)
            for step1 in (-1, 0, 1)
            for step2 in (-1, 0, 1)
            if (step1, step2) != (0, 0)
            and abs(current[0] + step1 + current[1] + step2) <= 2 * reach
            and abs(current[0] + step1 - current[1] - step2) <= 2 * reach
        ]
        best = min(neighbours, key=mean)
        if mean(best) >= mean(current):
            return current
        current = best


def _line_means(
    gray100: NDArray[np.uint8],
    rows: NDArray[np.int64],
    first: tuple[float, float],
    second: tuple[float, float],
    shifts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Mean brightness along the line with its ends shifted sideways.

    Args:
        gray100 (numpy.ndarray of numpy.uint8): (H, W) page at 100 dpi.
        rows (numpy.ndarray of numpy.int64): ``arange(H)``.
        first (tuple[float, float]): One end of the line.
        second (tuple[float, float]): The other end, at a different y.
        shifts (numpy.ndarray of numpy.int64): (N, 2) shifts of the first and
            second end.

    Returns:
        numpy.ndarray of numpy.float64: N means, infinite for a shift that takes
        any part of the line outside the image.
    """
    width = gray100.shape[1]
    (x1, y1), (x2, y2) = first, second
    along = (rows + 0.5 - y1) / (y2 - y1)
    xs1 = x1 + shifts[:, :1]
    xs2 = x2 + shifts[:, 1:]
    columns = np.rint(xs1 + (xs2 - xs1) * along[None, :]).astype(np.int64)
    inside = (columns >= 0).all(axis=1) & (columns < width).all(axis=1)
    means = np.full(len(shifts), np.inf, dtype=np.float64)
    if inside.any():
        means[inside] = gray100[rows, columns[inside]].mean(axis=1, dtype=np.float64)
    return means


def _group_lines(hough_lines: list[HoughLine], height: int) -> list[_LineGroup]:
    groups: list[_LineGroup] = []
    for hough_line in hough_lines:
        candidate = _QualityLine.of(
            hough_line.point_at_y(0.0),
            hough_line.point_at_y(float(height)),
            hough_line.quality,
        )
        home: _LineGroup | None = None
        index = 0
        while index < len(groups):
            group = groups[index]
            if group.belongs_here(candidate):
                if home is not None:
                    home.merge(group)
                    del groups[index]
                    continue
                group.add(candidate)
                home = group
            index += 1
        if home is None:
            groups.append(_LineGroup(candidate))
    return groups


def _reduce_to_working_dpi(gray: NDArray[np.uint8], dpi: float) -> NDArray[np.uint8]:
    """Average the page down to 100 dpi, ignoring what lies outside it.

    The smallest averaging box is a fixed size on the page, not in source
    pixels, so scans of one page at different resolutions are smoothed alike.

    Args:
        gray (numpy.ndarray of numpy.uint8): (H, W) gray page.
        dpi (float): Resolution of ``gray``, the same on both axes.

    Returns:
        numpy.ndarray of numpy.uint8: The page at 100 dpi.
    """
    scale = _WORKING_DPI / dpi
    height, width = gray.shape
    size = (
        max(1, _round_half_up(width * scale)),
        max(1, _round_half_up(height * scale)),
    )
    min_window = _SMOOTHING_MM / _MM_PER_INCH * dpi
    return transform_scale_to_gray(gray, scale, size, min_window)


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _vertical_raster_lines(
    gray100: NDArray[np.uint8], dbg: DebugImages | None = None
) -> NDArray[np.uint8]:
    """Highlight dark strokes that are thin across and at least 19 pixels tall.

    Black is the foreground here, so OpenCV's dilation (a maximum) is the
    erosion of dark shapes. Outside the image counts as black so that nothing
    survives within 9 rows of the top or bottom edge.
    """
    wide = cv2.dilate(
        gray100,
        _WIDE_WINDOW,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    tall = cv2.dilate(
        gray100,
        _TALL_WINDOW,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    strokes = cv2.subtract(wide, tall)
    if dbg is not None:
        dbg.add(strokes, "vert_raster_lines")
    column = np.ones((_GAP_CLOSING_HEIGHT, 1), np.uint8)
    narrowed = cv2.erode(strokes, column, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    closed = cv2.dilate(narrowed, column, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if dbg is not None:
        dbg.add(closed, "short_segments_removed")
    return np.asarray(closed, dtype=np.uint8)


def _build_weight_table() -> NDArray[np.uint32]:
    """Map a stroke strength to a vote weight that grows as the bands double."""
    table = np.zeros(256, dtype=np.uint32)
    gray_level = 0
    weight = 2
    segment = 2
    previous_segment = 1
    while gray_level < 256:
        limit = min(256, gray_level + segment)
        table[gray_level:limit] = weight
        gray_level = limit
        weight += 1
        segment += previous_segment
        previous_segment = segment
    return table


_ANGLE_STEPS_TO_MAX = int(_MAX_ANGLE / _ANGLE_STEP)
_TOTAL_ANGLE_STEPS = _ANGLE_STEPS_TO_MAX * 2 + 1
_WEIGHT_TABLE = _build_weight_table()
