"""Find long thin dark vertical lines, such as a book's folding line.

The page is reduced to 100 dpi and a gray-level detector keeps dark features
that are thin horizontally but tall. A Hough transform over a few degrees
either side of vertical then votes for lines, and lines whose horizontal extents
overlap are merged into one.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.imageproc.area_scale import transform_scale_to_gray
from scantailor.imageproc.hough_line_detector import HoughLine, HoughLineDetector

Line = tuple[tuple[float, float], tuple[float, float]]

_WORKING_DPI = 100
_LINE_THICKNESS = 5.0
_MAX_ANGLE = 7.0
_ANGLE_STEP = 0.25
_MARGIN_MM = 3.5
_MM_PER_INCH = 25.4
_WIDE_WINDOW = np.ones((1, 11), np.uint8)
_TALL_WINDOW = np.ones((11, 1), np.uint8)
_GAP_CLOSING_HEIGHT = 19
_MIN_QUALITY_FACTOR = 1.8
_MIN_WEIGHTED_LEVEL = 1


@dataclass(frozen=True)
class _QualityLine:
    """A candidate line with its top and bottom points ordered left to right."""

    left: tuple[float, float]
    right: tuple[float, float]
    quality: int

    @classmethod
    def of(
        cls, top: tuple[float, float], bottom: tuple[float, float], quality: int
    ) -> "_QualityLine":
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

    def merge(self, other: "_LineGroup") -> None:
        self.left = min(self.left, other.left)
        self.right = max(self.right, other.right)
        if other.leader.quality > self.leader.quality:
            self.leader = other.leader


def find_lines(gray: NDArray[np.uint8], dpi: float, max_lines: int = 8) -> list[Line]:
    """Find nearly vertical dark lines in a page.

    Args:
        gray (numpy.ndarray of numpy.uint8): (H, W) gray page.
        dpi (float): Resolution of ``gray``.
        max_lines (int): How many lines to return at most; the rest are
            dropped in order of the quality of their best candidate.

    Returns:
        list[tuple[tuple[float, float], tuple[float, float]]]:
            Lines as ``((x1, y1), (x2, y2))`` in ``gray`` pixels, each
        running from top to bottom of the page, the point with the smaller x first.
    """
    gray100 = _reduce_to_working_dpi(gray, dpi)
    raster_lines = _vertical_raster_lines(gray100)
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
    groups = _group_lines(detector.find_lines(min_quality), height)

    scale = dpi / _WORKING_DPI
    lines: list[Line] = []
    for group in groups[:max_lines]:
        left, right = group.leader.left, group.leader.right
        lines.append(
            (
                (left[0] * scale, left[1] * scale),
                (right[0] * scale, right[1] * scale),
            )
        )
    return lines


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
    """Average the page down to 100 dpi, ignoring what lies outside it."""
    scale = _WORKING_DPI / dpi
    height, width = gray.shape
    size = (
        max(1, _round_half_up(width * scale)),
        max(1, _round_half_up(height * scale)),
    )
    return transform_scale_to_gray(gray, scale, size)


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _vertical_raster_lines(gray100: NDArray[np.uint8]) -> NDArray[np.uint8]:
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
    column = np.ones((_GAP_CLOSING_HEIGHT, 1), np.uint8)
    narrowed = cv2.erode(strokes, column, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    closed = cv2.dilate(narrowed, column, borderType=cv2.BORDER_CONSTANT, borderValue=0)
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
