"""Page box detection: where the background around the paper ends.

The image is binarised at 150 dpi, so that a dark scanner bed or lid around the
paper turns black. Each side of the page box then moves inward from the image
border for as long as the lines it crosses are almost all black. An edge the
scanner trimmed off has no background to cross, so the page reaches the image
border there.

A light background does not turn black, so no side moves. The paper edges are
then looked for as long straight lines near the image border instead.
"""

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.imageproc.binarize import (
    binarize_mokji,
    binarize_otsu,
    binarize_sauvola,
    binarize_wolf,
    peak_threshold,
)
from scantailor.imageproc.filters import bgr_to_gray

from .content_box import PageBox
from .content_box_finder import Rect
from .content_finder import paint_outside_darkest

_DETECTION_DPI = 150.0
_MM_PER_INCH = 25.4
# A line counts as background when at least this share of it is black.
_BACKGROUND_SHARE = 0.95
# An edge stops after more than this many lines in a row that are not
# background, so that specks of paper-coloured dust on the bed are crossed.
_MAX_GAP = 10

# Upstream's debug labels of the binarisations ``_binarisations`` returns, in order.
_BINARISATION_LABELS = (
    "peakThreshold",
    "OtsuThreshold",
    "MokjiThreshold",
    "SauvolaThreshold",
    "WolfThreshold",
)

# The C++ defaults of these binarisations, which differ from the Python ones.
_SAUVOLA_K = 0.34
_WOLF_K = 0.3


def find_page_box(
    image: NDArray[np.uint8],
    dpi: float = 300.0,
    fine_tune: bool = False,
    expected_size_mm: tuple[float, float] = (0.0, 0.0),
    tolerance: float = 0.1,
    outside: NDArray[np.bool_] | None = None,
    dbg: DebugImages | None = None,
    frame_size: tuple[int, int] | None = None,
) -> PageBox | None:
    """Find the paper in a scan that shows background around it.

    Without an expected size one adaptive binarisation is used. With one, five
    binarisations are tried and, separately for the width and the height, the
    one whose result comes closest to the expected size is kept. When that
    leaves the whole image, as on a light background, the box is taken from
    long straight lines near the image border; fine tuning, the expected size
    and the tolerance do not apply to that box.

    Args:
        image (numpy.ndarray of numpy.uint8): Gray or BGR image, dark content
            on a light page. Not modified. It may be the frame shrunk ahead,
            ideally to 150 dpi, which spares resizing it.
        dpi (float): Resolution of the frame in DPI.
        fine_tune (bool): Also move each corner of the box diagonally inward
            while it lies on background, which trims the dark wedges left at
            the corners of a page lying slightly crooked.
        expected_size_mm (tuple[float, float]): Width and height of the paper
            in millimetres, or zeros when unknown.
        tolerance (float): Share by which fine tuning may make the box smaller
            than ``expected_size_mm``; unused without an expected size.
        outside (numpy.ndarray of numpy.bool | None): Pixels of ``image`` with
            no source pixel behind them, such as the corners exposed by a
            rotation. They count as background.
        dbg (DebugImages | None): Receives the 150 dpi gray page and every
            binarisation of it, including those an unknown size leaves unused.
        frame_size (tuple[int, int] | None): Width and height of the view the
            box is given in, which ``image`` shows whole; None when that is
            ``image`` itself.

    Returns:
        PageBox | None: The page in pixels of the frame, or None when no box
            was found, or none came near the expected size.
    """
    source = bgr_to_gray(image)
    gray = source
    if outside is not None and outside.any():
        if outside.all():
            return None
        gray = paint_outside_darkest(source, outside)
    h, w = gray.shape[:2]
    frame_w, frame_h = (w, h) if frame_size is None else frame_size
    scaled_w = max(1, round(frame_w * _DETECTION_DPI / dpi))
    scaled_h = max(1, round(frame_h * _DETECTION_DPI / dpi))
    interpolation = cv2.INTER_AREA if scaled_w < w else cv2.INTER_LINEAR
    scaled = (
        gray
        if gray.shape[:2] == (scaled_h, scaled_w)
        else np.asarray(
            cv2.resize(gray, (scaled_w, scaled_h), interpolation=interpolation),
            dtype=np.uint8,
        )
    )
    width_mm, height_mm = expected_size_mm
    sized = width_mm > 0 and height_mm > 0
    binarisations = _binarisations(scaled) if sized or dbg is not None else []
    if dbg is not None:
        dbg.add(scaled, "gray150")
        for label, black in zip(_BINARISATION_LABELS, binarisations, strict=True):
            dbg.add(black, label)

    if not sized:
        rect = _find_in(_sauvola(scaled), fine_tune)
    else:
        expected = (
            int(width_mm * _DETECTION_DPI / _MM_PER_INCH),
            int(height_mm * _DETECTION_DPI / _MM_PER_INCH),
        )
        rect = _closest_to_size(binarisations, expected, fine_tune, tolerance)
    if rect is None or rect.is_empty:
        return None
    if rect == Rect(0, 0, scaled_w - 1, scaled_h - 1):
        logger.debug("dark-bed finder found the whole image; line-edge fallback")
        lines = _find_line_edges(source, dpi * w / frame_w)
        if lines is None:
            return None
        return PageBox(
            x=lines.x * frame_w / w,
            y=lines.y * frame_h / h,
            width=lines.width * frame_w / w,
            height=lines.height * frame_h / h,
        )

    x_scale = frame_w / scaled_w
    y_scale = frame_h / scaled_h
    left = rect.left * x_scale
    top = rect.top * y_scale
    right = min(float(frame_w), (rect.right + 1) * x_scale)
    bottom = min(float(frame_h), (rect.bottom + 1) * y_scale)
    return PageBox(x=left, y=top, width=right - left, height=bottom - top)


def _sauvola(gray: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Black pixels by Sauvola's threshold over a window the size of the image."""
    height, width = gray.shape
    return binarize_sauvola(gray, (width, height), k=_SAUVOLA_K) == 0


def _binarisations(gray: NDArray[np.uint8]) -> list[NDArray[np.bool_]]:
    """Black pixels of ``gray`` by each method tried against an expected size."""
    height, width = gray.shape
    return [
        gray < peak_threshold(gray),
        binarize_otsu(gray) == 0,
        binarize_mokji(gray) == 0,
        _sauvola(gray),
        binarize_wolf(gray, (width, height), k=_WOLF_K) == 0,
    ]


def _find_in(
    background: NDArray[np.bool_],
    fine_tune: bool,
    expected: tuple[int, int] | None = None,
    tolerance: float = 1.0,
) -> Rect:
    """Detect the borders in one binarisation and optionally fine tune them."""
    rect = detect_borders(background)
    if fine_tune:
        rect = fine_tune_corners(background, rect, expected, tolerance)
    return rect


def _closest_to_size(
    binarisations: list[NDArray[np.bool_]],
    expected: tuple[int, int],
    fine_tune: bool,
    tolerance: float,
) -> Rect | None:
    """Take left and right, then top and bottom, from the closest binarisation.

    A side pair is only taken from a result that is off by less than the whole
    expected length; None when no binarisation came that close for either.
    """
    expected_w, expected_h = expected
    best_columns: tuple[float, int, int] | None = None
    best_rows: tuple[float, int, int] | None = None
    for background in binarisations:
        rect = _find_in(background, fine_tune, expected, tolerance)
        width_error = abs(expected_w - rect.width) / max(expected_w, 1)
        height_error = abs(expected_h - rect.height) / max(expected_h, 1)
        if width_error < (1.0 if best_columns is None else best_columns[0]):
            best_columns = (width_error, rect.left, rect.right)
        if height_error < (1.0 if best_rows is None else best_rows[0]):
            best_rows = (height_error, rect.top, rect.bottom)
    if best_columns is None or best_rows is None:
        return None
    return Rect(best_columns[1], best_rows[1], best_columns[2], best_rows[2])


def detect_borders(background: NDArray[np.bool_]) -> Rect:
    """Move each side of the image inward across background lines.

    A side stops on the last background line before more than ``_MAX_GAP``
    lines that are not. The bottom side stops at the top one at the latest.

    Args:
        background (numpy.ndarray of numpy.bool): True for black pixels.

    Returns:
        Rect: The page, inclusive of the last background line on each side;
            the whole image when no side starts on background.
    """
    h, w = background.shape
    # Like C++, a line is judged over its first 2 * ((length - 1) // 2) pixels.
    column_span = 2 * ((h - 1) // 2)
    row_span = 2 * ((w - 1) // 2)
    background_columns = background[:column_span].sum(axis=0) >= int(
        column_span * _BACKGROUND_SHARE
    )
    background_rows = background[:, :row_span].sum(axis=1) >= int(
        row_span * _BACKGROUND_SHARE
    )
    left = _last_background_line(background_columns, 0, w - 1, 1)
    top = _last_background_line(background_rows, 0, h - 1, 1)
    right = _last_background_line(background_columns, w - 1, 0, -1)
    bottom = _last_background_line(background_rows, h - 1, top, -1)
    return Rect(left, top, right, bottom)


def _last_background_line(
    is_background: NDArray[np.bool_], start: int, stop: int, step: int
) -> int:
    """Walk from ``start`` towards ``stop`` (excluded) and return the edge."""
    edge = start
    gap = 0
    for line in range(start, stop, step):
        if is_background[line]:
            edge = line
            gap = 0
        else:
            gap += 1
            if gap > _MAX_GAP:
                break
    return edge


# Corners as (index of x, index of y, step of x, step of y) into
# [left, top, right, bottom]; the opposite corner is at (2 - x, 4 - y).
_CORNERS = ((0, 1, 1, 1), (2, 1, -1, 1), (0, 3, 1, -1), (2, 3, -1, -1))


def fine_tune_corners(
    background: NDArray[np.bool_],
    rect: Rect,
    expected: tuple[int, int] | None,
    tolerance: float,
) -> Rect:
    """Move each corner diagonally inward, one pixel a round, while on background.

    Moving a corner moves the two sides that meet there, so a side is pushed
    as far as the more crooked of its two corners needs.

    Args:
        background (numpy.ndarray of numpy.bool): True for black pixels.
        rect (~scantailor.filters.select_content.content_box_finder.Rect): Box
            found by ``detect_borders``.
        expected (tuple[int, int] | None): Expected page width and height in
            pixels of ``background``, or None when unknown.
        tolerance (float): A corner stops once the box would be narrower or
            shorter than the expected size by more than this share.

    Returns:
        Rect: The tuned box; empty when the corners crossed.
    """
    h, w = background.shape
    sides = [rect.left, rect.top, rect.right, rect.bottom]
    minimum = (
        None
        if expected is None or expected[0] <= 0 or expected[1] <= 0
        else (int(expected[0] * (1 - tolerance)), int(expected[1] * (1 - tolerance)))
    )
    moved = True
    while moved:
        moved = False
        for x_side, y_side, x_step, y_step in _CORNERS:
            x, y = sides[x_side], sides[y_side]
            if minimum is not None and (
                abs(sides[2 - x_side] - x) < minimum[0]
                or abs(sides[4 - y_side] - y) < minimum[1]
            ):
                continue
            next_x, next_y = x + x_step, y + y_step
            if background[y, x] and 0 <= next_x < w and 0 <= next_y < h:
                sides[x_side], sides[y_side] = next_x, next_y
                moved = True
    return Rect(*sides)


def _find_line_edges(gray: NDArray[np.uint8], dpi: float) -> PageBox | None:
    """Find the paper edges as long straight lines near the image border.

    Args:
        gray (numpy.ndarray of numpy.uint8): 2-D gray image. Not modified.
        dpi (float): Image resolution in DPI.

    Returns:
        PageBox | None: The page in pixels of ``gray``, or None when lines were
            not found on both sides along both axes.
    """
    h, w = gray.shape[:2]
    scale = _DETECTION_DPI / dpi
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if scale < 1.0 and size != (w, h):
        scaled = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    else:
        scaled = gray
        scale = 1.0

    edges = cv2.Canny(scaled, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    shortest = min(scaled.shape)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(shortest * 0.3),
        minLineLength=int(shortest * 0.2),
        maxLineGap=int(shortest * 0.05),
    )
    if lines is None or len(lines) < 4:
        return None

    sh, sw = scaled.shape[:2]
    # Only lines within this distance of the image border can be paper edges.
    reach = 0.1 * min(sw, sh)
    left, right, top, bottom = sw, 0, sh, 0
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x2 - x1) < abs(y2 - y1) * 0.2:
            x = (x1 + x2) / 2
            if x < reach:
                left = min(left, int(x))
            elif x > sw - reach:
                right = max(right, int(x))
        if abs(y2 - y1) < abs(x2 - x1) * 0.2:
            y = (y1 + y2) / 2
            if y < reach:
                top = min(top, int(y))
            elif y > sh - reach:
                bottom = max(bottom, int(y))
    if left >= right or top >= bottom:
        return None

    return PageBox(
        x=left / scale,
        y=top / scale,
        width=(right - left) / scale,
        height=(bottom - top) / scale,
    )
