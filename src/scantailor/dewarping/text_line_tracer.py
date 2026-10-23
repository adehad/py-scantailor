"""Trace the text lines of a page, as the output stage's automatic dewarping does.

The page is reduced to 200 dpi, binarised and cleaned to find the left and right
content bounds. A ridge map built from the second derivative of the blurred
image along the text columns then marks the lines. Seeds are taken where the
middle line between the bounds crosses a ridge, and each seed is followed
towards both bounds along the ridge. The traced lines are refined with snakes.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.models import Dpi
from scantailor.dewarping.grid_filters import (
    directional_gradient,
    gauss_blur,
    sobel_xy,
)
from scantailor.dewarping.line_geometry import (
    Line,
    Point,
    grid_line_cells,
    line_bounded_by_rect,
    line_intersection,
    lines_within_image,
    normal_vector,
    projection_dist,
    projection_point,
    q_round,
)
from scantailor.dewarping.text_line_refiner import refine_text_lines
from scantailor.dewarping.vert_content_bounds import detect_vert_content_bounds
from scantailor.imageproc import debug_drawing as draw
from scantailor.imageproc.binarize import binarize_wolf
from scantailor.imageproc.filters import bgr_to_gray, stretch_gray_range

# Resolution the tracer works at, and the band around it that is left as it is.
_WORKING_DPI = 200
_KEEP_DPI_MIN = 180
_KEEP_DPI_MAX = 220
_WOLF_WINDOW = 31
_WOLF_K = 0.3
# Standard deviations of the Gaussian blurs, in pixels at the working resolution.
_FIRST_DERIVATIVE_SIGMA = 6.0
_ABS_SECOND_DERIVATIVE_SIGMA = 12.0
# Responses below this share (15/255) of the strongest one are not ridges.
_RIDGE_THRESHOLD_NUMERATOR = 15.0
_RIDGE_THRESHOLD_DENOMINATOR = 255.0
_CLOSING_BRICK = 21
_TRACE_STEP_DISTANCE = 10.0
_SNAKE_ITERATIONS = 100
# A line is dropped when its ends are further than this share of its chord from
# the bounds.
_MAX_END_GAP_FRACTION = 0.3
# A turn of at least this many degrees between neighbouring segments is a bend.
_CURVATURE_ANGLE_DEG = 6.0
_FLOAT_EPS = 1.1920929e-07
_FLOAT_MAX = 3.4028235e38
# Upstream's colour of the mid-line seeds in the debug image, as BGR.
_SEED_PURPLE = (0x6D, 0, 0x2D)
# Neighbour steps of a ridge walk, clockwise from the top left.
_TRACE_DIRECTIONS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
)


@dataclass(frozen=True)
class VerticalBounds:
    """The left and right content bounds as lines running from top to bottom.

    Attributes:
        first (tuple[tuple[float, float], tuple[float, float]]): The left bound.
        second (tuple[tuple[float, float], tuple[float, float]]): The right bound.
    """

    first: Line
    second: Line


@dataclass
class TextLineResult:
    """What the tracer found.

    Attributes:
        lines (list[numpy.ndarray of numpy.float64]):
            Traced lines, each of shape ``(n, 2)``,
            running from the left bound to the right one, in the input image's
            pixels.
        vertical_bounds (VerticalBounds | None): The content bounds in the input
            image's pixels.
        traced_count (int): Lines the ridge tracing found before the filters
            that drop short, stray and doubly bent ones; ``lines`` holds the
            survivors.
    """

    lines: list[NDArray[np.float64]] = field(default_factory=list)
    vertical_bounds: VerticalBounds | None = None
    traced_count: int = 0


def trace_text_lines(
    image: NDArray[np.uint8],
    dpi: Dpi | tuple[float, float] | float = 300.0,
    content_rect: tuple[int, int, int, int] | None = None,
    dbg: DebugImages | None = None,
) -> TextLineResult:
    """Trace the text lines of a page.

    Args:
        image (numpy.ndarray of numpy.uint8):
            2-D gray, or BGR(A), page with dark text on a
            light background.
        dpi (Dpi | tuple[float, float] | float): Resolution of ``image``, as one
            value or (horizontal, vertical).
        content_rect (tuple[int, int, int, int] | None): (x, y, width, height)
            of the content in ``image`` pixels; ink outside it is ignored. None
            takes the whole image.
        dbg (DebugImages | None): Receives each step's image, at the
            tracer's working resolution.

    Returns:
        TextLineResult: The traced lines and the vertical content bounds, in
            ``image`` pixels.
    """
    dpi_h, dpi_v = _resolution(dpi)
    gray = np.asarray(bgr_to_gray(image) if image.ndim == 3 else image, dtype=np.uint8)
    height, width = gray.shape

    downscaled = _downscale(gray, dpi_h, dpi_v)
    if dbg is not None:
        dbg.add(downscaled, "downscaled")
    scaled_height, scaled_width = downscaled.shape
    factor_x = scaled_width / width
    factor_y = scaled_height / height

    if content_rect is None:
        content_rect = (0, 0, width, height)
    scaled_rect = _scale_rect(content_rect, factor_x, factor_y)

    ink = binarize_wolf(downscaled, _WOLF_WINDOW, k=_WOLF_K) == 0
    if dbg is not None:
        dbg.add(ink, "binarized")
    ink = _sanitize_binary_image(ink, scaled_rect)
    if dbg is not None:
        dbg.add(ink, "sanitized")
    bounds = detect_vert_content_bounds(ink, dbg)
    if dbg is not None:
        dbg.add(_visualize_vertical_bounds(ink, bounds), "vertBounds")

    polylines = _extract_text_lines(
        stretch_gray_range(downscaled, 0.0, 0.0), bounds, dbg
    )
    traced_count = len(polylines)
    if dbg is not None:
        dbg.add(_visualize_polylines(downscaled, polylines), "traced")
    polylines = _filter_short_curves(polylines, bounds)
    polylines = _filter_out_of_bounds_curves(polylines, bounds)
    if dbg is not None:
        dbg.add(_visualize_polylines(downscaled, polylines), "filtered1")

    down = _average_unit_vector(bounds)
    down_norm = math.hypot(*down)
    down = (down[0] / down_norm, down[1] / down_norm)
    if down[1] < 0:
        down = (-down[0], -down[1])
    polylines = refine_text_lines(downscaled, polylines, down, _SNAKE_ITERATIONS, dbg)
    polylines = [line for line in polylines if _is_curvature_consistent(line)]
    if dbg is not None:
        dbg.add(_visualize_polylines(downscaled, polylines), "filtered2")

    to_original = np.array([1.0 / factor_x, 1.0 / factor_y])
    return TextLineResult(
        lines=[line * to_original for line in polylines],
        vertical_bounds=VerticalBounds(
            first=_scale_line(bounds[0], to_original),
            second=_scale_line(bounds[1], to_original),
        ),
        traced_count=traced_count,
    )


def _resolution(dpi: Dpi | tuple[float, float] | float) -> tuple[int, int]:
    """Horizontal and vertical resolution as whole numbers, as C++ holds them."""
    if isinstance(dpi, tuple):
        dpi_h, dpi_v = dpi
    elif isinstance(dpi, int | float):
        dpi_h = dpi_v = float(dpi)
    else:
        dpi_h, dpi_v = float(dpi.horizontal), float(dpi.vertical)
    return max(1, round(dpi_h)), max(1, round(dpi_v))


def _downscale(gray: NDArray[np.uint8], dpi_h: int, dpi_v: int) -> NDArray[np.uint8]:
    """Resample to about 200 dpi; a resolution already near 200 is left alone."""
    height, width = gray.shape
    if all(_KEEP_DPI_MIN <= value <= _KEEP_DPI_MAX for value in (dpi_h, dpi_v)):
        return gray
    new_width = max(1, width * _WORKING_DPI // dpi_h)
    new_height = max(1, height * _WORKING_DPI // dpi_v)
    if (new_width, new_height) == (width, height):
        return gray
    shrinking = new_width <= width and new_height <= height
    resized = cv2.resize(
        gray,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )
    return np.asarray(resized, dtype=np.uint8)


def _scale_rect(
    rect: tuple[int, int, int, int], factor_x: float, factor_y: float
) -> tuple[int, int, int, int]:
    """Scale a rectangle, rounding position and size separately, like Qt does."""
    x, y, width, height = rect
    return (
        q_round(factor_x * x),
        q_round(factor_y * y),
        q_round(factor_x * width),
        q_round(factor_y * height),
    )


def _scale_line(line: Line, factors: NDArray[np.float64]) -> Line:
    (x1, y1), (x2, y2) = line
    return (
        (x1 * float(factors[0]), y1 * float(factors[1])),
        (x2 * float(factors[0]), y2 * float(factors[1])),
    )


def _seed_fill(
    seed: NDArray[np.bool_], mask: NDArray[np.bool_], connectivity: int
) -> NDArray[np.bool_]:
    """Components of ``mask`` that contain at least one ``seed`` pixel."""
    count, labels = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=connectivity
    )
    tagged = np.zeros(count, dtype=bool)
    tagged[labels[seed & mask]] = True
    tagged[0] = False
    return tagged[labels]


def _open_brick(ink: NDArray[np.bool_], width: int, height: int) -> NDArray[np.bool_]:
    """Opening by a rectangle that cannot fit by going off the image."""
    kernel = np.ones((height, width), dtype=np.uint8)
    eroded = cv2.erode(
        ink.astype(np.uint8),
        kernel,
        anchor=(width // 2, height // 2),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # OpenCV's dilation does not mirror the kernel, so an even-sized brick needs
    # the mirrored anchor to undo the shift of the erosion.
    dilated = cv2.dilate(
        eroded, kernel, anchor=(width - 1 - width // 2, height - 1 - height // 2)
    )
    return np.asarray(dilated, dtype=bool)


def _sanitize_binary_image(
    ink: NDArray[np.bool_], content_rect: tuple[int, int, int, int]
) -> NDArray[np.bool_]:
    """Remove what would mislead the bounds detection.

    Components touching the image border go, then everything that is not at
    least part of a 2x3 or a 3x2 block of ink, then everything outside the
    content rectangle.

    Args:
        ink (numpy.ndarray of numpy.bool): True for ink.
        content_rect (tuple[int, int, int, int]): ``(x, y, width, height)`` to keep.

    Returns:
        numpy.ndarray of numpy.bool: The cleaned page.
    """
    height, width = ink.shape
    frame = np.zeros_like(ink)
    frame[0, :] = frame[-1, :] = True
    frame[:, 0] = frame[:, -1] = True
    ink = ink & ~_seed_fill(frame, ink, 8)

    seeds = _open_brick(ink, 2, 3) | _open_brick(ink, 3, 2)
    ink = _seed_fill(seeds, ink, 8)

    x, y, rect_width, rect_height = content_rect
    inside = np.zeros_like(ink)
    inside[
        max(0, y) : max(0, min(height, y + rect_height)),
        max(0, x) : max(0, min(width, x + rect_width)),
    ] = True
    return ink & inside


def _average_unit_vector(bounds: tuple[Line, Line]) -> tuple[float, float]:
    """Unit vector along the mean direction of the two bounds."""
    units = []
    for (x1, y1), (x2, y2) in bounds:
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy)
        units.append((dx / norm, dy / norm))
    sx = units[0][0] + units[1][0]
    sy = units[0][1] + units[1][1]
    norm = math.hypot(sx, sy)
    return sx / norm, sy / norm


def _mid_line(line1: Line, line2: Line) -> Line:
    """The line half way between two bounds, through their crossing if they cross."""
    crossing = line_intersection(line1, line2)
    if crossing is None:
        p1 = line2[0]
        p2 = projection_point(line1, p1)
        origin = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
        vector = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
        return (origin, (origin[0] + vector[0], origin[1] + vector[1]))
    units = []
    for (x1, y1), (x2, y2) in (line1, line2):
        norm = math.hypot(x2 - x1, y2 - y1)
        units.append(((x2 - x1) / norm, (y2 - y1) / norm))
    return (
        crossing,
        (
            crossing[0] + 0.5 * (units[0][0] + units[1][0]),
            crossing[1] + 0.5 * (units[0][1] + units[1][1]),
        ),
    )


def _squared_distance_map(ink: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Squared distance of each ink pixel to the nearest background or border.

    Args:
        ink (numpy.ndarray of numpy.bool): True for ink.

    Returns:
        numpy.ndarray of numpy.int64: Squared distances, zero on the background.
    """
    padded = np.pad(ink, 1, constant_values=False).astype(np.uint8)
    # The precise mask gives exact Euclidean distances; squaring them in double
    # precision keeps the float32 rounding well below the 0.5 that rint absorbs.
    distances = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    squared = np.square(distances[1:-1, 1:-1], dtype=np.float64)
    return np.rint(squared).astype(np.int64)


def _find_mid_line_seeds(
    sedm: NDArray[np.int64], mid_line: Line
) -> list[tuple[int, int]]:
    """Cells along the middle line where the distance map peaks, then drops."""
    height, width = sedm.shape
    clipped = line_bounded_by_rect(mid_line, (0.0, 0.0, width - 1.0, height - 1.0))
    if clipped is not None:
        mid_line = clipped
    seeds = []
    previous = (0, 0)
    previous_level = 0
    direction = 1
    for x, y in grid_line_cells(mid_line):
        if not (0 <= x < width and 0 <= y < height):
            continue
        level = int(sedm[y, x])
        if (level - previous_level) * direction < 0:
            if direction > 0:
                seeds.append(previous)
            direction *= -1
        previous = (x, y)
        previous_level = level
    return seeds


def _trace_towards_line(
    sedm: NDArray[np.int64],
    ridge: NDArray[np.float32],
    line: NDArray[np.float64],
    seed_x: int,
    seed_y: int,
    max_dist: float,
) -> NDArray[np.int64]:
    """Follow the ridge from a seed until the line is crossed, as a list of cells.

    Each step goes to one of the neighbours that lie towards the line: the one
    with the highest ridge response, unless another has a larger distance than
    the current cell, in which case the one with the largest distance wins.

    Args:
        sedm (numpy.ndarray of numpy.int64): Squared distance map of the ridge mask.
        ridge (numpy.ndarray of numpy.float32): Ridge response.
        line (numpy.ndarray of numpy.float64):
            ``[[x1, y1], [x2, y2]]`` of the line to reach.
        seed_x (int): Start column.
        seed_y (int): Start row.
        max_dist (float): A point is reported each time the path is further than this
            from the last reported point.

    Returns:
        numpy.ndarray of numpy.int64: Cells ``(x, y)`` that were reported, in path
            order, shape ``(k, 2)``.
    """
    height, width = sedm.shape
    # The walk reads single cells; flat memoryviews hand them back as Python
    # numbers several times faster than NumPy scalar indexing.
    distance_at = memoryview(np.ascontiguousarray(sedm, dtype=np.int64)).cast("B")
    distance_at = distance_at.cast("q")
    ridge_at = memoryview(np.ascontiguousarray(ridge, dtype=np.float32)).cast("B")
    ridge_at = ridge_at.cast("f")
    p1x = float(line[0, 0])
    p1y = float(line[0, 1])
    nx, ny = _normal_towards_line(line, seed_x, seed_y)
    # Each step with its offset in the flat arrays.
    moves = [(dx, dy, dy * width + dx) for dx, dy in _steps_towards(nx, ny)]

    max_sqdist = math.floor(max_dist * max_dist + 0.5)
    cells: list[tuple[int, int]] = []
    last_x = seed_x
    last_y = seed_y
    finished = False
    while not finished:
        cur_x = last_x
        cur_y = last_y
        last_side = nx * (last_x - p1x) + ny * (last_y - p1y)
        reported = False
        while True:
            here = cur_y * width + cur_x
            # Away from the border every neighbour is inside the grid.
            inside = 0 < cur_x < width - 1 and 0 < cur_y < height - 1
            best_dm = None
            best_pm = None
            best_sqd = 0
            best_prob = -_FLOAT_MAX
            for move in moves:
                dx, dy, offset = move
                if not inside and not (
                    0 <= cur_x + dx < width and 0 <= cur_y + dy < height
                ):
                    continue
                sqd = distance_at[here + offset]
                if sqd > best_sqd:
                    best_sqd = sqd
                    best_dm = move
                probability = ridge_at[here + offset]
                if probability > best_prob:
                    best_prob = probability
                    best_pm = move
            if best_dm is None or best_pm is None:
                finished = True
                break
            best = best_pm
            # A neighbour deeper in the mask than this cell wins over the ridge.
            if best_sqd > distance_at[here]:
                best = best_dm
            new_x = cur_x + best[0]
            new_y = cur_y + best[1]
            if (nx * (new_x - p1x) + ny * (new_y - p1y)) * last_side < 0:
                finished = True
                break
            cur_x = new_x
            cur_y = new_y
            vx = cur_x - last_x
            vy = cur_y - last_y
            if vx * vx + vy * vy > max_sqdist:
                reported = True
                break
        if not reported and (last_x == cur_x and last_y == cur_y):
            break
        last_x = cur_x
        last_y = cur_y
        cells.append((last_x, last_y))
    return np.array(cells, dtype=np.int64).reshape(-1, 2)


def _normal_towards_line(
    line: NDArray[np.float64], seed_x: int, seed_y: int
) -> tuple[float, float]:
    """The line's normal, turned to point from the seed towards the line."""
    p1x = float(line[0, 0])
    p1y = float(line[0, 1])
    dx = float(line[1, 0]) - p1x
    dy = float(line[1, 1]) - p1y
    nx = dy
    ny = -dx
    start_side = nx * (seed_x - p1x) + ny * (seed_y - p1y)
    tip_side = nx * nx + ny * ny
    if start_side * tip_side > 0:
        return -nx, -ny
    return nx, ny


def _steps_towards(nx: float, ny: float) -> list[tuple[int, int]]:
    """Neighbour steps facing ``(nx, ny)``, best aligned first.

    The sort is stable, so equally aligned steps keep the order of
    ``_TRACE_DIRECTIONS``.
    """
    facing = []
    for vx, vy in _TRACE_DIRECTIONS:
        if nx * vx + ny * vy > 0.0:
            norm = math.sqrt(float(vx * vx + vy * vy))
            facing.append(((vx, vy), nx * (vx / norm) + ny * (vy / norm)))
    facing.sort(key=lambda entry: -entry[1])
    return [step for step, _ in facing]


def _extract_text_lines(
    stretched: NDArray[np.uint8],
    bounds: tuple[Line, Line],
    dbg: DebugImages | None = None,
) -> list[NDArray[np.float64]]:
    """Find ridges in the page and trace a line along each, from bound to bound."""
    direction = _average_unit_vector(bounds)

    # The ridge response is the second derivative along the text columns of the
    # image, taken from a blurred first derivative.
    first = directional_gradient(stretched, direction)
    if dbg is not None:
        dbg.add(draw.signed_overlay(stretched, first), "first_dir_deriv")
    first = gauss_blur(first, _FIRST_DERIVATIVE_SIGMA, _FIRST_DERIVATIVE_SIGMA)
    if dbg is not None:
        dbg.add(draw.signed_overlay(stretched, first), "first_dir_deriv_blurred")
    grad_x, grad_y = sobel_xy(first)
    ridge = grad_x * np.float32(direction[0]) + grad_y * np.float32(direction[1])
    if dbg is not None:
        dbg.add(draw.signed_overlay(stretched, ridge), "second_dir_deriv")

    strongest = max(float(ridge.max()), 0.0)
    # Single precision, as the C++ code computes it: the threshold is compared
    # with single precision responses.
    threshold = np.float32(
        np.float32(strongest)
        * np.float32(_RIDGE_THRESHOLD_NUMERATOR)
        / np.float32(_RIDGE_THRESHOLD_DENOMINATOR)
    )
    initial = ridge > threshold
    if dbg is not None:
        dbg.add(initial, "initialBinarization")

    magnitude = np.abs(ridge)
    if dbg is not None:
        dbg.add(draw.signed_overlay(stretched, magnitude), "abs")
    blurred = gauss_blur(
        magnitude, _ABS_SECOND_DERIVATIVE_SIGMA, _ABS_SECOND_DERIVATIVE_SIGMA
    )
    if dbg is not None:
        dbg.add(draw.signed_overlay(stretched, blurred), "blurred")
    blurred += ridge - magnitude
    if dbg is not None:
        dbg.add(draw.signed_overlay(stretched, blurred), "+= diff")
    post = blurred > threshold
    obstacles = blurred < -threshold
    if dbg is not None:
        dbg.add(post, "postBinarization")
        dbg.add(obstacles, "obstacles")

    closed = cv2.morphologyEx(
        initial.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((_CLOSING_BRICK, _CLOSING_BRICK), dtype=np.uint8),
    ).astype(bool)
    initial = _seed_fill(initial, closed & ~obstacles, 4)
    if dbg is not None:
        dbg.add(initial, "initial_closed")
    post &= initial
    if dbg is not None:
        dbg.add(post, "post &&= initial")

    sedm = _squared_distance_map(post)
    mid_line = _mid_line(bounds[0], bounds[1])
    seeds = _find_mid_line_seeds(sedm, mid_line)
    if dbg is not None:
        dbg.add(
            _visualize_mid_line_seeds(stretched, post, bounds, mid_line, seeds),
            "seeds",
        )

    first_bound = np.array(bounds[0], dtype=np.float64)
    second_bound = np.array(bounds[1], dtype=np.float64)
    lines = []
    for seed_x, seed_y in seeds:
        towards_first = _trace_towards_line(
            sedm, ridge, first_bound, seed_x, seed_y, _TRACE_STEP_DISTANCE
        )
        towards_second = _trace_towards_line(
            sedm, ridge, second_bound, seed_x, seed_y, _TRACE_STEP_DISTANCE
        )
        cells = np.concatenate(
            [towards_first[::-1], [[seed_x, seed_y]], towards_second]
        )
        lines.append(cells.astype(np.float64))
    return lines


def _visualize_vertical_bounds(
    ink: NDArray[np.bool_], bounds: tuple[Line, Line]
) -> NDArray[np.uint8]:
    """Draw the left and right content bounds over the binarised page.

    Args:
        ink (numpy.ndarray of numpy.bool): True for ink.
        bounds (tuple[Line, Line]): The left and the right bound.

    Returns:
        numpy.ndarray of numpy.uint8: New ``(H, W, 3)`` BGR image.
    """
    canvas = draw.to_bgr(ink)
    drawn = canvas.copy()
    draw.draw_lines(drawn, bounds, draw.BLUE, 2.0)
    draw.blend(canvas, drawn, 0.7)
    return canvas


def _visualize_polylines(
    background: NDArray[np.uint8], polylines: list[NDArray[np.float64]]
) -> NDArray[np.uint8]:
    """Draw traced text lines over the page.

    Args:
        background (numpy.ndarray of numpy.uint8): Gray or BGR page.
        polylines (list[numpy.ndarray of numpy.float64]): ``(n, 2)`` points of
            each line.

    Returns:
        numpy.ndarray of numpy.uint8: New ``(H, W, 3)`` BGR image.
    """
    canvas = draw.to_bgr(background)
    draw.draw_polylines(canvas, polylines, draw.BLUE, 3.0)
    return canvas


def _visualize_mid_line_seeds(
    background: NDArray[np.uint8],
    ridges: NDArray[np.bool_],
    bounds: tuple[Line, Line],
    mid_line: Line,
    seeds: list[tuple[int, int]],
) -> NDArray[np.uint8]:
    """Draw the ridges, the bounds, the mid line and the seeds tracing starts from.

    Args:
        background (numpy.ndarray of numpy.uint8): Gray or BGR page.
        ridges (numpy.ndarray of numpy.bool): Ridge pixels, tinted red.
        bounds (tuple[Line, Line]): The left and the right content bound.
        mid_line (Line): The line between them that seeds are taken on.
        seeds (list[tuple[int, int]]): ``(x, y)`` of each seed.

    Returns:
        numpy.ndarray of numpy.uint8: New ``(H, W, 3)`` BGR image.
    """
    canvas = draw.to_bgr(background)
    draw.tint(canvas, ridges, draw.RED, 120 / 255)
    drawn = canvas.copy()
    draw.draw_lines(drawn, lines_within_image(bounds, drawn.shape), draw.BLUE, 5.0)
    draw.blend(canvas, drawn, 180 / 255)
    drawn = canvas.copy()
    draw.draw_lines(drawn, lines_within_image([mid_line], drawn.shape), draw.GREEN, 5.0)
    draw.blend(canvas, drawn, 180 / 255)
    draw.draw_knots(canvas, np.array(seeds, dtype=np.float64) + 0.5, _SEED_PURPLE, 7.0)
    return canvas


def _filter_short_curves(
    polylines: list[NDArray[np.float64]], bounds: tuple[Line, Line]
) -> list[NDArray[np.float64]]:
    """Drop lines whose ends stay far from the bounds compared with their length."""
    kept = []
    for polyline in polylines:
        front = (float(polyline[0, 0]), float(polyline[0, 1]))
        back = (float(polyline[-1, 0]), float(polyline[-1, 1]))
        gap = projection_dist(bounds[0], front) + projection_dist(bounds[1], back)
        chord = math.hypot(back[0] - front[0], back[1] - front[1])
        if gap <= _MAX_END_GAP_FRACTION * chord:
            kept.append(polyline)
    return kept


def _is_inside_bounds(point: Point, left: Line, right: Line) -> bool:
    left_normal = normal_vector(left)
    if left_normal[0] < 0:
        left_normal = (-left_normal[0], -left_normal[1])
    left_dot = left_normal[0] * (point[0] - left[0][0]) + left_normal[1] * (
        point[1] - left[0][1]
    )
    if left_dot < 0:
        return False
    right_normal = normal_vector(right)
    if right_normal[0] > 0:
        right_normal = (-right_normal[0], -right_normal[1])
    right_dot = right_normal[0] * (point[0] - right[0][0]) + right_normal[1] * (
        point[1] - right[0][1]
    )
    return right_dot >= 0


def _filter_out_of_bounds_curves(
    polylines: list[NDArray[np.float64]], bounds: tuple[Line, Line]
) -> list[NDArray[np.float64]]:
    """Drop lines with both ends outside the bounds."""
    kept = []
    for polyline in polylines:
        front = (float(polyline[0, 0]), float(polyline[0, 1]))
        back = (float(polyline[-1, 0]), float(polyline[-1, 1]))
        if _is_inside_bounds(front, *bounds) or _is_inside_bounds(back, *bounds):
            kept.append(polyline)
    return kept


def _is_curvature_consistent(polyline: NDArray[np.float64]) -> bool:
    """False for a line that bends significantly both ways.

    Args:
        polyline (numpy.ndarray of numpy.float64): Line of shape ``(n, 2)``.

    Returns:
        bool: True when the line has no significant convexity together with a
            significant concavity. A line of fewer than two points is rejected.
    """
    count = len(polyline)
    if count <= 1:
        return False
    if count == 2:
        return True
    threshold = math.cos(math.radians(90.0 - _CURVATURE_ANGLE_DEG))
    threshold_sq = threshold * threshold

    segments = np.diff(polyline, axis=0)
    normals = np.column_stack([-segments[:-1, 1], segments[:-1, 0]])
    following = segments[1:]
    normal_sqlen = (normals**2).sum(axis=1)
    segment_sqlen = (following**2).sum(axis=1)
    product = normal_sqlen * segment_sqlen
    dot = (normals * following).sum(axis=1)
    cos_sq = np.where(
        product > _FLOAT_EPS,
        np.abs(dot) * dot / np.where(product > 0, product, 1.0),
        0.0,
    )
    significant = np.abs(cos_sq) >= threshold_sq
    return not (
        bool((significant & (cos_sq > 0)).any())
        and bool((significant & (cos_sq < 0)).any())
    )
