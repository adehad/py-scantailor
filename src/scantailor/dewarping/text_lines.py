"""Text line detection and tracing for dewarping.

This module provides functions to detect and trace curved text lines
in scanned document images, which are used to build distortion models
for page dewarping.

The algorithm works as follows:
1. Downscale the image to ~200 DPI for performance
2. Binarize, dropping border-touching components and specks
3. Detect the vertical content bounds
4. Merge letters into line blobs and trace the middle of each blob
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d

from scantailor.core.models import Dpi

# Resolution the tracer works at.
_WORKING_DPI = 200.0
_KEEP_SCALE_TOLERANCE = 0.1
# Smallest component, in pixels at the working resolution, that can be a letter.
_MIN_SPECK_AREA = 6
# Letters closer than this horizontally belong to one line.
_WORD_GAP_INCH = 0.2
# Runs narrower than this fraction of the content width are not lines.
_MIN_LINE_SPAN = 0.4
# Blobs thicker than this are several lines merged together.
_MAX_LINE_THICKNESS_INCH = 0.4
_SAMPLE_SPACING_INCH = 0.04
# Width over which a line's ink is averaged to find its centre.
_CENTRE_WINDOW_INCH = 0.2


@dataclass
class VerticalBounds:
    """Left and right vertical content boundaries.

    Each boundary is represented as a line from top to bottom of the image,
    defined by two points: (x1, y1) at top and (x2, y2) at bottom.
    """

    left_top: tuple[float, float]
    left_bottom: tuple[float, float]
    right_top: tuple[float, float]
    right_bottom: tuple[float, float]

    @property
    def left_line(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get left boundary as (start_point, end_point)."""
        return (self.left_top, self.left_bottom)

    @property
    def right_line(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get right boundary as (start_point, end_point)."""
        return (self.right_top, self.right_bottom)


@dataclass
class TextLine:
    """A traced text line represented as a polyline.

    Attributes:
        points: List of (x, y) points along the text line.
    """

    points: list[tuple[float, float]] = field(default_factory=list)

    def __len__(self) -> int:
        """Return number of points in the line."""
        return len(self.points)

    @property
    def is_valid(self) -> bool:
        """Check if line has enough points to be valid."""
        return len(self.points) >= 2


@dataclass
class TextLineResult:
    """Result of text line detection.

    Attributes:
        lines: List of detected text lines.
        vertical_bounds: Detected vertical content boundaries.
    """

    lines: list[TextLine] = field(default_factory=list)
    vertical_bounds: VerticalBounds | None = None


def detect_vertical_bounds(
    binary: NDArray[np.uint8],
) -> VerticalBounds:
    """Detect left and right vertical content boundaries.

    Uses RANSAC to find the best fitting lines for the left and right
    edges of the text content.

    Args:
        binary: Binary image (0=background, 255=foreground).

    Returns:
        VerticalBounds with left and right boundary lines.
    """
    height, width = binary.shape[:2]

    # Calculate vertical ranges for each column
    vert_ranges = _calculate_vertical_ranges(binary)

    # Process from left for left boundary
    left_path = _build_convex_envelope(vert_ranges, width, direction="left")
    left_line = _fit_line_ransac(left_path, height, side="left")

    # Process from right for right boundary
    right_path = _build_convex_envelope(vert_ranges, width, direction="right")
    right_line = _fit_line_ransac(right_path, height, side="right")

    return VerticalBounds(
        left_top=(left_line[0], 0.0),
        left_bottom=(left_line[1], float(height)),
        right_top=(right_line[0], 0.0),
        right_bottom=(right_line[1], float(height)),
    )


def _calculate_vertical_ranges(
    binary: NDArray[np.uint8],
) -> list[tuple[int, int] | None]:
    """Calculate top and bottom black pixel for each column.

    Args:
        binary: Binary image.

    Returns:
        List of (top, bottom) tuples for each column, or None if column is empty.
    """
    height = binary.shape[0]
    foreground = binary != 0
    occupied = foreground.any(axis=0)
    tops = foreground.argmax(axis=0)
    bottoms = height - 1 - foreground[::-1].argmax(axis=0)
    return [
        (int(top), int(bottom)) if has_ink else None
        for has_ink, top, bottom in zip(occupied, tops, bottoms, strict=True)
    ]


def _build_convex_envelope(
    vert_ranges: list[tuple[int, int] | None],
    width: int,
    direction: str,
) -> list[tuple[int, int]]:
    """Build a convex envelope from vertical ranges.

    Args:
        vert_ranges: Vertical ranges for each column.
        width: Image width.
        direction: "left" or "right".

    Returns:
        List of points forming the convex envelope.
    """
    path: list[tuple[int, int]] = []

    if direction == "left":
        x_range = range(width)
    else:
        x_range = range(width - 1, -1, -1)

    for x in x_range:
        vrange = vert_ranges[x]
        if vrange is None:
            continue

        top, bottom = vrange

        if not path:
            path.append((x, top))
            if top != bottom:
                path.append((x, bottom))
            continue

        # Add top point maintaining convexity
        while len(path) >= 2:
            if not _is_convex(path[-2], path[-1], (x, top), direction):
                path.pop()
            else:
                break
        path.append((x, top))

        # Add bottom point maintaining convexity
        while len(path) >= 2:
            if not _is_convex(path[-2], path[-1], (x, bottom), direction):
                path.pop()
            else:
                break
        if (x, bottom) != path[-1]:
            path.append((x, bottom))

    return path


def _is_convex(
    p1: tuple[int, int],
    p2: tuple[int, int],
    p3: tuple[int, int],
    direction: str,
) -> bool:
    """Check if three points form a convex turn.

    Args:
        p1: First point.
        p2: Second point (middle).
        p3: Third point.
        direction: "left" or "right".

    Returns:
        True if the turn is convex for the given direction.
    """
    cross_z = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p3[0] - p2[0]) * (p2[1] - p1[1])
    if direction == "left":
        return cross_z >= 0
    return cross_z <= 0


def _fit_line_ransac(
    path: list[tuple[int, int]],
    height: int,
    side: str = "left",
    cos_threshold: float = 0.9976,  # cos(4 degrees)
) -> tuple[float, float]:
    """Fit a line to path points using RANSAC.

    Args:
        path: List of points.
        height: Image height.
        side: Which side of content ("left" or "right").
        cos_threshold: Cosine threshold for RANSAC inliers.

    Returns:
        Tuple of (x_at_top, x_at_bottom) for the fitted line.
    """
    if len(path) < 2:
        if path:
            return (float(path[0][0]), float(path[0][0]))
        return (0.0, 0.0)

    # Build segments from path
    segments: list[tuple[tuple[int, int], tuple[int, int], NDArray, int]] = []
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        if p2[1] <= p1[1]:
            continue  # Skip non-vertical segments

        vec = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float64)
        if abs(vec[0]) > abs(vec[1]):
            continue  # Skip horizontal segments

        vec /= np.linalg.norm(vec)
        vert_dist = p2[1] - p1[1]
        segments.append((p1, p2, vec, vert_dist))

    if not segments:
        return (float(path[0][0]), float(path[0][0]))

    # RANSAC
    best_model: list[int] = []
    best_total_dist = 0
    rng = np.random.default_rng(0)  # Reproducible

    # Try best segments first
    segments_sorted = sorted(
        segments,
        key=lambda s: min(abs(s[0][0] - path[0][0]), abs(s[1][0] - path[0][0])),
    )

    for i, seed_seg in enumerate(segments_sorted[:6]):
        inliers = []
        total_dist = 0
        for j, seg in enumerate(segments):
            cos_val = np.dot(seed_seg[2], seg[2])
            if cos_val > cos_threshold:
                inliers.append(j)
                total_dist += seg[3]
        if total_dist > best_total_dist:
            best_model = inliers
            best_total_dist = total_dist

    # Random iterations
    for _ in range(200):
        seed_idx = rng.integers(len(segments))
        seed_seg = segments[seed_idx]
        inliers = []
        total_dist = 0
        for j, seg in enumerate(segments):
            cos_val = np.dot(seed_seg[2], seg[2])
            if cos_val > cos_threshold:
                inliers.append(j)
                total_dist += seg[3]
        if total_dist > best_total_dist:
            best_model = inliers
            best_total_dist = total_dist

    if not best_model:
        return (float(path[0][0]), float(path[0][0]))

    # Interpolate best model
    accum_vec = np.zeros(2, dtype=np.float64)
    accum_weight = 0.0
    for idx in best_model:
        seg = segments[idx]
        weight = np.sqrt(seg[3])
        accum_vec += weight * seg[2]
        accum_weight += weight

    if accum_weight > 0:
        accum_vec /= accum_weight
    else:
        accum_vec = np.array([0.0, 1.0])

    # Find point to pass through
    # Normal points towards the inside of the content
    normal = np.array([-accum_vec[1], accum_vec[0]])
    # For left side, normal should point right (positive x)
    # For right side, normal should point left (negative x)
    if (side == "left") != (normal[0] > 0):
        normal = -normal

    best_point = np.array(path[0], dtype=np.float64)

    for pt in path:
        pt_arr = np.array(pt, dtype=np.float64)
        if np.dot(normal, pt_arr - best_point) < 0:
            best_point = pt_arr

    # Extend line to top and bottom
    if abs(accum_vec[1]) < 1e-6:
        return (best_point[0], best_point[0])

    t_top = -best_point[1] / accum_vec[1]
    t_bottom = (height - best_point[1]) / accum_vec[1]

    x_top = best_point[0] + t_top * accum_vec[0]
    x_bottom = best_point[0] + t_bottom * accum_vec[0]

    return (x_top, x_bottom)


def _binarize_wolf(
    image: np.ndarray,
    window_size: int = 31,
    k: float = 0.5,
) -> NDArray[np.uint8]:
    """Binarize using Wolf's method.

    A simplified Wolf binarization for text line detection.
    Uses local mean and standard deviation.

    Args:
        image: Grayscale image.
        window_size: Local window size.
        k: Wolf parameter.

    Returns:
        Binary image (0 or 255).
    """
    # Use cv2 for local statistics
    img_float = image.astype(np.float64)

    # Local mean
    mean = cv2.blur(img_float, (window_size, window_size))

    # Local variance and std dev
    sq_mean = cv2.blur(img_float**2, (window_size, window_size))
    variance = sq_mean - mean**2
    variance = np.maximum(variance, 0)  # Handle numerical issues
    std = np.sqrt(variance)

    # Wolf's threshold:
    # t = (1 - k) * mean + k * min_val + k * std / max_std * (mean - min_val)
    min_val = np.min(image)
    max_std = np.max(std)

    if max_std > 0:
        threshold = (
            (1 - k) * mean + k * min_val + k * (std / max_std) * (mean - min_val)
        )
    else:
        threshold = mean

    # Binarize
    binary = np.where(img_float > threshold, 255, 0).astype(np.uint8)
    return binary


def trace_text_lines(
    image: NDArray[np.uint8],
    dpi: Dpi | tuple[float, float] | float = 300.0,
    content_rect: tuple[int, int, int, int] | None = None,
) -> TextLineResult:
    """Trace text lines in a page image.

    The page is reduced to about 200 DPI and binarised. Letters are merged into
    line-shaped blobs and each wide blob yields one polyline through the middle
    of its ink, whatever its curvature.

    Args:
        image (NDArray[np.uint8]): 2-D gray, or BGR(A), page with dark text on a
            light background.
        dpi (Dpi | tuple[float, float] | float): Resolution of ``image``, as one
            value or (horizontal, vertical).
        content_rect (tuple[int, int, int, int] | None): (x, y, width, height)
            of the area to search, in ``image`` pixels. None searches everything.

    Returns:
        TextLineResult: Lines and vertical bounds in ``image`` pixels.
    """
    if isinstance(dpi, tuple):
        dpi_h, dpi_v = dpi
    elif isinstance(dpi, int | float):
        dpi_h = dpi_v = float(dpi)
    else:
        dpi_h, dpi_v = float(dpi.horizontal), float(dpi.vertical)

    gray = np.asarray(
        cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image,
        dtype=np.uint8,
    )
    height, width = gray.shape[:2]

    scale_x = _WORKING_DPI / dpi_h
    scale_y = _WORKING_DPI / dpi_v
    if (
        abs(scale_x - 1.0) <= _KEEP_SCALE_TOLERANCE
        and abs(scale_y - 1.0) <= _KEEP_SCALE_TOLERANCE
    ):
        scale_x = scale_y = 1.0
        reduced = gray
    else:
        reduced = np.asarray(
            cv2.resize(
                gray,
                (max(1, int(width * scale_x)), max(1, int(height * scale_y))),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )
    working_dpi = 0.5 * (dpi_h * scale_x + dpi_v * scale_y)

    scaled_rect = None
    if content_rect is not None:
        x, y, rect_width, rect_height = content_rect
        scaled_rect = (
            int(x * scale_x),
            int(y * scale_y),
            int(rect_width * scale_x),
            int(rect_height * scale_y),
        )
    ink = _ink_mask(reduced, scaled_rect, working_dpi)
    bounds = detect_vertical_bounds(ink)
    lines = _trace_lines(ink, bounds, working_dpi)

    return TextLineResult(
        lines=[
            TextLine(points=[(px / scale_x, py / scale_y) for px, py in line])
            for line in lines
        ],
        vertical_bounds=VerticalBounds(
            left_top=(bounds.left_top[0] / scale_x, bounds.left_top[1] / scale_y),
            left_bottom=(
                bounds.left_bottom[0] / scale_x,
                bounds.left_bottom[1] / scale_y,
            ),
            right_top=(bounds.right_top[0] / scale_x, bounds.right_top[1] / scale_y),
            right_bottom=(
                bounds.right_bottom[0] / scale_x,
                bounds.right_bottom[1] / scale_y,
            ),
        ),
    )


def _ink_mask(
    gray: NDArray[np.uint8],
    content_rect: tuple[int, int, int, int] | None,
    dpi: float,
) -> NDArray[np.uint8]:
    """Binarise a page and keep only components that can be text.

    Components that touch the image border (scanner shadows, page edges) and
    specks too small to be letters are dropped, as is everything outside
    ``content_rect``.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page.
        content_rect (tuple[int, int, int, int] | None): (x, y, width, height)
            to search in, or None.
        dpi (float): Resolution of ``gray``.

    Returns:
        NDArray[np.uint8]: 255 where there is ink, 0 elsewhere.
    """
    ink = (_binarize_wolf(gray) == 0).astype(np.uint8)
    if content_rect is not None:
        x, y, rect_width, rect_height = content_rect
        inside = np.zeros_like(ink)
        inside[max(0, y) : y + rect_height, max(0, x) : x + rect_width] = 1
        ink &= inside

    height, width = ink.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    left, top, box_w, box_h, area = stats.T
    touches_border = (
        (left == 0) | (top == 0) | (left + box_w == width) | (top + box_h == height)
    )
    min_area = _MIN_SPECK_AREA * (dpi / _WORKING_DPI) ** 2
    keep = ~touches_border & (area >= min_area)
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def _trace_lines(
    ink: NDArray[np.uint8],
    bounds: VerticalBounds,
    dpi: float,
) -> list[list[tuple[float, float]]]:
    """Find a polyline through the middle of every long, thin run of text.

    Args:
        ink (NDArray[np.uint8]): 255 where there is ink.
        bounds (VerticalBounds): Content bounds; runs much narrower than the
            distance between them are not lines.
        dpi (float): Resolution of ``ink``.

    Returns:
        list[list[tuple[float, float]]]: One left-to-right polyline per line,
            in ``ink`` pixels.
    """
    width = ink.shape[1]
    closing_width = max(3, round(dpi * _WORD_GAP_INCH))
    closed = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, np.ones((1, closing_width), dtype=np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        return []

    content_width = 0.5 * (
        bounds.right_top[0]
        - bounds.left_top[0]
        + bounds.right_bottom[0]
        - bounds.left_bottom[0]
    )
    long_enough = stats[:, cv2.CC_STAT_WIDTH] >= _MIN_LINE_SPAN * content_width
    long_enough[0] = False
    selected = np.flatnonzero(long_enough)
    if selected.size == 0:
        return []

    # Per (line, column) pixel count and sum of y, from one pass over the pixels.
    line_of_label = np.full(count, -1, dtype=np.int64)
    line_of_label[selected] = np.arange(selected.size)
    ys, xs = np.nonzero(long_enough[labels])
    flat = line_of_label[labels[ys, xs]] * width + xs
    size = selected.size * width
    counts = np.bincount(flat, minlength=size).reshape(-1, width)
    y_sums = np.bincount(flat, weights=ys, minlength=size).reshape(-1, width)

    # A single column's ink centre jumps with the letter under it (an "h"
    # against an "a"); the centre of mass over a few letters does not.
    window = max(3, round(dpi * _CENTRE_WINDOW_INCH))
    smooth_counts = uniform_filter1d(counts.astype(np.float64), window, axis=1)
    smooth_sums = uniform_filter1d(y_sums, window, axis=1)

    max_thickness = _MAX_LINE_THICKNESS_INCH * dpi
    sample_step = max(1, round(dpi * _SAMPLE_SPACING_INCH))
    polylines = []
    for row_counts, row_smooth_counts, row_smooth_sums in zip(
        counts, smooth_counts, smooth_sums, strict=True
    ):
        columns = np.flatnonzero(row_counts)
        if np.median(row_counts[columns]) > max_thickness:
            continue
        picked = columns[::sample_step]
        centres = row_smooth_sums[picked] / row_smooth_counts[picked] + 0.5
        polylines.append(
            list(zip((picked + 0.5).tolist(), centres.tolist(), strict=True))
        )
    return polylines
