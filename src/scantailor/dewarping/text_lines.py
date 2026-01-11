"""Text line detection and tracing for dewarping.

This module provides functions to detect and trace curved text lines
in scanned document images, which are used to build distortion models
for page dewarping.

The algorithm works as follows:
1. Downscale image to ~200 DPI for performance
2. Binarize and detect vertical content boundaries
3. Extract text lines using directional gradient analysis
4. Refine text lines using snake/active contour optimization
5. Filter out invalid lines (too short, wrong curvature, etc.)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scantailor.core import Dpi


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
    height, width = binary.shape[:2]
    ranges: list[tuple[int, int] | None] = []

    for x in range(width):
        col = binary[:, x]
        nonzero = np.nonzero(col)[0]
        if len(nonzero) == 0:
            ranges.append(None)
        else:
            ranges.append((int(nonzero[0]), int(nonzero[-1])))

    return ranges


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


def trace_text_lines(
    image: NDArray[np.uint8],
    dpi: Dpi | tuple[float, float] | float = 300.0,
    content_rect: tuple[int, int, int, int] | None = None,
) -> TextLineResult:
    """Trace text lines in an image.

    This is the main entry point for text line detection. It:
    1. Downscales the image to ~200 DPI
    2. Binarizes and detects vertical bounds
    3. Extracts text lines using gradient analysis
    4. Refines and filters the lines

    Args:
        image: Grayscale input image.
        dpi: Image resolution (scalar or (horizontal, vertical)).
        content_rect: Optional (x, y, width, height) content rectangle.

    Returns:
        TextLineResult with detected lines and vertical bounds.
    """
    # Normalize DPI
    if isinstance(dpi, tuple):
        dpi_h, dpi_v = dpi
    elif hasattr(dpi, "horizontal"):
        dpi_h, dpi_v = dpi.horizontal, dpi.vertical
    else:
        dpi_h = dpi_v = float(dpi)

    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape[:2]

    # Downscale to ~200 DPI
    target_dpi = 200.0
    downscale_x = target_dpi / dpi_h
    downscale_y = target_dpi / dpi_v

    if downscale_x < 0.9 or downscale_x > 1.1 or downscale_y < 0.9 or downscale_y > 1.1:
        new_width = max(1, int(width * downscale_x))
        new_height = max(1, int(height * downscale_y))
        downscaled = cv2.resize(
            gray, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
    else:
        downscaled = gray
        downscale_x = downscale_y = 1.0

    # Binarize using Wolf's method (similar to C++)
    binary = _binarize_wolf(downscaled, window_size=31)

    # Sanitize: remove border-touching components and despeckle
    if content_rect is not None:
        scaled_rect = (
            int(content_rect[0] * downscale_x),
            int(content_rect[1] * downscale_y),
            int(content_rect[2] * downscale_x),
            int(content_rect[3] * downscale_y),
        )
    else:
        scaled_rect = None

    binary = _sanitize_binary_image(binary, scaled_rect)

    # Detect vertical bounds
    vert_bounds = detect_vertical_bounds(binary)

    # Extract text lines
    polylines = _extract_text_lines(downscaled, vert_bounds)

    # Filter lines
    polylines = _filter_short_curves(polylines, vert_bounds)
    polylines = _filter_out_of_bounds_curves(polylines, vert_bounds)
    polylines = _filter_edgy_curves(polylines)

    # Refine lines
    unit_down = _calc_avg_unit_vector(vert_bounds)
    polylines = _refine_text_lines(downscaled, polylines, unit_down, iterations=100)

    # Filter again after refinement
    polylines = _filter_edgy_curves(polylines)

    # Scale back to original coordinates
    scale_x = 1.0 / downscale_x
    scale_y = 1.0 / downscale_y

    result_lines = []
    for polyline in polylines:
        scaled_points = [(p[0] * scale_x, p[1] * scale_y) for p in polyline]
        result_lines.append(TextLine(points=scaled_points))

    scaled_bounds = VerticalBounds(
        left_top=(vert_bounds.left_top[0] * scale_x, vert_bounds.left_top[1] * scale_y),
        left_bottom=(
            vert_bounds.left_bottom[0] * scale_x,
            vert_bounds.left_bottom[1] * scale_y,
        ),
        right_top=(
            vert_bounds.right_top[0] * scale_x,
            vert_bounds.right_top[1] * scale_y,
        ),
        right_bottom=(
            vert_bounds.right_bottom[0] * scale_x,
            vert_bounds.right_bottom[1] * scale_y,
        ),
    )

    return TextLineResult(lines=result_lines, vertical_bounds=scaled_bounds)


def _binarize_wolf(
    image: NDArray[np.uint8],
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


def _sanitize_binary_image(
    binary: NDArray[np.uint8],
    content_rect: tuple[int, int, int, int] | None = None,
) -> NDArray[np.uint8]:
    """Remove border-touching components and despeckle.

    Args:
        binary: Binary image.
        content_rect: Optional (x, y, w, h) content rectangle.

    Returns:
        Sanitized binary image.
    """
    h, w = binary.shape[:2]

    # Remove components touching borders using morphological reconstruction
    seed = np.zeros_like(binary)
    seed[0, :] = binary[0, :]
    seed[-1, :] = binary[-1, :]
    seed[:, 0] = binary[:, 0]
    seed[:, -1] = binary[:, -1]

    # Reconstruct
    kernel = np.ones((3, 3), dtype=np.uint8)
    prev = seed.copy()
    for _ in range(max(h, w)):
        dilated = cv2.dilate(prev, kernel)
        current = cv2.bitwise_and(dilated, binary)
        if np.array_equal(current, prev):
            break
        prev = current

    # Subtract border-touching components
    result = cv2.subtract(binary, prev)

    # Despeckle using opening
    se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    seeds = cv2.morphologyEx(result, cv2.MORPH_OPEN, se_h)
    seeds = cv2.bitwise_or(seeds, cv2.morphologyEx(result, cv2.MORPH_OPEN, se_v))

    # Reconstruct from seeds
    prev = seeds.copy()
    for _ in range(max(h, w)):
        dilated = cv2.dilate(prev, kernel)
        current = cv2.bitwise_and(dilated, result)
        if np.array_equal(current, prev):
            break
        prev = current

    result = prev

    # Clear outside content rect
    if content_rect is not None:
        x, y, rw, rh = content_rect
        mask = np.zeros_like(result)
        mask[y : y + rh, x : x + rw] = 255
        result = cv2.bitwise_and(result, mask)

    return result


def _calc_avg_unit_vector(bounds: VerticalBounds) -> NDArray[np.float64]:
    """Calculate average unit vector from vertical bounds.

    Args:
        bounds: Vertical bounds.

    Returns:
        Unit vector pointing downward.
    """
    v1 = np.array(
        [
            bounds.left_bottom[0] - bounds.left_top[0],
            bounds.left_bottom[1] - bounds.left_top[1],
        ]
    )
    v1 /= np.linalg.norm(v1) + 1e-10

    v2 = np.array(
        [
            bounds.right_bottom[0] - bounds.right_top[0],
            bounds.right_bottom[1] - bounds.right_top[1],
        ]
    )
    v2 /= np.linalg.norm(v2) + 1e-10

    v3 = v1 + v2
    norm = np.linalg.norm(v3)
    if norm > 1e-10:
        v3 /= norm

    # Ensure pointing downward
    if v3[1] < 0:
        v3 = -v3

    return v3


def _extract_text_lines(
    image: NDArray[np.uint8],
    bounds: VerticalBounds,
) -> list[list[tuple[float, float]]]:
    """Extract text lines using gradient analysis.

    Uses directional second derivative to find text line centers.

    Args:
        image: Grayscale image.
        bounds: Vertical content bounds.

    Returns:
        List of polylines (each a list of (x, y) points).
    """
    height, width = image.shape[:2]
    direction = _calc_avg_unit_vector(bounds).astype(np.float32)

    # Compute directional gradient
    # First derivative in the direction perpendicular to text lines
    img_f32 = image.astype(np.float32)
    sobel_x = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3) / 8.0 / 255.0
    sobel_y = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3) / 8.0 / 255.0

    first_deriv = (sobel_x * direction[0] + sobel_y * direction[1]).astype(np.float32)

    # Blur
    first_deriv = cv2.GaussianBlur(first_deriv, (0, 0), 6.0)

    # Second derivative
    sobel_x2 = cv2.Sobel(first_deriv, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y2 = cv2.Sobel(first_deriv, cv2.CV_32F, 0, 1, ksize=3)
    second_deriv = (sobel_x2 * direction[0] + sobel_y2 * direction[1]).astype(
        np.float32
    )

    # Find threshold
    max_val = np.max(second_deriv)
    threshold = max_val * 15.0 / 255.0

    # Initial binarization (positive second derivative = ridge)
    initial_binary = (second_deriv > threshold).astype(np.uint8) * 255

    # Obstacles (negative second derivative = valley)
    obstacles = (second_deriv < -threshold).astype(np.uint8) * 255

    # Close with obstacles
    closed = _close_with_obstacles(initial_binary, obstacles, (21, 21))

    # Compute SEDM (Squared Euclidean Distance Map)
    _, binary_mask = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
    sedm = cv2.distanceTransform(binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Find seeds along mid-line
    mid_line = _calc_mid_line(bounds)
    seeds = _find_mid_line_seeds(sedm, mid_line, height, width)

    # Trace from each seed
    polylines = []
    for seed in seeds:
        polyline = _trace_from_seed(sedm, second_deriv, seed, bounds, height, width)
        if len(polyline) >= 2:
            polylines.append(polyline)

    return polylines


def _close_with_obstacles(
    image: NDArray[np.uint8],
    obstacles: NDArray[np.uint8],
    brick_size: tuple[int, int],
) -> NDArray[np.uint8]:
    """Morphological closing avoiding obstacles.

    Args:
        image: Binary image.
        obstacles: Obstacle mask.
        brick_size: Structuring element size.

    Returns:
        Closed image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, brick_size)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    closed = cv2.subtract(closed, obstacles)

    # Reconstruct
    h, w = image.shape[:2]
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    prev = image.copy()
    for _ in range(max(h, w)):
        dilated = cv2.dilate(prev, kernel3)
        current = cv2.bitwise_and(dilated, closed)
        if np.array_equal(current, prev):
            break
        prev = current

    return prev


def _calc_mid_line(
    bounds: VerticalBounds,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calculate mid-line between vertical bounds.

    Args:
        bounds: Vertical bounds.

    Returns:
        Mid-line as (start_point, end_point).
    """
    left_top = np.array(bounds.left_top)
    left_bottom = np.array(bounds.left_bottom)
    right_top = np.array(bounds.right_top)
    right_bottom = np.array(bounds.right_bottom)

    mid_top = (left_top + right_top) / 2
    mid_bottom = (left_bottom + right_bottom) / 2

    return (tuple(mid_top), tuple(mid_bottom))


def _find_mid_line_seeds(
    sedm: NDArray[np.float32],
    mid_line: tuple[tuple[float, float], tuple[float, float]],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    """Find seed points along mid-line at local maxima of SEDM.

    Args:
        sedm: Squared Euclidean Distance Map.
        mid_line: Mid-line (start, end).
        height: Image height.
        width: Image width.

    Returns:
        List of seed points.
    """
    start, end = mid_line
    start = np.array(start)
    end = np.array(end)

    # Clip to image bounds
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1:
        return []
    direction /= length

    seeds = []
    prev_level = 0.0
    prev_pt = None
    going_up = True

    # Walk along line
    num_steps = int(length)
    for i in range(num_steps):
        t = i / num_steps
        pt = start + t * (end - start)
        x, y = round(pt[0]), round(pt[1])

        if 0 <= x < width and 0 <= y < height:
            level = sedm[y, x]
            if prev_pt is not None:
                if going_up and level < prev_level:
                    # Found local maximum
                    seeds.append(prev_pt)
                    going_up = False
                elif not going_up and level > prev_level:
                    going_up = True
            prev_level = level
            prev_pt = (x, y)

    return seeds


def _trace_from_seed(
    sedm: NDArray[np.float32],
    gradient: NDArray[np.float32],
    seed: tuple[int, int],
    bounds: VerticalBounds,
    height: int,
    width: int,
) -> list[tuple[float, float]]:
    """Trace a text line from a seed point.

    Args:
        sedm: Distance transform.
        gradient: Second derivative field.
        seed: Starting point.
        bounds: Vertical bounds.
        height: Image height.
        width: Image width.

    Returns:
        List of points along the line.
    """
    polyline: list[tuple[float, float]] = []

    # Trace towards left bound
    left_points = _trace_towards_line(
        sedm, gradient, seed, bounds.left_line, height, width
    )
    polyline.extend(reversed(left_points))

    # Add seed
    polyline.append((float(seed[0]), float(seed[1])))

    # Trace towards right bound
    right_points = _trace_towards_line(
        sedm, gradient, seed, bounds.right_line, height, width
    )
    polyline.extend(right_points)

    return polyline


def _trace_towards_line(
    sedm: NDArray[np.float32],
    gradient: NDArray[np.float32],
    start: tuple[int, int],
    target_line: tuple[tuple[float, float], tuple[float, float]],
    height: int,
    width: int,
    step_size: float = 10.0,
) -> list[tuple[float, float]]:
    """Trace from start towards target line following SEDM ridge.

    Args:
        sedm: Distance transform.
        gradient: Second derivative field.
        start: Starting point.
        target_line: Target line to trace towards.
        height: Image height.
        width: Image width.
        step_size: Step size for tracing.

    Returns:
        List of traced points.
    """
    points = []
    current = np.array(start, dtype=np.float64)

    line_start = np.array(target_line[0])
    line_end = np.array(target_line[1])
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1:
        return points
    line_vec /= line_len

    for _ in range(1000):  # Max iterations
        x, y = round(current[0]), round(current[1])
        if not (0 <= x < width and 0 <= y < height):
            break

        # Check if reached target line
        to_line = current - line_start
        proj_len = np.dot(to_line, line_vec)
        perp = to_line - proj_len * line_vec
        dist_to_line = np.linalg.norm(perp)
        if dist_to_line < step_size:
            break

        # Direction towards line
        if np.linalg.norm(perp) > 0:
            towards_line = -perp / np.linalg.norm(perp)
        else:
            towards_line = np.array([1.0, 0.0])

        # Find ridge direction using gradient
        gx = 0.0
        gy = 0.0
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            gx = float(sedm[y, x + 1] - sedm[y, x - 1])
            gy = float(sedm[y + 1, x] - sedm[y - 1, x])

        # Move along ridge towards target
        grad = np.array([gx, gy])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            # Move perpendicular to gradient (along ridge)
            ridge_dir = np.array([-grad[1], grad[0]]) / grad_norm
            # Choose direction towards target
            if np.dot(ridge_dir, towards_line) < 0:
                ridge_dir = -ridge_dir
            direction = 0.5 * ridge_dir + 0.5 * towards_line
        else:
            direction = towards_line

        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction /= direction_norm

        current = current + step_size * direction
        points.append((current[0], current[1]))

    return points


def _filter_short_curves(
    polylines: list[list[tuple[float, float]]],
    bounds: VerticalBounds,
) -> list[list[tuple[float, float]]]:
    """Filter out curves that don't span most of the content width.

    Args:
        polylines: List of polylines.
        bounds: Vertical bounds.

    Returns:
        Filtered list of polylines.
    """
    result = []
    for polyline in polylines:
        if len(polyline) < 2:
            continue

        front = np.array(polyline[0])
        back = np.array(polyline[-1])

        # Distance to left bound
        left_dist = _point_to_line_distance(front, bounds.left_line)
        # Distance to right bound
        right_dist = _point_to_line_distance(back, bounds.right_line)

        chord_len = np.linalg.norm(back - front)
        if chord_len > 0 and (left_dist + right_dist) <= 0.3 * chord_len:
            result.append(polyline)

    return result


def _point_to_line_distance(
    point: NDArray[np.float64],
    line: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Calculate distance from point to line.

    Args:
        point: Point coordinates.
        line: Line as (start, end).

    Returns:
        Distance to line.
    """
    start = np.array(line[0])
    end = np.array(line[1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return float(np.linalg.norm(point - start))

    line_vec /= line_len
    to_point = point - start
    proj_len = np.dot(to_point, line_vec)
    proj_point = start + proj_len * line_vec
    return float(np.linalg.norm(point - proj_point))


def _filter_out_of_bounds_curves(
    polylines: list[list[tuple[float, float]]],
    bounds: VerticalBounds,
) -> list[list[tuple[float, float]]]:
    """Filter curves with endpoints outside bounds.

    Args:
        polylines: List of polylines.
        bounds: Vertical bounds.

    Returns:
        Filtered list of polylines.
    """
    result = []
    for polyline in polylines:
        if len(polyline) < 2:
            continue

        front_inside = _is_inside_bounds(polyline[0], bounds)
        back_inside = _is_inside_bounds(polyline[-1], bounds)

        if front_inside or back_inside:
            result.append(polyline)

    return result


def _is_inside_bounds(
    point: tuple[float, float],
    bounds: VerticalBounds,
) -> bool:
    """Check if point is inside vertical bounds.

    Args:
        point: Point coordinates.
        bounds: Vertical bounds.

    Returns:
        True if inside bounds.
    """
    pt = np.array(point)

    # Check left bound
    left_start = np.array(bounds.left_top)
    left_end = np.array(bounds.left_bottom)
    left_vec = left_end - left_start
    left_normal = np.array([-left_vec[1], left_vec[0]])
    if left_normal[0] < 0:
        left_normal = -left_normal
    if np.dot(left_normal, pt - left_start) < 0:
        return False

    # Check right bound
    right_start = np.array(bounds.right_top)
    right_end = np.array(bounds.right_bottom)
    right_vec = right_end - right_start
    right_normal = np.array([-right_vec[1], right_vec[0]])
    if right_normal[0] > 0:
        right_normal = -right_normal
    if np.dot(right_normal, pt - right_start) < 0:
        return False

    return True


def _filter_edgy_curves(
    polylines: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Filter curves with inconsistent curvature.

    Removes curves that have both significant convex and concave segments.

    Args:
        polylines: List of polylines.

    Returns:
        Filtered list of polylines.
    """
    result = []
    for polyline in polylines:
        if _is_curvature_consistent(polyline):
            result.append(polyline)
    return result


def _is_curvature_consistent(
    polyline: list[tuple[float, float]],
    angle_threshold_deg: float = 6.0,
) -> bool:
    """Check if polyline has consistent curvature.

    Args:
        polyline: List of points.
        angle_threshold_deg: Threshold angle in degrees.

    Returns:
        True if curvature is consistent.
    """
    n = len(polyline)
    if n <= 1:
        return False
    if n == 2:
        return True

    cos_threshold = np.cos(np.radians(90.0 - angle_threshold_deg))
    cos_sq_threshold = cos_threshold**2

    significant_positive = False
    significant_negative = False

    prev_segment = np.array(polyline[1]) - np.array(polyline[0])
    prev_normal = np.array([-prev_segment[1], prev_segment[0]])
    prev_normal_sqlen = np.dot(prev_normal, prev_normal)

    for i in range(1, n - 1):
        next_segment = np.array(polyline[i + 1]) - np.array(polyline[i])
        next_segment_sqlen = np.dot(next_segment, next_segment)

        sqlen_mult = prev_normal_sqlen * next_segment_sqlen
        if sqlen_mult > 1e-10:
            dot = np.dot(prev_normal, next_segment)
            cos_sq = abs(dot) * dot / sqlen_mult

            if abs(cos_sq) >= cos_sq_threshold:
                if cos_sq > 0:
                    significant_positive = True
                else:
                    significant_negative = True

        prev_normal = np.array([-next_segment[1], next_segment[0]])
        prev_normal_sqlen = next_segment_sqlen

    return not (significant_positive and significant_negative)


def _refine_text_lines(
    image: NDArray[np.uint8],
    polylines: list[list[tuple[float, float]]],
    unit_down: NDArray[np.float64],
    iterations: int = 100,
) -> list[list[tuple[float, float]]]:
    """Refine text lines using snake optimization.

    Uses active contour/snake approach to align lines with text.

    Args:
        image: Grayscale image.
        polylines: Initial polylines.
        unit_down: Downward unit vector.
        iterations: Number of optimization iterations.

    Returns:
        Refined polylines.
    """
    if not polylines:
        return polylines

    # Compute gradient for external energy
    gradient = _compute_gradient(image, sigma=3.0)

    refined = []
    for polyline in polylines:
        if len(polyline) < 3:
            refined.append(polyline)
            continue

        # Convert to snake
        snake = _make_snake(polyline, unit_down)

        # Evolve snake
        for _ in range(iterations):
            snake = _evolve_snake(snake, gradient, unit_down)

        # Convert back to polyline
        refined_polyline = [(node[0], node[1]) for node in snake]
        refined.append(refined_polyline)

    return refined


def _compute_gradient(
    image: NDArray[np.uint8],
    sigma: float = 3.0,
) -> NDArray[np.float32]:
    """Compute gradient magnitude for snake external energy.

    Args:
        image: Grayscale image.
        sigma: Gaussian blur sigma.

    Returns:
        Gradient magnitude field.
    """
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude


def _make_snake(
    polyline: list[tuple[float, float]],
    unit_down: NDArray[np.float64],
    rib_half_length: float = 10.0,
) -> list[tuple[float, float, float]]:
    """Convert polyline to snake (nodes with rib half-lengths).

    Args:
        polyline: Input polyline.
        unit_down: Downward unit vector.
        rib_half_length: Half-length of perpendicular ribs.

    Returns:
        Snake as list of (x, y, rib_half_length).
    """
    return [(p[0], p[1], rib_half_length) for p in polyline]


def _evolve_snake(
    snake: list[tuple[float, float, float]],
    gradient: NDArray[np.float32],
    unit_down: NDArray[np.float64],
    step_size: float = 0.5,
) -> list[tuple[float, float, float]]:
    """Evolve snake one iteration.

    Args:
        snake: Current snake state.
        gradient: Gradient field.
        unit_down: Downward unit vector.
        step_size: Movement step size.

    Returns:
        Updated snake.
    """
    height, width = gradient.shape[:2]
    n = len(snake)

    if n < 3:
        return snake

    new_snake = []
    for i in range(n):
        x, y, rib = snake[i]

        # Sample gradient along rib
        best_offset = 0.0
        best_energy = _sample_gradient(gradient, x, y, width, height)

        for offset in np.linspace(-rib, rib, 11):
            test_x = x + offset * unit_down[0]
            test_y = y + offset * unit_down[1]
            energy = _sample_gradient(gradient, test_x, test_y, width, height)
            if energy > best_energy:
                best_energy = energy
                best_offset = offset

        # Move towards best position
        new_x = x + step_size * best_offset * unit_down[0]
        new_y = y + step_size * best_offset * unit_down[1]

        # Apply smoothing with neighbors
        if 0 < i < n - 1:
            prev = snake[i - 1]
            next_node = snake[i + 1]
            smooth_x = (prev[0] + next_node[0]) / 2
            smooth_y = (prev[1] + next_node[1]) / 2
            alpha = 0.1  # Smoothing weight
            new_x = (1 - alpha) * new_x + alpha * smooth_x
            new_y = (1 - alpha) * new_y + alpha * smooth_y

        new_snake.append((new_x, new_y, rib))

    return new_snake


def _sample_gradient(
    gradient: NDArray[np.float32],
    x: float,
    y: float,
    width: int,
    height: int,
) -> float:
    """Sample gradient at a position with bilinear interpolation.

    Args:
        gradient: Gradient field.
        x: X coordinate.
        y: Y coordinate.
        width: Field width.
        height: Field height.

    Returns:
        Interpolated gradient value.
    """
    if x < 0 or x >= width - 1 or y < 0 or y >= height - 1:
        return 0.0

    x0 = int(x)
    y0 = int(y)
    fx = x - x0
    fy = y - y0

    v00 = gradient[y0, x0]
    v10 = gradient[y0, x0 + 1]
    v01 = gradient[y0 + 1, x0]
    v11 = gradient[y0 + 1, x0 + 1]

    return float(
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )
