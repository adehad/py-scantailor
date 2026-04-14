"""Image analysis utilities.

This module provides functions for analyzing images to detect properties
like skew angle, content boundaries, and other characteristics.
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SkewResult:
    """Result of skew detection.

    Attributes:
        angle: Detected skew angle in degrees. Positive values indicate
            clockwise skew (text slants to the right), negative values
            indicate counter-clockwise skew (text slants to the left).
        confidence: Confidence score for the detection. Higher values
            indicate more reliable detection. A value >= 2.0 is considered
            good confidence.
    """

    angle: float
    confidence: float

    GOOD_CONFIDENCE: float = 2.0

    def is_confident(self) -> bool:
        """Return True if confidence meets the threshold for reliability."""
        return self.confidence >= self.GOOD_CONFIDENCE


# Default parameters for skew detection (matching C++ SkewFinder defaults)
DEFAULT_MAX_ANGLE: float = 7.0
DEFAULT_MIN_ANGLE: float = 0.1
DEFAULT_ACCURACY: float = 0.1
DEFAULT_COARSE_STEP: float = 1.0  # Degrees per step in coarse search


def find_skew(
    image: NDArray[np.uint8],
    max_angle: float = DEFAULT_MAX_ANGLE,
    accuracy: float = DEFAULT_ACCURACY,
    min_angle: float = DEFAULT_MIN_ANGLE,
    resolution_ratio: float = 1.0,
    coarse_step: float = DEFAULT_COARSE_STEP,
) -> SkewResult:
    """Detect the skew angle of a document image.

    Uses a two-phase algorithm:
    1. Coarse search: Linear scan from -max_angle to +max_angle
       in coarse_step increments
    2. Fine search: Binary search around the best coarse angle for accuracy

    The algorithm works by applying vertical shear transforms at various
    angles and scoring each based on horizontal alignment of text lines.
    Well-aligned text has consistent pixel counts per row; skewed text
    shows variation.

    Note: The fine search can extend up to coarse_step/2 beyond max_angle
    to find the true optimum near boundaries. This matches C++ SkewFinder behavior.

    Args:
        image: Grayscale or binary image (uint8). Binary images work best.
        max_angle: Maximum angle to search in degrees (default 7.0).
        accuracy: Target accuracy in degrees for the fine search (default 0.1).
        min_angle: Minimum angle to report; smaller angles are returned as 0
            (default 0.1).
        resolution_ratio: Ratio of horizontal to vertical DPI. Used to adjust
            shear transform for non-square pixels (default 1.0).
        coarse_step: Step size in degrees for the coarse search (default 1.0).

    Returns:
        SkewResult with detected angle and confidence score.

    Example:
        >>> import numpy as np
        >>> from scantailor.imageproc import find_skew
        >>> # Create a simple test image with horizontal lines
        >>> image = np.zeros((100, 200), dtype=np.uint8)
        >>> image[20, :] = 255
        >>> image[50, :] = 255
        >>> image[80, :] = 255
        >>> result = find_skew(image)
        >>> abs(result.angle) < 1.0  # Should detect near-zero skew
        True
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Convert to binary if not already
    if gray.max() > 1:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = gray * 255

    # Reduce image size for faster processing
    reduced = _reduce_image(np.asarray(binary, dtype=np.uint8), factor=2)

    # Phase 1: Coarse search
    best_angle = 0.0
    best_score = 0.0
    scores: list[float] = []

    angle = -max_angle
    while angle <= max_angle:
        score = _calc_score(reduced, angle, resolution_ratio)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += coarse_step

    # Calculate confidence from coarse search
    sum_scores = sum(scores)
    if sum_scores > 0:
        confidence = (best_score / sum_scores) * len(scores) - 1.0
    else:
        confidence = 0.0

    # Phase 2: Fine binary search around best angle
    # Use original (less reduced) image for fine search
    # Note: Fine search can extend slightly beyond max_angle (by up to 0.5°)
    # to find the true optimum near boundaries. This matches C++ behavior.
    fine_reduced = _reduce_image(np.asarray(binary, dtype=np.uint8), factor=1)
    half_range = coarse_step / 2.0

    while half_range >= accuracy:
        left_angle = best_angle - half_range
        right_angle = best_angle + half_range

        left_score = _calc_score(fine_reduced, left_angle, resolution_ratio)
        right_score = _calc_score(fine_reduced, right_angle, resolution_ratio)

        if left_score > best_score and left_score >= right_score:
            best_angle = left_angle
            best_score = left_score
        elif right_score > best_score:
            best_angle = right_angle
            best_score = right_score

        half_range /= 2.0

    # Apply minimum angle threshold
    if abs(best_angle) < min_angle:
        best_angle = 0.0

    return SkewResult(angle=best_angle, confidence=confidence)


def _reduce_image(image: NDArray[np.uint8], factor: int) -> NDArray[np.uint8]:
    """Reduce image size by a power of 2 factor.

    Args:
        image: Input image.
        factor: Reduction factor (1 = 2x, 2 = 4x, etc.).

    Returns:
        Reduced image.
    """
    result: NDArray[np.uint8] = image.copy()
    for _ in range(factor):
        if result.shape[0] < 4 or result.shape[1] < 4:
            break
        # Use area interpolation for downscaling binary images
        result = np.asarray(
            cv2.resize(
                result,
                (result.shape[1] // 2, result.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )
    return result


def _calc_score(
    image: NDArray[np.uint8], angle: float, resolution_ratio: float
) -> float:
    """Calculate alignment score for a given skew angle.

    The score measures how well horizontal text lines align after
    applying a shear transform. Higher scores indicate better alignment.

    Args:
        image: Binary image.
        angle: Angle in degrees to test.
        resolution_ratio: DPI ratio for shear adjustment.

    Returns:
        Alignment score (higher is better).
    """
    if abs(angle) < 0.001:
        # No shear needed for zero angle
        sheared = image
    else:
        sheared = _apply_shear(image, angle, resolution_ratio)

    # Count black pixels per row (assuming black text on white background)
    # If image is inverted (white text), this still works for alignment
    row_counts = np.sum(sheared > 127, axis=1).astype(np.float64)

    # Score: sum of squared differences between consecutive rows
    # Well-aligned text has similar counts; skewed text varies more
    if len(row_counts) < 2:
        return 0.0

    diffs = np.diff(row_counts)
    score = float(np.sum(diffs * diffs))

    return score


def _apply_shear(
    image: NDArray[np.uint8], angle: float, resolution_ratio: float
) -> NDArray[np.uint8]:
    """Apply vertical shear transform to an image.

    Args:
        image: Input image.
        angle: Shear angle in degrees.
        resolution_ratio: DPI ratio adjustment.

    Returns:
        Sheared image.
    """
    h, w = image.shape[:2]

    # Calculate shear factor
    # Positive angle = clockwise skew = shear to the left at bottom
    shear_factor = np.tan(np.radians(angle)) / resolution_ratio

    # Shear matrix (vertical shear, centered)
    # [1, shear, -shear*center_x]
    # [0, 1,     0              ]
    center_x = w / 2.0

    # Create affine transform matrix
    # For vertical shear: y' = y + shear * (x - center_x)
    # Which translates to: x' = x, y' = y + shear*x - shear*center_x
    # Affine matrix: [[1, 0, 0], [shear, 1, -shear*center_x]]
    matrix = np.array(
        [[1.0, 0.0, 0.0], [shear_factor, 1.0, -shear_factor * center_x]],
        dtype=np.float32,
    )

    # Apply transform
    result = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return np.asarray(result, dtype=np.uint8)


def connected_components(
    image: NDArray[np.uint8],
    connectivity: Literal[4, 8] = 8,
) -> tuple[int, NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    """Find connected components in a binary image.

    Args:
        image: Binary image (0 and 255 values).
        connectivity: 4 or 8 connectivity (default 8).

    Returns:
        Tuple of (num_labels, labels, stats, centroids) where:
        - num_labels: Number of labels including background (label 0)
        - labels: Label matrix same size as input
        - stats: Stats matrix with columns [x, y, width, height, area]
        - centroids: Centroid matrix with columns [x, y]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=connectivity
    )
    return (
        num_labels,
        np.asarray(labels, dtype=np.int32),
        np.asarray(stats, dtype=np.int32),
        np.asarray(centroids, dtype=np.float64),
    )


def distance_transform(
    image: NDArray[np.uint8],
    distance_type: Literal["l1", "l2", "c"] = "l2",
) -> NDArray[np.float32]:
    """Compute distance transform of a binary image.

    For each non-zero pixel, computes the distance to the nearest zero pixel.

    Args:
        image: Binary image (0 and 255 values).
        distance_type: Distance metric - "l1" (Manhattan), "l2" (Euclidean),
            or "c" (Chebyshev/chessboard).

    Returns:
        Distance transform as float32 array.
    """
    dist_types = {
        "l1": cv2.DIST_L1,
        "l2": cv2.DIST_L2,
        "c": cv2.DIST_C,
    }
    result = cv2.distanceTransform(image, dist_types[distance_type], maskSize=5)
    return np.asarray(result, dtype=np.float32)


def hough_lines(
    image: NDArray[np.uint8],
    rho: float = 1.0,
    theta: float = np.pi / 180,
    threshold: int = 100,
) -> NDArray[np.float32] | None:
    """Detect lines in a binary image using the standard Hough transform.

    Args:
        image: Binary edge image (e.g., from Canny).
        rho: Distance resolution in pixels.
        theta: Angle resolution in radians.
        threshold: Accumulator threshold (minimum votes for a line).

    Returns:
        Array of lines as (rho, theta) pairs, or None if no lines found.
    """
    lines = cv2.HoughLines(image, rho, theta, threshold)
    if lines is None:
        return None
    return np.asarray(lines, dtype=np.float32)


def hough_lines_p(
    image: NDArray[np.uint8],
    rho: float = 1.0,
    theta: float = np.pi / 180,
    threshold: int = 50,
    min_line_length: float = 50,
    max_line_gap: float = 10,
) -> NDArray[np.int32] | None:
    """Detect line segments using the probabilistic Hough transform.

    Args:
        image: Binary edge image (e.g., from Canny).
        rho: Distance resolution in pixels.
        theta: Angle resolution in radians.
        threshold: Accumulator threshold (minimum votes for a line).
        min_line_length: Minimum line length to accept.
        max_line_gap: Maximum gap between line segments to merge.

    Returns:
        Array of line segments as (x1, y1, x2, y2), or None if no lines found.
    """
    lines = cv2.HoughLinesP(
        image,
        rho,
        theta,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None
    return np.asarray(lines, dtype=np.int32).reshape(-1, 4)


def find_contours(
    image: NDArray[np.uint8],
    mode: Literal["external", "list", "tree"] = "external",
) -> list[NDArray[np.int32]]:
    """Find contours in a binary image.

    Args:
        image: Binary image (0 and 255 values).
        mode: Retrieval mode:
            - "external": Only outermost contours
            - "list": All contours as flat list
            - "tree": All contours with hierarchy

    Returns:
        List of contours, each as an Nx1x2 array of (x, y) points.
    """
    mode_map = {
        "external": cv2.RETR_EXTERNAL,
        "list": cv2.RETR_LIST,
        "tree": cv2.RETR_TREE,
    }
    contours, _ = cv2.findContours(image, mode_map[mode], cv2.CHAIN_APPROX_SIMPLE)
    return [np.asarray(c, dtype=np.int32) for c in contours]


def max_whitespace_rect(
    image: NDArray[np.uint8],
) -> tuple[int, int, int, int] | None:
    """Find the largest axis-aligned white rectangle in a binary image.

    This is useful for finding margins or blank areas in document images.

    Args:
        image: Binary image (0=black, 255=white).

    Returns:
        Tuple (x, y, width, height) of the largest white rectangle,
        or None if no white pixels found.
    """
    # Use connected components to find white regions
    # Invert so white becomes foreground
    inverted = cv2.bitwise_not(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted)

    if num_labels <= 1:
        # No white regions (only background)
        return None

    # Find largest white region (skip label 0 which is the black background)
    max_area = 0
    best_rect = None

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)

    return best_rect
