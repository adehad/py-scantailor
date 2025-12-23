"""Content box detection algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .content_box import ContentBox


@dataclass(frozen=True)
class ContentDetectionResult:
    """Result of content detection."""

    content_box: ContentBox
    confidence: float

    def is_confident(self) -> bool:
        """Return True if detection is confident."""
        return self.confidence >= 0.5


# Target DPI for content detection processing
_TARGET_DPI = 150.0
_DEFAULT_DPI = 300.0


def find_content_box(
    image: NDArray[np.uint8],
    dpi: float = _DEFAULT_DPI,
    margin: int = 10,
) -> ContentDetectionResult:
    """Detect content boundaries in an image.

    Uses a multi-stage algorithm:
    1. Scale to processing resolution
    2. Binarize the image
    3. Detect and remove shadows
    4. Despeckle to remove noise
    5. Find content boundaries

    Args:
        image: Grayscale or color image as numpy array.
        dpi: Image resolution in DPI.
        margin: Minimum margin from image edges.

    Returns:
        ContentDetectionResult with detected box and confidence.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]

    # Scale to target DPI for processing
    scale = _TARGET_DPI / dpi
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scaled = gray
        scale = 1.0

    # Binarize using adaptive threshold
    binary = cv2.adaptiveThreshold(
        scaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,  # Block size
        10,  # C constant
    )

    # Remove shadows using morphological operations
    binary = _remove_shadows(np.asarray(binary, dtype=np.uint8))

    # Despeckle to remove noise
    binary = _despeckle(binary, min_area=50)

    # Find content boundaries
    content_rect = _find_bounding_rect(binary, margin=int(margin * scale))

    # Scale back to original resolution
    if content_rect is not None:
        x, y, cw, ch = content_rect
        inv_scale = 1.0 / scale
        content_box = ContentBox(
            x=x * inv_scale,
            y=y * inv_scale,
            width=cw * inv_scale,
            height=ch * inv_scale,
        )
        # Calculate confidence based on content area ratio
        content_area = cw * ch
        image_area = binary.shape[0] * binary.shape[1]
        # Content should be substantial but not the entire image
        area_ratio = content_area / image_area if image_area > 0 else 0
        if 0.1 <= area_ratio <= 0.95:
            confidence = 1.0
        elif area_ratio < 0.1:
            confidence = area_ratio / 0.1
        else:
            confidence = (1.0 - area_ratio) / 0.05
        confidence = max(0.0, min(1.0, confidence))
    else:
        # No content found - return full image minus margin
        content_box = ContentBox(
            x=float(margin),
            y=float(margin),
            width=float(max(0, w - 2 * margin)),
            height=float(max(0, h - 2 * margin)),
        )
        confidence = 0.0

    return ContentDetectionResult(content_box=content_box, confidence=confidence)


def _remove_shadows(binary: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Remove shadows from a binary image using morphological operations."""
    h, w = binary.shape[:2]

    # Detect horizontal shadows (long horizontal structures)
    h_kernel_size = max(1, min(200, w // 3))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 3))
    h_shadows = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical shadows (long vertical structures)
    v_kernel_size = max(1, min(300, h // 3))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, v_kernel_size))
    v_shadows = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Combine shadows
    shadows = cv2.bitwise_or(h_shadows, v_shadows)

    # Dilate shadows to extend them fully
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    shadows = cv2.dilate(shadows, dilate_kernel, iterations=2)

    # Subtract shadows from binary
    result = cv2.subtract(binary, shadows)

    return np.asarray(result, dtype=np.uint8)



def _despeckle(
    binary: NDArray[np.uint8], min_area: int = 50
) -> NDArray[np.uint8]:
    """Remove small noise specks from a binary image."""
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Create output image
    result = np.zeros_like(binary)

    # Keep only components larger than min_area
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == i] = 255

    return result


def _find_bounding_rect(
    binary: NDArray[np.uint8], margin: int = 5
) -> tuple[int, int, int, int] | None:
    """Find the bounding rectangle of content in a binary image.

    Returns (x, y, width, height) or None if no content found.
    """
    h, w = binary.shape[:2]

    # Find non-zero pixels
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    # Get bounding rectangle
    x, y, bw, bh = cv2.boundingRect(coords)

    # Apply margin, staying within image bounds
    x = max(margin, x - margin)
    y = max(margin, y - margin)
    x2 = min(w - margin, x + bw + margin)
    y2 = min(h - margin, y + bh + margin)

    if x2 <= x or y2 <= y:
        return None

    return (x, y, x2 - x, y2 - y)


def find_page_edges(
    image: NDArray[np.uint8],
    dpi: float = _DEFAULT_DPI,
    tolerance: float = 0.1,
    fine_tune: bool = False,
) -> ContentBox | None:
    """Detect page boundaries in an image.

    Looks for strong edges that indicate the physical page boundary
    (as opposed to content within the page).

    Args:
        image: Grayscale or color image.
        dpi: Image resolution.
        tolerance: Edge detection threshold (0-1).
        fine_tune: Whether to refine corner detection.

    Returns:
        ContentBox representing page boundaries, or None if not found.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]

    # Scale to processing resolution
    scale = _TARGET_DPI / dpi
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scaled = gray
        scale = 1.0

    # Detect edges
    edges = cv2.Canny(scaled, 50, 150)

    # Dilate edges to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(min(scaled.shape) * 0.3),
        minLineLength=int(min(scaled.shape) * 0.2),
        maxLineGap=int(min(scaled.shape) * 0.05),
    )

    if lines is None or len(lines) < 4:
        return None

    # Find boundary lines (approximately horizontal/vertical at edges)
    sh, sw = scaled.shape[:2]
    edge_threshold = 0.1 * min(sw, sh)

    left_edge = sw
    right_edge = 0
    top_edge = sh
    bottom_edge = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check if line is approximately vertical (for left/right edges)
        if abs(x2 - x1) < abs(y2 - y1) * 0.2:
            avg_x = (x1 + x2) / 2
            if avg_x < edge_threshold:
                left_edge = min(left_edge, int(avg_x))
            elif avg_x > sw - edge_threshold:
                right_edge = max(right_edge, int(avg_x))

        # Check if line is approximately horizontal (for top/bottom edges)
        if abs(y2 - y1) < abs(x2 - x1) * 0.2:
            avg_y = (y1 + y2) / 2
            if avg_y < edge_threshold:
                top_edge = min(top_edge, int(avg_y))
            elif avg_y > sh - edge_threshold:
                bottom_edge = max(bottom_edge, int(avg_y))

    # Check if we found valid edges
    if left_edge >= right_edge or top_edge >= bottom_edge:
        return None

    # Scale back to original resolution
    inv_scale = 1.0 / scale
    return ContentBox(
        x=left_edge * inv_scale,
        y=top_edge * inv_scale,
        width=(right_edge - left_edge) * inv_scale,
        height=(bottom_edge - top_edge) * inv_scale,
    )
