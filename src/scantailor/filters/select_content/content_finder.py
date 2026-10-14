"""Content box and page edge detection."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.result_memo import ResultMemo, array_digest

from .content_box import ContentBox
from .content_box_finder import PROCESSING_DPI, Rect, find_content_rect


@dataclass(frozen=True)
class ContentDetectionResult:
    """Result of content detection."""

    content_box: ContentBox
    confidence: float

    def is_confident(self) -> bool:
        """Return True if detection is confident."""
        return self.confidence >= 0.5


# Target DPI for content detection processing
_TARGET_DPI = float(PROCESSING_DPI)
_DEFAULT_DPI = 300.0

# Detection results by processed image; revisiting a page or a stage looks at
# the same pixels again.
_CONTENT_MEMO = ResultMemo[Rect | None](capacity=32)


def find_content_box(
    image: NDArray[np.uint8],
    dpi: float = _DEFAULT_DPI,
    margin: int = 10,
    page_box: ContentBox | None = None,
    outside: NDArray[np.bool_] | Callable[[], NDArray[np.bool_]] | None = None,
    outside_key: Hashable | None = None,
) -> ContentDetectionResult:
    """Detect content boundaries in an image.

    The image is processed at 150 dpi: it is binarised, shadows along the edges
    are separated as garbage, the rest is grouped into blocks of text and
    trimmed from every side where what lies beyond is closer to garbage than
    to content.

    Args:
        image (numpy.ndarray of numpy.uint8): Gray or BGR image, dark content on a light
            page. Not modified.
        dpi (float): Image resolution in DPI.
        margin (int): Inset from the image edges of the box returned when no
            content is found, in pixels of ``image``.
        page_box (ContentBox | None): Area of the page in pixels of ``image``;
            anything outside it counts as shadow.
        outside (numpy.ndarray of numpy.bool | None): Pixels of ``image`` with no source
            pixel behind them, such as the corners exposed by a rotation. They
            are painted with the darkest gray level of the rest of the image,
            so that a shadow along the edge of the page still reaches the edge
            of the image. A callable is only called when the result is not
            already remembered, which needs ``outside_key``.
        outside_key (Hashable | None): Identifies the mask a callable
            ``outside`` returns for this image; equal keys must mean equal
            masks. Without it a callable is called at once.

    Returns:
        ContentDetectionResult: The detected box and a confidence from 0 to 1;
            the confidence is 0 when no content was found.
    """
    gray = np.asarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image,
        dtype=np.uint8,
    )
    deferred: Callable[[], NDArray[np.bool_]] | None = None
    mask: NDArray[np.bool_] | None = None
    if callable(outside):
        if outside_key is None:
            mask = outside()
        else:
            deferred = outside
    else:
        mask = outside
    h, w = gray.shape[:2]

    scaled_w = max(1, round(w * _TARGET_DPI / dpi))
    scaled_h = max(1, round(h * _TARGET_DPI / dpi))
    interpolation = cv2.INTER_AREA if scaled_w < w else cv2.INTER_LINEAR

    def scale(source: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return np.asarray(
            cv2.resize(source, (scaled_w, scaled_h), interpolation=interpolation),
            dtype=np.uint8,
        )

    scaled = scale(gray)
    x_scale = w / scaled_w
    y_scale = h / scaled_h

    page_rect = None
    if page_box is not None:
        page_rect = Rect(
            max(0, round(page_box.x / x_scale)),
            max(0, round(page_box.y / y_scale)),
            min(scaled_w - 1, round(page_box.right / x_scale) - 1),
            min(scaled_h - 1, round(page_box.bottom / y_scale) - 1),
        )
    if deferred is not None:

        def search() -> Rect | None:
            made = deferred()
            if made.all():
                return None
            painted = scaled
            if made.any():
                painted = scale(_paint_outside_darkest(gray, made))
            return find_content_rect(painted, page_rect)

        found = _CONTENT_MEMO.get_or_compute(
            (array_digest(scaled), page_rect, outside_key), search
        )
    else:
        everything_outside = mask is not None and bool(mask.all())
        if mask is not None and mask.any() and not everything_outside:
            scaled = scale(_paint_outside_darkest(gray, mask))
        found = (
            None
            if everything_outside
            else _CONTENT_MEMO.get_or_compute(
                (array_digest(scaled), page_rect),
                lambda: find_content_rect(scaled, page_rect),
            )
        )

    if found is None:
        return ContentDetectionResult(
            content_box=ContentBox(
                x=float(margin),
                y=float(margin),
                width=float(max(0, w - 2 * margin)),
                height=float(max(0, h - 2 * margin)),
            ),
            confidence=0.0,
        )

    left = max(0.0, found.left * x_scale)
    top = max(0.0, found.top * y_scale)
    right = min(float(w), (found.right + 1) * x_scale)
    bottom = min(float(h), (found.bottom + 1) * y_scale)
    content_box = ContentBox(x=left, y=top, width=right - left, height=bottom - top)

    # Content should be substantial but not the entire image
    area_ratio = content_box.width * content_box.height / (w * h)
    if 0.1 <= area_ratio <= 0.95:
        confidence = 1.0
    elif area_ratio < 0.1:
        confidence = area_ratio / 0.1
    else:
        confidence = (1.0 - area_ratio) / 0.05
    return ContentDetectionResult(
        content_box=content_box, confidence=max(0.0, min(1.0, confidence))
    )


def _paint_outside_darkest(
    gray: NDArray[np.uint8], outside: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    """Return ``gray`` with the ``outside`` pixels set to the darkest level inside."""
    inside = (~outside).view(np.uint8)
    darkest = np.uint8(cv2.minMaxLoc(gray, mask=inside)[0])
    painted = gray.copy()
    np.copyto(painted, darkest, where=outside)
    return painted


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
