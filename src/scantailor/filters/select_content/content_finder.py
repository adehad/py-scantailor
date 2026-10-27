"""Content box detection."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
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
    page_box: ContentBox | None = None,
    outside: NDArray[np.bool_] | Callable[[], NDArray[np.bool_]] | None = None,
    outside_key: Hashable | None = None,
    dbg: DebugImages | None = None,
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
        dbg (DebugImages | None): Receives the finder's intermediate images.
            The finder then runs even when its result is remembered, since a
            remembered result has no images.

    Returns:
        ContentDetectionResult: The detected box and a confidence from 0 to 1.
            When no content is found, the box is empty and the confidence 0, as
            upstream; Page Layout then centres the page and Output leaves it blank.
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
                painted = scale(paint_outside_darkest(gray, made))
            return find_content_rect(painted, page_rect, dbg)

        found = (
            search()
            if dbg is not None
            else _CONTENT_MEMO.get_or_compute(
                (array_digest(scaled), page_rect, outside_key), search
            )
        )
    else:
        everything_outside = mask is not None and bool(mask.all())
        if mask is not None and mask.any() and not everything_outside:
            scaled = scale(paint_outside_darkest(gray, mask))
        if everything_outside:
            found = None
        elif dbg is not None:
            found = find_content_rect(scaled, page_rect, dbg)
        else:
            found = _CONTENT_MEMO.get_or_compute(
                (array_digest(scaled), page_rect),
                lambda: find_content_rect(scaled, page_rect),
            )

    if found is None:
        return ContentDetectionResult(content_box=ContentBox(), confidence=0.0)

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


def paint_outside_darkest(
    gray: NDArray[np.uint8], outside: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    """Return ``gray`` with the ``outside`` pixels set to the darkest level inside.

    Args:
        gray (numpy.ndarray of numpy.uint8): 2-D gray image. Not modified.
        outside (numpy.ndarray of numpy.bool): Pixels to paint; not all of them.

    Returns:
        numpy.ndarray of numpy.uint8: A painted copy of ``gray``.
    """
    inside = (~outside).view(np.uint8)
    darkest = np.uint8(cv2.minMaxLoc(gray, mask=inside)[0])
    painted = gray.copy()
    np.copyto(painted, darkest, where=outside)
    return painted
