"""Synthetic scanned pages for the Deskew tests."""

import cv2
import numpy as np
from numpy.typing import NDArray

_WORD_WIDTHS = (30, 22, 38, 16, 26, 34, 20, 28, 36, 18)
_LINE_PITCH = 34
_LINE_HEIGHT = 16
_MARGIN = 60
_WORD_GAP = 10


def text_page(height: int = 900, width: int = 700) -> NDArray[np.uint8]:
    """Return a gray page of justified-looking text lines, black on white.

    Every line has the same words, so the left and right edges of the text block
    are straight, which is what the oblique detection looks for.
    """
    page = np.full((height, width), 255, dtype=np.uint8)
    for top in range(_MARGIN, height - _MARGIN, _LINE_PITCH):
        left = _MARGIN
        for word_width in _WORD_WIDTHS * 3:
            if left + word_width > width - _MARGIN:
                break
            page[top : top + _LINE_HEIGHT, left : left + word_width] = 0
            left += word_width + _WORD_GAP
    return page


def sheared_page(
    shear_degrees: float, height: int = 900, width: int = 700
) -> NDArray[np.uint8]:
    """Return ``text_page`` with every row moved sideways by tan(angle) per pixel down.

    A positive angle leans the vertical axis of the page to the right going down;
    horizontal lines stay horizontal.
    """
    page = text_page(height, width)
    factor = np.tan(np.radians(shear_degrees))
    matrix = np.array([[1, factor, -factor * height / 2], [0, 1, 0]], dtype=np.float32)
    sheared = cv2.warpAffine(
        page, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255
    )
    return np.asarray(sheared, dtype=np.uint8)
