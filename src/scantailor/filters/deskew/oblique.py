"""Automatic oblique (shear) detection.

The page's horizontal skew is measured on a binary mask and again on a
"sausage" version of it rotated by 90 degrees. The sausage replaces every row's
ink by one run of the same length centred on the ink's mean x, which keeps how
the left-right position of the content drifts down the page but discards the
glyph detail. Measuring that drift as a skew of the rotated mask gives the
angle of the page's vertical axis; its difference to the horizontal skew is
the oblique angle.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.filters.deskew.params import MAX_OBLIQUE_DEG
from scantailor.imageproc.analysis import find_skew

_BLACK = 0
_WHITE = 255
_BINARY_THRESHOLD = 127


def build_sausage_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Replace the ink of each row by a centred run of the same length.

    Args:
        mask (numpy.ndarray of numpy.uint8):
            Binary image, 0 for ink and 255 for background.

    Returns:
        numpy.ndarray of numpy.uint8: Binary image of the same size, with the same
        convention. Rows without ink stay white.
    """
    height, width = mask.shape[:2]
    ink = mask < _BINARY_THRESHOLD
    counts = ink.sum(axis=1)
    sums = (ink * np.arange(width)).sum(axis=1)
    centers = np.divide(sums, counts, out=np.zeros(height), where=counts > 0)

    left = np.maximum(0, np.floor(centers - counts / 2.0)).astype(np.int64)
    length = np.minimum(counts, width - left)
    columns = np.arange(width)[None, :]
    filled = (columns >= left[:, None]) & (columns < (left + length)[:, None])
    filled &= (counts > 0)[:, None]
    return np.where(filled, np.uint8(_BLACK), np.uint8(_WHITE))


def rotate_mask(mask: NDArray[np.uint8], degrees: float) -> NDArray[np.uint8]:
    """Rotate a binary mask about its centre, growing the canvas to fit.

    Args:
        mask (numpy.ndarray of numpy.uint8):
            Binary image, 0 for ink and 255 for background.
        degrees (float): Clockwise rotation in degrees.

    Returns:
        numpy.ndarray of numpy.uint8: Rotated binary image; the corners are background.
    """
    if abs(degrees) < 1e-6:
        return mask
    height, width = mask.shape[:2]
    # cv2 angles are counter-clockwise for a y axis pointing up.
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -degrees, 1.0)
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=float
    )
    moved = corners @ matrix.T
    low = np.floor(moved.min(axis=0)).astype(int)
    high = np.ceil(moved.max(axis=0)).astype(int)
    out_width, out_height = (high - low).tolist()
    if out_width < 1 or out_height < 1:
        return mask
    matrix[:, 2] -= low
    rotated = cv2.warpAffine(
        mask,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_WHITE,
    )
    return np.where(rotated > _BINARY_THRESHOLD, np.uint8(_WHITE), np.uint8(_BLACK))


def _invert(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return np.ascontiguousarray(_WHITE - mask)


def find_oblique_degrees(
    mask: NDArray[np.uint8], max_oblique_deg: float = MAX_OBLIQUE_DEG
) -> float | None:
    """Estimate the oblique angle of a deskewed (horizontal) binary page.

    Each skew is weighted by confidence / (confidence + 1), so a weak
    measurement contributes less.

    Args:
        mask (numpy.ndarray of numpy.uint8):
            Binary image, 0 for ink and 255 for background,
            rotated so the text lines are horizontal.
        max_oblique_deg (float): The result is clamped to plus or minus this.

    Returns:
        float | None: Vertical-axis skew minus horizontal skew, in the
        convention of ``find_skew``; None for an empty mask. The oblique
        correction to apply is the negative of this.
    """
    if mask.size == 0:
        return None

    sausage = build_sausage_mask(mask)
    rotated = np.ascontiguousarray(np.rot90(sausage, k=-1))

    # find_skew counts the bright pixels per row and fills the corners that a
    # shear uncovers with black, so the ink has to be the bright side; otherwise
    # the uncovered corners outweigh the faint signal of the sausage edges.
    horizontal = find_skew(_invert(mask))
    vertical = find_skew(_invert(rotated))

    angle_horizontal = (
        horizontal.angle * horizontal.confidence / (horizontal.confidence + 1.0)
    )
    angle_vertical = vertical.angle * vertical.confidence / (vertical.confidence + 1.0)
    oblique = angle_vertical - angle_horizontal
    return float(max(-max_oblique_deg, min(max_oblique_deg, oblique)))
