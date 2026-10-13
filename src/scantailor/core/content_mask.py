"""Where a page has content, at a fixed low resolution.

Select Content and Page Layout use the mask to snap a box edge or the page
content to what is really on the page when the user double clicks on it.
"""

import copy
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation
from scantailor.imageproc.binarize import binarize_wolf
from scantailor.imageproc.despeckle import despeckle

MASK_DPI = 150
_WOLF_WINDOW = 51
# Darker than this always counts as content, which keeps dim paper texture out.
_WOLF_LOWER_BOUND = 50
_WOLF_K = 0.3
_DESPECKLE_LEVEL = 2.0

_IDENTITY = np.eye(3)


@dataclass(frozen=True, eq=False)
class ContentMask:
    """Black-on-white content of a page at 150 dpi.

    Attributes:
        image: True where the page has content; empty for a null mask.
        original_to_content: 3x3 matrix, column-vector convention, from the
            coordinates of the image the mask was computed from to mask pixels.
    """

    image: NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=bool)
    )
    original_to_content: NDArray[np.float64] = field(
        default_factory=lambda: _IDENTITY.copy()
    )

    def is_null(self) -> bool:
        """Return True if the mask holds nothing, as when the page is empty."""
        return self.image.size == 0

    @property
    def content_to_original(self) -> NDArray[np.float64]:
        """The inverse of ``original_to_content``."""
        return np.asarray(np.linalg.inv(self.original_to_content), dtype=np.float64)

    def find_content_in_area(
        self, x: int, y: int, width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """Return the bounding box of the content inside an area.

        Args:
            x (int): Left of the area in mask pixels.
            y (int): Top of the area in mask pixels.
            width (int): Width of the area.
            height (int): Height of the area.

        Returns:
            tuple[int, int, int, int] | None: Left, top, width and height in mask
            pixels of the content found, grown by one pixel on every side; None
            if the area holds no content.
        """
        if self.is_null():
            return None
        rows, columns = self.image.shape
        left, top = max(x, 0), max(y, 0)
        right, bottom = min(x + width, columns), min(y + height, rows)
        if left >= right or top >= bottom:
            return None
        found = np.argwhere(self.image[top:bottom, left:right])
        if found.size == 0:
            return None
        first_row, first_column = found.min(axis=0)
        last_row, last_column = found.max(axis=0)
        return (
            left + int(first_column) - 1,
            top + int(first_row) - 1,
            int(last_column - first_column) + 3,
            int(last_row - first_row) + 3,
        )


def compute_content_mask(
    pixels: NDArray[np.uint8], xform: ImageTransformation
) -> ContentMask:
    """Find the content of a page.

    Args:
        pixels (NDArray[np.uint8]): The image ``xform`` starts from, gray or BGR.
        xform (ImageTransformation): Maps the image to the page; the mask covers
            its resulting area and ignores what its pre-crop area leaves out.

    Returns:
        ContentMask: The mask, null when the page area is empty.
    """
    scaled = copy.deepcopy(xform)
    scaled.pre_scale_to_dpi(Dpi.uniform(MASK_DPI))
    rect = scaled.resulting_rect
    width, height = round(rect.width), round(rect.height)
    if width <= 0 or height <= 0:
        return ContentMask()

    to_content = np.eye(3)
    to_content[0, 2], to_content[1, 2] = -rect.x, -rect.y
    to_content = to_content @ scaled.transform

    gray = (
        cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        if pixels.ndim == 3 and pixels.shape[2] == 3
        else np.ascontiguousarray(pixels)
    )
    page = np.asarray(
        cv2.warpAffine(
            gray,
            to_content[:2],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        ),
        dtype=np.uint8,
    )
    ink = binarize_wolf(page, _WOLF_WINDOW, k=_WOLF_K, lower_bound=_WOLF_LOWER_BOUND)
    inside = _inside_pre_crop_area(scaled, rect.x, rect.y, (height, width))
    ink[~inside] = 255
    kept = despeckle(ink, MASK_DPI, _DESPECKLE_LEVEL)
    return ContentMask(kept == 0, to_content)


def _inside_pre_crop_area(
    xform: ImageTransformation, origin_x: float, origin_y: float, shape: tuple[int, int]
) -> NDArray[np.bool_]:
    area = xform.resulting_pre_crop_area
    if len(area) < 3:
        return np.ones(shape, dtype=bool)
    polygon = np.round(area - (origin_x, origin_y)).astype(np.int32)
    inside = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(inside, [polygon], 1)
    return inside.astype(bool)
