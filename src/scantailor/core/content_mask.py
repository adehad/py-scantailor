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
from scantailor.imageproc.filters import bgr_to_gray

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
    pixels: NDArray[np.uint8], xform: ImageTransformation, black_on_white: bool = True
) -> ContentMask:
    """Find the content of a page.

    Args:
        pixels (NDArray[np.uint8]): The image ``xform`` starts from, gray or BGR.
        xform (ImageTransformation): Maps the image to the page; the mask covers
            its resulting area and ignores what its pre-crop area leaves out.
        black_on_white (bool): False for a page of light content on a dark
            background, which is inverted before its content is looked for.

    Returns:
        ContentMask: The mask, null when the page area is empty.
    """
    page = render_page_at_dpi(pixels, xform, MASK_DPI, black_on_white=black_on_white)
    if page is None:
        return ContentMask()
    ink = binarize_wolf(
        page.gray, _WOLF_WINDOW, k=_WOLF_K, lower_bound=_WOLF_LOWER_BOUND
    )
    ink[~page.inside] = 255
    kept = despeckle(ink, MASK_DPI, _DESPECKLE_LEVEL)
    return ContentMask(kept == 0, page.to_page)


@dataclass(frozen=True, eq=False)
class LowResolutionPage:
    """A page resampled to a fixed resolution in gray.

    Attributes:
        gray: The page; white where it has no source pixel.
        inside: True where the page lies inside the transformation's pre-crop
            area.
        to_page: 3x3 matrix, column-vector convention, from the coordinates of
            the source image to pixels of ``gray``.
    """

    gray: NDArray[np.uint8]
    inside: NDArray[np.bool_]
    to_page: NDArray[np.float64]


def render_page_at_dpi(
    pixels: NDArray[np.uint8],
    xform: ImageTransformation,
    dpi: int,
    *,
    black_on_white: bool = True,
    source_scale: tuple[float, float] = (1.0, 1.0),
) -> LowResolutionPage | None:
    """Resample the page an image transformation describes, at ``dpi``.

    Args:
        pixels (NDArray[np.uint8]): The image ``xform`` starts from, gray or BGR.
            Not modified.
        xform (ImageTransformation): Maps the image to the page. Not modified.
        dpi (int): Resolution of the result.
        black_on_white (bool): False inverts the image before resampling, so
            that the area with no source pixel stays white.
        source_scale (tuple[float, float]): Width and height of ``pixels``
            over those of the image ``xform`` was made for, when ``pixels`` is
            a shrunk copy of it.

    Returns:
        LowResolutionPage | None: The page, or None when its area is empty.
    """
    scaled = copy.deepcopy(xform)
    scaled.pre_scale_to_dpi(Dpi.uniform(dpi))
    rect = scaled.resulting_rect
    width, height = round(rect.width), round(rect.height)
    if width <= 0 or height <= 0:
        return None

    to_page = np.eye(3)
    to_page[0, 2], to_page[1, 2] = -rect.x, -rect.y
    to_page = to_page @ scaled.transform
    to_page = to_page @ np.diag([1 / source_scale[0], 1 / source_scale[1], 1.0])

    gray = np.ascontiguousarray(bgr_to_gray(pixels))
    if not black_on_white:
        gray = cv2.bitwise_not(gray)
    page = np.asarray(
        cv2.warpAffine(
            gray,
            to_page[:2],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        ),
        dtype=np.uint8,
    )
    inside = _inside_pre_crop_area(scaled, rect.x, rect.y, (height, width))
    return LowResolutionPage(page, inside, to_page)


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
