"""Entry point of automatic page split detection.

The Split Pages stage calls :func:`detect_split` for each scan. It converts the
image to gray and passes it to the layout estimator, which looks for a folding
line first and for whitespace second. See
:mod:`scantailor.filters.page_split.page_layout_estimator` for the steps.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation, render_transformed
from scantailor.imageproc.binary_threshold import to_gray

from .layout_type import LayoutType
from .page_layout import PageLayout
from .page_layout_estimator import estimate_page_layout


@dataclass(frozen=True)
class SplitResult:
    """Result of split detection.

    Attributes:
        layout (PageLayout): The detected layout, in the pixels of the image given.
    """

    layout: PageLayout


def detect_split(
    image: NDArray[np.uint8],
    layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE,
    dpi: Dpi | float = 300.0,
    bw_threshold: int | None = None,
    region: tuple[int, int, int, int] | None = None,
    dbg: DebugImages | None = None,
) -> SplitResult:
    """Detect how to split an upright page image into pages.

    Args:
        image (numpy.ndarray of numpy.uint8):
            (H, W) gray or (H, W, 3 or 4) BGR(A) image.
        layout_type (LayoutType): Requested layout; ``AUTO_LAYOUT_TYPE`` lets the
            detector choose.
        dpi (Dpi | float): Resolution of the image; a number means the same on
            both axes. The estimator needs square pixels, so an image whose
            axes differ is analysed resampled to the smaller resolution.
        bw_threshold (int | None): Gray level below which a pixel counts as black
            when whitespace is analysed. By default Otsu's threshold of the
            analysed part of the image.
        region (tuple[int, int, int, int] | None): ``(x, y, width, height)`` of
            the only part of the image to analyse, such as what a manual trim
            keeps. The layout is still in the pixels of the whole image.
        dbg (DebugImages | None): Receives the estimator's intermediate images.

    Returns:
        SplitResult: The layout, always of the requested type unless that was
        automatic.
    """
    gray = to_gray(image)
    if not isinstance(dpi, Dpi):
        dpi = Dpi.uniform(round(dpi))
    if dpi.horizontal == dpi.vertical:
        return SplitResult(
            estimate_page_layout(
                gray, layout_type, dpi.horizontal, bw_threshold, region, dbg
            )
        )
    height, width = gray.shape
    xform = ImageTransformation.from_size(width, height, dpi)
    square = render_transformed(gray, xform)
    x_scale = square.shape[1] / width
    y_scale = square.shape[0] / height
    if region is not None:
        x, y, region_width, region_height = region
        region = (
            round(x * x_scale),
            round(y * y_scale),
            max(1, round(region_width * x_scale)),
            max(1, round(region_height * y_scale)),
        )
    layout = estimate_page_layout(
        square,
        layout_type,
        xform.pre_scaled_dpi.horizontal,
        bw_threshold,
        region,
        dbg,
    )
    return SplitResult(layout.scaled(1 / x_scale, 1 / y_scale))
