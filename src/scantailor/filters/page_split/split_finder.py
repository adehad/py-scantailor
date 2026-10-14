"""Entry point of automatic page split detection.

The Split Pages stage calls :func:`detect_split` for each scan. It converts the
image to gray and passes it to the layout estimator, which looks for a folding
line first and for whitespace second. See
:mod:`scantailor.filters.page_split.page_layout_estimator` for the steps.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
    dpi: float = 300.0,
    bw_threshold: int | None = None,
    region: tuple[int, int, int, int] | None = None,
) -> SplitResult:
    """Detect how to split an upright page image into pages.

    Args:
        image (numpy.ndarray of numpy.uint8):
            (H, W) gray or (H, W, 3 or 4) BGR(A) image.
        layout_type (LayoutType): Requested layout; ``AUTO_LAYOUT_TYPE`` lets the
            detector choose.
        dpi (float): Resolution of the image, the same horizontally and vertically.
        bw_threshold (int | None): Gray level below which a pixel counts as black
            when whitespace is analysed. By default Otsu's threshold of the
            analysed part of the image.
        region (tuple[int, int, int, int] | None): ``(x, y, width, height)`` of
            the only part of the image to analyse, such as what a manual trim
            keeps. The layout is still in the pixels of the whole image.

    Returns:
        SplitResult: The layout, always of the requested type unless that was
        automatic.
    """
    return SplitResult(
        estimate_page_layout(to_gray(image), layout_type, dpi, bw_threshold, region)
    )
