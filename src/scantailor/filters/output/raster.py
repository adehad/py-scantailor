"""Polygon rasterisation and filling shared by the output generator modules."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

_MIN_POLYGON_VERTICES = 3


def polygon_mask(
    shape: tuple[int, int],
    polygon: Sequence[tuple[float, float]] | NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Rasterise a polygon given as (x, y) vertices.

    Args:
        shape (tuple[int, int]): (height, width) of the mask.
        polygon (Sequence[tuple[float, float]] | NDArray[np.float64]): Vertices
            in pixel coordinates, as pairs or an (N, 2) array.

    Returns:
        NDArray[np.bool_]: True inside the polygon. Fewer than three vertices
            enclose nothing, so the whole mask is True (no polygon).
    """
    if len(polygon) < _MIN_POLYGON_VERTICES:
        return np.ones(shape, dtype=bool)
    mask = np.zeros(shape, dtype=np.uint8)
    vertices = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [vertices], 1)
    return mask.astype(bool)


def encloses_no_pixel(
    polygon: Sequence[tuple[float, float]] | NDArray[np.float64],
) -> bool:
    """Tell whether a polygon is too small to cover a pixel.

    Its bounding box, rounded to whole pixels as upstream rounds the content
    area, is empty. Fewer than three vertices is no polygon at all, which
    ``polygon_mask`` takes for the whole image, so it is not too small.

    Args:
        polygon (Sequence[tuple[float, float]] | NDArray[np.float64]): Vertices
            in pixel coordinates, as pairs or an (N, 2) array.

    Returns:
        bool: True when the polygon covers no pixel.
    """
    if len(polygon) < _MIN_POLYGON_VERTICES:
        return False
    width, height = np.ptp(np.asarray(polygon, dtype=np.float64), axis=0)
    return round(width) == 0 or round(height) == 0


def fill_masked(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    color: int | tuple[int, ...],
) -> NDArray[np.uint8]:
    """Return a copy of an image with the masked pixels set to a colour.

    Args:
        image (NDArray[np.uint8]): 2-D gray or (H, W, C) image. Not modified.
        mask (NDArray[np.bool_]): True for the pixels to set; (H, W).
        color (int | tuple[int, ...]): A gray level, or one value per channel.

    Returns:
        NDArray[np.uint8]: The filled copy.
    """
    result = image.copy()
    result[mask] = color
    return result
