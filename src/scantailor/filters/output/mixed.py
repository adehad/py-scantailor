"""Picture masks and layer composition for mixed output.

Mixed output keeps the gray levels or colours of pictures and binarises the
rest. This module finds the picture regions, applies manual picture zones and
combines the two parts.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scantailor.core.zones import PictureLayer, PictureZoneProperties, Zone
from scantailor.imageproc.pictures import (
    estimate_picture_mask,
    find_picture_rectangles,
)

from .picture_shape import PictureShape, PictureShapeOptions
from .raster import polygon_mask

_PURE_BLACK = 0
_PURE_WHITE = 255
# Replacements that keep pictures free of the two levels the text part uses.
_NEAR_BLACK = 1
_NEAR_WHITE = 254
_COLOR_CHANNELS = 3


def detect_picture_mask(
    gray: NDArray[np.uint8],
    dpi: tuple[int, int],
    options: PictureShapeOptions,
) -> NDArray[np.bool_]:
    """Find picture regions automatically.

    Args:
        gray (NDArray[np.uint8]): 2-D gray page. Not modified.
        dpi (tuple[int, int]): Horizontal and vertical resolution of ``gray``.
        options (PictureShapeOptions): Detection mode and sensitivity.

    Returns:
        NDArray[np.bool_]: True where there is a picture; all False when
            detection is off.
    """
    if options.shape == PictureShape.OFF:
        return np.zeros(gray.shape, dtype=bool)
    mask = estimate_picture_mask(
        gray,
        dpi,
        higher_search_sensitivity=options.higher_search_sensitivity,
    )
    if options.shape == PictureShape.FREE:
        return mask
    boxes = np.zeros(gray.shape, dtype=bool)
    for left, top, right, bottom in find_picture_rectangles(mask, options.sensitivity):
        boxes[top : bottom + 1, left : right + 1] = True
    return boxes


@dataclass(frozen=True)
class PictureLayers:
    """Union of the zones of each picture layer, as masks.

    Attributes:
        eraser1: Zones that return pixels to the binarised part first.
        foreground: Zones that keep original colours only where there is ink.
        background: Zones that keep original colours only where there is no ink.
        painter: Zones that make pixels part of a picture.
        eraser3: Zones that return pixels to the binarised part last.
    """

    eraser1: NDArray[np.bool_]
    foreground: NDArray[np.bool_]
    background: NDArray[np.bool_]
    painter: NDArray[np.bool_]
    eraser3: NDArray[np.bool_]


def rasterize_picture_zones(
    zones: Sequence[Zone],
    shape: tuple[int, int],
) -> PictureLayers:
    """Rasterise picture zones into one mask per layer.

    Args:
        zones (Sequence[Zone]): Zones in image coordinates. Fill zones, zones
            with fewer than three vertices and zones on the no-op layer do nothing.
        shape (tuple[int, int]): (height, width) of the image.

    Returns:
        PictureLayers: The layer masks.
    """
    masks = {
        layer: np.zeros(shape, dtype=bool)
        for layer in (
            PictureLayer.ERASER1,
            PictureLayer.FOREGROUND,
            PictureLayer.BACKGROUND,
            PictureLayer.PAINTER2,
            PictureLayer.ERASER3,
        )
    }
    for zone in zones:
        properties = zone.properties
        if not isinstance(properties, PictureZoneProperties) or not zone.is_valid():
            continue
        if properties.layer in masks:
            masks[properties.layer] |= polygon_mask(shape, zone.spline.to_polygon())
    return PictureLayers(
        eraser1=masks[PictureLayer.ERASER1],
        foreground=masks[PictureLayer.FOREGROUND],
        background=masks[PictureLayer.BACKGROUND],
        painter=masks[PictureLayer.PAINTER2],
        eraser3=masks[PictureLayer.ERASER3],
    )


def apply_picture_layers(
    text_region: NDArray[np.bool_],
    ink: NDArray[np.bool_],
    layers: PictureLayers,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Apply manual picture zones to the text mask and the ink.

    The passes run in a fixed order. The foreground and background passes
    compare the mask with the ink (xnor and xor) so that inside such a zone the
    text region follows the ink or its complement. Inside a picture the
    returned ink is the complement of the binarised ink and is not drawn.

    Args:
        text_region (NDArray[np.bool_]): True where output is binarised.
        ink (NDArray[np.bool_]): True where binarisation found ink.
        layers (PictureLayers): The zones by layer.

    Returns:
        tuple[NDArray[np.bool_], NDArray[np.bool_]]: New text region and the ink
            to draw inside it. Neither input is modified.
    """
    region = text_region | layers.eraser1

    region = ~(region ^ ink) & ~layers.foreground
    ink_without_foreground = ink & ~layers.foreground
    region = ~(region ^ ink)

    region = (region ^ ink_without_foreground) & ~layers.background
    region = region ^ ink_without_foreground

    region = (region & ~layers.painter) | layers.eraser3
    return region, ~(ink ^ region)


def compose_mixed(
    layer: NDArray[np.uint8],
    text_region: NDArray[np.bool_],
    ink: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Draw the binarised text over the picture layer.

    Args:
        layer (NDArray[np.uint8]): 2-D gray or (H, W, 3|4) BGR(A) original
            content. Not modified.
        text_region (NDArray[np.bool_]): True where the output is binarised.
        ink (NDArray[np.bool_]): True where binarised pixels are black.

    Returns:
        NDArray[np.uint8]: New image. Pixels in the text region are 0 or 255 in
            every colour channel; the rest keep ``layer`` except that pure black
            and white become 1 and 254.
    """
    result = _reserve_black_and_white(layer, ~text_region)
    levels = np.where(ink, _PURE_BLACK, _PURE_WHITE).astype(np.uint8)
    if result.ndim == 2:
        result[text_region] = levels[text_region]
    else:
        result[text_region, :_COLOR_CHANNELS] = levels[text_region][:, None]
    return result


def _reserve_black_and_white(
    image: NDArray[np.uint8],
    region: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Copy an image, moving pure black and white inside a region to 1 and 254."""
    result = image.copy()
    if result.ndim == 2:
        result[region & (image == _PURE_BLACK)] = _NEAR_BLACK
        result[region & (image == _PURE_WHITE)] = _NEAR_WHITE
        return result
    color = image[..., :_COLOR_CHANNELS]
    black = region & (color == _PURE_BLACK).all(axis=2)
    white = region & (color == _PURE_WHITE).all(axis=2)
    result[black, :_COLOR_CHANNELS] = _NEAR_BLACK
    result[white, :_COLOR_CHANNELS] = _NEAR_WHITE
    return result
