"""Picture masks and layer composition for mixed output.

Mixed output keeps the gray levels or colours of pictures and binarises the
rest. This module finds the picture regions, applies manual picture zones and
combines the two parts. It also holds the steps for colouring the ink and for
keeping the original background, which the generator applies in binary and
mixed output.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.zones import (
    PictureLayer,
    PictureZoneProperties,
    Zone,
    ZoneCategory,
    ZoneSpline,
)
from scantailor.imageproc.color_segmenter import segment_colors
from scantailor.imageproc.filters import wiener_color_filter
from scantailor.imageproc.pictures import (
    estimate_picture_mask,
    find_picture_rectangles,
)
from scantailor.imageproc.posterizer import posterize

from .binarization import ColorSegmenterOptions
from .color_options import ColorCommonOptions, PosterizationOptions
from .picture_shape import PictureShape, PictureShapeOptions
from .raster import polygon_mask
from .render_params import RenderParams

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
    dbg: DebugImages | None = None,
) -> NDArray[np.bool_]:
    """Find free-form picture regions.

    Rectangular pictures are not part of the mask: they become auto picture
    zones (see ``find_auto_picture_zones``), which the user can edit.

    Args:
        gray (numpy.ndarray of numpy.uint8): 2-D gray page. Not modified.
        dpi (tuple[int, int]): Horizontal and vertical resolution of ``gray``.
        options (PictureShapeOptions): Detection mode and sensitivity.
        dbg (DebugImages | None): Receives the picture detector's steps.

    Returns:
        numpy.ndarray of numpy.bool: True where there is a picture; all False
            unless the shape is free.
    """
    if options.shape != PictureShape.FREE:
        return np.zeros(gray.shape, dtype=bool)
    return estimate_picture_mask(
        gray,
        dpi,
        higher_search_sensitivity=options.higher_search_sensitivity,
        dbg=dbg,
    )


def find_auto_picture_zones(
    gray: NDArray[np.uint8],
    dpi: tuple[int, int],
    options: PictureShapeOptions,
    dbg: DebugImages | None = None,
) -> list[Zone]:
    """Find rectangular pictures and return them as auto picture zones.

    Each zone adds its rectangle to the pictures, as a zone on the painter
    layer does.

    Args:
        gray (numpy.ndarray of numpy.uint8): 2-D gray page. Not modified.
        dpi (tuple[int, int]): Horizontal and vertical resolution of ``gray``.
        options (PictureShapeOptions): Sensitivity and search options; the
            shape is taken to be rectangular.
        dbg (DebugImages | None): Receives the picture detector's steps.

    Returns:
        list[Zone]: One zone per picture, in the pixel coordinates of ``gray``.
            The corners lie on the outermost picture pixels, which is what
            ``polygon_mask`` fills.
    """
    mask = estimate_picture_mask(
        gray,
        dpi,
        higher_search_sensitivity=options.higher_search_sensitivity,
        dbg=dbg,
    )
    properties = PictureZoneProperties(
        layer=PictureLayer.PAINTER2, category=ZoneCategory.AUTO
    )
    return [
        Zone(
            spline=ZoneSpline(
                points=np.array(
                    [(left, top), (right, top), (right, bottom), (left, bottom)],
                    dtype=np.float64,
                )
            ),
            properties=properties,
        )
        for left, top, right, bottom in find_picture_rectangles(
            mask, options.sensitivity
        )
    ]


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
        text_region (numpy.ndarray of numpy.bool): True where output is binarised.
        ink (numpy.ndarray of numpy.bool): True where binarisation found ink.
        layers (PictureLayers): The zones by layer.

    Returns:
        tuple[numpy.ndarray of numpy.bool, numpy.ndarray of numpy.bool]:
            New text region and the ink
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
        layer (numpy.ndarray of numpy.uint8): 2-D gray or (H, W, 3|4) BGR(A) original
            content. Not modified.
        text_region (numpy.ndarray of numpy.bool): True where the output is binarised.
        ink (numpy.ndarray of numpy.bool): True where binarised pixels are black.

    Returns:
        numpy.ndarray of numpy.uint8:
            New image. Pixels in the text region are 0 or 255 in
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


def build_mixed_layer(
    original: NDArray[np.uint8],
    normalized: NDArray[np.uint8] | None,
    render: RenderParams,
    options: ColorCommonOptions,
) -> NDArray[np.uint8]:
    """Pick the picture layer of a mixed page.

    With binarisation on, the layer is the untouched original unless colour
    illumination normalisation is on. Only then does it get the Wiener colour
    filter, and it is the normalised page only when the binarisation also
    normalises illumination.

    =============  =============  ===============================
    colour norm    binarisation   layer
    =============  =============  ===============================
    off            any            original, no Wiener filter
    on             on             normalised, Wiener filtered
    on             off            original, Wiener filtered
    =============  =============  ===============================

    Without binarisation (colour foreground), the layer is the normalised page
    if ``render.normalize_illumination`` says so, otherwise the original, and
    is always Wiener filtered.

    Args:
        original (numpy.ndarray of numpy.uint8):
            Original page, gray or (H, W, 3|4) BGR(A).
        normalized (numpy.ndarray of numpy.uint8 | None): The page with illumination
            normalised, same shape; only read when the layer is the normalised page.
        render (RenderParams): Flags of the page.
        options (ColorCommonOptions): Wiener filter settings.

    Returns:
        numpy.ndarray of numpy.uint8: ``original`` itself when it is used unchanged,
            otherwise a new array. Callers must not modify it.

    Raises:
        ValueError: If the layer must be normalised but ``normalized`` is None.
    """
    if render.need_binarization and not render.normalize_illumination_color:
        return original
    layer = original
    if render.normalize_illumination:
        if normalized is None:
            msg = "normalized page is required when illumination is normalised"
            raise ValueError(msg)
        layer = normalized
    return wiener_color_filter(layer, options.wiener_window_size, options.wiener_coef)


def background_lightness(color: int | tuple[int, ...]) -> int:
    """Lightness of a colour on the 0 to 255 scale, the posterizer's white level.

    Args:
        color (int | tuple[int, ...]): A gray level, or BGR(A) levels.

    Returns:
        int: Mean of the largest and smallest colour channel, halves rounded up.
    """
    if isinstance(color, int):
        return color
    channels = color[:_COLOR_CHANNELS]
    return (max(channels) + min(channels) + 1) // 2


def colorize_ink(
    image: NDArray[np.uint8],
    ink: NDArray[np.bool_],
    dpi: tuple[int, int],
    segmenter: ColorSegmenterOptions,
    posterization: PosterizationOptions | None = None,
    background_color: int | tuple[int, ...] | None = None,
) -> NDArray[np.uint8]:
    """Colour the ink with the colours of the original and optionally posterise it.

    Args:
        image (numpy.ndarray of numpy.uint8): Original page, gray or (H, W, 3|4) BGR(A).
            Not modified.
        ink (numpy.ndarray of numpy.bool): True where binarisation found ink.
        dpi (tuple[int, int]): Horizontal and vertical resolution of ``image``.
        segmenter (ColorSegmenterOptions): Noise reduction and thresholds.
        posterization (PosterizationOptions | None): Posterise the result with
            these options when given and enabled.
        background_color (int | tuple[int, ...] | None): Page background, whose
            lightness is the white level when stretching colours before
            posterising; white when None.

    Returns:
        numpy.ndarray of numpy.uint8:
            Ink in colour on white paper, same shape as ``image``.
    """
    segmented = segment_colors(
        ink,
        image,
        dpi,
        segmenter.noise_reduction,
        (
            segmenter.red_threshold_adjustment,
            segmenter.green_threshold_adjustment,
            segmenter.blue_threshold_adjustment,
        ),
    )
    if posterization is None or not posterization.enabled:
        return segmented
    white_level = (
        _PURE_WHITE
        if background_color is None
        else background_lightness(background_color)
    )
    return posterize(
        segmented,
        posterization.level,
        normalize=posterization.normalization_enabled,
        force_black_and_white=posterization.force_black_and_white,
        normalize_white_level=white_level,
    )


def compose_mixed_colored(
    layer: NDArray[np.uint8],
    text_region: NDArray[np.bool_],
    colored_text: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Put coloured text, from ``colorize_ink``, over the picture layer.

    Args:
        layer (numpy.ndarray of numpy.uint8): Picture layer, gray or (H, W, 3|4) BGR(A).
            Not modified.
        text_region (numpy.ndarray of numpy.bool):
            True where the text replaces the layer.
        colored_text (numpy.ndarray of numpy.uint8):
            Text on white paper, same shape as ``layer``.

    Returns:
        numpy.ndarray of numpy.uint8: New image. Outside the text region pure black and
            white of the layer become 1 and 254.
    """
    result = _reserve_black_and_white(layer, ~text_region)
    result[text_region] = colored_text[text_region]
    return result


def compose_over_original(
    layer: NDArray[np.uint8],
    ink: NDArray[np.bool_],
    colored_text: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Draw ink over the original page, which stays as the background.

    Args:
        layer (numpy.ndarray of numpy.uint8): Original page, gray or (H, W, 3|4) BGR(A).
            Not modified.
        ink (numpy.ndarray of numpy.bool): True where ink is drawn.
        colored_text (numpy.ndarray of numpy.uint8 | None):
            Ink colours from ``colorize_ink``;
            pure white pixels of it are not drawn. Ink is black when None.

    Returns:
        numpy.ndarray of numpy.uint8:
            New image. Pure black and white of the page become 1
            and 254 away from the ink, so only ink is exactly black.
    """
    result = layer.copy()
    if colored_text is None:
        drawn = ink
        source = np.zeros_like(layer)
    else:
        colors = (
            colored_text[..., :_COLOR_CHANNELS]
            if colored_text.ndim == 3
            else colored_text
        )
        is_color = colors != _PURE_WHITE
        drawn = ink & (is_color if is_color.ndim == 2 else is_color.any(axis=2))
        source = colored_text
    if result.ndim == 2:
        result[drawn] = source[drawn]
    else:
        result[drawn, :_COLOR_CHANNELS] = source[drawn, :_COLOR_CHANNELS]
    return _reserve_black_and_white(result, ~ink)
