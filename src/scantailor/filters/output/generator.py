"""Output image generator for the output filter.

This module provides functions for generating final output images
with binarization, despeckling, dewarping, and other processing.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.estimate_background import normalize_illumination_gray
from scantailor.core.models import Dpi
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.imageproc.binarize import (
    binarize_bradley,
    binarize_edge_div,
    binarize_otsu,
    binarize_sauvola,
    binarize_wolf,
)
from scantailor.imageproc.despeckle import despeckle as despeckle_image

from .binarization import BinarizationMethod, BinarizationOptions
from .color_mode import ColorMode
from .color_options import ColorCommonOptions, FillingColor
from .despeckle import DespeckleLevel
from .params import Params


@dataclass
class OutputResult:
    """Result of output generation."""

    image: NDArray[np.uint8]
    is_binary: bool = False
    was_dewarped: bool = False


def generate_output(
    image: NDArray[np.uint8],
    params: Params,
    dewarper: CylindricalSurfaceDewarper | None = None,
    *,
    color_options: ColorCommonOptions | None = None,
    content_polygon: Sequence[tuple[float, float]] | None = None,
) -> OutputResult:
    """Generate the output image according to parameters.

    Args:
        image (NDArray[np.uint8]): Input image, 2-D gray or BGR(A) colour. Not modified.
        params (Params): Output parameters.
        dewarper (CylindricalSurfaceDewarper | None): Optional dewarper for
            curved page correction. Required if dewarping mode is AUTO or MANUAL.
        color_options (ColorCommonOptions | None): Options for the grayscale and
            colour paths; margin filling also applies to bilevel output.
            Defaults to ``params.color_common``.
        content_polygon (Sequence[tuple[float, float]] | None): Content area as
            (x, y) vertices in the coordinates of the image after dewarping.
            Margins outside it are filled when ``fill_margins`` is set. None
            means the whole image is content.

    Returns:
        OutputResult with the processed image.
    """
    color_options = color_options or params.color_common
    processed = image
    was_dewarped = False

    # Apply dewarping if enabled and dewarper provided
    if params.needs_dewarping() and dewarper is not None:
        processed = _apply_dewarping(processed, params, dewarper)
        was_dewarped = True

    # Convert based on color mode
    if params.color_mode == ColorMode.BLACK_AND_WHITE:
        result = _generate_binary_output(
            processed, params, color_options, content_polygon
        )
    elif params.color_mode == ColorMode.COLOR_GRAYSCALE:
        result = _generate_grayscale_output(
            processed, params, color_options, content_polygon
        )
    else:
        result = _generate_mixed_output(
            processed, params, color_options, content_polygon
        )

    result.was_dewarped = was_dewarped
    return result


def _apply_dewarping(
    image: NDArray[np.uint8],
    params: Params,
    dewarper: CylindricalSurfaceDewarper,
) -> NDArray[np.uint8]:
    """Apply dewarping to an image.

    Args:
        image: Input image.
        params: Output parameters with dewarping options.
        dewarper: The dewarper model.

    Returns:
        Dewarped image.
    """
    from scantailor.dewarping.raster_dewarper import compute_dewarped_size, dewarp_image

    # Compute output size based on source size and dewarper
    src_size = (image.shape[1], image.shape[0])  # (width, height)
    dst_size = compute_dewarped_size(src_size, dewarper)

    # Determine background color based on image type
    if len(image.shape) == 2:
        background = 255  # White for grayscale
    elif image.shape[2] == 3:
        background = (255, 255, 255)  # White for RGB
    else:
        background = (255, 255, 255, 255)  # White for RGBA

    # Apply dewarping with full model domain
    dewarped = dewarp_image(
        image,
        dewarper,
        dst_size=dst_size,
        model_domain=(0.0, 0.0, 1.0, 1.0),
        background_color=background,
    )

    # Apply post-deskew if requested
    if params.dewarping.post_deskew and params.dewarping.post_deskew_angle != 0.0:
        dewarped = _apply_post_deskew(dewarped, params.dewarping.post_deskew_angle)

    return dewarped


def _apply_post_deskew(
    image: NDArray[np.uint8],
    angle: float,
) -> NDArray[np.uint8]:
    """Apply post-dewarping deskew rotation.

    Args:
        image: Dewarped image.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding box size
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix for the new size
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Determine background color
    if len(image.shape) == 2:
        border_value = 255
    else:
        border_value = (255,) * image.shape[2]

    # Apply rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return np.asarray(rotated, dtype=np.uint8)


def _generate_binary_output(
    image: NDArray[np.uint8],
    params: Params,
    color_options: ColorCommonOptions,
    content_polygon: Sequence[tuple[float, float]] | None,
) -> OutputResult:
    """Generate black and white output.

    Args:
        image (NDArray[np.uint8]): Input image, 2-D gray or BGR(A).
        params (Params): Output parameters.
        color_options (ColorCommonOptions): Margin filling options.
        content_polygon (Sequence[tuple[float, float]] | None): Content area.

    Returns:
        OutputResult: Binary image (uint8, 0 or 255) of the input's height and width.
    """
    if not params.black_on_white:
        image = _invert_polarity(image)
    gray = _to_gray(image)

    if params.binarization.normalize_illumination:
        gray = normalize_illumination_gray(gray)

    binary = _binarize(gray, params.binarization)

    if color_options.fill_margins and content_polygon is not None:
        margin = 0 if color_options.filling_color == FillingColor.BLACK else 255
        binary = _fill_outside(binary, content_polygon, margin)

    if params.binarization.morphological_smoothing:
        binary = _morphological_smooth(binary)

    # Despeckling stays the last step that changes which pixels are ink.
    if params.despeckle_level.is_enabled():
        binary = _despeckle(binary, params.despeckle_level, params.output_dpi)

    if not params.black_on_white:
        binary = 255 - binary

    return OutputResult(image=binary, is_binary=True)


def _generate_grayscale_output(
    image: NDArray[np.uint8],
    params: Params,
    color_options: ColorCommonOptions,
    content_polygon: Sequence[tuple[float, float]] | None,
) -> OutputResult:
    """Generate grayscale or colour output.

    With illumination normalisation on, the background is estimated from the
    gray version of the page and, for colour input, the normalised brightness is
    transferred back onto the original colour differences. Margins outside the
    content area are then filled and the result is inverted for white-on-black
    output.

    Args:
        image (NDArray[np.uint8]): Input image, 2-D gray or BGR(A).
        params (Params): Output parameters.
        color_options (ColorCommonOptions): Normalisation and margin options.
        content_polygon (Sequence[tuple[float, float]] | None): Content area.

    Returns:
        OutputResult: New uint8 array with the shape of the input.
    """
    if not params.black_on_white:
        image = _invert_polarity(image)
    result = image.copy()
    if color_options.normalize_illumination:
        brightness = normalize_illumination_gray(_to_gray(image))
        result = (
            brightness if image.ndim == 2 else _replace_brightness(image, brightness)
        )

    if color_options.fill_margins and content_polygon is not None:
        fill = _margin_color(result, params, color_options, content_polygon)
        result = _fill_outside(result, content_polygon, fill)

    if not params.black_on_white:
        result = _invert_polarity(result)

    return OutputResult(image=result, is_binary=False)


def _generate_mixed_output(
    image: NDArray[np.uint8],
    params: Params,
    color_options: ColorCommonOptions,
    content_polygon: Sequence[tuple[float, float]] | None,
) -> OutputResult:
    """Generate mixed mode output.

    Picture zone detection is not implemented, so the whole page is binarised.

    Args:
        image (NDArray[np.uint8]): Input image.
        params (Params): Output parameters.
        color_options (ColorCommonOptions): Margin filling options.
        content_polygon (Sequence[tuple[float, float]] | None): Content area.

    Returns:
        OutputResult: Binary result.
    """
    return _generate_binary_output(image, params, color_options, content_polygon)


# Luma weights (R, G, B) in 1/32 units.
_WEIGHT_R = 11
_WEIGHT_G = 16
_WEIGHT_B = 5
_WEIGHT_SCALE = 32


def _to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR(A) to gray with weights 11/16/5 over 32.

    Args:
        image (NDArray[np.uint8]): 2-D gray or (H, W, 3|4) BGR(A) image.

    Returns:
        NDArray[np.uint8]: 2-D gray image. Aliases ``image`` for gray input.
    """
    if image.ndim == 2:
        return image
    blue = image[..., 0].astype(np.uint16)
    green = image[..., 1].astype(np.uint16)
    red = image[..., 2].astype(np.uint16)
    luma = (_WEIGHT_R * red + _WEIGHT_G * green + _WEIGHT_B * blue) // _WEIGHT_SCALE
    return luma.astype(np.uint8)


def _invert_polarity(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return a copy with gray levels or colour channels inverted; alpha is kept."""
    inverted = image.copy()
    if image.ndim == 2:
        inverted[:] = 255 - image
    else:
        inverted[..., :3] = 255 - image[..., :3]
    return inverted


def _replace_brightness(
    image: NDArray[np.uint8],
    brightness: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Give a colour image a new luma while keeping its colour differences.

    Args:
        image (NDArray[np.uint8]): BGR(A) image. Not modified.
        brightness (NDArray[np.uint8]): Target luma, same height and width.

    Returns:
        NDArray[np.uint8]: New image; an alpha channel is carried over unchanged.
    """
    result = image.copy()
    color = image[..., :3].astype(np.float32)
    luma = (
        _WEIGHT_R * color[..., 2]
        + _WEIGHT_G * color[..., 1]
        + _WEIGHT_B * color[..., 0]
    ) / _WEIGHT_SCALE
    shifted = color + (brightness.astype(np.float32) - luma)[..., None]
    result[..., :3] = np.clip(np.floor(shifted + 0.5), 0, 255).astype(np.uint8)
    return result


_MIN_POLYGON_VERTICES = 3


def _polygon_mask(
    shape: tuple[int, int],
    polygon: Sequence[tuple[float, float]],
) -> NDArray[np.bool_]:
    """Rasterise a polygon given as (x, y) vertices.

    Args:
        shape (tuple[int, int]): (height, width) of the mask.
        polygon (Sequence[tuple[float, float]]): Vertices in pixel coordinates.

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


def _fill_outside(
    image: NDArray[np.uint8],
    polygon: Sequence[tuple[float, float]],
    color: int | tuple[int, ...],
) -> NDArray[np.uint8]:
    """Return a copy with everything outside the polygon set to a colour."""
    inside = _polygon_mask(image.shape[:2], polygon)
    result = image.copy()
    result[~inside] = color
    return result


def _margin_color(
    image: NDArray[np.uint8],
    params: Params,
    color_options: ColorCommonOptions,
    content_polygon: Sequence[tuple[float, float]],
) -> int | tuple[int, ...]:
    """Pick the margin colour before any white-on-black inversion.

    Args:
        image (NDArray[np.uint8]): Image whose content decides the dominant colour.
        params (Params): Output parameters; ``black_on_white`` flips white and black.
        color_options (ColorCommonOptions): Filling colour choice.
        content_polygon (Sequence[tuple[float, float]]): Content area.

    Returns:
        int | tuple[int, ...]: A gray level, or one value per channel for colour images.
    """
    if color_options.filling_color == FillingColor.BACKGROUND:
        return _dominant_color(image, _polygon_mask(image.shape[:2], content_polygon))
    wants_white = (
        color_options.filling_color == FillingColor.WHITE
    ) == params.black_on_white
    level = 255 if wants_white else 0
    return level if image.ndim == 2 else (level,) * image.shape[2]


def _dominant_color(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
) -> int | tuple[int, ...]:
    """Most frequent value of each channel over the masked pixels."""
    pixels = image[mask] if mask.any() else image.reshape(-1, *image.shape[2:])
    if image.ndim == 2:
        return int(np.bincount(pixels, minlength=256).argmax())
    return tuple(
        int(np.bincount(pixels[:, c], minlength=256).argmax())
        for c in range(image.shape[2])
    )


def _binarize(
    gray: NDArray[np.uint8],
    options: BinarizationOptions,
) -> NDArray[np.uint8]:
    """Apply binarization to a grayscale image.

    Args:
        gray: Grayscale input image.
        options: Binarization options.

    Returns:
        Binary image (0 or 255).
    """
    # Apply threshold adjustment
    if options.threshold_adjustment != 0:
        # Adjust by shifting pixel values before binarization
        adjusted = gray.astype(np.int16) - options.threshold_adjustment
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    else:
        adjusted = gray

    if options.method == BinarizationMethod.OTSU:
        return binarize_otsu(adjusted)
    if options.method == BinarizationMethod.SAUVOLA:
        return binarize_sauvola(
            adjusted,
            window_size=options.window_size,
            k=options.sauvola_coef,
        )
    if options.method == BinarizationMethod.WOLF:
        return binarize_wolf(
            adjusted,
            window_size=options.window_size,
            k=options.wolf_coef,
        )
    if options.method == BinarizationMethod.BRADLEY:
        return binarize_bradley(
            adjusted,
            window_size=options.window_size,
            k=options.bradley_coef,
        )
    # EdgeDiv method
    return binarize_edge_div(
        adjusted,
        window_size=options.window_size,
        kep=options.edge_div_kep,
        kbd=options.edge_div_kbd,
    )


def _despeckle(
    binary: NDArray[np.uint8],
    level: DespeckleLevel,
    dpi: Dpi,
) -> NDArray[np.uint8]:
    """Remove small noise components that are not near text.

    Args:
        binary (NDArray[np.uint8]): Binary image, 0 is ink. Not modified.
        level (DespeckleLevel): Despeckle level.
        dpi (Dpi): Resolution of ``binary``; the smaller axis sets the thresholds.

    Returns:
        NDArray[np.uint8]: Despeckled binary image.
    """
    if not level.is_enabled():
        return binary
    return despeckle_image(binary, min(dpi.horizontal, dpi.vertical), level.to_factor())


def _morphological_smooth(binary: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply morphological smoothing to a binary image.

    This helps smooth jagged edges.

    Args:
        binary: Binary input image.

    Returns:
        Smoothed binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Opening removes small protrusions
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Closing fills small holes
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return np.asarray(result, dtype=np.uint8)
