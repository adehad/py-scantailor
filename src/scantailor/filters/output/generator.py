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
from scantailor.core.zones import FillZoneProperties, Zone
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.imageproc.background_color import dominant_background_color
from scantailor.imageproc.binarize import (
    binarize_bradley,
    binarize_edge_div,
    binarize_otsu,
    binarize_sauvola,
    binarize_wolf,
)
from scantailor.imageproc.despeckle import despeckle as despeckle_image
from scantailor.imageproc.filters import (
    bgr_to_gray,
    savgol_filter_2d,
    wiener_color_filter,
    wiener_filter,
)
from scantailor.imageproc.posterizer import posterize

from .binarization import BinarizationMethod, BinarizationOptions
from .color_mode import ColorMode
from .color_options import ColorCommonOptions, FillingColor
from .despeckle import DespeckleLevel
from .mixed import (
    apply_picture_layers,
    compose_mixed,
    detect_picture_mask,
    rasterize_picture_zones,
)
from .params import Params
from .raster import fill_masked, polygon_mask
from .render_params import RenderParams

Polygon = Sequence[tuple[float, float]]

# Window side and polynomial degree of the smoothing before binarisation, by the
# smaller output DPI: (largest DPI of the band, window, degree).
_SMOOTHING_BANDS = ((200, 5, 3), (400, 7, 4), (800, 11, 4))
_SMOOTHING_ABOVE_BANDS = (11, 2)
_BILEVEL_GRAY_SPLIT = 128
_COLOR_CHANNELS = 3


@dataclass
class OutputResult:
    """Result of output generation.

    Attributes:
        image: The output image, uint8, 2-D gray or BGR(A) like the input.
        is_binary: True when every pixel is 0 or 255 in a single channel.
        was_dewarped: True when the dewarper was applied.
        picture_mask: Mixed mode only; True where the output keeps the original
            gray levels or colours instead of black and white. None otherwise.
    """

    image: NDArray[np.uint8]
    is_binary: bool = False
    was_dewarped: bool = False
    picture_mask: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class _PageLayout:
    """Where the page and its kept content lie in the image.

    Attributes:
        page_outline: The page outline as given, or None.
        page: True inside the page outline, or None when it covers the image.
        keep: True where pixels are kept; everything else gets the margin colour.
            None when nothing is to be filled.
    """

    page_outline: Polygon | None
    page: NDArray[np.bool_] | None
    keep: NDArray[np.bool_] | None

    @classmethod
    def build(
        cls,
        shape: tuple[int, int],
        render: RenderParams,
        content_polygon: Polygon | None,
        page_outline: Polygon | None,
    ) -> "_PageLayout":
        """Rasterise the page outline and the mask of the area to keep.

        Args:
            shape (tuple[int, int]): (height, width) of the image.
            render (RenderParams): Decides which fills are on.
            content_polygon (Polygon | None): Content area.
            page_outline (Polygon | None): Page outline.

        Returns:
            _PageLayout: The layout.
        """
        page = None if page_outline is None else polygon_mask(shape, page_outline)
        keep = page if render.fill_offcut else None
        if render.fill_margins and content_polygon is not None:
            content = polygon_mask(shape, content_polygon)
            keep = content if keep is None else keep & content
        return cls(page_outline, page, keep)


def generate_output(
    image: NDArray[np.uint8],
    params: Params,
    dewarper: CylindricalSurfaceDewarper | None = None,
    *,
    color_options: ColorCommonOptions | None = None,
    content_polygon: Polygon | None = None,
    page_outline: Polygon | None = None,
    picture_zones: Sequence[Zone] = (),
    fill_zones: Sequence[Zone] = (),
) -> OutputResult:
    """Generate the output image according to parameters.

    Polygons and zones are in the coordinates of the image after dewarping.

    Args:
        image (NDArray[np.uint8]): Input image, 2-D gray or BGR(A) colour. Not modified.
        params (Params): Output parameters.
        dewarper (CylindricalSurfaceDewarper | None): Optional dewarper for
            curved page correction. Required if dewarping mode is AUTO or MANUAL.
        color_options (ColorCommonOptions | None): Options for the grayscale and
            colour paths; margin filling also applies to bilevel output.
            Defaults to ``params.color_common``.
        content_polygon (Polygon | None): Content area as (x, y) vertices.
            Margins outside it are filled when ``fill_margins`` is set. None
            means the whole image is content.
        page_outline (Polygon | None): Outline of the page as (x, y) vertices.
            What lies outside it is filled when ``fill_offcut`` is set. None
            means the whole image is page.
        picture_zones (Sequence[Zone]): Manual picture zones, used by mixed mode.
        fill_zones (Sequence[Zone]): Fill zones, painted with their own colour
            after everything else.

    Returns:
        OutputResult with the processed image.
    """
    color_options = color_options or params.color_common
    params = params.with_color_common(color_options)
    render = RenderParams.from_params(params)
    processed = image
    was_dewarped = False

    # Apply dewarping if enabled and dewarper provided
    if params.needs_dewarping() and dewarper is not None:
        processed = _apply_dewarping(processed, params, dewarper)
        was_dewarped = True

    layout = _PageLayout.build(
        processed.shape[:2], render, content_polygon, page_outline
    )
    if params.color_mode == ColorMode.BLACK_AND_WHITE:
        result = _generate_binary_output(processed, params, render, layout)
    elif params.color_mode == ColorMode.COLOR_GRAYSCALE:
        result = _generate_grayscale_output(processed, params, render, layout)
    else:
        result = _generate_mixed_output(
            processed, params, render, layout, picture_zones
        )

    if fill_zones:
        result = _apply_fill_zones(result, fill_zones)
    if render.posterize and not render.need_binarization:
        result.image = _posterize(result.image, color_options)
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
    render: RenderParams,
    layout: _PageLayout,
) -> OutputResult:
    """Generate black and white output.

    Args:
        image (NDArray[np.uint8]): Input image, 2-D gray or BGR(A).
        params (Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page and fill areas.

    Returns:
        OutputResult: Binary image (uint8, 0 or 255) of the input's height and width.
    """
    if not params.black_on_white:
        image = _invert_polarity(image)
    gray = bgr_to_gray(image)
    if render.normalize_illumination:
        gray = normalize_illumination_gray(gray, layout.page_outline)
    gray = _wiener_gray(gray, params.color_common)

    binary = _binarize_page(gray, params, render)

    if layout.keep is not None:
        margin = 0 if params.color_common.filling_color == FillingColor.BLACK else 255
        binary = fill_masked(binary, ~layout.keep, margin)

    if render.morphological_smoothing:
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
    render: RenderParams,
    layout: _PageLayout,
) -> OutputResult:
    """Generate grayscale or colour output.

    With illumination normalisation on, the background is estimated from the
    gray version of the page and, for colour input, the normalised brightness is
    transferred back onto the original colour differences. The Wiener colour
    filter runs next, then margins and offcut are filled, the result is
    inverted for white-on-black output. Posterizing is left to the caller so that
    fill zones are posterized too.

    Args:
        image (NDArray[np.uint8]): Input image, 2-D gray or BGR(A).
        params (Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page and fill areas.

    Returns:
        OutputResult: New uint8 array with the shape of the input.
    """
    color_options = params.color_common
    if not params.black_on_white:
        image = _invert_polarity(image)
    result = image
    if render.normalize_illumination:
        brightness = normalize_illumination_gray(
            bgr_to_gray(image), layout.page_outline
        )
        result = _with_brightness(image, brightness)
    result = wiener_color_filter(
        result, color_options.wiener_window_size, color_options.wiener_coef
    )

    if layout.keep is not None:
        fill = _margin_color(result, params, layout)
        result = fill_masked(result, ~layout.keep, fill)

    if not params.black_on_white:
        result = _invert_polarity(result)

    return OutputResult(image=result, is_binary=False)


def _generate_mixed_output(
    image: NDArray[np.uint8],
    params: Params,
    render: RenderParams,
    layout: _PageLayout,
    picture_zones: Sequence[Zone],
) -> OutputResult:
    """Generate mixed output: pictures keep their grays or colours, text is binarised.

    Picture regions come from automatic detection (see ``PictureShapeOptions``)
    and from manual zones, whose layers add to, remove from or invert the
    detected picture mask. Everything outside the kept area is filled with
    white, or with black when the filling colour says so.

    Args:
        image (NDArray[np.uint8]): Input image, 2-D gray or BGR(A).
        params (Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page and fill areas.
        picture_zones (Sequence[Zone]): Manual picture zones.

    Returns:
        OutputResult: New uint8 array with the shape of the input. Text pixels
            are 0 or 255 in every channel and picture pixels are never pure
            black or white.
    """
    color_options = params.color_common
    if not params.black_on_white:
        image = _invert_polarity(image)

    raw_gray = bgr_to_gray(image)
    normalized = raw_gray
    if render.normalize_illumination or render.normalize_illumination_color:
        normalized = normalize_illumination_gray(raw_gray, layout.page_outline)
    text_gray = _wiener_gray(
        normalized if render.normalize_illumination else raw_gray, color_options
    )

    dpi = params.output_dpi
    picture = detect_picture_mask(
        text_gray, (dpi.horizontal, dpi.vertical), params.picture_shape
    )
    binary = _binarize_page(text_gray, params, render)
    if layout.keep is not None:
        binary = fill_masked(binary, ~layout.keep, 255)
    if render.morphological_smoothing:
        binary = _morphological_smooth(binary)

    text_region, ink = apply_picture_layers(
        ~picture, binary == 0, rasterize_picture_zones(picture_zones, binary.shape)
    )
    if layout.keep is not None:
        text_region |= ~layout.keep
        ink &= layout.keep
    if params.despeckle_level.is_enabled():
        speckled = np.where(ink, 0, 255).astype(np.uint8)
        ink = _despeckle(speckled, params.despeckle_level, dpi) == 0

    layer = image
    if render.normalize_illumination_color:
        layer = _with_brightness(image, normalized)
    layer = wiener_color_filter(
        layer, color_options.wiener_window_size, color_options.wiener_coef
    )
    result = compose_mixed(layer, text_region, ink)

    if layout.keep is not None:
        margin = 0 if color_options.filling_color == FillingColor.BLACK else 255
        result = fill_masked(result, ~layout.keep, _per_channel(margin, result))
    if not params.black_on_white:
        result = _invert_polarity(result)
    return OutputResult(image=result, is_binary=False, picture_mask=~text_region)


def _apply_fill_zones(
    result: OutputResult,
    zones: Sequence[Zone],
) -> OutputResult:
    """Paint fill zones over an output.

    Text pixels of bilevel output are painted black or white by the zone
    colour's gray level; gray and colour pixels take the colour itself.

    Args:
        result (OutputResult): Output to paint. Not modified.
        zones (Sequence[Zone]): Zones in output coordinates; non-fill zones and
            zones with fewer than three vertices are skipped.

    Returns:
        OutputResult: A painted copy.
    """
    image = result.image.copy()
    shape = image.shape[:2]
    if result.is_binary:
        bilevel = np.ones(shape, dtype=bool)
    elif result.picture_mask is not None:
        bilevel = ~result.picture_mask
    else:
        bilevel = np.zeros(shape, dtype=bool)

    for zone in zones:
        properties = zone.properties
        if not isinstance(properties, FillZoneProperties) or not zone.is_valid():
            continue
        red, green, blue = properties.color
        level = (11 * red + 16 * green + 5 * blue) // 32
        inside = polygon_mask(shape, zone.spline.to_polygon())
        exact = (level, level, level) if image.ndim == 2 else (blue, green, red)
        snapped = 0 if level < _BILEVEL_GRAY_SPLIT else 255
        image[inside & ~bilevel] = _match_channels(exact, image)
        image[inside & bilevel] = _per_channel(snapped, image)
    return OutputResult(
        image=image,
        is_binary=result.is_binary,
        was_dewarped=result.was_dewarped,
        picture_mask=result.picture_mask,
    )


def _match_channels(
    color: tuple[int, ...],
    image: NDArray[np.uint8],
) -> int | tuple[int, ...]:
    """Shape a BGR (or gray triple) colour for ``image``; alpha is opaque."""
    if image.ndim == 2:
        return color[0]
    return color if image.shape[2] == len(color) else (*color, 255)


def _per_channel(level: int, image: NDArray[np.uint8]) -> int | tuple[int, ...]:
    """A gray level as a value for each channel of ``image``; alpha stays opaque."""
    if image.ndim == 2:
        return level
    return (level,) * _COLOR_CHANNELS + (255,) * (image.shape[2] - _COLOR_CHANNELS)


def _wiener_gray(
    gray: NDArray[np.uint8],
    options: ColorCommonOptions,
) -> NDArray[np.uint8]:
    """Wiener filter a gray page; returned as is when the coefficient is 0."""
    if options.wiener_coef <= 0.0:
        return gray
    return wiener_filter(gray, options.wiener_window_size, 255.0 * options.wiener_coef)


def _smooth_gray(gray: NDArray[np.uint8], dpi: Dpi) -> NDArray[np.uint8]:
    """Savitzky-Golay smooth a gray page with a window suited to its resolution."""
    min_dpi = min(dpi.horizontal, dpi.vertical)
    window, degree = next(
        ((w, d) for limit, w, d in _SMOOTHING_BANDS if min_dpi <= limit),
        _SMOOTHING_ABOVE_BANDS,
    )
    return savgol_filter_2d(gray, window, degree, degree)


def _binarize_page(
    gray: NDArray[np.uint8],
    params: Params,
    render: RenderParams,
) -> NDArray[np.uint8]:
    """Smooth if asked and binarise a gray page; 0 is ink."""
    if render.savitzky_golay_smoothing:
        gray = _smooth_gray(gray, params.output_dpi)
    return _binarize(gray, params.binarization)


def _posterize(
    image: NDArray[np.uint8],
    options: ColorCommonOptions,
) -> NDArray[np.uint8]:
    """Posterize with the posterization part of the colour options."""
    posterization = options.posterization
    return posterize(
        image,
        posterization.level,
        normalize=posterization.normalization_enabled,
        force_black_and_white=posterization.force_black_and_white,
    )


def _invert_polarity(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return a copy with gray levels or colour channels inverted; alpha is kept."""
    inverted = image.copy()
    if image.ndim == 2:
        inverted[:] = 255 - image
    else:
        inverted[..., :3] = 255 - image[..., :3]
    return inverted


# Luma weights (R, G, B) in 1/32 units.
_WEIGHT_R = 11
_WEIGHT_G = 16
_WEIGHT_B = 5
_WEIGHT_SCALE = 32


def _with_brightness(
    image: NDArray[np.uint8],
    brightness: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Give an image a new brightness while keeping its colour differences.

    Args:
        image (NDArray[np.uint8]): 2-D gray or BGR(A) image. Not modified.
        brightness (NDArray[np.uint8]): Target luma, same height and width.

    Returns:
        NDArray[np.uint8]: A new array; for gray input the brightness itself.
    """
    if image.ndim == 2:
        return brightness.copy()
    return _replace_brightness(image, brightness)


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


def _margin_color(
    image: NDArray[np.uint8],
    params: Params,
    layout: _PageLayout,
) -> int | tuple[int, ...]:
    """Pick the margin colour before any white-on-black inversion.

    Args:
        image (NDArray[np.uint8]): Image whose page decides the background colour.
        params (Params): Output parameters; ``black_on_white`` flips white and black.
        layout (_PageLayout): Page area the background colour is sampled from.

    Returns:
        int | tuple[int, ...]: A gray level, or one value per channel for colour images.
    """
    filling = params.color_common.filling_color
    if filling == FillingColor.BACKGROUND:
        return dominant_background_color(image, layout.page)
    wants_white = (filling == FillingColor.WHITE) == params.black_on_white
    return _per_channel(255 if wants_white else 0, image)


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
