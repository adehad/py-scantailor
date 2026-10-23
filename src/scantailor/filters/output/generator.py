"""Output image generator for the output filter.

This module provides functions for generating final output images
with binarization, despeckling, dewarping, and other processing.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.estimate_background import normalize_illumination_gray
from scantailor.core.models import Dpi
from scantailor.core.zones import (
    FillZoneProperties,
    PictureZoneProperties,
    Zone,
    ZoneCategory,
)
from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
from scantailor.dewarping.raster_dewarper import (
    InterpolationMethod,
    compute_dewarp_maps,
    output_model_domain,
    remap_with_maps,
)
from scantailor.imageproc.background_color import dominant_background_color
from scantailor.imageproc.binarize import (
    binarize_bradley,
    binarize_edge_div,
    binarize_fox,
    binarize_grad,
    binarize_otsu,
    binarize_sauvola,
    binarize_window,
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
from .dewarping_options import DewarpingMode
from .mixed import (
    PictureLayers,
    apply_picture_layers,
    build_mixed_layer,
    colorize_ink,
    compose_mixed,
    compose_mixed_colored,
    compose_over_original,
    detect_picture_mask,
    find_auto_picture_zones,
    rasterize_picture_zones,
)
from .params import Params
from .picture_shape import PictureShape
from .raster import fill_masked, polygon_mask
from .render_params import RenderParams
from .split_output import SplitImages, split_mixed_output

Polygon = Sequence[tuple[float, float]]
# Receives the illumination-normalised gray page and returns the page's
# dewarper, or None when no usable distortion model can be built from it.
DewarperFactory = Callable[[NDArray[np.uint8]], CylindricalSurfaceDewarper | None]
DewarperSource = CylindricalSurfaceDewarper | DewarperFactory
_DEWARPING_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError)
_POST_DESKEW_MODES = frozenset({DewarpingMode.MANUAL, DewarpingMode.MARGINAL})
_MASK_ON = 255
_WHITE = 255

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
        ink_mask: Mixed mode with binarisation only; True where text ink is
            drawn. None otherwise.
        split: Layers of the page when split output is on, always with black
            text on a white page whatever ``black_on_white`` says; None otherwise.
        despeckle_source: Only when asked for with ``keep_despeckle_source`` and
            the page is binarised: the text as a 0 (ink) and 255 bilevel page
            just before despeckling, whatever the despeckle level. None otherwise.
        auto_picture_zones: Mixed mode only: the auto picture zones that replace
            the page's stored ones, in the coordinates of the input image; empty
            when the page should have none. None when the stored ones stay.
    """

    image: NDArray[np.uint8]
    is_binary: bool = False
    was_dewarped: bool = False
    picture_mask: NDArray[np.bool_] | None = None
    ink_mask: NDArray[np.bool_] | None = None
    split: SplitImages | None = None
    despeckle_source: NDArray[np.uint8] | None = None
    auto_picture_zones: tuple[Zone, ...] | None = None


@dataclass(frozen=True)
class _PageLayout:
    """Where the page and its kept content lie in the image.

    Attributes:
        page_outline: The page outline as given, or None.
        page: True inside the page outline, or None when it covers the image.
        keep: True where pixels are kept; everything else gets the margin colour.
            None when nothing is to be filled.
        normalized: Illumination-normalised gray of the page, already computed
            (for instance before dewarping); None to compute it when needed.
    """

    page_outline: Polygon | None
    page: NDArray[np.bool_] | None
    keep: NDArray[np.bool_] | None
    normalized: NDArray[np.uint8] | None = None

    @classmethod
    def build(
        cls,
        shape: tuple[int, int],
        render: RenderParams,
        content_polygon: Polygon | None,
        page_outline: Polygon | None,
        source_outline: Polygon | None = None,
    ) -> _PageLayout:
        """Rasterise the page outline and the mask of the area to keep.

        Args:
            shape (tuple[int, int]): (height, width) of the image.
            render (RenderParams): Decides which fills are on.
            content_polygon (Sequence[tuple[float, float]] | None): Content area.
            page_outline (Sequence[tuple[float, float]] | None):
                Page outline; ignored when
                ``render.fill_outside_page_box`` is set, so that nothing is
                taken from the page area alone.
            source_outline (Sequence[tuple[float, float]] | None):
                Where the input has source pixels.
                Everything outside it is filled with the margin colour, and the
                background colour is sampled inside it. None means the whole
                image.

        Returns:
            _PageLayout: The layout.
        """
        if render.fill_outside_page_box:
            page_outline = None
        page = None if page_outline is None else polygon_mask(shape, page_outline)
        keep = page if render.fill_offcut else None
        if render.fill_margins and content_polygon is not None:
            content = polygon_mask(shape, content_polygon)
            keep = content if keep is None else keep & content
        source = None if source_outline is None else polygon_mask(shape, source_outline)
        if source is not None and not source.all():
            keep = source if keep is None else keep & source
            page = source if page is None else page & source
        return cls(page_outline, page, keep)


def generate_output(
    image: NDArray[np.uint8],
    params: Params,
    dewarper: DewarperSource | None = None,
    *,
    color_options: ColorCommonOptions | None = None,
    content_polygon: Polygon | None = None,
    page_outline: Polygon | None = None,
    picture_zones: Sequence[Zone] = (),
    fill_zones: Sequence[Zone] = (),
    keep_despeckle_source: bool = False,
    source_outline: Polygon | None = None,
    dbg: DebugImages | None = None,
) -> OutputResult:
    """Generate the output image according to parameters.

    Polygons and zones are in the coordinates of ``image``. When the page is
    dewarped, the masks they produce are dewarped with it.

    Args:
        image (numpy.ndarray of numpy.uint8):
            Input image, 2-D gray or BGR(A) colour. Not modified.
        params (scantailor.filters.output.params.Params): Output parameters.
        dewarper (Any): The page's dewarper, or a function
            that builds it from the illumination-normalised gray page. Used
            when ``params`` enable dewarping; None skips dewarping. A function
            that returns None skips it too.
        color_options (ColorCommonOptions | None): Options for the grayscale and
            colour paths; margin filling also applies to bilevel output.
            Defaults to ``params.color_common``.
        content_polygon (Sequence[tuple[float, float]] | None):
            Content area as (x, y) vertices.
            Margins outside it are filled when ``fill_margins`` is set. None
            means the whole image is content.
        page_outline (Sequence[tuple[float, float]] | None):
            Outline of the page as (x, y) vertices.
            What lies outside it is filled when ``fill_offcut`` is set. None
            means the whole image is page.
        picture_zones (Sequence[Zone]): Picture zones, used by mixed mode. Auto
            zones count only while ``params`` say they are up to date; see
            ``OutputResult.auto_picture_zones``.
        fill_zones (Sequence[Zone]): Fill zones, painted with their own colour
            after everything else.
        keep_despeckle_source (bool): Also return the page as it was before
            despeckling, for the despeckling preview.
        source_outline (Sequence[tuple[float, float]] | None):
            Outline of the source image in the
            coordinates of ``image``. Pixels outside it have nothing behind
            them and are always filled with the margin colour. None means
            ``image`` is all source.
        dbg (DebugImages | None): Receives the intermediate images of the
            generation, the illumination estimate and the despeckling.

    Returns:
        OutputResult with the processed image.
    """
    color_options = color_options or params.color_common
    params = params.with_color_common(color_options)
    render = RenderParams.from_params(params)

    layout = _PageLayout.build(
        image.shape[:2], render, content_polygon, page_outline, source_outline
    )
    drawn_zones, auto_zones, layout = _picture_zones_to_draw(
        image, params, render, layout, picture_zones, dbg
    )
    page = _PageGeometry.build(image, params, layout, drawn_zones, fill_zones)
    if params.needs_dewarping() and dewarper is not None:
        page = _dewarp_page(page, params, render, dewarper, dbg)

    if params.color_mode == ColorMode.BLACK_AND_WHITE:
        result = _generate_binary_output(
            page.image, params, render, page.layout, keep_despeckle_source, dbg
        )
    elif params.color_mode == ColorMode.COLOR_GRAYSCALE:
        result = _generate_grayscale_output(
            page.image, params, render, page.layout, dbg
        )
    else:
        assert page.picture_layers is not None
        result = _generate_mixed_output(
            page.image,
            params,
            render,
            page.layout,
            page.picture_layers,
            keep_despeckle_source,
            dbg,
        )

    if page.fill_zones:
        result = _apply_fill_zones(result, page.fill_zones)
    if dbg is not None:
        _record_filled(result, params, render, dbg)
    if render.posterize and not render.need_binarization:
        result.image = _posterize(result.image, color_options)
        if dbg is not None:
            dbg.add(result.image, "posterized")
    result.was_dewarped = page.was_dewarped
    result.auto_picture_zones = auto_zones
    if render.split_output and result.picture_mask is not None:
        result.split = _split_layers(result, params, render)
    return result


def _record_filled(
    result: OutputResult, params: Params, render: RenderParams, dbg: DebugImages
) -> None:
    """Record the page with its fill zones painted, under upstream's label.

    Upstream records nothing here for black and white output without colour
    segmentation.
    """
    if params.color_mode != ColorMode.BLACK_AND_WHITE:
        dbg.add(result.image, "fillZones")
    elif render.color_segmentation:
        dbg.add(result.image, "segmented_with_fill_zones")


def _split_layers(
    result: OutputResult, params: Params, render: RenderParams
) -> SplitImages:
    """Split a finished mixed page into its layers, with black text on white."""
    assert result.picture_mask is not None
    text_region = ~result.picture_mask
    image = result.image if params.black_on_white else _invert_polarity(result.image)
    ink = result.ink_mask
    if ink is None:
        ink = np.zeros(text_region.shape, dtype=bool)
    return split_mixed_output(image, render, text_region, ink)


@dataclass(frozen=True)
class _FillMask:
    """A fill zone rasterised to the output's pixel grid."""

    inside: NDArray[np.bool_]
    properties: FillZoneProperties


@dataclass(frozen=True)
class _PageGeometry:
    """The page's pixels and every mask derived from its geometry.

    All arrays share one pixel grid: the input image's, or the dewarped one.

    Attributes:
        image: Pixels the output stages work on.
        layout: Page and fill areas.
        picture_layers: Manual picture zones by layer; None unless the output is
            mixed.
        fill_zones: Fill zones to paint on the result.
        was_dewarped: True once the grid is the dewarped one.
    """

    image: NDArray[np.uint8]
    layout: _PageLayout
    picture_layers: PictureLayers | None
    fill_zones: tuple[_FillMask, ...]
    was_dewarped: bool = False

    @classmethod
    def build(
        cls,
        image: NDArray[np.uint8],
        params: Params,
        layout: _PageLayout,
        picture_zones: Sequence[Zone],
        fill_zones: Sequence[Zone],
    ) -> _PageGeometry:
        """Rasterise the zones on the input image's grid."""
        shape = image.shape[:2]
        return cls(
            image=image,
            layout=layout,
            picture_layers=(
                rasterize_picture_zones(picture_zones, shape)
                if params.color_mode == ColorMode.MIXED
                else None
            ),
            fill_zones=_rasterize_fill_zones(fill_zones, shape),
        )


def _is_auto_zone(zone: Zone) -> bool:
    properties = zone.properties
    return (
        isinstance(properties, PictureZoneProperties)
        and properties.category is ZoneCategory.AUTO
    )


def _picture_zones_to_draw(
    image: NDArray[np.uint8],
    params: Params,
    render: RenderParams,
    layout: _PageLayout,
    zones: Sequence[Zone],
    dbg: DebugImages | None = None,
) -> tuple[list[Zone], tuple[Zone, ...] | None, _PageLayout]:
    """Pick the picture zones a render draws, detecting auto zones when due.

    Auto zones count only in mixed mode with rectangular pictures, and only
    when they were detected with the current picture shape options. Otherwise
    detection runs again on ``image``, before any dewarping, so that the zones
    share the coordinates of the other zones.

    Args:
        image (numpy.ndarray of numpy.uint8): Input image, 2-D gray or BGR(A).
        params (scantailor.filters.output.params.Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page and fill areas of ``image``.
        zones (Sequence[Zone]): The page's picture zones, auto ones included.
        dbg (DebugImages | None): Receives the illumination estimate and the
            picture detector's steps when detection runs.

    Returns:
        tuple[list[Zone], tuple[Zone, ...] | None, _PageLayout]: The zones to
            draw; the auto zones that replace the stored ones, or None when
            those stay; and ``layout``, holding the illumination-normalised
            gray when detection made it.
    """
    if params.color_mode != ColorMode.MIXED:
        return list(zones), None, layout
    manual = [zone for zone in zones if not _is_auto_zone(zone)]
    options = params.picture_shape
    if options.shape != PictureShape.RECTANGULAR:
        outdated = params.auto_picture_zones_shape is not None or len(manual) < len(
            zones
        )
        return manual, () if outdated else None, layout
    if params.auto_picture_zones_shape == options:
        return list(zones), None, layout

    gray = bgr_to_gray(image if params.black_on_white else _invert_polarity(image))
    if render.normalize_illumination:
        gray = _normalized_gray(gray, layout, dbg)
        layout = replace(layout, normalized=gray)
    dpi = params.output_dpi
    found = tuple(
        find_auto_picture_zones(
            _wiener_gray(gray, params.color_common),
            (dpi.horizontal, dpi.vertical),
            options,
            dbg,
        )
    )
    return manual + list(found), found, layout


def _rasterize_fill_zones(
    zones: Sequence[Zone], shape: tuple[int, int]
) -> tuple[_FillMask, ...]:
    """Rasterise the valid fill zones; other zones are skipped."""
    return tuple(
        _FillMask(polygon_mask(shape, zone.spline.to_polygon()), zone.properties)
        for zone in zones
        if isinstance(zone.properties, FillZoneProperties) and zone.is_valid()
    )


def _dewarp_page(
    page: _PageGeometry,
    params: Params,
    render: RenderParams,
    source: DewarperSource,
    dbg: DebugImages | None = None,
) -> _PageGeometry:
    """Dewarp the page with its masks, normalising illumination first.

    Illumination is estimated on the warped page, where the page outline still
    describes the paper, then dewarped with the pixels so that the background
    surface is not distorted.

    Args:
        page (_PageGeometry): The page on the input image's grid.
        params (scantailor.filters.output.params.Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        source (Any): A dewarper, or a function that builds one from
            the normalised gray page.
        dbg (DebugImages | None): Receives the illumination estimate and the
            dewarped page.

    Returns:
        _PageGeometry: The dewarped page, or ``page`` with its illumination
            already normalised when no usable dewarper exists.
    """
    oriented = page.image if params.black_on_white else _invert_polarity(page.image)
    gray = bgr_to_gray(oriented)
    normalized: NDArray[np.uint8] | None = None
    if render.normalize_illumination or render.normalize_illumination_color:
        normalized = _normalized_gray(gray, page.layout, dbg)
        gray = normalized

    dewarper = (
        source if isinstance(source, CylindricalSurfaceDewarper) else source(gray)
    )
    if dewarper is None:
        return replace(page, layout=replace(page.layout, normalized=normalized))

    height, width = page.image.shape[:2]
    try:
        maps = compute_dewarp_maps(
            dewarper, (width, height), output_model_domain(dewarper, (width, height))
        )
    except _DEWARPING_ERRORS as error:
        logger.warning("Dewarping skipped, the distortion model is unusable: {}", error)
        return replace(page, layout=replace(page.layout, normalized=normalized))

    angle = applied_post_deskew_angle(params)

    def warp(array: NDArray[np.uint8], background: int) -> NDArray[np.uint8]:
        warped = remap_with_maps(array, maps, background)
        return _apply_post_deskew(warped, angle, background) if angle else warped

    def warp_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        warped = remap_with_maps(
            mask.astype(np.uint8) * _MASK_ON, maps, 0, InterpolationMethod.NEAREST
        )
        return (_apply_post_deskew(warped, angle, 0) if angle else warped) == _MASK_ON

    layout = page.layout
    layers = page.picture_layers
    dewarped = warp(page.image, _WHITE if params.black_on_white else 0)
    if dbg is not None:
        dbg.add(dewarped, "dewarped")
    return _PageGeometry(
        image=dewarped,
        layout=_PageLayout(
            page_outline=None,
            page=None if layout.page is None else warp_mask(layout.page),
            keep=None if layout.keep is None else warp_mask(layout.keep),
            normalized=None if normalized is None else warp(normalized, _WHITE),
        ),
        picture_layers=None
        if layers is None
        else PictureLayers(
            eraser1=warp_mask(layers.eraser1),
            foreground=warp_mask(layers.foreground),
            background=warp_mask(layers.background),
            painter=warp_mask(layers.painter),
            eraser3=warp_mask(layers.eraser3),
        ),
        fill_zones=tuple(
            _FillMask(warp_mask(zone.inside), zone.properties)
            for zone in page.fill_zones
        ),
        was_dewarped=True,
    )


def applied_post_deskew_angle(params: Params) -> float:
    """Return the rotation a dewarped page gets after dewarping, in degrees.

    Args:
        params (scantailor.filters.output.params.Params): Output parameters.

    Returns:
        float: The angle; positive turns the page clockwise. Zero unless the
            dewarping mode and options ask for it.
    """
    options = params.dewarping
    if options.post_deskew and options.mode in _POST_DESKEW_MODES:
        return options.post_deskew_angle
    return 0.0


def _apply_post_deskew(
    image: NDArray[np.uint8],
    angle: float,
    border: int,
) -> NDArray[np.uint8]:
    """Rotate a dewarped image about its centre, keeping its size.

    Args:
        image (numpy.ndarray of numpy.uint8): Dewarped image.
        angle (float): Rotation in degrees; positive turns the page clockwise.
        border (int): Gray level of the newly exposed pixels, in every channel.

    Returns:
        numpy.ndarray of numpy.uint8: Rotated image of the same shape.
    """
    h, w = image.shape[:2]
    # cv2 counts positive angles counter-clockwise.
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border,) * image.shape[2] if image.ndim == 3 else border,
    )
    return np.asarray(rotated, dtype=np.uint8)


def _generate_binary_output(
    image: NDArray[np.uint8],
    params: Params,
    render: RenderParams,
    layout: _PageLayout,
    keep_despeckle_source: bool = False,
    dbg: DebugImages | None = None,
) -> OutputResult:
    """Generate black and white output.

    Args:
        image (numpy.ndarray of numpy.uint8): Input image, 2-D gray or BGR(A).
        params (scantailor.filters.output.params.Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page and fill areas.
        keep_despeckle_source (bool): Keep the page before despeckling in the
            result.
        dbg (DebugImages | None): Receives each step's image.

    Returns:
        OutputResult: Binary image (uint8, 0 or 255) of the input's height and width.
    """
    if not params.black_on_white:
        image = _invert_polarity(image)
    gray = bgr_to_gray(image)
    if render.normalize_illumination:
        gray = _normalized_gray(gray, layout, dbg)
    if dbg is not None:
        dbg.add(gray, "maybeNormalized")
    gray = _wiener_gray(gray, params.color_common)
    color_page = _segmentation_source(image, gray, params, render)

    binary = _binarize_page(gray, params, render, dbg)

    if layout.keep is not None:
        margin = 0 if params.color_common.filling_color == FillingColor.BLACK else 255
        binary = fill_masked(binary, ~layout.keep, margin)
    if dbg is not None:
        dbg.add(binary, "binarized_and_cropped")

    if render.morphological_smoothing:
        binary = _morphological_smooth(binary, dbg)

    # Despeckling stays the last step that changes which pixels are ink.
    source = binary.copy() if keep_despeckle_source else None
    if params.despeckle_strength > 0.0:
        binary = _despeckle(binary, params.despeckle_strength, params.output_dpi, dbg)

    if color_page is not None:
        result = _colored_binary_output(color_page, binary, params, render, layout, dbg)
    else:
        if not params.black_on_white:
            binary = 255 - binary
        result = OutputResult(image=binary, is_binary=True)
    result.despeckle_source = source
    return result


def _segmentation_source(
    image: NDArray[np.uint8],
    gray: NDArray[np.uint8],
    params: Params,
    render: RenderParams,
) -> NDArray[np.uint8] | None:
    """Colour page whose colours the segmented text takes; None without segmentation.

    Args:
        image (numpy.ndarray of numpy.uint8):
            Page in black-on-white polarity. Not modified.
        gray (numpy.ndarray of numpy.uint8): Its illumination-normalised gray, when
            ``render`` normalises illumination.
        params (scantailor.filters.output.params.Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.

    Returns:
        numpy.ndarray of numpy.uint8 | None:
            The page, normalised and Wiener filtered like
            the binarisation input; None unless colour segmentation is on.
    """
    if not render.color_segmentation:
        return None
    page = _with_brightness(image, gray) if render.normalize_illumination else image
    options = params.color_common
    return wiener_color_filter(page, options.wiener_window_size, options.wiener_coef)


def _colored_binary_output(
    color_page: NDArray[np.uint8],
    binary: NDArray[np.uint8],
    params: Params,
    render: RenderParams,
    layout: _PageLayout,
    dbg: DebugImages | None = None,
) -> OutputResult:
    """Colour the ink of a binarised page with the colours of the original.

    Args:
        color_page (numpy.ndarray of numpy.uint8):
            Colour source from ``_segmentation_source``.
        binary (numpy.ndarray of numpy.uint8): Despeckled binary page in black-on-white
            polarity, 0 is ink.
        params (scantailor.filters.output.params.Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page and fill areas.
        dbg (DebugImages | None): Receives the coloured text.

    Returns:
        OutputResult: Coloured text on white, inverted for white-on-black output.
    """
    background = dominant_background_color(color_page, layout.page)
    if layout.keep is not None:
        color_page = fill_masked(
            color_page, ~layout.keep, _per_channel(_WHITE, color_page)
        )
    dpi = params.output_dpi
    text = colorize_ink(
        color_page,
        binary == 0,
        (dpi.horizontal, dpi.vertical),
        params.binarization.color_segmenter,
        params.color_common.posterization if render.posterize else None,
        background,
    )
    if dbg is not None:
        dbg.add(text, "segmented")
    if not params.black_on_white:
        text = _invert_polarity(text)
    return OutputResult(image=text, is_binary=False)


def _generate_grayscale_output(
    image: NDArray[np.uint8],
    params: Params,
    render: RenderParams,
    layout: _PageLayout,
    dbg: DebugImages | None = None,
) -> OutputResult:
    """Generate grayscale or colour output.

    With illumination normalisation on, the background is estimated from the
    gray version of the page and, for colour input, the normalised brightness is
    transferred back onto the original colour differences. The Wiener colour
    filter runs next, then margins and offcut are filled, the result is
    inverted for white-on-black output. Posterizing is left to the caller so that
    fill zones are posterized too.

    Args:
        image (numpy.ndarray of numpy.uint8): Input image, 2-D gray or BGR(A).
        params (scantailor.filters.output.params.Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page and fill areas.
        dbg (DebugImages | None): Receives the illumination estimate and the
            normalised page.

    Returns:
        OutputResult: New uint8 array with the shape of the input.
    """
    color_options = params.color_common
    if not params.black_on_white:
        image = _invert_polarity(image)
    result = image
    if render.normalize_illumination:
        brightness = _normalized_gray(bgr_to_gray(image), layout, dbg)
        result = _with_brightness(image, brightness)
    if dbg is not None:
        dbg.add(result, "maybeNormalized")
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
    picture_layers: PictureLayers,
    keep_despeckle_source: bool = False,
    dbg: DebugImages | None = None,
) -> OutputResult:
    """Generate mixed output: pictures keep their grays or colours, text is binarised.

    Picture regions come from free-form detection (see ``PictureShapeOptions``)
    and from picture zones, whose layers add to, remove from or invert the
    detected picture mask; rectangular detection arrives as auto zones.
    Everything outside the kept area is filled with white, or with black when
    the filling colour says so.

    Args:
        image (numpy.ndarray of numpy.uint8): Input image, 2-D gray or BGR(A).
        params (scantailor.filters.output.params.Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page and fill areas.
        picture_layers (PictureLayers): Rasterised picture zones.
        keep_despeckle_source (bool): Keep the text before despeckling in the
            result.
        dbg (DebugImages | None): Receives each step's image.

    Returns:
        OutputResult: New uint8 array with the shape of the input. Text pixels
            are 0 or 255 in every channel and picture pixels are never pure
            black or white.
    """
    color_options = params.color_common
    if not params.black_on_white:
        image = _invert_polarity(image)

    raw_gray = bgr_to_gray(image)
    normalized_gray = (
        _normalized_gray(raw_gray, layout, dbg)
        if render.normalize_illumination
        else None
    )
    if dbg is not None:
        dbg.add(
            image
            if normalized_gray is None
            else _with_brightness(image, normalized_gray),
            "maybeNormalized",
        )
    text_gray = _wiener_gray(
        raw_gray if normalized_gray is None else normalized_gray, color_options
    )

    dpi = params.output_dpi
    dpi_pair = (dpi.horizontal, dpi.vertical)
    binary = _binarize_page(text_gray, params, render, dbg)
    if layout.keep is not None:
        binary = fill_masked(binary, ~layout.keep, 255)
    if dbg is not None:
        dbg.add(binary, "binarized_and_cropped")
    if render.morphological_smoothing:
        binary = _morphological_smooth(binary, dbg)
    picture = detect_picture_mask(text_gray, dpi_pair, params.picture_shape, dbg)
    if dbg is not None:
        dbg.add(~picture, "bwMask")

    raw_ink: NDArray[np.bool_] = binary == 0
    text_region, _ = apply_picture_layers(~picture, raw_ink, picture_layers)
    if layout.keep is not None:
        text_region |= ~layout.keep
    if dbg is not None:
        dbg.add(text_region, "bwMask with zones")

    normalized_page = (
        None if normalized_gray is None else _with_brightness(image, normalized_gray)
    )
    layer = build_mixed_layer(image, normalized_page, render, color_options)
    ink: NDArray[np.bool_] | None = None
    source: NDArray[np.uint8] | None = None
    if render.need_binarization:
        ink = raw_ink
        if keep_despeckle_source:
            source = np.where(raw_ink & text_region, 0, 255).astype(np.uint8)
        if params.despeckle_strength > 0.0:
            speckled = np.where(ink, 0, 255).astype(np.uint8)
            ink = _despeckle(speckled, params.despeckle_strength, dpi, dbg) == 0
        result = _compose_text(
            layer, ink, text_region, params, render, layout, dpi_pair, dbg
        )
        if dbg is not None:
            dbg.add(result, "combined")
    else:
        result = layer.copy()

    if layout.keep is not None:
        result = fill_masked(
            result, ~layout.keep, _mixed_margin_color(result, params, render, layout)
        )
    if not params.black_on_white:
        result = _invert_polarity(result)
    return OutputResult(
        image=result,
        is_binary=False,
        picture_mask=~text_region,
        ink_mask=None if ink is None else ink & text_region,
        despeckle_source=source,
    )


def _compose_text(
    layer: NDArray[np.uint8],
    ink: NDArray[np.bool_],
    text_region: NDArray[np.bool_],
    params: Params,
    render: RenderParams,
    layout: _PageLayout,
    dpi: tuple[int, int],
    dbg: DebugImages | None = None,
) -> NDArray[np.uint8]:
    """Put the page's ink, black or coloured, over the picture layer.

    Args:
        layer (numpy.ndarray of numpy.uint8): Picture layer from ``build_mixed_layer``.
        ink (numpy.ndarray of numpy.bool): Despeckled ink of the whole page, pictures
            included; colour segmentation sees all of it and the text region
            then limits where it is drawn.
        text_region (numpy.ndarray of numpy.bool): True where the output is binarised.
        params (scantailor.filters.output.params.Params): Output parameters.
        render (RenderParams): Flags derived from ``params``.
        layout (_PageLayout): Page area, for the background colour.
        dpi (tuple[int, int]): Horizontal and vertical output resolution.
        dbg (DebugImages | None): Receives the coloured ink.

    Returns:
        numpy.ndarray of numpy.uint8: The composed page.
    """
    colored = None
    if render.color_segmentation:
        colored = colorize_ink(
            layer,
            ink,
            dpi,
            params.binarization.color_segmenter,
            params.color_common.posterization if render.posterize else None,
            dominant_background_color(layer, layout.page),
        )
        if dbg is not None:
            dbg.add(colored, "segmented")
    if render.original_background:
        return compose_over_original(layer, ink, colored)
    if colored is None:
        return compose_mixed(layer, text_region, ink)
    return compose_mixed_colored(layer, text_region, colored)


def _mixed_margin_color(
    result: NDArray[np.uint8],
    params: Params,
    render: RenderParams,
    layout: _PageLayout,
) -> int | tuple[int, ...]:
    """Margin colour of a mixed page, before any white-on-black inversion."""
    if render.need_binarization and not render.original_background:
        black = params.color_common.filling_color == FillingColor.BLACK
        return _per_channel(0 if black else 255, result)
    return _margin_color(result, params, layout)


def _normalized_gray(
    gray: NDArray[np.uint8], layout: _PageLayout, dbg: DebugImages | None = None
) -> NDArray[np.uint8]:
    """Illumination-normalise a gray page, unless that was already done."""
    if layout.normalized is not None:
        return layout.normalized
    return normalize_illumination_gray(gray, layout.page_outline, dbg)


def _apply_fill_zones(
    result: OutputResult,
    zones: Sequence[_FillMask],
) -> OutputResult:
    """Paint fill zones over an output.

    Text pixels of bilevel output are painted black or white by the zone
    colour's gray level; gray and colour pixels take the colour itself.

    Args:
        result (OutputResult): Output to paint. Not modified.
        zones (Sequence[_FillMask]): Zones rasterised on the output's grid.

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
        red, green, blue = zone.properties.color
        level = (11 * red + 16 * green + 5 * blue) // 32
        inside = zone.inside
        exact = (level, level, level) if image.ndim == 2 else (blue, green, red)
        snapped = 0 if level < _BILEVEL_GRAY_SPLIT else 255
        image[inside & ~bilevel] = _match_channels(exact, image)
        image[inside & bilevel] = _per_channel(snapped, image)
    return replace(result, image=image)


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
    dbg: DebugImages | None = None,
) -> NDArray[np.uint8]:
    """Smooth if asked and binarise a gray page; 0 is ink."""
    if render.savitzky_golay_smoothing:
        gray = _smooth_gray(gray, params.output_dpi)
        if dbg is not None:
            dbg.add(gray, "smoothed")
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
        image (numpy.ndarray of numpy.uint8): 2-D gray or BGR(A) image. Not modified.
        brightness (numpy.ndarray of numpy.uint8): Target luma, same height and width.

    Returns:
        numpy.ndarray of numpy.uint8: A new array; for gray input the brightness itself.
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
        image (numpy.ndarray of numpy.uint8): BGR(A) image. Not modified.
        brightness (numpy.ndarray of numpy.uint8): Target luma, same height and width.

    Returns:
        numpy.ndarray of numpy.uint8:
            New image; an alpha channel is carried over unchanged.
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
        image (numpy.ndarray of numpy.uint8):
            Image whose page decides the background colour.
        params (scantailor.filters.output.params.Params):
            Output parameters; ``black_on_white`` flips white and black.
        layout (_PageLayout): Page area the background colour is sampled from.

    Returns:
        int | tuple[int, ...]: A gray level, or one value per channel for colour images.
    """
    filling = params.color_common.filling_color
    if filling == FillingColor.BACKGROUND:
        return dominant_background_color(image, layout.page)
    wants_white = (filling == FillingColor.WHITE) == params.black_on_white
    return _per_channel(255 if wants_white else 0, image)


# C++ hard-bounds the adjusted Otsu threshold so that a page whose content is
# all masked out does not turn entirely black or white.
_OTSU_BOUNDS = (30, 225)

# Methods that take the Wolf coefficient and bounds of the options.
_WOLF_FAMILY = {
    BinarizationMethod.WOLF: binarize_wolf,
    BinarizationMethod.FOX: binarize_fox,
    BinarizationMethod.WINDOW: binarize_window,
    BinarizationMethod.GRAD: binarize_grad,
}


def _binarize(
    gray: NDArray[np.uint8],
    options: BinarizationOptions,
) -> NDArray[np.uint8]:
    """Apply binarization to a grayscale image.

    The threshold adjustment shifts the global Otsu threshold (clamped to
    30..225) and is the ``delta`` of the local methods, each of which applies it
    as described in ``scantailor.imageproc.binarize``.

    Args:
        gray (numpy.ndarray of numpy.uint8): 2-D gray image. Not modified.
        options (BinarizationOptions): Binarization options.

    Returns:
        numpy.ndarray of numpy.uint8:
            New binary image, 0 for ink and 255 for background.
    """
    adjustment = options.threshold_adjustment
    if options.method == BinarizationMethod.OTSU:
        return binarize_otsu(gray, adjustment, _OTSU_BOUNDS)
    if options.method == BinarizationMethod.SAUVOLA:
        return binarize_sauvola(
            gray,
            window_size=options.window_size,
            k=options.sauvola_coef,
            delta=adjustment,
        )
    wolf_family = _WOLF_FAMILY.get(options.method)
    if wolf_family is not None:
        return wolf_family(
            gray,
            window_size=options.window_size,
            k=options.wolf_coef,
            delta=adjustment,
            lower_bound=options.wolf_lower_bound,
            upper_bound=options.wolf_upper_bound,
        )
    if options.method == BinarizationMethod.BRADLEY:
        return binarize_bradley(
            gray,
            window_size=options.window_size,
            k=options.bradley_coef,
            delta=adjustment,
        )
    kep = 0.0 if options.method == BinarizationMethod.BLURDIV else options.edge_div_kep
    kbd = 0.0 if options.method == BinarizationMethod.EDGEPLUS else options.edge_div_kbd
    return binarize_edge_div(
        gray, window_size=options.window_size, kep=kep, kbd=kbd, delta=adjustment
    )


def _despeckle(
    binary: NDArray[np.uint8],
    strength: float,
    dpi: Dpi,
    dbg: DebugImages | None = None,
) -> NDArray[np.uint8]:
    """Remove small noise components that are not near text.

    Args:
        binary (numpy.ndarray of numpy.uint8): Binary image, 0 is ink. Not modified.
        strength (float): Despeckle strength, 0 for off; 1, 2 and 3 are the
            cautious, normal and aggressive presets.
        dpi (Dpi): Resolution of ``binary``; the smaller axis sets the thresholds.
        dbg (DebugImages | None): Receives the despeckler's steps and its result.

    Returns:
        numpy.ndarray of numpy.uint8: Despeckled binary image.
    """
    if strength <= 0.0:
        return binary
    despeckled = despeckle_image(
        binary, min(dpi.horizontal, dpi.vertical), strength, dbg
    )
    if dbg is not None:
        dbg.add(despeckled, "despeckled")
    return despeckled


def _morphological_smooth(
    binary: NDArray[np.uint8], dbg: DebugImages | None = None
) -> NDArray[np.uint8]:
    """Apply morphological smoothing to a binary image.

    This helps smooth jagged edges.

    Args:
        binary (NDArray[np.uint8]): Binary input image.
        dbg (DebugImages | None): Receives the smoothed image.

    Returns:
        NDArray[np.uint8]: Smoothed binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Opening removes small protrusions
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Closing fills small holes
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    smoothed = np.asarray(result, dtype=np.uint8)
    if dbg is not None:
        dbg.add(smoothed, "edges_smoothed")
    return smoothed
