"""Reading Output settings from a C++ ScanTailor project file."""

from xml.etree import ElementTree as ET

from loguru import logger

from scantailor.core.models import Dpi, PageId
from scantailor.core.project import StageParamsEntry
from scantailor.core.xml_reader import (
    Point,
    XmlProjectContext,
    attr_flag,
    attr_float,
    attr_int,
    attr_text,
    child,
    children,
    page_elements,
    parse_float,
    read_float32_polyline,
    read_polygon,
    skip_malformed,
)
from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
    ZoneCategory,
)

from .binarization import (
    BinarizationMethod,
    BinarizationOptions,
    ColorSegmenterOptions,
)
from .color_mode import ColorMode
from .color_options import ColorCommonOptions, FillingColor, PosterizationOptions
from .despeckle import DespeckleLevel
from .dewarping_options import DewarpingMode, DewarpingOptions
from .params import (
    DistortionModelSpec,
    FillZoneSpec,
    Params,
    PictureZoneSpec,
)
from .picture_shape import PictureShape, PictureShapeOptions
from .splitting_options import SplittingOptions

_DEFAULT_DPI = 600
_DEFAULT_DEPTH_PERCEPTION = 2.0
_MIN_DEPTH_PERCEPTION = 1.0
_MAX_DEPTH_PERCEPTION = 3.0
_MIN_WIENER_WINDOW = 3
_DEFAULT_WIENER_WINDOW = 5
_MIN_POINTS_PER_ZONE = 3
_MIN_POINTS_PER_CURVE = 2
_SHORT_HEX_DIGITS = 3
_HEX_RGB_DIGITS = (_SHORT_HEX_DIGITS, 6, 8)
# Midpoints between the stored strengths 0, 1, 2 and 3.
_DESPECKLE_UPPER_BOUNDS = (
    (DespeckleLevel.OFF, 0.5),
    (DespeckleLevel.CAUTIOUS, 1.5),
    (DespeckleLevel.NORMAL, 2.5),
)

_COLOR_MODES = {
    "bw": ColorMode.BLACK_AND_WHITE,
    "colorOrGray": ColorMode.COLOR_GRAYSCALE,
    "mixed": ColorMode.MIXED,
}
_FILLING_COLORS = {"white": FillingColor.WHITE, "black": FillingColor.BLACK}
_PICTURE_SHAPES = {
    "off": PictureShape.OFF,
    "rectangular": PictureShape.RECTANGULAR,
}
_DEWARPING_MODES = {
    "auto": DewarpingMode.AUTO,
    "manual": DewarpingMode.MANUAL,
    "marginal": DewarpingMode.MARGINAL,
}
_PICTURE_LAYERS = {
    "eraser1": PictureLayer.ERASER1,
    "painter2": PictureLayer.PAINTER2,
    "eraser3": PictureLayer.ERASER3,
    "foreground": PictureLayer.FOREGROUND,
    "background": PictureLayer.BACKGROUND,
}


def read_output(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> list[StageParamsEntry]:
    """Read the per-page settings of the ``<output>`` element.

    C++ keeps the zones beside the parameters; here they are part of the
    parameters. The page's ``<processing-params>`` (auto-zone bookkeeping) and
    ``<output-params>`` (a record of the files a previous run wrote, used by the
    C++ cache) have no counterpart in the port and are not read.

    Args:
        filter_element (ET.Element | None): The ``<output>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        list[StageParamsEntry]: One entry per page that has parameters or
            zones; malformed pages are logged and left out.
    """
    entries: list[StageParamsEntry] = []
    for page_id, element in page_elements(filter_element, context):
        params_element = child(element, "params")
        with skip_malformed("output settings", element):
            picture_zones = _read_picture_zones(child(element, "zones"))
            fill_zones = _read_fill_zones(child(element, "fill-zones"))
            if params_element is None and not picture_zones and not fill_zones:
                continue
            params = _read_params(params_element).model_copy(
                update={
                    "picture_zones": tuple(picture_zones),
                    "fill_zones": tuple(fill_zones),
                }
            )
            entries.append(
                StageParamsEntry(page=page_id, params=params.model_dump(mode="json"))
            )
    return entries


def read_fill_outside_page_box(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> frozenset[PageId]:
    """Find the pages whose Output settings fill outside the page box.

    Version 4 projects store this flag in the colour options; the port has no
    such option yet, so it is read apart from the parameters.

    Args:
        filter_element (ET.Element | None): The ``<output>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        frozenset[PageId]: The pages with the flag set.
    """
    pages: set[PageId] = set()
    for page_id, element in page_elements(filter_element, context):
        color_params = child(child(element, "params"), "color-params")
        options = child(color_params, "color-or-grayscale")
        if attr_flag(options, "fillOutsidePageBox"):
            pages.add(page_id)
    return frozenset(pages)


def despeckle_level_from_factor(factor: float) -> DespeckleLevel:
    """Map the C++ despeckle strength to the nearest named level.

    C++ stores a number: 0 is off and 1, 2 and 3 are cautious, normal and
    aggressive, with the slider moving in tenths in between.

    Args:
        factor (float): The stored ``despeckleLevel``.

    Returns:
        DespeckleLevel: The nearest level.
    """
    for level, upper_bound in _DESPECKLE_UPPER_BOUNDS:
        if factor < upper_bound:
            return level
    return DespeckleLevel.AGGRESSIVE


def _clamp(value: float, low: float, high: float) -> float:
    """Limit ``value`` to the range ``low`` to ``high``."""
    return max(low, min(high, value))


def _read_params(element: ET.Element | None) -> Params:
    """Read a ``<params>`` element; an absent one gives the port's defaults."""
    if element is None:
        return Params()
    color_params = child(element, "color-params")
    return Params(
        output_dpi=_read_dpi(child(element, "dpi")),
        color_mode=_COLOR_MODES.get(
            attr_text(color_params, "colorMode"), ColorMode.BLACK_AND_WHITE
        ),
        binarization=_read_binarization(child(color_params, "bw")),
        despeckle_level=despeckle_level_from_factor(
            attr_float(element, "despeckleLevel")
        ),
        black_on_white=attr_flag(element, "blackOnWhite"),
        dewarping=_read_dewarping(child(element, "dewarping-options")),
        color_common=_read_color_common(child(color_params, "color-or-grayscale")),
        picture_shape=_read_picture_shape(child(element, "picture-shape-options")),
        depth_perception=_read_depth_perception(attr_text(element, "depthPerception")),
        distortion_model=_read_distortion_model(child(element, "distortion-model")),
        splitting=_read_splitting(child(element, "splitting")),
    )


def _read_dpi(element: ET.Element | None) -> Dpi:
    """Read a ``<dpi>`` element; an undefined resolution gets the default."""
    if element is None:
        return Dpi.uniform(_DEFAULT_DPI)
    dpi = Dpi(
        horizontal=attr_int(element, "horizontal"),
        vertical=attr_int(element, "vertical"),
    )
    return Dpi.uniform(_DEFAULT_DPI) if dpi.is_null() else dpi


def _read_binarization(element: ET.Element | None) -> BinarizationOptions:
    """Read a ``<bw>`` element.

    C++ drives every local method with the single Sauvola coefficient; the edge
    methods differ only in which of the two edge-div weights get it.
    """
    if element is None:
        return BinarizationOptions()
    coef = _clamp(attr_float(element, "sauvolaCoef"), 0.0, 1.0)
    method_name = attr_text(element, "binarizationMethod")
    edge_weights = {
        "edgeplus": (coef, 0.0),
        "blurdiv": (0.0, coef),
        "edgediv": (coef, coef),
    }
    method = {
        "sauvola": BinarizationMethod.SAUVOLA,
        "wolf": BinarizationMethod.WOLF,
        "bradley": BinarizationMethod.BRADLEY,
    }.get(method_name, BinarizationMethod.OTSU)
    kep, kbd = (
        BinarizationOptions().edge_div_kep,
        BinarizationOptions().edge_div_kbd,
    )
    if method_name in edge_weights:
        method = BinarizationMethod.EDGEDIV
        kep, kbd = edge_weights[method_name]
    return BinarizationOptions(
        method=method,
        threshold_adjustment=int(_clamp(attr_int(element, "thresholdAdj"), -100, 100)),
        window_size=int(_clamp(attr_int(element, "windowSize"), 10, 1000)),
        sauvola_coef=coef,
        wolf_coef=_clamp(attr_float(element, "wolfCoef"), 0.0, 1.0),
        wolf_lower_bound=int(_clamp(attr_int(element, "wolfLowerBound"), 0, 255)),
        wolf_upper_bound=int(_clamp(attr_int(element, "wolfUpperBound"), 0, 255)),
        bradley_coef=coef,
        edge_div_kep=kep,
        edge_div_kbd=kbd,
        normalize_illumination=attr_flag(element, "normalizeIlluminationBW"),
        savitzky_golay_smoothing=attr_flag(element, "savitzkyGolaySmoothing"),
        morphological_smoothing=attr_flag(element, "morphologicalSmoothing"),
        color_segmenter=_read_color_segmenter(
            child(element, "color-segmenter-options")
        ),
    )


def _read_color_segmenter(element: ET.Element | None) -> ColorSegmenterOptions:
    """Read a ``<color-segmenter-options>`` element."""
    if element is None:
        return ColorSegmenterOptions()
    return ColorSegmenterOptions(
        enabled=attr_flag(element, "enabled"),
        noise_reduction=max(0, attr_int(element, "noiseReduction")),
        red_threshold_adjustment=attr_int(element, "redThresholdAdjustment"),
        green_threshold_adjustment=attr_int(element, "greenThresholdAdjustment"),
        blue_threshold_adjustment=attr_int(element, "blueThresholdAdjustment"),
    )


def _read_color_common(element: ET.Element | None) -> ColorCommonOptions:
    """Read a ``<color-or-grayscale>`` element, with the C++ range checks."""
    if element is None:
        return ColorCommonOptions()
    wiener_coef = attr_float(element, "wienerCoef")
    if not 0.0 <= wiener_coef <= 1.0:
        wiener_coef = 0.0
    wiener_window = attr_int(element, "wienerWinSize")
    if wiener_window < _MIN_WIENER_WINDOW:
        wiener_window = _DEFAULT_WIENER_WINDOW
    return ColorCommonOptions(
        fill_offcut=attr_flag(element, "fillOffcut"),
        fill_margins=attr_flag(element, "fillMargins"),
        normalize_illumination=attr_flag(element, "normalizeIlluminationColor"),
        filling_color=_FILLING_COLORS.get(
            attr_text(element, "fillingColor"), FillingColor.BACKGROUND
        ),
        wiener_coef=wiener_coef,
        wiener_window_size=wiener_window,
        posterization=_read_posterization(child(element, "posterization-options")),
    )


def _read_posterization(element: ET.Element | None) -> PosterizationOptions:
    """Read a ``<posterization-options>`` element."""
    if element is None:
        return PosterizationOptions()
    return PosterizationOptions(
        enabled=attr_flag(element, "enabled"),
        level=int(_clamp(attr_int(element, "level"), 2, 255)),
        normalization_enabled=attr_flag(element, "normalizationEnabled"),
        force_black_and_white=attr_flag(element, "forceBlackAndWhite"),
    )


def _read_splitting(element: ET.Element | None) -> SplittingOptions:
    """Read a ``<splitting>`` element."""
    if element is None:
        return SplittingOptions()
    return SplittingOptions(
        split_output=attr_flag(element, "splitOutput"),
        original_background=attr_flag(element, "originalBackground"),
        color_foreground=attr_text(element, "splittingMode") == "color",
    )


def _read_picture_shape(element: ET.Element | None) -> PictureShapeOptions:
    """Read a ``<picture-shape-options>`` element."""
    if element is None:
        return PictureShapeOptions()
    return PictureShapeOptions(
        shape=_PICTURE_SHAPES.get(
            attr_text(element, "pictureShape"), PictureShape.FREE
        ),
        sensitivity=int(_clamp(attr_int(element, "sensitivity"), 0, 100)),
        higher_search_sensitivity=attr_flag(element, "higherSearchSensitivity"),
    )


def _read_dewarping(element: ET.Element | None) -> DewarpingOptions:
    """Read a ``<dewarping-options>`` element."""
    if element is None:
        return DewarpingOptions()
    return DewarpingOptions(
        mode=_DEWARPING_MODES.get(attr_text(element, "mode"), DewarpingMode.OFF),
        post_deskew=attr_flag(element, "postDeskew", default=True),
        post_deskew_angle=attr_float(element, "postDeskewAngle"),
    )


def _read_depth_perception(text: str) -> float:
    """Parse the ``depthPerception`` attribute, clamped like the C++ class."""
    if not text:
        return _DEFAULT_DEPTH_PERCEPTION
    try:
        value = parse_float(text, "depthPerception")
    except ValueError:
        return _DEFAULT_DEPTH_PERCEPTION
    return _clamp(value, _MIN_DEPTH_PERCEPTION, _MAX_DEPTH_PERCEPTION)


def _read_distortion_model(element: ET.Element | None) -> DistortionModelSpec:
    """Read a ``<distortion-model>`` element; both curves must be usable."""
    top = _read_curve(child(element, "top-curve"))
    bottom = _read_curve(child(element, "bottom-curve"))
    if not top or not bottom:
        return DistortionModelSpec()
    return DistortionModelSpec(top=top, bottom=bottom)


def _read_curve(element: ET.Element | None) -> tuple[Point, ...]:
    """Read a curve as its polyline, or as its control points without one.

    Returns:
        tuple[Point, ...]: The points; empty for a curve C++ would not write
            (fewer than two points, or ending where it starts).
    """
    points = read_float32_polyline(child(element, "polyline"))
    if not points:
        points = read_polygon(child(element, "xspline"))
    if len(points) < _MIN_POINTS_PER_CURVE or points[0] == points[-1]:
        return ()
    return points


def _read_outline(element: ET.Element | None) -> tuple[Point, ...] | None:
    """Read a zone's ``<spline>``; None when C++ would drop the zone as invalid."""
    points = read_polygon(child(element, "spline"))
    if len(points) < _MIN_POINTS_PER_ZONE:
        return None
    if len(points) == _MIN_POINTS_PER_ZONE and points[0] == points[-1]:
        return None
    return points


def _properties(element: ET.Element | None) -> dict[str, ET.Element]:
    """Index a ``<properties>`` element by property type; the last one wins."""
    return {prop.get("type", ""): prop for prop in children(element, "property")}


def _read_picture_zones(element: ET.Element | None) -> list[PictureZoneSpec]:
    """Read a ``<zones>`` element."""
    zones: list[PictureZoneSpec] = []
    for zone in children(element, "zone"):
        points = _read_outline(zone)
        if points is None:
            logger.debug("Dropping a picture zone with too few points")
            continue
        props = _properties(child(zone, "properties"))
        layer = _PICTURE_LAYERS.get(
            attr_text(props.get("PictureZoneProperty"), "layer"), PictureLayer.NOOP
        )
        category = (
            ZoneCategory.AUTO
            if attr_text(props.get("ZoneCategoryProperty"), "zoneCategory") == "auto"
            else ZoneCategory.MANUAL
        )
        zones.append(
            PictureZoneSpec(
                points=points,
                properties=PictureZoneProperties(layer=layer, category=category),
            )
        )
    return zones


def _read_fill_zones(element: ET.Element | None) -> list[FillZoneSpec]:
    """Read a ``<fill-zones>`` element."""
    zones: list[FillZoneSpec] = []
    for zone in children(element, "zone"):
        points = _read_outline(zone)
        if points is None:
            logger.debug("Dropping a fill zone with too few points")
            continue
        props = _properties(child(zone, "properties"))
        color = _parse_color(attr_text(props.get("FillColorProperty"), "color"))
        zones.append(
            FillZoneSpec(points=points, properties=FillZoneProperties(color=color))
        )
    return zones


def _parse_color(text: str) -> tuple[int, int, int]:
    """Parse a ``#rgb``, ``#rrggbb`` or ``#aarrggbb`` colour as QColor does.

    Returns:
        tuple[int, int, int]: The red, green and blue values; black for text
            QColor would reject.
    """
    digits = text[1:] if text.startswith("#") else ""
    if len(digits) not in _HEX_RGB_DIGITS:
        logger.debug("Unreadable fill colour {!r}, using black", text)
        return (0, 0, 0)
    try:
        value = int(digits, 16)
    except ValueError:
        logger.debug("Unreadable fill colour {!r}, using black", text)
        return (0, 0, 0)
    if len(digits) == _SHORT_HEX_DIGITS:
        red, green, blue = ((value >> shift) & 0xF for shift in (8, 4, 0))
        return (red * 17, green * 17, blue * 17)
    return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
