"""Reading Default Parameters profiles saved by the C++ ScanTailor Advanced.

C++ keeps each profile as an XML file, ``<name>.stp`` (or ``<name>.xml``), whose
``<profile>`` root holds one ``<default-params>`` element. The options below it
are the ones a project file stores per page, so the readers of the project
import parse them. The result is a
:class:`~scantailor.core.default_params.DefaultParams`, and the profile store
saves it as JSON.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

from pydantic import ValidationError

from scantailor.core.default_params import (
    DefaultParams,
    DeskewDefaults,
    FixOrientationDefaults,
    OutputDefaults,
    PageLayoutDefaults,
    PageSplitDefaults,
    SelectContentDefaults,
)
from scantailor.core.default_params_profiles import ProfileError
from scantailor.core.models import Margins, OrthogonalRotation
from scantailor.core.settings import Units
from scantailor.core.xml_compat import NEWEST_PROJECT_VERSION
from scantailor.core.xml_reader import (
    attr_float,
    attr_int,
    attr_text,
    child,
    decode_project_bytes,
    read_size_f,
)
from scantailor.filters.deskew.params import AutoManualMode
from scantailor.filters.output.xml_import import read_profile_output
from scantailor.filters.page_layout.xml_import import read_alignment
from scantailor.filters.page_split.xml_import import layout_type_from_string
from scantailor.filters.select_content.content_box import PhysicalSize
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)

PROFILE_SUFFIXES = (".stp", ".xml")

# The deskew angle a profile may hold, as the Default Parameters dialog limits it.
_MAX_DESKEW_DEG = 45.0
_PAGE_DETECTION_MODES = {
    "manual": PageDetectionMode.MANUAL,
    "disabled": PageDetectionMode.DISABLED,
}


def read_xml_profile(path: Path) -> DefaultParams:
    """Read a profile file written by the C++ ScanTailor Advanced.

    Args:
        path (pathlib.Path): A ``.stp`` or ``.xml`` profile.

    Returns:
        DefaultParams: The profile's parameters.

    Raises:
        ProfileError: If the file cannot be read or is not a profile this
            application understands.
    """
    try:
        content = decode_project_bytes(path.read_bytes())
    except OSError as error:
        raise ProfileError(f"Cannot read profile {path}: {error}") from error
    try:
        return parse_xml_profile(content)
    except ProfileError as error:
        raise ProfileError(f"{path}: {error}") from error


def parse_xml_profile(content: str) -> DefaultParams:
    """Convert the text of a C++ profile.

    Attributes the file lacks read as C++ reads them, so a hand-edited file
    gets the same values it would get in ScanTailor Advanced. Values outside the
    ranges this application allows are clamped.

    Args:
        content (str): The XML text.

    Returns:
        DefaultParams: The profile's parameters.

    Raises:
        ProfileError: If the text is not XML, has no ``<profile>`` root with a
            ``<default-params>`` element, or has a version this application does
            not read.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise ProfileError(f"The profile is not valid XML: {error}") from error
    if root.tag != "profile":
        raise ProfileError(f"The root element is <{root.tag}>, not <profile>.")
    _check_version(root)
    params = child(root, "default-params")
    if params is None:
        raise ProfileError("The profile has no <default-params> element.")
    try:
        return DefaultParams(
            fix_orientation=_read_fix_orientation(
                child(params, "fix-orientation-params")
            ),
            deskew=_read_deskew(child(params, "deskew-params")),
            page_split=PageSplitDefaults(
                layout_type=layout_type_from_string(
                    attr_text(child(params, "page-split-params"), "layoutType")
                )
            ),
            select_content=_read_select_content(child(params, "select-content-params")),
            page_layout=_read_page_layout(child(params, "page-layout-params")),
            output=_read_output(child(params, "output-params")),
            units=_read_units(attr_text(params, "units")),
        )
    except (ValueError, ValidationError) as error:
        raise ProfileError(f"The profile holds an invalid value: {error}") from error


def _check_version(root: ET.Element) -> None:
    """Accept the project format versions this application reads.

    C++ refuses any version but its own; an older profile holds a subset of
    the same attributes, so it is read too.
    """
    text = root.get("version", "")
    try:
        version = int(text)
    except ValueError as error:
        raise ProfileError(f"The profile has no readable version: {text!r}.") from error
    if not 1 <= version <= NEWEST_PROJECT_VERSION:
        raise ProfileError(
            f"The profile is version {version}; this application reads versions "
            f"1 to {NEWEST_PROJECT_VERSION}."
        )


def _read_fix_orientation(element: ET.Element | None) -> FixOrientationDefaults:
    degrees = attr_int(child(element, "imageRotation"), "degrees") % 360
    return FixOrientationDefaults(
        image_rotation=OrthogonalRotation.model_validate({"degrees": degrees})
    )


def _read_deskew(element: ET.Element | None) -> DeskewDefaults:
    """Read the deskew defaults.

    C++ hands the stored angle to new pages as their deskew angle unchanged, so
    it is in the convention
    :class:`~scantailor.core.default_params.DeskewDefaults` stores, not the one
    the dialog shows.
    """
    angle = attr_float(element, "deskewAngleDeg")
    return DeskewDefaults(
        deskew_angle_deg=max(-_MAX_DESKEW_DEG, min(_MAX_DESKEW_DEG, angle)),
        mode=AutoManualMode.MANUAL
        if attr_text(element, "mode") == "manual"
        else AutoManualMode.AUTO,
        auto_oblique=attr_text(element, "autoOblique") not in ("", "0"),
    )


def _read_select_content(element: ET.Element | None) -> SelectContentDefaults:
    width, height = read_size_f(child(element, "pageRectSize"))
    return SelectContentDefaults(
        page_rect_size_mm=PhysicalSize(
            width_mm=max(0.0, width), height_mm=max(0.0, height)
        ),
        content_detection_mode=ContentDetectionMode.AUTO
        if attr_text(element, "contentDetectEnabled") == "1"
        else ContentDetectionMode.DISABLED,
        page_detection_mode=_PAGE_DETECTION_MODES.get(
            attr_text(element, "pageDetectMode"), PageDetectionMode.AUTO
        ),
        fine_tune_corners=attr_text(element, "fineTuneCorners") == "1",
    )


def _read_page_layout(element: ET.Element | None) -> PageLayoutDefaults:
    margins = child(element, "hardMargins")
    return PageLayoutDefaults(
        hard_margins_mm=Margins(
            top=attr_float(margins, "top"),
            bottom=attr_float(margins, "bottom"),
            left=attr_float(margins, "left"),
            right=attr_float(margins, "right"),
        ),
        alignment=read_alignment(child(element, "alignment")),
        auto_margins=attr_text(element, "autoMargins") == "1",
    )


def _read_output(element: ET.Element | None) -> OutputDefaults:
    """Read the Output defaults.

    The Wolf upper bound is taken from the file. ScanTailor Advanced's dialog
    never stores the one the user typed, so a profile it saved holds the
    default, 254, which is also what it gave new pages.
    """
    params = read_profile_output(element)
    return OutputDefaults(
        dpi=params.output_dpi,
        color_mode=params.color_mode,
        binarization=params.binarization,
        despeckle_level=params.despeckle_level,
        color_common=params.color_common,
        picture_shape=params.picture_shape,
        splitting=params.splitting,
        dewarping=params.dewarping,
        depth_perception=params.depth_perception,
    )


def _read_units(text: str) -> Units:
    """Parse the ``units`` attribute; anything unknown means millimetres, as in C++."""
    try:
        return Units(text)
    except ValueError:
        return Units.MILLIMETERS
