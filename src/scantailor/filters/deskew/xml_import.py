"""Reading Deskew settings from a C++ ScanTailor project file."""

from xml.etree import ElementTree as ET

from scantailor.core.project import StageParamsEntry
from scantailor.core.xml_reader import (
    XmlProjectContext,
    attr_float,
    attr_int,
    attr_text,
    child,
    page_elements,
    read_polygon,
    skip_malformed,
)

from .dependencies import Dependencies
from .params import MAX_OBLIQUE_DEG, AutoManualMode, Params


def read_deskew(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> list[StageParamsEntry]:
    """Read the per-page angles of the ``<deskew>`` element.

    The oblique angle is clamped to the range the port allows.

    The element's ``<image-settings>`` (a per-page binarization threshold and
    black-on-white flag) have no counterpart in the port and are not read.

    Args:
        filter_element (ET.Element | None): The ``<deskew>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        list[StageParamsEntry]: One entry per page with stored parameters;
            malformed ones are logged and left out.
    """
    entries: list[StageParamsEntry] = []
    for page_id, element in page_elements(filter_element, context):
        params_element = child(element, "params")
        if params_element is None:
            continue
        with skip_malformed("deskew settings", element):
            mode = _read_mode(attr_text(params_element, "mode"))
            oblique = attr_float(params_element, "oblique")
            params = Params(
                deskew_angle_deg=attr_float(params_element, "angle"),
                mode=mode,
                oblique_angle_deg=max(-MAX_OBLIQUE_DEG, min(MAX_OBLIQUE_DEG, oblique)),
                oblique_mode=_read_oblique_mode(params_element, mode),
                dependencies=_read_dependencies(child(params_element, "dependencies")),
            )
            entries.append(
                StageParamsEntry(page=page_id, params=params.model_dump(mode="json"))
            )
    return entries


def _read_mode(text: str) -> AutoManualMode:
    return AutoManualMode.MANUAL if text == "manual" else AutoManualMode.AUTO


def _read_oblique_mode(
    params_element: ET.Element, deskew_mode: AutoManualMode
) -> AutoManualMode:
    """Read the oblique mode as the C++ reader does.

    ``oblique-mode`` wins, then the transitional ``autoOblique`` flag, then the
    rotation's mode when only an ``oblique`` angle is present. A file that
    predates the oblique correction has none of them and stays manual, so that
    opening it never adds a correction.
    """
    if "oblique-mode" in params_element.attrib:
        return _read_mode(attr_text(params_element, "oblique-mode"))
    if "autoOblique" in params_element.attrib:
        return (
            AutoManualMode.MANUAL
            if attr_text(params_element, "autoOblique") == "0"
            else AutoManualMode.AUTO
        )
    if "oblique" in params_element.attrib:
        return deskew_mode
    return AutoManualMode.MANUAL


def _read_dependencies(element: ET.Element | None) -> Dependencies:
    """Read a ``<dependencies>`` element; an absent one is unknown (null)."""
    if element is None:
        return Dependencies()
    return Dependencies(
        page_outline=read_polygon(child(element, "page-outline")),
        rotation_degrees=attr_int(child(element, "rotation"), "degrees"),
    )
