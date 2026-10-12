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
from .params import AutoManualMode, Params


def read_deskew(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> list[StageParamsEntry]:
    """Read the per-page angles of the ``<deskew>`` element.

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
            params = Params(
                deskew_angle_deg=attr_float(params_element, "angle"),
                mode=(
                    AutoManualMode.MANUAL
                    if attr_text(params_element, "mode") == "manual"
                    else AutoManualMode.AUTO
                ),
                dependencies=_read_dependencies(child(params_element, "dependencies")),
            )
            entries.append(
                StageParamsEntry(page=page_id, params=params.model_dump(mode="json"))
            )
    return entries


def _read_dependencies(element: ET.Element | None) -> Dependencies:
    """Read a ``<dependencies>`` element; an absent one is unknown (null)."""
    if element is None:
        return Dependencies()
    return Dependencies(
        page_outline=read_polygon(child(element, "page-outline")),
        rotation_degrees=attr_int(child(element, "rotation"), "degrees"),
    )
