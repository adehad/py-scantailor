"""Reading Page Layout settings from a C++ ScanTailor project file."""

from xml.etree import ElementTree as ET

from scantailor.core.models import Margins
from scantailor.core.project import StageParamsEntry
from scantailor.core.xml_reader import (
    XmlProjectContext,
    attr_flag,
    attr_float,
    attr_int,
    attr_text,
    child,
    page_elements,
    read_rect_f,
    read_size_f,
    skip_malformed,
)

from .alignment import Alignment, HorizontalAlignment, VerticalAlignment
from .params import ContentRect, ContentSize, Params

_VERTICAL = {
    "top": VerticalAlignment.TOP,
    "bottom": VerticalAlignment.BOTTOM,
    "auto": VerticalAlignment.VAUTO,
    "original": VerticalAlignment.VORIGINAL,
}
_HORIZONTAL = {
    "left": HorizontalAlignment.LEFT,
    "right": HorizontalAlignment.RIGHT,
    "auto": HorizontalAlignment.HAUTO,
    "original": HorizontalAlignment.HORIGINAL,
}


def read_page_layout(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> list[StageParamsEntry]:
    """Read the per-page margins and alignment of the ``<page-layout>`` element.

    The element's ``showMiddleRect`` flag and ``<guides>`` have no counterpart
    in the port and are not read.

    Args:
        filter_element (ET.Element | None): The ``<page-layout>`` element.
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
        with skip_malformed("page layout settings", element):
            params = _read_params(params_element)
            entries.append(
                StageParamsEntry(page=page_id, params=params.model_dump(mode="json"))
            )
    return entries


def _read_params(element: ET.Element) -> Params:
    """Read one ``<params>`` element."""
    margins = child(element, "hardMarginsMM")
    content_x, content_y, content_w, content_h = read_rect_f(
        child(element, "contentRect")
    )
    page_x, page_y, page_w, page_h = read_rect_f(child(element, "pageRect"))
    size_w, size_h = read_size_f(child(element, "contentSizeMM"))
    return Params(
        hard_margins_mm=Margins(
            top=attr_float(margins, "top"),
            bottom=attr_float(margins, "bottom"),
            left=attr_float(margins, "left"),
            right=attr_float(margins, "right"),
        ),
        alignment=_read_alignment(child(element, "alignment")),
        content_rect=ContentRect(
            x=content_x, y=content_y, width=content_w, height=content_h
        ),
        page_rect=ContentRect(x=page_x, y=page_y, width=page_w, height=page_h),
        content_size_mm=ContentSize(width_mm=size_w, height_mm=size_h),
        auto_margins=attr_flag(element, "autoMargins"),
    )


def _read_alignment(element: ET.Element | None) -> Alignment:
    """Read an ``<alignment>`` element; unknown names mean centred, as in C++."""
    return Alignment(
        vertical=_VERTICAL.get(attr_text(element, "vert"), VerticalAlignment.VCENTER),
        horizontal=_HORIZONTAL.get(
            attr_text(element, "hor"), HorizontalAlignment.HCENTER
        ),
        is_null=attr_int(element, "null") != 0,
    )
