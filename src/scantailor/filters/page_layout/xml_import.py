"""Reading Page Layout settings from a C++ ScanTailor project file."""

from dataclasses import dataclass, field
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
    children,
    page_elements,
    read_rect_f,
    read_size_f,
    skip_malformed,
)

from .alignment import Alignment, HorizontalAlignment, VerticalAlignment
from .guide import Guide
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


@dataclass(frozen=True)
class PageLayoutImport:
    """What a ``<page-layout>`` element holds.

    Attributes:
        entries (list[StageParamsEntry]): Stored parameters, one per page.
        middle_rect_shown (bool): Whether the hard margins rectangle is shown.
        guides (tuple[Guide, ...]): The alignment guides.
    """

    entries: list[StageParamsEntry] = field(default_factory=list)
    middle_rect_shown: bool = True
    guides: tuple[Guide, ...] = ()


def read_page_layout(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> list[StageParamsEntry]:
    """Read the per-page margins and alignment of the ``<page-layout>`` element.

    Args:
        filter_element (ET.Element | None): The ``<page-layout>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        list[StageParamsEntry]: One entry per page with stored parameters;
            malformed ones are logged and left out.
    """
    return read_page_layout_stage(filter_element, context).entries


def read_page_layout_stage(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> PageLayoutImport:
    """Read the ``<page-layout>`` element with its project-wide settings.

    ``showMiddleRect`` reads as shown when it is absent. C++ reads an absent
    attribute as hidden, which leaves the rectangle hidden after a project
    saved before the attribute existed is opened.

    Args:
        filter_element (ET.Element | None): The ``<page-layout>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        PageLayoutImport: The per-page parameters, the rectangle flag and the
            guides; malformed pages and guides are logged and left out.
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
    return PageLayoutImport(
        entries=entries,
        middle_rect_shown=attr_flag(filter_element, "showMiddleRect", default=True),
        guides=_read_guides(child(filter_element, "guides")),
    )


def _read_guides(element: ET.Element | None) -> tuple[Guide, ...]:
    """Read the ``<guides>`` element; any orientation but vertical is horizontal."""
    guides: list[Guide] = []
    for guide in children(element, "guide"):
        with skip_malformed("page layout guide", guide):
            guides.append(
                Guide(
                    position_mm=attr_float(guide, "position"),
                    is_horizontal=attr_text(guide, "orientation") != "vertical",
                )
            )
    return tuple(guides)


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
        alignment=read_alignment(child(element, "alignment")),
        content_rect=ContentRect(
            x=content_x, y=content_y, width=content_w, height=content_h
        ),
        page_rect=ContentRect(x=page_x, y=page_y, width=page_w, height=page_h),
        content_size_mm=ContentSize(width_mm=size_w, height_mm=size_h),
        auto_margins=attr_flag(element, "autoMargins"),
    )


def read_alignment(element: ET.Element | None) -> Alignment:
    """Read an ``<alignment>`` element; unknown names mean centred, as in C++."""
    return Alignment(
        vertical=_VERTICAL.get(attr_text(element, "vert"), VerticalAlignment.VCENTER),
        horizontal=_HORIZONTAL.get(
            attr_text(element, "hor"), HorizontalAlignment.HCENTER
        ),
        is_null=attr_int(element, "null") != 0,
    )
