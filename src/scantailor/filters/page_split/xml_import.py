"""Reading Page Split settings from a C++ ScanTailor project file."""

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from scantailor.core.models import ImageId
from scantailor.core.project import StageParamsEntry
from scantailor.core.xml_reader import (
    Line,
    XmlFormatError,
    XmlProjectContext,
    attr_int,
    attr_text,
    child,
    entry_for_image,
    image_elements,
    read_line,
    read_polygon,
    read_size,
    read_text,
    skip_malformed,
)

from .dependencies import Dependencies
from .layout_type import LayoutType
from .page_layout import PageLayout, PageLayoutType
from .params import AutoManualMode, Params

_PAGE_LAYOUT_TYPES = {
    "single-uncut": PageLayoutType.SINGLE_PAGE_UNCUT,
    "single-cut": PageLayoutType.SINGLE_PAGE_CUT,
    "two-pages": PageLayoutType.TWO_PAGES,
}


@dataclass(frozen=True)
class PageSplitImport:
    """What a ``<page-split>`` element holds.

    Attributes:
        default_layout_type (LayoutType): Layout type of images without a record.
        entries (list[StageParamsEntry]): Stored parameters, one per image.
    """

    default_layout_type: LayoutType
    entries: list[StageParamsEntry]


def layout_type_from_string(text: str) -> LayoutType:
    """Parse a layout type name; anything unknown means auto-detect, as in C++."""
    return LayoutType.from_string(text)


def read_page_split(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> PageSplitImport:
    """Read the ``<page-split>`` element.

    Images that carry no record follow the default layout type, so when it is
    not auto-detect they get an entry of their own: the port keeps that default
    in the stage settings, not in the project file.

    Args:
        filter_element (ET.Element | None): The ``<page-split>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        PageSplitImport: The default layout type and the per-image entries;
            malformed records are logged and left out.
    """
    default_layout = layout_type_from_string(
        attr_text(filter_element, "defaultLayoutType")
    )
    entries: dict[ImageId, StageParamsEntry] = {}
    for image_id, element in image_elements(filter_element, context):
        with skip_malformed("page split settings", element):
            layout_type = attr_text(element, "layoutType")
            params = _read_params(
                child(element, "params"),
                layout_type_from_string(layout_type) if layout_type else default_layout,
            )
            entries[image_id] = entry_for_image(
                image_id, params.model_dump(mode="json")
            )

    if default_layout is not LayoutType.AUTO_LAYOUT_TYPE:
        default_params = Params(layout_type=default_layout).model_dump(mode="json")
        for image_id in dict.fromkeys(context.image_ids.values()):
            entries.setdefault(image_id, entry_for_image(image_id, default_params))
    return PageSplitImport(default_layout, list(entries.values()))


def _read_params(element: ET.Element | None, layout_type: LayoutType) -> Params:
    """Read a ``<params>`` element; an absent one gives a layout-type-only record."""
    if element is None:
        return Params(layout_type=layout_type)
    return Params(
        layout_type=layout_type,
        page_layout=_read_layout(child(element, "pages")),
        split_line_mode=(
            AutoManualMode.MANUAL
            if element.get("mode") == "manual"
            else AutoManualMode.AUTO
        ),
        dependencies=_read_dependencies(child(element, "dependencies")),
    )


def _read_layout(element: ET.Element | None) -> PageLayout | None:
    """Read a ``<pages>`` element into a layout; None when it has no outline.

    Raises:
        XmlFormatError: If a cutter the layout type needs is missing.
    """
    if element is None:
        return None
    outline = read_polygon(child(element, "outline"))
    if not outline:
        return None
    xs = [x for x, _ in outline]
    ys = [y for _, y in outline]
    rect = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    layout_type = _PAGE_LAYOUT_TYPES.get(
        attr_text(element, "type"), PageLayoutType.SINGLE_PAGE_UNCUT
    )
    cutter1: Line | None = None
    cutter2: Line | None = None
    if layout_type is not PageLayoutType.SINGLE_PAGE_UNCUT:
        cutter1 = _read_cutter(child(element, "cutter1"), "cutter1")
    if layout_type is PageLayoutType.SINGLE_PAGE_CUT:
        cutter2 = _read_cutter(child(element, "cutter2"), "cutter2")
    elif layout_type is PageLayoutType.TWO_PAGES:
        # Both cutters are the split line, as PageLayout.two_pages builds it.
        cutter2 = cutter1
    return PageLayout(
        outline=rect, layout_type=layout_type, cutter1=cutter1, cutter2=cutter2
    )


def _read_cutter(element: ET.Element | None, name: str) -> Line:
    """Read a cutter line that the layout type requires."""
    if element is None:
        msg = f"page layout lacks its {name}"
        raise XmlFormatError(msg)
    return read_line(element)


def _read_dependencies(element: ET.Element | None) -> Dependencies:
    """Read a ``<dependencies>`` element; an absent one is unknown (null)."""
    if element is None:
        return Dependencies()
    width, height = read_size(child(element, "size"))
    rotation_degrees = attr_int(child(element, "rotation"), "degrees")
    if rotation_degrees % 180 == 90:
        # C++ records the size of the unrotated source; the port records the
        # oriented image the layout outline is measured on.
        width, height = height, width
    return Dependencies(
        image_size=(width, height),
        rotation_degrees=rotation_degrees,
        layout_type=layout_type_from_string(read_text(child(element, "layoutType"))),
    )
