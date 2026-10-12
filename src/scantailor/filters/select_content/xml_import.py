"""Reading Select Content settings from a C++ ScanTailor project file."""

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from scantailor.core.project import StageParamsEntry
from scantailor.core.xml_reader import (
    XmlProjectContext,
    attr_flag,
    attr_float,
    attr_text,
    child,
    page_elements,
    read_polygon,
    read_rect_f,
    read_size_f,
    skip_malformed,
)

from .content_box import ContentBox, PageBox, PhysicalSize
from .dependencies import Dependencies
from .detection import ContentDetectionMode, PageDetectionMode
from .params import Params

_DEFAULT_TOLERANCE = 0.1


@dataclass(frozen=True)
class SelectContentImport:
    """What a ``<select-content>`` element holds.

    Attributes:
        page_detection_box (tuple[float, float]): Width and height in millimetres
            of the box page detection looks for.
        page_detection_tolerance (float): How far a detected page may deviate
            from that box, as a fraction.
        entries (list[StageParamsEntry]): Stored parameters, one per page.
    """

    page_detection_box: tuple[float, float]
    page_detection_tolerance: float
    entries: list[StageParamsEntry]


def read_select_content(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> SelectContentImport:
    """Read the ``<select-content>`` element.

    Args:
        filter_element (ET.Element | None): The ``<select-content>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        SelectContentImport: Stage-wide values and the per-page entries;
            malformed pages are logged and left out. Unreadable stage-wide
            values fall back to their defaults.
    """
    box = (0.0, 0.0)
    tolerance = _DEFAULT_TOLERANCE
    with skip_malformed("select content page detection box"):
        box = read_size_f(child(filter_element, "page-detection-box"))
    with skip_malformed("select content page detection tolerance"):
        tolerance = attr_float(
            filter_element, "pageDetectionTolerance", _DEFAULT_TOLERANCE
        )

    entries: list[StageParamsEntry] = []
    for page_id, element in page_elements(filter_element, context):
        params_element = child(element, "params")
        if params_element is None:
            continue
        with skip_malformed("select content settings", element):
            params = _read_params(params_element)
            entries.append(
                StageParamsEntry(page=page_id, params=params.model_dump(mode="json"))
            )
    return SelectContentImport(box, tolerance, entries)


def _mode(text: str) -> str:
    """Normalise a detection mode name; anything unknown means auto, as in C++."""
    return text if text in {"manual", "disabled"} else "auto"


def _read_params(element: ET.Element) -> Params:
    """Read one ``<params>`` element."""
    content_x, content_y, content_w, content_h = read_rect_f(
        child(element, "content-rect")
    )
    page_x, page_y, page_w, page_h = read_rect_f(child(element, "page-rect"))
    size_w, size_h = read_size_f(child(element, "content-size-mm"))
    return Params(
        content_box=ContentBox(
            x=content_x, y=content_y, width=content_w, height=content_h
        ),
        page_box=PageBox(x=page_x, y=page_y, width=page_w, height=page_h),
        content_size_mm=PhysicalSize(width_mm=size_w, height_mm=size_h),
        content_detection_mode=ContentDetectionMode(
            _mode(attr_text(element, "contentDetectionMode"))
        ),
        page_detection_mode=PageDetectionMode(
            _mode(attr_text(element, "pageDetectionMode"))
        ),
        fine_tune_corners=attr_flag(element, "fineTuneCorners"),
        dependencies=_read_dependencies(child(element, "dependencies")),
    )


def _read_dependencies(element: ET.Element | None) -> Dependencies:
    """Read a ``<dependencies>`` element; an absent one reads as all-auto, as in C++."""
    modes = child(element, "params")
    return Dependencies(
        content_detection_mode=ContentDetectionMode(
            _mode(attr_text(modes, "contentDetectionMode"))
        ),
        page_detection_mode=PageDetectionMode(
            _mode(attr_text(modes, "pageDetectionMode"))
        ),
        fine_tune_corners=attr_flag(modes, "fineTuneCorners"),
        rotated_page_outline=list(read_polygon(child(element, "rotated-page-outline"))),
    )
