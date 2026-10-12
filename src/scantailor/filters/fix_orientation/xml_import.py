"""Reading Fix Orientation settings from a C++ ScanTailor project file."""

from xml.etree import ElementTree as ET

from scantailor.core.models import OrthogonalRotation
from scantailor.core.project import StageParamsEntry
from scantailor.core.xml_reader import (
    XmlProjectContext,
    attr_int,
    child,
    entry_for_image,
    image_elements,
    skip_malformed,
)


def read_rotations(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> list[StageParamsEntry]:
    """Read the per-image rotations of the ``<fix-orientation>`` element.

    The element's ``<image-settings>`` (a per-page binarization threshold and
    black-on-white flag) have no counterpart in the port and are not read.

    Args:
        filter_element (ET.Element | None): The ``<fix-orientation>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        list[StageParamsEntry]: One entry per image with a stored rotation;
            malformed ones are logged and left out.
    """
    entries: list[StageParamsEntry] = []
    for image_id, element in image_elements(filter_element, context):
        with skip_malformed("fix orientation rotation", element):
            degrees = attr_int(child(element, "rotation"), "degrees")
            rotation = OrthogonalRotation.model_validate({"degrees": degrees})
            entries.append(entry_for_image(image_id, rotation.model_dump(mode="json")))
    return entries
