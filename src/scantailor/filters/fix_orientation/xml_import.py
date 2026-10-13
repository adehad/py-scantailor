"""Reading Fix Orientation settings from a C++ ScanTailor project file."""

from xml.etree import ElementTree as ET

from scantailor.core.models import ImageId, OrthogonalRotation
from scantailor.core.project import StageParamsEntry
from scantailor.core.xml_reader import (
    XmlProjectContext,
    attr_int,
    child,
    entry_for_image,
    image_elements,
    skip_malformed,
)
from scantailor.filters.fix_orientation.image_trim import ImageTrim

__all__ = ["ImageTrim", "read_rotations", "read_trims"]


def read_rotations(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> list[StageParamsEntry]:
    """Read the per-image rotations and trims of the ``<fix-orientation>`` element.

    The element's ``<image-settings>`` (a per-page binarization threshold and
    black-on-white flag) have no counterpart in the port and are not read.

    Args:
        filter_element (ET.Element | None): The ``<fix-orientation>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        list[StageParamsEntry]: One entry per image with a stored rotation or an
            enabled trim, shaped as ``Settings.dump_params`` writes it; malformed
            ones are logged and left out.
    """
    trims = read_trims(filter_element, context)
    entries: list[StageParamsEntry] = []
    for image_id, element in image_elements(filter_element, context):
        with skip_malformed("fix orientation rotation", element):
            params: dict[str, object] = {}
            rotation_element = child(element, "rotation")
            if rotation_element is not None:
                degrees = attr_int(rotation_element, "degrees")
                rotation = OrthogonalRotation.model_validate({"degrees": degrees})
                params.update(rotation.model_dump(mode="json"))
            trim = trims.get(image_id)
            if trim is not None:
                params["trim"] = trim.model_dump(mode="json")
            if params:
                entries.append(entry_for_image(image_id, params))
    return entries


def read_trims(
    filter_element: ET.Element | None, context: XmlProjectContext
) -> dict[ImageId, ImageTrim]:
    """Read the enabled ``<trim>`` elements that version 4 projects store per image.

    Args:
        filter_element (ET.Element | None): The ``<fix-orientation>`` element.
        context (XmlProjectContext): Id lookup of the project.

    Returns:
        dict[ImageId, ImageTrim]: The enabled trim of each image that has one;
            malformed ones are logged and left out.
    """
    trims: dict[ImageId, ImageTrim] = {}
    for image_id, element in image_elements(filter_element, context):
        trim_element = child(element, "trim")
        if trim_element is None:
            continue
        with skip_malformed("fix orientation trim", element):
            if (trim_element.get("enabled") or "0") == "0":
                continue
            trims[image_id] = ImageTrim(
                enabled=True,
                left=attr_int(trim_element, "left"),
                right=attr_int(trim_element, "right"),
                top=attr_int(trim_element, "top"),
                bottom=attr_int(trim_element, "bottom"),
            )
    return trims
