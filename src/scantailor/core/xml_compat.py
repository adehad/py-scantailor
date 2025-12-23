"""XML compatibility layer for reading C++ ScanTailor project files.

This module provides read-only support for the XML format used by the
original C++ ScanTailor application. Projects loaded from XML can be
saved in the new JSON format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from loguru import logger

from scantailor.core.models import Dpi, ImageId, PageId, SubPage

if TYPE_CHECKING:
    from scantailor.core.project import ImageInfo, ImageMetadata, Project


def parse_xml_project(path: Path, content: str) -> Project:
    """Parse a C++ ScanTailor XML project file.

    Args:
        path (Path): Path to the project file (for resolving relative paths).
        content (str): XML content.

    Returns:
        Project: The parsed project.

    Raises:
        ValueError: If the XML is invalid or missing required elements.
    """
    # Import here to avoid circular imports
    from scantailor.core.project import ImageInfo, ImageMetadata, Project

    root = ET.fromstring(content)

    if root.tag != "project":
        msg = f"Expected <project> root element, got <{root.tag}>"
        raise ValueError(msg)

    # Parse version
    version_str = root.get("version", "1")
    version = int(version_str) if version_str.isdigit() else 1

    # Parse output directory
    output_dir_str = root.get("outputDirectory", "out")
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir

    # Parse layout direction
    layout_dir = root.get("layoutDirection", "LTR")
    layout_direction = "RTL" if layout_dir == "RTL" else "LTR"

    # Build directory map
    dir_map: dict[int, str] = {}
    dirs_el = root.find("directories")
    if dirs_el is not None:
        for dir_el in dirs_el.findall("directory"):
            dir_id = _parse_int_attr(dir_el, "id")
            dir_path = dir_el.get("path", "")
            if dir_id is not None and dir_path:
                dir_map[dir_id] = dir_path

    # Build file map
    file_map: dict[int, tuple[str, bool]] = {}
    files_el = root.find("files")
    if files_el is not None:
        for file_el in files_el.findall("file"):
            file_id = _parse_int_attr(file_el, "id")
            dir_id = _parse_int_attr(file_el, "dirId")
            name = file_el.get("name", "")
            multi_page = file_el.get("multiPage") == "1"

            if file_id is not None and dir_id is not None and name:
                dir_path = dir_map.get(dir_id, "")
                if dir_path:
                    full_path = str(Path(dir_path) / name)
                    file_map[file_id] = (full_path, multi_page)

    # Parse images
    images: list[ImageInfo] = []
    image_map: dict[int, ImageInfo] = {}
    images_el = root.find("images")
    if images_el is not None:
        for image_el in images_el.findall("image"):
            image_info = _parse_image_element(image_el, file_map)
            if image_info is not None:
                image_id = _parse_int_attr(image_el, "id")
                images.append(image_info)
                if image_id is not None:
                    image_map[image_id] = image_info

    # Parse pages to find selected page
    selected_page: PageId | None = None
    pages_el = root.find("pages")
    if pages_el is not None:
        for page_el in pages_el.findall("page"):
            if page_el.get("selected") == "selected":
                image_id = _parse_int_attr(page_el, "imageId")
                sub_page_str = page_el.get("subPage", "single")
                if image_id is not None and image_id in image_map:
                    sub_page = _parse_sub_page(sub_page_str)
                    selected_page = PageId(
                        image_id=image_map[image_id].id,
                        sub_page=sub_page,
                    )
                break

    logger.info(
        "Loaded XML project with {} images from {}",
        len(images),
        path,
    )

    return Project(
        version=version,
        output_directory=output_dir,
        layout_direction=layout_direction,
        images=images,
        selected_page=selected_page,
    )


def _parse_int_attr(element: ET.Element, attr: str) -> int | None:
    """Parse an integer attribute from an XML element.

    Args:
        element (ET.Element): The XML element.
        attr (str): The attribute name.

    Returns:
        int | None: The parsed integer, or None if not valid.
    """
    value = element.get(attr)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_sub_page(value: str) -> SubPage:
    """Parse a SubPage enum from a string.

    Args:
        value (str): The string value (single, left, right).

    Returns:
        SubPage: The parsed enum value.
    """
    value_lower = value.lower()
    if value_lower == "left":
        return SubPage.LEFT_PAGE
    if value_lower == "right":
        return SubPage.RIGHT_PAGE
    return SubPage.SINGLE_PAGE


def _parse_image_element(
    image_el: ET.Element,
    file_map: dict[int, tuple[str, bool]],
) -> ImageInfo | None:
    """Parse an image element from XML.

    Args:
        image_el (ET.Element): The <image> XML element.
        file_map (dict[int, tuple[str, bool]]): Map of file IDs to (path, multiPage).

    Returns:
        ImageInfo | None: The parsed image info, or None if invalid.
    """
    from scantailor.core.project import ImageInfo, ImageMetadata

    file_id = _parse_int_attr(image_el, "fileId")
    file_image = _parse_int_attr(image_el, "fileImage") or 0
    sub_pages = _parse_int_attr(image_el, "subPages") or 1

    if file_id is None or file_id not in file_map:
        return None

    file_path, compat_multi_page = file_map[file_id]
    # Backwards compatibility: adjust page number
    page = file_image + (1 if compat_multi_page else 0)

    image_id = ImageId(file_path=Path(file_path), page=page)

    # Parse removed half
    removed = image_el.get("removed", "")
    left_half_removed = removed == "L"
    right_half_removed = removed == "R"

    # Parse metadata
    metadata = _parse_image_metadata(image_el)

    return ImageInfo(
        id=image_id,
        metadata=metadata,
        num_sub_pages=sub_pages,
        left_half_removed=left_half_removed,
        right_half_removed=right_half_removed,
    )


def _parse_image_metadata(image_el: ET.Element) -> ImageMetadata:
    """Parse image metadata from an XML element.

    Args:
        image_el (ET.Element): The <image> XML element.

    Returns:
        ImageMetadata: The parsed metadata.
    """
    from scantailor.core.project import ImageMetadata

    width = 0
    height = 0
    dpi = Dpi()

    size_el = image_el.find("size")
    if size_el is not None:
        width = _parse_int_attr(size_el, "width") or 0
        height = _parse_int_attr(size_el, "height") or 0

    dpi_el = image_el.find("dpi")
    if dpi_el is not None:
        h_dpi = _parse_int_attr(dpi_el, "horizontal") or 0
        v_dpi = _parse_int_attr(dpi_el, "vertical") or 0
        dpi = Dpi(horizontal=h_dpi, vertical=v_dpi)

    return ImageMetadata(width=width, height=height, dpi=dpi)
