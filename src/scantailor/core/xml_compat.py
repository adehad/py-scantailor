"""XML compatibility layer for reading C++ ScanTailor project files.

This module provides read-only support for the XML format used by the
original C++ ScanTailor application: the images, pages and sub-pages, per-page
DPI, and the settings of every filter stage. A project read from XML is saved
in the JSON format, which stays the only format the port writes.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from loguru import logger

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.project import (
    ImageInfo,
    ImageMetadata,
    OutputFileLabel,
    Project,
    StageParamsEntry,
)
from scantailor.core.stage_params import STAGE_KEYS
from scantailor.core.stage_sequence import FilterStage
from scantailor.core.xml_reader import (
    XmlFormatError,
    XmlProjectContext,
    attr_int,
    attr_text,
    child,
    children,
    decode_project_bytes,
    skip_malformed,
    sub_page_from_string,
)
from scantailor.filters.deskew.xml_import import read_deskew
from scantailor.filters.fix_orientation.xml_import import read_rotations
from scantailor.filters.output.xml_import import read_output
from scantailor.filters.page_layout.xml_import import read_page_layout
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.xml_import import read_page_split
from scantailor.filters.select_content.xml_import import read_select_content

if TYPE_CHECKING:
    from scantailor.core.stage_sequence import StageSequence

# The newest project format the C++ application reads and writes.
NEWEST_PROJECT_VERSION = 3

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class XmlStageDefaults:
    """Stage-wide values a C++ project stores outside any page.

    Attributes:
        page_split_layout_type (LayoutType): Layout type of images that were never
            given one.
        page_detection_box (tuple[float, float]): Width and height in millimetres
            of the box Select Content's page detection looks for.
        page_detection_tolerance (float): How far a detected page may deviate
            from that box, as a fraction.
    """

    page_split_layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE
    page_detection_box: tuple[float, float] = (0.0, 0.0)
    page_detection_tolerance: float = 0.1

    def as_project_defaults(self) -> dict[str, object]:
        """Return the values in the form ``Project.stage_defaults`` stores."""
        return {
            "page_split_layout_type": self.page_split_layout_type.value,
            "page_detection_box": list(self.page_detection_box),
            "page_detection_tolerance": self.page_detection_tolerance,
        }


@dataclass(frozen=True)
class XmlProjectImport:
    """Everything read from a C++ project file.

    Attributes:
        project (Project): Images, pages and per-page stage parameters.
        stage_defaults (XmlStageDefaults): Stage-wide values, to install into the
            stages with :func:`apply_xml_stage_defaults`.
        output_file_labels (dict[Path, int]): The number C++ gave each source
            file whose name another source file shares; it appears in the
            output file name as ``name(label).tif``. Files absent from the
            mapping have label 0.
    """

    project: Project
    stage_defaults: XmlStageDefaults = field(default_factory=XmlStageDefaults)
    output_file_labels: dict[Path, int] = field(default_factory=dict)


def parse_xml_project(path: Path, content: str) -> Project:
    """Parse a C++ ScanTailor XML project file.

    Args:
        path (Path): Path to the project file (for resolving relative paths).
        content (str): XML content.

    Returns:
        Project: The parsed project, with the filter settings in
            ``stage_params``.

    Raises:
        ValueError: If the XML is invalid, is not a project, or was written by
            a newer version of the application than this one reads.
    """
    return import_xml_project(path, content).project


def import_xml_project(path: Path, content: str) -> XmlProjectImport:
    """Parse a C++ ScanTailor XML project file, keeping what lies outside the pages.

    Malformed pages and settings are logged and skipped; the rest of the
    project is still read.

    Args:
        path (Path): Path to the project file (for resolving relative paths).
        content (str): XML content.

    Returns:
        XmlProjectImport: The project with the stage-wide values and the output
            file name labels.

    Raises:
        ValueError: If the XML is invalid, is not a project, or was written by
            a newer version of the application than this one reads.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        msg = f"The project file is not well-formed XML: {error}"
        raise ValueError(msg) from error

    if root.tag != "project":
        msg = f"Expected <project> root element, got <{root.tag}>"
        raise ValueError(msg)

    version = _read_version(root)

    base = path.resolve().parent
    output_dir = Path(root.get("outputDirectory", "out"))
    if not _is_rooted(str(output_dir)):
        output_dir = base / output_dir
    layout_direction = "RTL" if root.get("layoutDirection") == "RTL" else "LTR"

    dir_map = _read_directories(child(root, "directories"))
    file_map = _read_files(child(root, "files"), dir_map, base)
    images, image_map = _read_images(child(root, "images"), file_map)
    page_map, selected_page = _read_pages(child(root, "pages"), image_map)

    context = XmlProjectContext(
        image_ids={key: info.id for key, info in image_map.items()},
        page_ids=page_map,
    )
    stage_params, stage_defaults = _read_filters(child(root, "filters"), context)
    labels = _read_output_file_labels(child(root, "file-name-disambiguation"), file_map)

    logger.info(
        "Loaded XML project with {} images from {}",
        len(images),
        path,
    )
    project = Project(
        version=version,
        output_directory=output_dir,
        layout_direction=layout_direction,
        images=images,
        selected_page=selected_page,
        stage_params=stage_params,
        stage_defaults=stage_defaults.as_project_defaults(),
        output_file_labels=[
            OutputFileLabel(file_path=file_path, label=max(label, 0))
            for file_path, label in labels.items()
        ],
    )
    return XmlProjectImport(project, stage_defaults, labels)


def import_xml_project_file(path: Path) -> XmlProjectImport | None:
    """Read a project file if it is in the C++ XML format.

    Args:
        path (Path): The project file.

    Returns:
        XmlProjectImport | None: The import, or None if the file is not XML
            (a JSON project, for instance).

    Raises:
        OSError: If the file cannot be read.
        ValueError: As :func:`import_xml_project`.
    """
    content = decode_project_bytes(path.read_bytes())
    if not content.lstrip().startswith(("<?xml", "<project")):
        return None
    return import_xml_project(path, content)


def apply_xml_stage_defaults(
    stages: "StageSequence", defaults: XmlStageDefaults
) -> None:
    """Install the stage-wide values of a C++ project into the stages.

    Args:
        stages (StageSequence): The stages of the project being opened.
        defaults (XmlStageDefaults): Values from :func:`import_xml_project`.
    """
    stages.page_split_filter.settings.set_default_layout_type(
        defaults.page_split_layout_type
    )
    select_content = stages.select_content_filter.settings
    select_content.set_page_detection_box(*defaults.page_detection_box)
    select_content.set_page_detection_tolerance(defaults.page_detection_tolerance)


def _read_version(root: ET.Element) -> int:
    """Read and check the project format version.

    Raises:
        ValueError: If the version is missing, unreadable, or newer than the
            newest this application reads.
    """
    text = root.get("version")
    if text is None:
        msg = "The project file has no version attribute."
        raise ValueError(msg)
    try:
        version = int(text)
    except ValueError as error:
        msg = f"The project file has an unreadable version {text!r}."
        raise ValueError(msg) from error
    if version > NEWEST_PROJECT_VERSION:
        msg = (
            f"The project file is version {version}, written by a newer "
            f"application; this one reads up to version {NEWEST_PROJECT_VERSION}."
        )
        raise ValueError(msg)
    if version < 1:
        msg = f"The project file has an invalid version {version}."
        raise ValueError(msg)
    return version


def _is_rooted(path: str) -> bool:
    """Tell whether a stored path is absolute on either Windows or POSIX."""
    return bool(
        _WINDOWS_DRIVE.match(path)
        or path.startswith(("/", "\\"))
        or Path(path).is_absolute()
    )


def _read_directories(element: ET.Element | None) -> dict[int, str]:
    """Read the ``<directories>`` table: id to path."""
    directories: dict[int, str] = {}
    for directory in children(element, "directory"):
        try:
            directory_id = int(directory.get("id", ""))
        except ValueError:
            continue
        directory_path = directory.get("path", "")
        if directory_path:
            directories[directory_id] = directory_path
    return directories


def _read_files(
    element: ET.Element | None, directories: dict[int, str], base: Path
) -> dict[int, tuple[Path, bool]]:
    """Read the ``<files>`` table: id to (path, legacy multi-page flag).

    A relative directory is taken relative to the project file's folder.
    """
    files: dict[int, tuple[Path, bool]] = {}
    for file in children(element, "file"):
        try:
            file_id = int(file.get("id", ""))
            directory_id = int(file.get("dirId", ""))
        except ValueError:
            continue
        name = file.get("name", "")
        directory = directories.get(directory_id, "")
        if not name or not directory:
            continue
        directory_path = Path(directory)
        if not _is_rooted(directory):
            directory_path = base / directory_path
        files[file_id] = (directory_path / name, file.get("multiPage") == "1")
    return files


def _read_images(
    element: ET.Element | None, files: dict[int, tuple[Path, bool]]
) -> tuple[list[ImageInfo], dict[int, ImageInfo]]:
    """Read the ``<images>`` table."""
    images: list[ImageInfo] = []
    by_id: dict[int, ImageInfo] = {}
    for image in children(element, "image"):
        with skip_malformed("image", image):
            info = _read_image(image, files)
            if info is None:
                continue
            images.append(info)
            by_id[attr_int(image, "id")] = info
    return images, by_id


def _read_image(
    image: ET.Element, files: dict[int, tuple[Path, bool]]
) -> ImageInfo | None:
    """Read one ``<image>`` element; None if it refers to an unknown file.

    Raises:
        ValueError: If a required attribute is missing or unreadable.
    """
    for required in ("id", "subPages", "fileId", "fileImage"):
        if required not in image.attrib:
            msg = f"image lacks the {required} attribute"
            raise XmlFormatError(msg)
    record = files.get(attr_int(image, "fileId"))
    if record is None:
        return None
    file_path, legacy_multi_page = record
    # Older projects flagged multi-page files on the file instead of numbering
    # the pages from one.
    page = attr_int(image, "fileImage") + (1 if legacy_multi_page else 0)
    removed = attr_text(image, "removed")
    return ImageInfo(
        id=ImageId(file_path=file_path, page=page),
        metadata=_read_metadata(image),
        num_sub_pages=attr_int(image, "subPages"),
        left_half_removed=removed == "L",
        right_half_removed=removed == "R",
    )


def _read_metadata(image: ET.Element) -> ImageMetadata:
    """Read the size and DPI of an ``<image>`` element."""
    width, height = 0, 0
    size = child(image, "size")
    if size is not None:
        width = max(0, attr_int(size, "width"))
        height = max(0, attr_int(size, "height"))
    dpi = Dpi()
    dpi_element = child(image, "dpi")
    if dpi_element is not None:
        dpi = Dpi(
            horizontal=max(0, attr_int(dpi_element, "horizontal")),
            vertical=max(0, attr_int(dpi_element, "vertical")),
        )
    return ImageMetadata(width=width, height=height, dpi=dpi)


def _read_pages(
    element: ET.Element | None, images: dict[int, ImageInfo]
) -> tuple[dict[int, PageId], PageId | None]:
    """Read the ``<pages>`` table: page id to page, and the selected page."""
    pages: dict[int, PageId] = {}
    selected: PageId | None = None
    for page in children(element, "page"):
        with skip_malformed("page", page):
            try:
                page_id = int(page.get("id", ""))
                image_id = int(page.get("imageId", ""))
            except ValueError:
                continue
            sub_page: SubPage = sub_page_from_string(page.get("subPage", ""))
            image = images.get(image_id)
            if image is None:
                continue
            pages[page_id] = PageId(image_id=image.id, sub_page=sub_page)
            if page.get("selected") == "selected":
                selected = pages[page_id]
    return pages, selected


def _read_filters(
    element: ET.Element | None, context: XmlProjectContext
) -> tuple[dict[str, list[StageParamsEntry]], XmlStageDefaults]:
    """Read the ``<filters>`` element into stage parameters and stage defaults."""
    page_split = read_page_split(child(element, "page-split"), context)
    select_content = read_select_content(child(element, "select-content"), context)
    per_stage: dict[FilterStage, list[StageParamsEntry]] = {
        FilterStage.FIX_ORIENTATION: read_rotations(
            child(element, "fix-orientation"), context
        ),
        FilterStage.PAGE_SPLIT: page_split.entries,
        FilterStage.DESKEW: read_deskew(child(element, "deskew"), context),
        FilterStage.SELECT_CONTENT: select_content.entries,
        FilterStage.PAGE_LAYOUT: read_page_layout(
            child(element, "page-layout"), context
        ),
        FilterStage.OUTPUT: read_output(child(element, "output"), context),
    }
    stage_params = {
        STAGE_KEYS[stage]: entries for stage, entries in per_stage.items() if entries
    }
    defaults = XmlStageDefaults(
        page_split_layout_type=page_split.default_layout_type,
        page_detection_box=select_content.page_detection_box,
        page_detection_tolerance=select_content.page_detection_tolerance,
    )
    return stage_params, defaults


def _read_output_file_labels(
    element: ET.Element | None, files: dict[int, tuple[Path, bool]]
) -> dict[Path, int]:
    """Read the ``<file-name-disambiguation>`` mappings: source file to label."""
    labels: dict[Path, int] = {}
    for mapping in children(element, "mapping"):
        with skip_malformed("file name mapping"):
            try:
                file_id = int(mapping.get("file", ""))
            except ValueError:
                continue
            record = files.get(file_id)
            if record is not None:
                labels[record[0]] = attr_int(mapping, "label")
    return labels
