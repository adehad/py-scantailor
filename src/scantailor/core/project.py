"""Project management for ScanTailor.

Handles loading, saving, and managing ScanTailor projects. Supports:
- JSON format (primary, for new projects)
- XML format (read-only, for C++ ScanTailor compatibility)
"""

import json
import os
from collections.abc import Callable, Iterator
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field

from scantailor.core.models import Dpi, ImageId, PageId, SubPage

LayoutDirection = Literal["LTR", "RTL"]

_MM_PER_INCH = 25.4
_MIN_RELIABLE_DPI = 150
_MAX_DPI = 9999
_MAX_PHYSICAL_SIZE_MM = 500


class DpiStatus(IntEnum):
    """Status of DPI validation for an image."""

    OK = 0
    UNDEFINED = 1
    TOO_LARGE = 2
    TOO_SMALL = 3
    TOO_SMALL_FOR_PIXEL_SIZE = 4


class ImageMetadata(BaseModel):
    """Metadata for an image file.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        dpi (Dpi): Resolution of the image.
    """

    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    dpi: Dpi = Field(default_factory=Dpi)

    def is_dpi_ok(self) -> bool:
        """Check that the DPI is defined on both axes.

        Implausible but defined values still count as OK; only the finer
        ``*_dpi_status`` methods report them.
        """
        return (
            self.horizontal_dpi_status() != DpiStatus.UNDEFINED
            and self.vertical_dpi_status() != DpiStatus.UNDEFINED
        )

    def has_plausible_dpi(self) -> bool:
        """Check that both DPI values are usable for processing.

        Unlike ``is_dpi_ok`` this also rejects values that are defined but out
        of range for the image size, which is when the user should be asked to
        correct them.
        """
        return (
            self.horizontal_dpi_status() == DpiStatus.OK
            and self.vertical_dpi_status() == DpiStatus.OK
        )

    def horizontal_dpi_status(self) -> DpiStatus:
        """Check the status of horizontal DPI."""
        return self._dpi_status(self.width, self.dpi.horizontal)

    def vertical_dpi_status(self) -> DpiStatus:
        """Check the status of vertical DPI."""
        return self._dpi_status(self.height, self.dpi.vertical)

    @staticmethod
    def _dpi_status(pixel_size: int, dpi: int) -> DpiStatus:
        """Determine DPI status based on pixel size and DPI value.

        Args:
            pixel_size (int): Size in pixels along the axis the DPI applies to.
            dpi (int): DPI value.

        Returns:
            DpiStatus: The status of the DPI value.
        """
        if dpi <= 1:
            return DpiStatus.UNDEFINED
        if dpi < _MIN_RELIABLE_DPI:
            return DpiStatus.TOO_SMALL
        if dpi > _MAX_DPI:
            return DpiStatus.TOO_LARGE
        # A physical size this large usually means the DPI is wrong, and a DPI
        # that is too low inflates memory use quadratically.
        if _MM_PER_INCH * pixel_size / dpi > _MAX_PHYSICAL_SIZE_MM:
            return DpiStatus.TOO_SMALL_FOR_PIXEL_SIZE
        return DpiStatus.OK


class ImageInfo(BaseModel):
    """Information about an image in the project.

    Args:
        id (ImageId): Identifier for the image.
        metadata (ImageMetadata): Image dimensions and DPI.
        num_sub_pages (int): Number of logical pages (1 or 2).
        left_half_removed (bool): Whether left half was removed.
        right_half_removed (bool): Whether right half was removed.
    """

    id: ImageId
    metadata: ImageMetadata = Field(default_factory=ImageMetadata)
    num_sub_pages: int = Field(default=1, ge=1, le=2)
    left_half_removed: bool = False
    right_half_removed: bool = False


class PageInfo(BaseModel):
    """Information about a logical page.

    Args:
        page_id (PageId): Identifier for the page.
        image_info (ImageInfo): Information about the source image.
    """

    page_id: PageId
    image_info: ImageInfo


class StageParamsEntry(BaseModel):
    """Serialised parameters of one filter stage for one page.

    Args:
        page (PageId): The page the parameters belong to. Stages that store
            parameters per image use the image's single-page id.
        params (dict[str, Any]): The stage's Params model dumped as JSON data.
    """

    page: PageId
    params: dict[str, Any]


class Project(BaseModel):
    """A ScanTailor project.

    Args:
        version (int): Project file format version.
        output_directory (Path): Directory for output files.
        layout_direction (LayoutDirection): Text layout direction (LTR or RTL).
        images (list[ImageInfo]): List of images in the project.
        selected_page (PageId | None): Currently selected page.
        stage_params (dict[str, list[StageParamsEntry]]): Explicitly set per-page
            parameters, keyed by stage key (see ``core.stage_params``).
    """

    version: int = 1
    output_directory: Path = Field(default_factory=lambda: Path("out"))
    layout_direction: LayoutDirection = "LTR"
    images: list[ImageInfo] = Field(default_factory=list)
    selected_page: PageId | None = None
    stage_params: dict[str, list[StageParamsEntry]] = Field(default_factory=dict)

    def get_pages(self) -> list[PageId]:
        """Get all page IDs in the project, respecting layout direction.

        Returns:
            list[PageId]: All pages in display order.
        """
        pages: list[PageId] = []
        for image_info in self.images:
            if image_info.num_sub_pages == 1:
                # Single page or one half removed
                if image_info.left_half_removed:
                    sub_page = SubPage.RIGHT_PAGE
                elif image_info.right_half_removed:
                    sub_page = SubPage.LEFT_PAGE
                else:
                    sub_page = SubPage.SINGLE_PAGE
                pages.append(PageId(image_id=image_info.id, sub_page=sub_page))
            else:
                # Two pages
                if self.layout_direction == "LTR":
                    pages.append(
                        PageId(image_id=image_info.id, sub_page=SubPage.LEFT_PAGE)
                    )
                    pages.append(
                        PageId(image_id=image_info.id, sub_page=SubPage.RIGHT_PAGE)
                    )
                else:  # RTL
                    pages.append(
                        PageId(image_id=image_info.id, sub_page=SubPage.RIGHT_PAGE)
                    )
                    pages.append(
                        PageId(image_id=image_info.id, sub_page=SubPage.LEFT_PAGE)
                    )
        return pages

    def add_image(
        self, image_id: ImageId, metadata: ImageMetadata | None = None
    ) -> ImageInfo:
        """Add an image to the project.

        Args:
            image_id (ImageId): The image identifier.
            metadata (ImageMetadata | None): Optional metadata for the image.

        Returns:
            ImageInfo: The created image info.
        """
        info = ImageInfo(
            id=image_id,
            metadata=metadata or ImageMetadata(),
        )
        self.images.append(info)
        return info

    def image_ids(self) -> list[ImageId]:
        """Return the id of every image in the project."""
        return [info.id for info in self.images]

    def remap_image_paths(self, remap: Callable[[Path], Path]) -> None:
        """Replace the file path of every image, including in stored parameters.

        Args:
            remap (Callable[[Path], Path]): Maps an image file path to its new path.
        """

        def remap_id(image_id: ImageId) -> ImageId:
            return ImageId(file_path=remap(image_id.file_path), page=image_id.page)

        def remap_page(page: PageId) -> PageId:
            return PageId(image_id=remap_id(page.image_id), sub_page=page.sub_page)

        self.images = [
            info.model_copy(update={"id": remap_id(info.id)}) for info in self.images
        ]
        if self.selected_page is not None:
            self.selected_page = remap_page(self.selected_page)
        self.stage_params = {
            stage: [
                StageParamsEntry(page=remap_page(entry.page), params=entry.params)
                for entry in entries
            ]
            for stage, entries in self.stage_params.items()
        }

    def save(self, path: Path) -> None:
        """Save the project to a JSON file.

        Image and output paths are written relative to the project file when
        they can be, so a project moved together with its images still opens.

        Args:
            path (Path): Path to save the project to.
        """
        data = self.model_dump(mode="json")
        base = path.resolve().parent
        _map_paths(data, lambda stored: _to_stored_path(Path(stored), base))
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a project from a file.

        Automatically detects JSON or XML format.

        Args:
            path (Path): Path to the project file.

        Returns:
            Project: The loaded project.

        Raises:
            ValueError: If the file format is not recognized.
        """
        content = path.read_text(encoding="utf-8")

        # Detect format by content
        stripped = content.lstrip()
        if stripped.startswith("{"):
            data = json.loads(content)
            base = path.resolve().parent
            _map_paths(data, lambda stored: _from_stored_path(stored, base))
            return cls.model_validate(data)
        if stripped.startswith("<?xml") or stripped.startswith("<project"):
            return cls._load_xml(path, content)

        msg = f"Unknown project file format: {path}"
        raise ValueError(msg)

    @classmethod
    def _load_xml(cls, path: Path, content: str) -> Self:
        """Load a project from C++ ScanTailor XML format.

        Args:
            path (Path): Path to the project file (for resolving relative paths).
            content (str): XML content.

        Returns:
            Project: The loaded project.
        """
        # Import here to avoid hard dependency on lxml at module load
        from scantailor.core.xml_compat import parse_xml_project

        return parse_xml_project(path, content)


def _is_drive_less_rooted(path: Path) -> bool:
    """Whether ``path`` starts at a root but names no drive (``/x`` on Windows).

    Such a path is neither absolute nor relative on Windows, so it is kept as is.
    """
    return bool(path.root) and not path.drive and not path.is_absolute()


def _to_stored_path(path: Path, base: Path) -> str:
    """Express ``path`` relative to ``base`` with forward slashes when possible.

    Args:
        path (Path): Path to store; relative paths are taken relative to the
            working directory.
        base (Path): Absolute directory of the project file.

    Returns:
        str: The relative path, or the absolute path when none exists (e.g. on
            another drive).
    """
    if _is_drive_less_rooted(path):
        return path.as_posix()
    absolute = Path(os.path.abspath(path))
    try:
        return Path(os.path.relpath(absolute, base)).as_posix()
    except ValueError:
        return absolute.as_posix()


def _from_stored_path(stored: str, base: Path) -> str:
    """Resolve a stored path against the project file's directory.

    Args:
        stored (str): Absolute path, or one relative to ``base``.
        base (Path): Absolute directory of the project file.

    Returns:
        str: The absolute, normalised path.
    """
    path = Path(stored)
    if path.is_absolute() or _is_drive_less_rooted(path):
        return str(path)
    return str(Path(os.path.normpath(base / path)))


def _dicts(value: object) -> list[dict[str, Any]]:
    """Return the mappings in a JSON list, or nothing for any other value."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _path_holders(data: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield every ``(mapping, key)`` of a dumped project that holds a path.

    Malformed parts are skipped so validation reports them instead.
    """
    yield data, "output_directory"
    image_ids = [info.get("id") for info in _dicts(data.get("images"))]
    pages = [data.get("selected_page")]
    stage_params = data.get("stage_params")
    if isinstance(stage_params, dict):
        for entries in stage_params.values():
            pages.extend(entry.get("page") for entry in _dicts(entries))
    image_ids.extend(page.get("image_id") for page in pages if isinstance(page, dict))
    for image_id in image_ids:
        if isinstance(image_id, dict):
            yield image_id, "file_path"


def _map_paths(data: dict[str, Any], convert: Callable[[str], str]) -> None:
    """Convert, in place, every path string of a dumped project.

    Args:
        data (dict[str, Any]): Project data as produced by ``model_dump(mode="json")``.
        convert (Callable[[str], str]): Maps one stored path string to another.
    """
    if not isinstance(data, dict):
        return
    for holder, key in _path_holders(data):
        value = holder.get(key)
        if isinstance(value, str):
            holder[key] = convert(value)
