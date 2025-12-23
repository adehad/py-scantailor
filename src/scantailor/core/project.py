"""Project management for ScanTailor.

Handles loading, saving, and managing ScanTailor projects. Supports:
- JSON format (primary, for new projects)
- XML format (read-only, for C++ ScanTailor compatibility)
"""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from scantailor.core.models import Dpi, ImageId, Margins, PageId, SubPage

LayoutDirection = Literal["LTR", "RTL"]


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
        """Check if DPI values are within acceptable range."""
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
            pixel_size (int): Size in pixels.
            dpi (int): DPI value.

        Returns:
            DpiStatus: The status of the DPI value.
        """
        if dpi <= 0:
            return DpiStatus.UNDEFINED
        if dpi > 9999:
            return DpiStatus.TOO_LARGE
        if dpi < 25:
            return DpiStatus.TOO_SMALL
        # Check if DPI is too small for the pixel size (huge images need higher DPI)
        if pixel_size > 0 and dpi < 150 and pixel_size > 10000:
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


class Project(BaseModel):
    """A ScanTailor project.

    Args:
        version (int): Project file format version.
        output_directory (Path): Directory for output files.
        layout_direction (LayoutDirection): Text layout direction (LTR or RTL).
        images (list[ImageInfo]): List of images in the project.
        selected_page (PageId | None): Currently selected page.
    """

    version: int = 1
    output_directory: Path = Field(default_factory=lambda: Path("out"))
    layout_direction: LayoutDirection = "LTR"
    images: list[ImageInfo] = Field(default_factory=list)
    selected_page: PageId | None = None

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
                    pages.append(PageId(image_id=image_info.id, sub_page=SubPage.LEFT_PAGE))
                    pages.append(PageId(image_id=image_info.id, sub_page=SubPage.RIGHT_PAGE))
                else:  # RTL
                    pages.append(PageId(image_id=image_info.id, sub_page=SubPage.RIGHT_PAGE))
                    pages.append(PageId(image_id=image_info.id, sub_page=SubPage.LEFT_PAGE))
        return pages

    def add_image(self, image_id: ImageId, metadata: ImageMetadata | None = None) -> ImageInfo:
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

    def save(self, path: Path) -> None:
        """Save the project to a JSON file.

        Args:
            path (Path): Path to save the project to.
        """
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Project:
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
            return cls.model_validate_json(content)
        if stripped.startswith("<?xml") or stripped.startswith("<project"):
            return cls._load_xml(path, content)

        msg = f"Unknown project file format: {path}"
        raise ValueError(msg)

    @classmethod
    def _load_xml(cls, path: Path, content: str) -> Project:
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
