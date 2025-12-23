"""Core data models for ScanTailor.

These models represent the fundamental data structures used throughout the
application: image identification, page references, resolution, margins,
and orientation.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

OrthogonalDegrees = Literal[0, 90, 180, 270]


class Dpi(BaseModel, frozen=True):
    """Dots per inch (horizontal and vertical resolution).

    Args:
        horizontal (int): Horizontal DPI value.
        vertical (int): Vertical DPI value.
    """

    horizontal: int = Field(default=0, ge=0)
    vertical: int = Field(default=0, ge=0)

    def is_null(self) -> bool:
        """Return True if either DPI value is effectively zero."""
        return self.horizontal <= 1 or self.vertical <= 1

    @classmethod
    def uniform(cls, dpi: int) -> Dpi:
        """Create a Dpi with equal horizontal and vertical values."""
        return cls(horizontal=dpi, vertical=dpi)


class ImageId(BaseModel, frozen=True):
    """Identifies a specific image, potentially within a multi-page file.

    Args:
        file_path (Path): Path to the image file.
        page (int): Page number (1-indexed) for multi-page files, 0 for single-page.
    """

    file_path: Path
    page: int = Field(default=0, ge=0)

    def is_null(self) -> bool:
        """Return True if this is a null/empty image ID."""
        path_str = str(self.file_path)
        return path_str == "" or path_str == "."

    def is_multi_page_file(self) -> bool:
        """Return True if this references a page in a multi-page file."""
        return self.page > 0

    def zero_based_page(self) -> int:
        """Return 0-indexed page number."""
        return self.page - 1 if self.page > 0 else 0

    def __hash__(self) -> int:
        """Return hash for use in sets and dicts."""
        return hash((self.file_path, self.page))


class SubPage(IntEnum):
    """Identifies which logical page within an image."""

    SINGLE_PAGE = 0
    LEFT_PAGE = 1
    RIGHT_PAGE = 2

    def __str__(self) -> str:
        """Return string representation (e.g., 'left-page')."""
        return self.name.lower().replace("_", "-")

    @classmethod
    def from_string(cls, value: str) -> SubPage:
        """Parse a SubPage from its string representation."""
        normalized = value.upper().replace("-", "_")
        return cls[normalized]


class PageId(BaseModel, frozen=True):
    """A logical page on an image.

    An image can contain one or two logical pages (e.g., a scan of an open book
    contains a left page and a right page).

    Args:
        image_id (ImageId): The source image.
        sub_page (SubPage): Which logical page within the image.
    """

    image_id: ImageId
    sub_page: SubPage = SubPage.SINGLE_PAGE

    def is_null(self) -> bool:
        """Return True if this is a null/empty page ID."""
        return self.image_id.is_null()

    def __hash__(self) -> int:
        """Return hash for use in sets and dicts."""
        return hash((self.image_id, self.sub_page))

    def __lt__(self, other: PageId) -> bool:
        """Enable sorting of PageIds."""
        if not isinstance(other, PageId):
            return NotImplemented
        if self.image_id.file_path != other.image_id.file_path:
            return self.image_id.file_path < other.image_id.file_path
        if self.image_id.page != other.image_id.page:
            return self.image_id.page < other.image_id.page
        return self.sub_page < other.sub_page


class Margins(BaseModel):
    """Margins around a page (in millimeters or other consistent units).

    Args:
        top (float): Top margin.
        bottom (float): Bottom margin.
        left (float): Left margin.
        right (float): Right margin.
    """

    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0

    @classmethod
    def uniform(cls, value: float) -> Margins:
        """Create margins with equal values on all sides."""
        return cls(top=value, bottom=value, left=value, right=value)


class OrthogonalRotation(BaseModel):
    """Represents a rotation by a multiple of 90 degrees.

    Args:
        degrees (OrthogonalDegrees): Rotation in degrees (must be 0, 90, 180, or 270).
    """

    degrees: OrthogonalDegrees = 0

    def rotate_clockwise(self) -> OrthogonalRotation:
        """Return a new rotation rotated 90 degrees clockwise."""
        new_degrees = cast(OrthogonalDegrees, (self.degrees + 90) % 360)
        return OrthogonalRotation(degrees=new_degrees)

    def rotate_counter_clockwise(self) -> OrthogonalRotation:
        """Return a new rotation rotated 90 degrees counter-clockwise."""
        new_degrees = cast(OrthogonalDegrees, (self.degrees - 90) % 360)
        return OrthogonalRotation(degrees=new_degrees)

    def rotate_dimensions(self, width: float, height: float) -> tuple[float, float]:
        """Apply rotation to dimensions, returning (new_width, new_height)."""
        if self.degrees in (90, 270):
            return (height, width)
        return (width, height)

    def unrotate_dimensions(self, width: float, height: float) -> tuple[float, float]:
        """Reverse rotation on dimensions.

        Returns:
            tuple[float, float]: The (original_width, original_height).
        """
        return self.rotate_dimensions(width, height)

    def rotate_point(
        self, x: float, y: float, max_x: float, max_y: float
    ) -> tuple[float, float]:
        """Rotate a point within a bounding box.

        Args:
            x (float): X coordinate of the point.
            y (float): Y coordinate of the point.
            max_x (float): Maximum X value (width - 1 or similar bound).
            max_y (float): Maximum Y value (height - 1 or similar bound).

        Returns:
            tuple[float, float]: The rotated (x, y) coordinates.
        """
        if self.degrees == 0:
            return (x, y)
        if self.degrees == 90:
            return (max_y - y, x)
        if self.degrees == 180:
            return (max_x - x, max_y - y)
        # 270 degrees
        return (y, max_x - x)

    def unrotate_point(
        self, x: float, y: float, max_x: float, max_y: float
    ) -> tuple[float, float]:
        """Reverse rotation on a point within a bounding box.

        Args:
            x (float): X coordinate of the rotated point.
            y (float): Y coordinate of the rotated point.
            max_x (float): Maximum X value in the rotated coordinate system.
            max_y (float): Maximum Y value in the rotated coordinate system.

        Returns:
            tuple[float, float]: The original (x, y) coordinates before rotation.
        """
        if self.degrees == 0:
            return (x, y)
        if self.degrees == 90:
            # Inverse of (max_y - y, x) is (y, max_x - x) applied to rotated coords
            return (y, max_x - x)
        if self.degrees == 180:
            # Inverse of 180 is same as 180
            return (max_x - x, max_y - y)
        # 270 degrees - inverse of (y, max_x - x) is (max_y - y, x)
        return (max_y - y, x)
