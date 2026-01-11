"""Content and page box data structures."""

from pydantic import BaseModel, Field


class ContentBox(BaseModel):
    """Represents the detected content rectangle.

    All coordinates are in pixels relative to the image origin.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def left(self) -> float:
        """Left edge x-coordinate."""
        return self.x

    @property
    def top(self) -> float:
        """Top edge y-coordinate."""
        return self.y

    @property
    def right(self) -> float:
        """Right edge x-coordinate."""
        return self.x + self.width

    @property
    def bottom(self) -> float:
        """Bottom edge y-coordinate."""
        return self.y + self.height

    @property
    def center_x(self) -> float:
        """Center x-coordinate."""
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        """Center y-coordinate."""
        return self.y + self.height / 2

    def is_empty(self) -> bool:
        """Return True if the box has zero area."""
        return self.width <= 0 or self.height <= 0

    def is_valid(self) -> bool:
        """Return True if the box has positive dimensions."""
        return self.width > 0 and self.height > 0

    def contains(self, x: float, y: float) -> bool:
        """Return True if the point is inside the box."""
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def intersects(self, other: ContentBox) -> bool:
        """Return True if this box overlaps with another."""
        return not (
            self.right < other.left
            or self.left > other.right
            or self.bottom < other.top
            or self.top > other.bottom
        )

    def intersection(self, other: ContentBox) -> ContentBox:
        """Return the intersection of this box with another."""
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        if right <= left or bottom <= top:
            return ContentBox()

        return ContentBox(x=left, y=top, width=right - left, height=bottom - top)

    def union(self, other: ContentBox) -> ContentBox:
        """Return the smallest box containing both boxes."""
        if self.is_empty():
            return other.model_copy()
        if other.is_empty():
            return self.model_copy()

        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)

        return ContentBox(x=left, y=top, width=right - left, height=bottom - top)

    def expanded(self, margin: float) -> ContentBox:
        """Return a new box expanded by the given margin on all sides."""
        return ContentBox(
            x=self.x - margin,
            y=self.y - margin,
            width=self.width + 2 * margin,
            height=self.height + 2 * margin,
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return box as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_ltrb(
        cls, left: float, top: float, right: float, bottom: float
    ) -> ContentBox:
        """Create a box from left, top, right, bottom coordinates."""
        return cls(x=left, y=top, width=right - left, height=bottom - top)


class PageBox(BaseModel):
    """Represents the detected page boundary rectangle.

    The page box is typically larger than or equal to the content box
    and represents the physical page boundaries.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def left(self) -> float:
        """Left edge x-coordinate."""
        return self.x

    @property
    def top(self) -> float:
        """Top edge y-coordinate."""
        return self.y

    @property
    def right(self) -> float:
        """Right edge x-coordinate."""
        return self.x + self.width

    @property
    def bottom(self) -> float:
        """Bottom edge y-coordinate."""
        return self.y + self.height

    def is_empty(self) -> bool:
        """Return True if the box has zero area."""
        return self.width <= 0 or self.height <= 0

    def is_valid(self) -> bool:
        """Return True if the box has positive dimensions."""
        return self.width > 0 and self.height > 0

    def to_content_box(self) -> ContentBox:
        """Convert to a ContentBox with the same dimensions."""
        return ContentBox(x=self.x, y=self.y, width=self.width, height=self.height)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return box as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_ltrb(cls, left: float, top: float, right: float, bottom: float) -> PageBox:
        """Create a box from left, top, right, bottom coordinates."""
        return cls(x=left, y=top, width=right - left, height=bottom - top)

    @classmethod
    def from_image_size(cls, width: int, height: int) -> PageBox:
        """Create a page box covering the entire image."""
        return cls(x=0, y=0, width=float(width), height=float(height))


class PhysicalSize(BaseModel):
    """Physical size in millimeters."""

    width_mm: float = Field(default=0.0, ge=0.0)
    height_mm: float = Field(default=0.0, ge=0.0)

    def is_empty(self) -> bool:
        """Return True if size is zero."""
        return self.width_mm <= 0 or self.height_mm <= 0

    @classmethod
    def from_pixels(
        cls, width_px: float, height_px: float, dpi: float = 300.0
    ) -> PhysicalSize:
        """Calculate physical size from pixel dimensions and DPI."""
        mm_per_inch = 25.4
        return cls(
            width_mm=width_px * mm_per_inch / dpi,
            height_mm=height_px * mm_per_inch / dpi,
        )
