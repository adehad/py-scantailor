"""Page layout of a split scan: the outline of the image and the cut lines in it.

A :class:`PageLayout` stores the outline of the whole image and up to two cutter
lines. Each cutter runs the full height of the image, and a layout reads its
position at mid height. :class:`PageLayoutType` tells how the cutters divide the
image: not at all, into a single page with two cutters that bound the content, or
into two pages with one cutter between them.
"""

from enum import Enum
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from scantailor.core.models import SubPage

from .layout_type import LayoutType


class PageLayoutType(Enum):
    """Which cutters a page layout has.

    SINGLE_PAGE_UNCUT: no cutters; the whole image is one page.
    SINGLE_PAGE_CUT: two cutters that bound the content of one page.
    TWO_PAGES: one cutter that divides the image into a left and a right page.
    """

    SINGLE_PAGE_UNCUT = "single-uncut"
    SINGLE_PAGE_CUT = "single-cut"
    TWO_PAGES = "two-pages"


class PageLayout(BaseModel):
    """Represents how an image is divided into pages.

    The page layout comprises:
    - A rectangular outline (the full image bounds).
    - Layout type indicator.
    - Zero, one, or two cutter lines.

    For SINGLE_PAGE_UNCUT: no cutters.
    For TWO_PAGES: one cutter (split line) dividing into left and right pages.
    For SINGLE_PAGE_CUT: two cutters defining the content area.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # The image bounds as (x, y, width, height)
    outline: tuple[float, float, float, float]

    # Layout type
    layout_type: PageLayoutType

    # Cutter lines as ((x1, y1), (x2, y2)) - can have 0, 1, or 2
    cutter1: tuple[tuple[float, float], tuple[float, float]] | None = None
    cutter2: tuple[tuple[float, float], tuple[float, float]] | None = None

    @field_validator("outline")
    @classmethod
    def validate_outline(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Validate outline dimensions are positive."""
        x, y, w, h = v
        if w <= 0 or h <= 0:
            msg = "Outline width and height must be positive"
            raise ValueError(msg)
        return v

    @classmethod
    def single_page_uncut(
        cls, width: float, height: float, x: float = 0, y: float = 0
    ) -> Self:
        """Create a single page layout with no cutting.

        Args:
            width: Image width.
            height: Image height.
            x: X offset (default 0).
            y: Y offset (default 0).

        Returns:
            PageLayout for a single uncut page.
        """
        return cls(
            outline=(x, y, width, height),
            layout_type=PageLayoutType.SINGLE_PAGE_UNCUT,
        )

    @classmethod
    def two_pages(
        cls,
        width: float,
        height: float,
        split_x: float,
        x: float = 0,
        y: float = 0,
    ) -> Self:
        """Create a two-page layout split at the given x position.

        Args:
            width: Image width.
            height: Image height.
            split_x: X coordinate of the split line.
            x: X offset (default 0).
            y: Y offset (default 0).

        Returns:
            PageLayout for two pages.
        """
        split_line = ((split_x, y), (split_x, y + height))
        return cls(
            outline=(x, y, width, height),
            layout_type=PageLayoutType.TWO_PAGES,
            cutter1=split_line,
            cutter2=split_line,  # Both cutters same for two-page
        )

    @classmethod
    def single_page_cut(
        cls,
        width: float,
        height: float,
        left_x: float,
        right_x: float,
        x: float = 0,
        y: float = 0,
    ) -> Self:
        """Create a single page layout with cutters on left and right.

        Args:
            width: Image width.
            height: Image height.
            left_x: X coordinate of the left cutter.
            right_x: X coordinate of the right cutter.
            x: X offset (default 0).
            y: Y offset (default 0).

        Returns:
            PageLayout for a single cut page.
        """
        left_line = ((left_x, y), (left_x, y + height))
        right_line = ((right_x, y), (right_x, y + height))
        return cls(
            outline=(x, y, width, height),
            layout_type=PageLayoutType.SINGLE_PAGE_CUT,
            cutter1=left_line,
            cutter2=right_line,
        )

    @property
    def num_cutters(self) -> int:
        """Return the number of cutters for this layout type."""
        if self.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT:
            return 0
        if self.layout_type == PageLayoutType.TWO_PAGES:
            return 1
        return 2  # SINGLE_PAGE_CUT

    @property
    def num_sub_pages(self) -> int:
        """Return the number of logical pages (1 or 2)."""
        return 2 if self.layout_type == PageLayoutType.TWO_PAGES else 1

    def get_outline_polygon(self) -> NDArray[np.floating]:
        """Get the outline as a polygon array.

        Returns:
            Array of shape (4, 2) with vertices in clockwise order
            starting from top-left.
        """
        x, y, w, h = self.outline
        return np.array(
            [
                [x, y],  # Top-left
                [x + w, y],  # Top-right
                [x + w, y + h],  # Bottom-right
                [x, y + h],  # Bottom-left
            ],
            dtype=np.float64,
        )

    def get_split_line_x(self) -> float | None:
        """Get the X coordinate of the split line for TWO_PAGES layout.

        Returns:
            The X coordinate or None if not a TWO_PAGES layout.
        """
        if self.layout_type != PageLayoutType.TWO_PAGES:
            return None
        if self.cutter1 is None:
            return None
        return (self.cutter1[0][0] + self.cutter1[1][0]) / 2

    def get_left_page_outline(self) -> NDArray[np.floating] | None:
        """Get the outline of the left page for TWO_PAGES layout.

        Returns:
            Polygon array or None if not TWO_PAGES.
        """
        if self.layout_type != PageLayoutType.TWO_PAGES:
            return None

        split_x = self.get_split_line_x()
        if split_x is None:
            return None

        assert self.cutter1 is not None
        if _is_vertical(self.cutter1):
            x, y, _w, h = self.outline
            return np.array(
                [[x, y], [split_x, y], [split_x, y + h], [x, y + h]],
                dtype=np.float64,
            )
        return _keep_side(self.get_outline_polygon(), self.cutter1, keep_left=True)

    def get_right_page_outline(self) -> NDArray[np.floating] | None:
        """Get the outline of the right page for TWO_PAGES layout.

        Returns:
            Polygon array or None if not TWO_PAGES.
        """
        if self.layout_type != PageLayoutType.TWO_PAGES:
            return None

        split_x = self.get_split_line_x()
        if split_x is None:
            return None

        assert self.cutter1 is not None
        if _is_vertical(self.cutter1):
            x, y, w, h = self.outline
            right_edge = x + w
            return np.array(
                [[split_x, y], [right_edge, y], [right_edge, y + h], [split_x, y + h]],
                dtype=np.float64,
            )
        return _keep_side(self.get_outline_polygon(), self.cutter1, keep_left=False)

    def get_single_page_outline(self) -> NDArray[np.floating] | None:
        """Get the outline for single page layouts.

        For SINGLE_PAGE_UNCUT, returns the full outline.
        For SINGLE_PAGE_CUT, returns the area between cutters.

        Returns:
            Polygon array or None if TWO_PAGES.
        """
        if self.layout_type == PageLayoutType.TWO_PAGES:
            return None

        if self.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT:
            return self.get_outline_polygon()

        # SINGLE_PAGE_CUT
        if self.cutter1 is None or self.cutter2 is None:
            return self.get_outline_polygon()

        left_line, right_line = self.cutter1, self.cutter2
        if _mid_x(left_line) > _mid_x(right_line):
            left_line, right_line = right_line, left_line

        if _is_vertical(left_line) and _is_vertical(right_line):
            x, y, w, h = self.outline
            left_x, right_x = _mid_x(left_line), _mid_x(right_line)
            return np.array(
                [[left_x, y], [right_x, y], [right_x, y + h], [left_x, y + h]],
                dtype=np.float64,
            )
        between = _keep_side(self.get_outline_polygon(), left_line, keep_left=False)
        return _keep_side(between, right_line, keep_left=True)

    def get_page_outline(self, sub_page: "SubPage") -> NDArray[np.floating] | None:
        """Get the outline for a specific sub-page.

        Args:
            sub_page: Which page to get the outline for.

        Returns:
            Polygon array for the requested page.
        """
        from scantailor.core.models import SubPage

        if sub_page == SubPage.LEFT_PAGE:
            return self.get_left_page_outline()
        if sub_page == SubPage.RIGHT_PAGE:
            return self.get_right_page_outline()
        return self.get_single_page_outline()

    def to_layout_type(self) -> LayoutType:
        """Return the layout type a user would request to get this layout."""
        return {
            PageLayoutType.SINGLE_PAGE_UNCUT: LayoutType.SINGLE_PAGE_UNCUT,
            PageLayoutType.SINGLE_PAGE_CUT: LayoutType.PAGE_PLUS_OFFCUT,
            PageLayoutType.TWO_PAGES: LayoutType.TWO_PAGES,
        }[self.layout_type]

    def with_type(self, layout_type: PageLayoutType) -> Self:
        """Change the layout type and keep the cutter lines.

        Keeping them lets a hand-placed line come back after the type was
        switched away and back. Two pages share the first cutter. A cutter the
        layout never had defaults to the left edge (first) or right edge
        (second) of the outline.

        Args:
            layout_type (PageLayoutType): The new type.

        Returns:
            PageLayout: A layout of that type.
        """
        x, y, width, height = self.outline
        cutter1, cutter2 = self.cutter1, self.cutter2
        if layout_type != PageLayoutType.SINGLE_PAGE_UNCUT:
            cutter1 = cutter1 or ((x, y), (x, y + height))
            cutter2 = cutter2 or ((x + width, y), (x + width, y + height))
        if layout_type == PageLayoutType.TWO_PAGES:
            cutter2 = cutter1
        return PageLayout(
            outline=self.outline,
            layout_type=layout_type,
            cutter1=cutter1,
            cutter2=cutter2,
        )

    def with_split_line(self, split_x: float) -> Self:
        """Create a new layout with a different split line position.

        Args:
            split_x: New X coordinate for the split line.

        Returns:
            New PageLayout with updated split line.
        """
        x, y, w, h = self.outline
        split_line = ((split_x, y), (split_x, y + h))
        return PageLayout(
            outline=self.outline,
            layout_type=PageLayoutType.TWO_PAGES,
            cutter1=split_line,
            cutter2=split_line,
        )

    def with_cutters(self, left_x: float, right_x: float) -> Self:
        """Create a new layout with different cutter positions.

        Args:
            left_x: X coordinate of the left cutter.
            right_x: X coordinate of the right cutter.

        Returns:
            New PageLayout with updated cutters.
        """
        x, y, _w, h = self.outline
        left_line = ((left_x, y), (left_x, y + h))
        right_line = ((right_x, y), (right_x, y + h))
        return PageLayout(
            outline=self.outline,
            layout_type=PageLayoutType.SINGLE_PAGE_CUT,
            cutter1=left_line,
            cutter2=right_line,
        )


Line = tuple[tuple[float, float], tuple[float, float]]


def _mid_x(line: Line) -> float:
    return (line[0][0] + line[1][0]) / 2


def _is_vertical(line: Line) -> bool:
    return line[0][0] == line[1][0]


def _keep_side(
    polygon: NDArray[np.floating], line: Line, *, keep_left: bool
) -> NDArray[np.floating]:
    """Clip a polygon to the side of the infinite ``line`` that is left or right of it.

    Left and right are judged along the horizontal, so the line must not be
    horizontal; a horizontal line leaves the polygon unchanged.

    Args:
        polygon (numpy.ndarray of numpy.floating): Vertices as an (N, 2) array.
        line (tuple[tuple[float, float], tuple[float, float]]):
            Two points on the cutter.
        keep_left (bool): Keep the part with smaller x at each y.

    Returns:
        numpy.ndarray of numpy.floating: The clipped polygon; empty if nothing is left.
    """
    (x1, y1), (x2, y2) = line
    if y1 == y2:
        return polygon
    sign = -1.0 if keep_left else 1.0

    def offset(point: NDArray[np.floating]) -> float:
        line_x = x1 + (x2 - x1) * (point[1] - y1) / (y2 - y1)
        return sign * (point[0] - line_x)

    kept: list[NDArray[np.floating]] = []
    for current, following in zip(polygon, np.roll(polygon, -1, axis=0)):
        before, after = offset(current), offset(following)
        if before >= 0:
            kept.append(current)
        if (before >= 0) != (after >= 0):
            along = before / (before - after)
            kept.append(current + along * (following - current))
    return np.array(kept, dtype=np.float64).reshape(-1, 2)
