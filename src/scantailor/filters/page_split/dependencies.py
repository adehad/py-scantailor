"""What a stored page split result was computed from."""

from pydantic import BaseModel, ConfigDict

from .layout_type import LayoutType


class Dependencies(BaseModel):
    """Inputs a detected page layout depends on.

    Attributes:
        image_size: Width and height in pixels of the oriented image that was
            analysed; (0, 0) when unknown.
        rotation_degrees: Orthogonal rotation applied to the image before the
            analysis.
        layout_type: Layout type that was requested at the time.
    """

    model_config = ConfigDict(frozen=True)

    image_size: tuple[int, int] = (0, 0)
    rotation_degrees: int = 0
    layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE

    def is_null(self) -> bool:
        """Return True when nothing is known about the original analysis."""
        return self.image_size == (0, 0)

    def is_compatible_with(
        self, current: "Dependencies", *, manual_split_line: bool
    ) -> bool:
        """Tell whether a result computed under ``self`` is still valid.

        A two-page request with a hand-placed split line has all the data it
        needs, and a single uncut page has no split line to depend on. A
        page-plus-offcut request cannot be reused for another type because
        which side the page is on is unknown.

        Args:
            current (Dependencies): The dependencies as they stand now.
            manual_split_line (bool): Whether the stored split line was placed
                by hand.

        Returns:
            bool: True if the stored layout can be kept.
        """
        if self.is_null():
            return True
        if self.image_size != current.image_size:
            return False
        if self.rotation_degrees != current.rotation_degrees:
            return False
        if self.layout_type == current.layout_type:
            return True
        if current.layout_type == LayoutType.SINGLE_PAGE_UNCUT:
            return True
        return current.layout_type == LayoutType.TWO_PAGES and manual_split_line
