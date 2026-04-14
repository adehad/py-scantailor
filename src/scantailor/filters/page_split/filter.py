"""Page Split filter implementation.

This filter is the second in the processing pipeline (after Fix Orientation).
It splits scanned images into individual pages, handling cases like:
- Single pages (no split needed)
- Two-page spreads (split down the middle)
- Pages with offcut garbage (cut off edges)
"""

import numpy as np
from numpy.typing import NDArray

from scantailor.core import PageId

from .layout_type import LayoutType
from .page_layout import PageLayout
from .params import Params
from .settings import Settings
from .split_finder import SplitResult, detect_split


class Filter:
    """Page Split filter.

    This filter handles splitting scanned images into logical pages.
    It can operate in automatic mode (detecting the split point) or
    manual mode (using user-specified split lines).

    Example:
        >>> from scantailor.filters.page_split import Filter, Settings
        >>> from scantailor.core import ImageId, PageId, SubPage
        >>> from pathlib import Path
        >>> import numpy as np
        >>>
        >>> settings = Settings()
        >>> filter = Filter(settings)
        >>>
        >>> # Auto-detect split for an image
        >>> image = np.zeros((100, 200), dtype=np.uint8)
        >>> page_id = PageId(
        ...     image_id=ImageId(file_path=Path("/scan.tiff")),
        ...     sub_page=SubPage.SINGLE_PAGE,
        ... )
        >>> result = filter.detect_split(image, page_id)
    """

    name: str = "Page Split"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the filter.

        Args:
            settings: Settings storage for per-page parameters.
                     If None, a new Settings instance is created.
        """
        self._settings = settings if settings is not None else Settings()

    @property
    def settings(self) -> Settings:
        """Get the settings storage."""
        return self._settings

    def get_params(self, page_id: "PageId") -> Params:
        """Get the current parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters or defaults if not set.
        """
        return self._settings.get_params(page_id)

    def set_params(self, page_id: "PageId", params: Params) -> None:
        """Set the parameters for a page.

        Args:
            page_id: The page to set parameters for.
            params: The parameters to store.
        """
        self._settings.set_params(page_id, params)

    def is_params_set(self, page_id: "PageId") -> bool:
        """Check if parameters have been set for a page."""
        return self._settings.is_params_set(page_id)

    def get_layout_type(self, page_id: "PageId") -> LayoutType:
        """Get the layout type for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The layout type for this page.
        """
        return self.get_params(page_id).layout_type

    def set_layout_type(self, page_id: "PageId", layout_type: LayoutType) -> Params:
        """Set the layout type for a page.

        Args:
            page_id: The page to update.
            layout_type: The new layout type.

        Returns:
            The updated Params.
        """
        params = self.get_params(page_id).with_layout_type(layout_type)
        self._settings.set_params(page_id, params)
        return params

    def get_page_layout(self, page_id: "PageId") -> PageLayout | None:
        """Get the page layout for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The page layout or None if not detected yet.
        """
        return self.get_params(page_id).page_layout

    def detect_split(
        self,
        image: NDArray[np.uint8],
        page_id: "PageId",
        dpi: float = 300.0,
        store_result: bool = True,
    ) -> SplitResult:
        """Detect how to split an image.

        Args:
            image: The image to analyze (grayscale or color).
            page_id: The page ID to associate with the result.
            dpi: Image resolution in dots per inch.
            store_result: If True, store the result in settings.

        Returns:
            SplitResult with detected layout and confidence.
        """
        params = self.get_params(page_id)
        layout_type = params.layout_type

        result = detect_split(image, layout_type, dpi)

        if store_result:
            new_params = params.with_auto_layout(result.layout)
            self._settings.set_params(page_id, new_params)

        return result

    def set_manual_split(
        self,
        page_id: "PageId",
        split_x: float,
        width: float,
        height: float,
    ) -> Params:
        """Set a manual split line for two-page layout.

        Args:
            page_id: The page to update.
            split_x: X coordinate of the split line.
            width: Image width.
            height: Image height.

        Returns:
            The updated Params.
        """
        layout = PageLayout.two_pages(width, height, split_x)
        params = self.get_params(page_id).with_manual_layout(layout)
        params = params.with_layout_type(LayoutType.TWO_PAGES)
        self._settings.set_params(page_id, params)
        return params

    def set_manual_cutters(
        self,
        page_id: "PageId",
        left_x: float,
        right_x: float,
        width: float,
        height: float,
    ) -> Params:
        """Set manual cutter lines for single-page-cut layout.

        Args:
            page_id: The page to update.
            left_x: X coordinate of the left cutter.
            right_x: X coordinate of the right cutter.
            width: Image width.
            height: Image height.

        Returns:
            The updated Params.
        """
        layout = PageLayout.single_page_cut(width, height, left_x, right_x)
        params = self.get_params(page_id).with_manual_layout(layout)
        params = params.with_layout_type(LayoutType.PAGE_PLUS_OFFCUT)
        self._settings.set_params(page_id, params)
        return params

    def reset_to_auto(self, page_id: "PageId") -> None:
        """Reset a page to auto mode.

        This clears any manual settings and reverts to auto-detection.

        Args:
            page_id: The page to reset.
        """
        params = Params(layout_type=LayoutType.AUTO_LAYOUT_TYPE)
        self._settings.set_params(page_id, params)

    def apply_to_pages(self, page_ids: list["PageId"], params: Params) -> None:
        """Apply parameters to multiple pages.

        Args:
            page_ids: List of pages to set parameters for.
            params: The parameters to apply.
        """
        self._settings.apply_params_to_pages(page_ids, params)

    def process(
        self,
        image: NDArray[np.uint8],
        page_id: "PageId",
        dpi: float = 300.0,
    ) -> Params:
        """Process an image to detect or apply page split.

        In AUTO mode, this will detect the split. In MANUAL mode,
        it preserves the existing layout.

        Args:
            image: The image to process.
            page_id: The page ID.
            dpi: Image resolution.

        Returns:
            The updated parameters.
        """
        params = self.get_params(page_id)

        # If manual mode and layout already exists, don't re-detect
        if params.is_manual() and params.page_layout is not None:
            return params

        # Detect split
        self.detect_split(image, page_id, dpi=dpi)
        return self.get_params(page_id)

    def get_sub_page_count(self, page_id: "PageId") -> int:
        """Get the number of sub-pages for an image.

        Args:
            page_id: The page to check.

        Returns:
            1 or 2 depending on the layout.
        """
        params = self.get_params(page_id)
        if params.page_layout is None:
            return 1
        return params.page_layout.num_sub_pages
