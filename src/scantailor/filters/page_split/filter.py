"""Page Split filter implementation.

This filter is the second in the processing pipeline (after Fix Orientation).
It splits scanned images into individual pages, handling cases like:
- Single pages (no split needed)
- Two-page spreads (split down the middle)
- Pages with offcut garbage (cut off edges)
"""

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import PageId

from .dependencies import Dependencies
from .layout_type import LayoutType
from .page_layout import PageLayout, PageLayoutType
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
        rotation_degrees: int = 0,
    ) -> SplitResult:
        """Detect how to split an image.

        Args:
            image: The image to analyze (grayscale or color).
            page_id: The page ID to associate with the result.
            dpi: Image resolution in dots per inch.
            store_result: If True, store the result in settings.
            rotation_degrees: Orthogonal rotation already applied to ``image``;
                recorded so a later rotation change invalidates the result.

        Returns:
            SplitResult with detected layout and confidence.
        """
        params = self.get_params(page_id)
        layout_type = params.layout_type

        result = detect_split(image, layout_type, dpi)

        if store_result:
            dependencies = self._dependencies_of(
                params.layout_type, image.shape, rotation_degrees
            )
            new_params = params.with_auto_layout(result.layout).model_copy(
                update={"dependencies": dependencies}
            )
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
        params = self._with_manual_layout_of_type(page_id, layout, LayoutType.TWO_PAGES)
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
        params = self._with_manual_layout_of_type(
            page_id, layout, LayoutType.PAGE_PLUS_OFFCUT
        )
        self._settings.set_params(page_id, params)
        return params

    def _with_manual_layout_of_type(
        self, page_id: "PageId", layout: PageLayout, layout_type: LayoutType
    ) -> Params:
        """Params holding a hand-placed layout, recorded as made for its own type."""
        params = self.get_params(page_id)
        dependencies = params.dependencies
        if not dependencies.is_null():
            dependencies = dependencies.model_copy(update={"layout_type": layout_type})
        return (
            params.with_manual_layout(layout)
            .with_layout_type(layout_type)
            .model_copy(update={"dependencies": dependencies})
        )

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

        An automatic split line is not copied: each target keeps the layout
        type and detects its own layout.

        Args:
            page_ids: List of pages to set parameters for.
            params: The parameters to apply.
        """
        if params.is_auto():
            params = Params(layout_type=params.layout_type)
        self._settings.apply_params_to_pages(page_ids, params)

    @staticmethod
    def _dependencies_of(
        layout_type: LayoutType, image_shape: tuple[int, ...], rotation_degrees: int
    ) -> Dependencies:
        height, width = image_shape[:2]
        return Dependencies(
            image_size=(width, height),
            rotation_degrees=rotation_degrees,
            layout_type=layout_type,
        )

    def needs_detection(
        self,
        page_id: "PageId",
        image_shape: tuple[int, ...],
        rotation_degrees: int = 0,
    ) -> bool:
        """Tell whether the stored layout is missing or stale.

        Args:
            page_id: The page to check.
            image_shape: Shape of the oriented image.
            rotation_degrees: Orthogonal rotation applied to that image.

        Returns:
            True if the layout must be detected again.
        """
        params = self.get_params(page_id)
        if params.page_layout is None:
            return True
        current = self._dependencies_of(
            params.layout_type, image_shape, rotation_degrees
        )
        if not params.dependencies.is_compatible_with(
            current, manual_split_line=params.is_manual()
        ):
            return True
        required = {
            LayoutType.TWO_PAGES: PageLayoutType.TWO_PAGES,
            LayoutType.PAGE_PLUS_OFFCUT: PageLayoutType.SINGLE_PAGE_CUT,
        }.get(params.layout_type)
        return required is not None and params.page_layout.layout_type != required

    def clear_detection(self, page_id: "PageId") -> None:
        """Forget the layout and split line mode but keep the requested type.

        The next run detects the layout again.

        Args:
            page_id: The page to reset.
        """
        layout_type = self.get_params(page_id).layout_type
        self._settings.set_params(page_id, Params(layout_type=layout_type))

    def process(
        self,
        image: NDArray[np.uint8],
        page_id: "PageId",
        dpi: float = 300.0,
        rotation_degrees: int = 0,
    ) -> Params:
        """Detect the page layout unless a valid one is stored.

        A stored layout is kept while its dependencies still hold. When the
        requested type became a single uncut page the layout is reduced to it
        without detection.

        Args:
            image: The oriented image to process.
            page_id: The page ID.
            dpi: Image resolution.
            rotation_degrees: Orthogonal rotation applied to ``image``.

        Returns:
            The updated parameters.
        """
        if self.needs_detection(page_id, image.shape, rotation_degrees):
            self.detect_split(
                image, page_id, dpi=dpi, rotation_degrees=rotation_degrees
            )
            return self.get_params(page_id)

        params = self.get_params(page_id)
        layout = params.page_layout
        if (
            params.layout_type == LayoutType.SINGLE_PAGE_UNCUT
            and layout is not None
            and layout.layout_type != PageLayoutType.SINGLE_PAGE_UNCUT
        ):
            x, y, width, height = layout.outline
            params = params.with_page_layout(
                PageLayout.single_page_uncut(width, height, x, y),
                params.split_line_mode,
            ).model_copy(
                update={
                    "dependencies": params.dependencies.model_copy(
                        update={"layout_type": LayoutType.SINGLE_PAGE_UNCUT}
                    )
                }
            )
            self._settings.set_params(page_id, params)
        return params

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
