"""Page Layout filter implementation.

This filter manages page margins and alignment across a project, ensuring
consistent page sizes in the final output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scantailor.core.models import Margins, PageId
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.params import ContentSize, Params
from scantailor.filters.page_layout.settings import Settings

if TYPE_CHECKING:
    pass


class Filter:
    """Page Layout filter.

    This filter is the fifth in the processing pipeline (after Select Content).
    It manages page margins and alignment to ensure consistent page sizes
    across all pages in a project.

    The filter handles:
    - Hard margins: User-specified or auto-calculated margins in millimeters
    - Soft margins: Additional padding to align pages with different sizes
    - Alignment: How content is positioned when soft margins are added

    Example:
        >>> from scantailor.filters.page_layout import Filter, Settings
        >>> from scantailor.core import ImageId, PageId, SubPage, Margins
        >>> from pathlib import Path
        >>>
        >>> settings = Settings()
        >>> filter = Filter(settings)
        >>>
        >>> page_id = PageId(
        ...     image_id=ImageId(file_path=Path("/scan.tiff")),
        ...     sub_page=SubPage.SINGLE_PAGE,
        ... )
        >>> filter.set_margins(page_id, Margins.uniform(10.0))
    """

    name: str = "Page Layout"

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

    def get_params(self, page_id: PageId) -> Params:
        """Get the current parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters (default margins and centered alignment if not set).
        """
        return self._settings.get_params(page_id)

    def get_margins(self, page_id: PageId) -> Margins:
        """Get the margins for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The margins in millimeters.
        """
        return self.get_params(page_id).hard_margins_mm

    def get_alignment(self, page_id: PageId) -> Alignment:
        """Get the alignment for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The alignment settings.
        """
        return self.get_params(page_id).alignment

    def set_params(self, page_id: PageId, params: Params) -> None:
        """Set the parameters for a page.

        Args:
            page_id: The page to set parameters for.
            params: The parameters to store.
        """
        self._settings.set_params(page_id, params)

    def set_margins(self, page_id: PageId, margins: Margins) -> None:
        """Set the margins for a page.

        This sets the hard margins (user-specified) for the page and
        disables auto-margin calculation.

        Args:
            page_id: The page to set margins for.
            margins: The margins in millimeters.
        """
        self._settings.set_margins(page_id, margins)

    def set_alignment(self, page_id: PageId, alignment: Alignment) -> None:
        """Set the alignment for a page.

        Args:
            page_id: The page to set alignment for.
            alignment: The alignment settings.
        """
        self._settings.set_alignment(page_id, alignment)

    def set_uniform_margins(self, page_id: PageId, margin_mm: float) -> None:
        """Set uniform margins on all sides.

        Args:
            page_id: The page to set margins for.
            margin_mm: The margin in millimeters for all sides.
        """
        self.set_margins(page_id, Margins.uniform(margin_mm))

    def apply_to_pages(self, page_ids: list[PageId], params: Params) -> None:
        """Apply parameters to multiple pages.

        Args:
            page_ids: List of pages to set parameters for.
            params: The parameters to apply.
        """
        self._settings.apply_params_to_pages(page_ids, params)

    def apply_margins_to_pages(self, page_ids: list[PageId], margins: Margins) -> None:
        """Apply margins to multiple pages.

        Args:
            page_ids: List of pages to set margins for.
            margins: The margins in millimeters.
        """
        self._settings.apply_margins_to_pages(page_ids, margins)

    def apply_alignment_to_pages(
        self, page_ids: list[PageId], alignment: Alignment
    ) -> None:
        """Apply alignment to multiple pages.

        Args:
            page_ids: List of pages to set alignment for.
            alignment: The alignment settings.
        """
        self._settings.apply_alignment_to_pages(page_ids, alignment)

    def get_aggregate_hard_size(self) -> ContentSize:
        """Get the aggregate hard size across all aligned pages.

        This is the maximum size among all pages that participate
        in alignment. Used to calculate soft margins.

        Returns:
            The aggregate size in millimeters.
        """
        return self._settings.get_aggregate_hard_size_mm()

    def calculate_soft_margins(
        self, page_id: PageId, aggregate_size: ContentSize | None = None
    ) -> Margins:
        """Calculate soft margins for a page.

        Soft margins are additional margins added to bring a page up to
        the aggregate size of all aligned pages. The distribution of
        soft margins is determined by the page's alignment.

        Args:
            page_id: The page to calculate soft margins for.
            aggregate_size: The target aggregate size. If None, calculated
                from all aligned pages.

        Returns:
            The soft margins in millimeters.
        """
        params = self.get_params(page_id)

        if params.alignment.is_null:
            # Independent page, no soft margins
            return Margins()

        if aggregate_size is None:
            aggregate_size = self.get_aggregate_hard_size()

        # Calculate size difference
        width_diff = aggregate_size.width_mm - params.hard_width_mm()
        height_diff = aggregate_size.height_mm - params.hard_height_mm()

        # Distribute based on alignment
        left = 0.0
        right = 0.0
        top = 0.0
        bottom = 0.0

        # Horizontal distribution
        if width_diff > 0:
            left, right = self._distribute_margin(
                width_diff,
                params.alignment.horizontal.value,
                is_horizontal=True,
            )

        # Vertical distribution
        if height_diff > 0:
            top, bottom = self._distribute_margin(
                height_diff,
                params.alignment.vertical.value,
                is_horizontal=False,
            )

        return Margins(top=top, bottom=bottom, left=left, right=right)

    def _distribute_margin(
        self, total: float, alignment: str, is_horizontal: bool
    ) -> tuple[float, float]:
        """Distribute a margin based on alignment.

        Args:
            total: Total margin to distribute.
            alignment: Alignment value string.
            is_horizontal: True for horizontal, False for vertical.

        Returns:
            Tuple of (before, after) margins.
        """
        if alignment in ("left", "top"):
            return (0.0, total)
        elif alignment in ("right", "bottom"):
            return (total, 0.0)
        elif alignment in ("hcenter", "vcenter"):
            half = total / 2.0
            return (half, half)
        elif alignment in ("hauto", "vauto"):
            # Auto uses 3:1 ratio favoring center
            # Simplified: treat as center for now
            half = total / 2.0
            return (half, half)
        elif alignment in ("horiginal", "voriginal"):
            # Original preserves proportion - simplified to center
            half = total / 2.0
            return (half, half)
        else:
            # Default to center
            half = total / 2.0
            return (half, half)

    def get_total_margins(self, page_id: PageId) -> Margins:
        """Get total margins (hard + soft) for a page.

        Args:
            page_id: The page to get margins for.

        Returns:
            Combined hard and soft margins in millimeters.
        """
        hard = self.get_margins(page_id)
        soft = self.calculate_soft_margins(page_id)

        return Margins(
            top=hard.top + soft.top,
            bottom=hard.bottom + soft.bottom,
            left=hard.left + soft.left,
            right=hard.right + soft.right,
        )
