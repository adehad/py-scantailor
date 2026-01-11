"""Settings for the Page Layout filter.

This module provides storage for per-page layout parameters.
"""

import threading
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from scantailor.core.models import Margins, PageId
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.params import ContentSize, Params

if TYPE_CHECKING:
    from collections.abc import Iterable


class Settings(BaseModel):
    """Thread-safe storage for per-page layout parameters.

    Each page can have associated layout parameters including margins,
    alignment, and content size. This class also calculates aggregate
    sizes across all aligned pages.

    This class is thread-safe for concurrent access.
    """

    params: dict[PageId, Params] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def get_params(self, page_id: PageId) -> Params:
        """Get the parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters, or default Params if not set.
        """
        with self._lock:
            return self.params.get(page_id, Params())

    def is_params_set(self, page_id: PageId) -> bool:
        """Check if parameters have been explicitly set for a page.

        Args:
            page_id: The page to check.

        Returns:
            True if parameters have been set, False otherwise.
        """
        with self._lock:
            return page_id in self.params

    def set_params(self, page_id: PageId, params: Params) -> None:
        """Set the parameters for a single page.

        Args:
            page_id: The page to set parameters for.
            params: The parameters to store.
        """
        with self._lock:
            self.params[page_id] = params

    def set_margins(self, page_id: PageId, margins: Margins) -> None:
        """Set just the margins for a page.

        Args:
            page_id: The page to set margins for.
            margins: The margins in millimeters.
        """
        with self._lock:
            existing = self.params.get(page_id, Params())
            self.params[page_id] = existing.with_margins(margins)

    def set_alignment(self, page_id: PageId, alignment: Alignment) -> None:
        """Set just the alignment for a page.

        Args:
            page_id: The page to set alignment for.
            alignment: The alignment settings.
        """
        with self._lock:
            existing = self.params.get(page_id, Params())
            self.params[page_id] = existing.with_alignment(alignment)

    def apply_params_to_pages(self, page_ids: Iterable[PageId], params: Params) -> None:
        """Set the parameters for multiple pages.

        Args:
            page_ids: The pages to set parameters for.
            params: The parameters to apply.
        """
        with self._lock:
            for page_id in page_ids:
                self.params[page_id] = params

    def apply_margins_to_pages(
        self, page_ids: Iterable[PageId], margins: Margins
    ) -> None:
        """Set the margins for multiple pages.

        Args:
            page_ids: The pages to set margins for.
            margins: The margins in millimeters.
        """
        with self._lock:
            for page_id in page_ids:
                existing = self.params.get(page_id, Params())
                self.params[page_id] = existing.with_margins(margins)

    def apply_alignment_to_pages(
        self, page_ids: Iterable[PageId], alignment: Alignment
    ) -> None:
        """Set the alignment for multiple pages.

        Args:
            page_ids: The pages to set alignment for.
            alignment: The alignment settings.
        """
        with self._lock:
            for page_id in page_ids:
                existing = self.params.get(page_id, Params())
                self.params[page_id] = existing.with_alignment(alignment)

    def clear(self) -> None:
        """Clear all stored parameters."""
        with self._lock:
            self.params.clear()

    def clear_page(self, page_id: PageId) -> None:
        """Clear the parameters for a specific page.

        Args:
            page_id: The page to clear parameters for.
        """
        with self._lock:
            self.params.pop(page_id, None)

    def get_aggregate_hard_size_mm(self) -> ContentSize:
        """Calculate the aggregate hard size across all aligned pages.

        This is the maximum width and height among all pages that
        participate in alignment (alignment.is_null is False).

        Returns:
            The aggregate size in millimeters.
        """
        with self._lock:
            max_width = 0.0
            max_height = 0.0

            for params in self.params.values():
                if params.alignment.aligned_with_others():
                    max_width = max(max_width, params.hard_width_mm())
                    max_height = max(max_height, params.hard_height_mm())

            return ContentSize(width_mm=max_width, height_mm=max_height)

    def get_all_margins(self) -> dict[PageId, Margins]:
        """Get all stored margins.

        Returns:
            Dictionary mapping page IDs to their margins.
        """
        with self._lock:
            return {
                page_id: params.hard_margins_mm
                for page_id, params in self.params.items()
            }
