"""Settings for the Page Layout filter.

This module provides storage for per-page layout parameters.
"""

import threading
from collections.abc import Iterable
from typing import NamedTuple

from pydantic import BaseModel, Field, PrivateAttr

from scantailor.core.models import Margins, PageId
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.params import ContentRect, ContentSize, Params


class ContentSizeUpdate(NamedTuple):
    """Outcome of Settings.update_content_size_and_get_aggregate.

    Attributes:
        params: The page's parameters after the update.
        aggregate_before: Aggregate hard size before the update.
        aggregate_after: Aggregate hard size after the update.
    """

    params: Params
    aggregate_before: ContentSize
    aggregate_after: ContentSize

    @property
    def aggregate_changed(self) -> bool:
        """True if the update changed the aggregate size other pages align to."""
        return self.aggregate_before != self.aggregate_after


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
            return self._aggregate_hard_size_locked()

    def update_content_size_and_get_aggregate(
        self,
        page_id: PageId,
        page_rect: ContentRect,
        content_rect: ContentRect,
        content_size_mm: ContentSize,
        auto_margins_mm: Margins | None = None,
    ) -> ContentSizeUpdate:
        """Record a page's content geometry and read the aggregate size atomically.

        Other threads cannot change any page between the update and the
        aggregate reads, so the returned aggregate is the one that includes
        exactly this update.

        Args:
            page_id (PageId): Page to update; created with default layout
                parameters if unknown.
            page_rect (ContentRect): Page rectangle in pixels.
            content_rect (ContentRect): Content rectangle in pixels.
            content_size_mm (ContentSize): Content size in millimetres.
            auto_margins_mm (Margins | None): Hard margins to adopt when the page
                has auto margins enabled; ignored otherwise.

        Returns:
            ContentSizeUpdate: Page parameters after the update plus the
            aggregate hard size before and after it.
        """
        with self._lock:
            existing = self.params.get(page_id)
            if (
                existing is not None
                and existing.auto_margins
                and auto_margins_mm is not None
            ):
                existing = existing.model_copy(
                    update={"hard_margins_mm": auto_margins_mm}
                )
                self.params[page_id] = existing
            before = self._aggregate_hard_size_locked()
            params = (existing or Params()).model_copy(
                update={
                    "page_rect": page_rect,
                    "content_rect": content_rect,
                    "content_size_mm": content_size_mm,
                }
            )
            self.params[page_id] = params
            after = self._aggregate_hard_size_locked()
        return ContentSizeUpdate(params, before, after)

    def check_everything_defined(
        self, pages: Iterable[PageId], ignore: PageId | None = None
    ) -> bool:
        """Check that every page has a recorded content geometry.

        Output sizes depend on the largest aligned page, so none can be fixed
        until each page has been measured. A page counts as measured once a
        processing run recorded a page rectangle or a content size for it;
        editing only margins or alignment does not.

        Args:
            pages (Iterable[PageId]): The pages of the project.
            ignore (PageId | None): A page to leave out, typically the one about
                to be processed, which records its own geometry.

        Returns:
            bool: True if every page that is not ignored has been measured.
        """
        with self._lock:
            recorded = [self.params.get(page) for page in pages if page != ignore]
        return all(
            params is not None
            and not (params.content_size_mm.is_empty() and params.page_rect.is_empty())
            for params in recorded
        )

    def invalidate_content_size(self, page_id: PageId) -> None:
        """Forget a page's measured geometry, keeping its margins and alignment.

        The page then counts as unmeasured until a processing run records it
        again. Unknown pages are left alone.

        Args:
            page_id (PageId): The page to invalidate.
        """
        with self._lock:
            existing = self.params.get(page_id)
            if existing is None:
                return
            self.params[page_id] = existing.model_copy(
                update={
                    "page_rect": ContentRect(),
                    "content_rect": ContentRect(),
                    "content_size_mm": ContentSize(),
                }
            )

    def _aggregate_hard_size_locked(self) -> ContentSize:
        """Aggregate hard size; the caller must hold the lock."""
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
