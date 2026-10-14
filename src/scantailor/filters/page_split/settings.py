"""Settings storage for the page split filter.

This module provides the Settings class which stores per-page parameters
for the page split filter.
"""

import threading

from scantailor.core.models import PageId

from .layout_type import LayoutType
from .params import Params


class Settings:
    """Storage for per-page page split parameters.

    The layout belongs to the image, not to one half of it, so both sub-pages
    of an image share one record.
    """

    def __init__(self) -> None:
        """Initialize empty settings."""
        self._lock = threading.Lock()
        self._params: dict[PageId, Params] = {}
        self._default_layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE

    @staticmethod
    def _key(page_id: PageId) -> PageId:
        return PageId(image_id=page_id.image_id)

    def get_params(self, page_id: PageId) -> Params:
        """Get parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters or default if not set.
        """
        with self._lock:
            params = self._params.get(self._key(page_id))
            default_layout_type = self._default_layout_type
        if params is None:
            return Params(layout_type=default_layout_type)
        return params

    def set_params(self, page_id: PageId, params: Params) -> None:
        """Set parameters for a page.

        Args:
            page_id: The page to set parameters for.
            params: The parameters to store.
        """
        with self._lock:
            self._params[self._key(page_id)] = params

    def is_params_set(self, page_id: PageId) -> bool:
        """Check if parameters have been explicitly set for a page.

        Args:
            page_id: The page to check.

        Returns:
            True if parameters have been set, False otherwise.
        """
        with self._lock:
            return self._key(page_id) in self._params

    def clear_params(self, page_id: PageId) -> None:
        """Clear parameters for a page, reverting to defaults.

        Args:
            page_id: The page to clear.
        """
        with self._lock:
            self._params.pop(self._key(page_id), None)

    def get_default_layout_type(self) -> LayoutType:
        """Get the default layout type for new pages."""
        with self._lock:
            return self._default_layout_type

    def set_default_layout_type(self, layout_type: LayoutType) -> None:
        """Set the default layout type for new pages.

        Args:
            layout_type: The new default layout type.
        """
        with self._lock:
            self._default_layout_type = layout_type

    def apply_params_to_pages(self, page_ids: list[PageId], params: Params) -> None:
        """Apply the same parameters to multiple pages.

        Args:
            page_ids: List of pages to update.
            params: The parameters to apply.
        """
        with self._lock:
            for page_id in page_ids:
                self._params[self._key(page_id)] = params

    def clear_all(self) -> None:
        """Clear all stored parameters and restore the default layout type."""
        with self._lock:
            self._params.clear()
            self._default_layout_type = LayoutType.AUTO_LAYOUT_TYPE
