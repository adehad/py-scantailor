"""Settings storage for the page split filter.

This module provides the Settings class which stores per-page parameters
for the page split filter.
"""

from typing import TYPE_CHECKING

from .layout_type import LayoutType
from .params import Params

if TYPE_CHECKING:
    from scantailor.core import PageId


class Settings:
    """Storage for per-page page split parameters.

    This class maintains a mapping from PageId to Params for the page split
    filter, allowing different pages to have different split settings.
    """

    def __init__(self) -> None:
        """Initialize empty settings."""
        self._params: dict[PageId, Params] = {}
        self._default_layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE

    def get_params(self, page_id: "PageId") -> Params:
        """Get parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters or default if not set.
        """
        return self._params.get(page_id, Params(layout_type=self._default_layout_type))

    def set_params(self, page_id: "PageId", params: Params) -> None:
        """Set parameters for a page.

        Args:
            page_id: The page to set parameters for.
            params: The parameters to store.
        """
        self._params[page_id] = params

    def is_params_set(self, page_id: "PageId") -> bool:
        """Check if parameters have been explicitly set for a page.

        Args:
            page_id: The page to check.

        Returns:
            True if parameters have been set, False otherwise.
        """
        return page_id in self._params

    def clear_params(self, page_id: "PageId") -> None:
        """Clear parameters for a page, reverting to defaults.

        Args:
            page_id: The page to clear.
        """
        self._params.pop(page_id, None)

    def get_default_layout_type(self) -> LayoutType:
        """Get the default layout type for new pages."""
        return self._default_layout_type

    def set_default_layout_type(self, layout_type: LayoutType) -> None:
        """Set the default layout type for new pages.

        Args:
            layout_type: The new default layout type.
        """
        self._default_layout_type = layout_type

    def apply_params_to_pages(self, page_ids: list["PageId"], params: Params) -> None:
        """Apply the same parameters to multiple pages.

        Args:
            page_ids: List of pages to update.
            params: The parameters to apply.
        """
        for page_id in page_ids:
            self._params[page_id] = params

    def clear_all(self) -> None:
        """Clear all stored parameters."""
        self._params.clear()
