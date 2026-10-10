"""Settings storage for the output filter.

This module provides the Settings class which stores per-page parameters
for the output filter.
"""

import threading

from scantailor.core.models import Dpi, PageId

from .color_mode import ColorMode
from .despeckle import DespeckleLevel
from .params import Params


class Settings:
    """Storage for per-page output parameters.

    This class maintains a mapping from PageId to Params for the output
    filter, allowing different pages to have different output settings.
    It is safe to use from several threads; Params are immutable, so values
    returned by getters can be used without holding the lock.
    """

    # Default settings
    DEFAULT_DPI: int = 600
    DEFAULT_COLOR_MODE: ColorMode = ColorMode.BLACK_AND_WHITE
    DEFAULT_DESPECKLE: DespeckleLevel = DespeckleLevel.NORMAL

    def __init__(self) -> None:
        """Initialize empty settings."""
        self._lock = threading.Lock()
        self._params: dict[PageId, Params] = {}
        self._default_dpi: Dpi = Dpi.uniform(self.DEFAULT_DPI)
        self._default_color_mode: ColorMode = self.DEFAULT_COLOR_MODE
        self._default_despeckle: DespeckleLevel = self.DEFAULT_DESPECKLE

    def get_params(self, page_id: "PageId") -> Params:
        """Get parameters for a page.

        Args:
            page_id (PageId): The page to look up.

        Returns:
            Params: The stored parameters or default if not set.
        """
        with self._lock:
            stored = self._params.get(page_id)
            dpi = self._default_dpi
            mode = self._default_color_mode
            despeckle = self._default_despeckle
        if stored is not None:
            return stored
        return Params(output_dpi=dpi, color_mode=mode, despeckle_level=despeckle)

    def set_params(self, page_id: "PageId", params: Params) -> None:
        """Set parameters for a page.

        Args:
            page_id (PageId): The page to set parameters for.
            params (Params): The parameters to store.
        """
        with self._lock:
            self._params[page_id] = params

    def is_params_set(self, page_id: "PageId") -> bool:
        """Check if parameters have been explicitly set for a page.

        Args:
            page_id (PageId): The page to check.

        Returns:
            bool: True if parameters have been set, False otherwise.
        """
        with self._lock:
            return page_id in self._params

    def clear_params(self, page_id: "PageId") -> None:
        """Clear parameters for a page, reverting to defaults.

        Args:
            page_id (PageId): The page to clear.
        """
        with self._lock:
            self._params.pop(page_id, None)

    def get_default_dpi(self) -> Dpi:
        """Get the default output DPI."""
        with self._lock:
            return self._default_dpi

    def set_default_dpi(self, dpi: Dpi) -> None:
        """Set the default output DPI.

        Args:
            dpi (Dpi): The new default DPI.
        """
        with self._lock:
            self._default_dpi = dpi

    def get_default_color_mode(self) -> ColorMode:
        """Get the default color mode."""
        with self._lock:
            return self._default_color_mode

    def set_default_color_mode(self, mode: ColorMode) -> None:
        """Set the default color mode.

        Args:
            mode (ColorMode): The new default color mode.
        """
        with self._lock:
            self._default_color_mode = mode

    def get_default_despeckle_level(self) -> DespeckleLevel:
        """Get the default despeckle level."""
        with self._lock:
            return self._default_despeckle

    def set_default_despeckle_level(self, level: DespeckleLevel) -> None:
        """Set the default despeckle level.

        Args:
            level (DespeckleLevel): The new default despeckle level.
        """
        with self._lock:
            self._default_despeckle = level

    def apply_params_to_pages(self, page_ids: list["PageId"], params: Params) -> None:
        """Apply the same parameters to multiple pages.

        Args:
            page_ids (list[PageId]): List of pages to update.
            params (Params): The parameters to apply.
        """
        updates = dict.fromkeys(page_ids, params)
        with self._lock:
            self._params.update(updates)

    def clear_all(self) -> None:
        """Clear all stored parameters."""
        with self._lock:
            self._params.clear()
