"""Settings storage for the output filter.

This module provides the Settings class which stores per-page parameters
for the output filter.
"""

from typing import TYPE_CHECKING

from scantailor.core import Dpi

from .color_mode import ColorMode
from .despeckle import DespeckleLevel
from .params import Params

if TYPE_CHECKING:
    from scantailor.core import PageId


class Settings:
    """Storage for per-page output parameters.

    This class maintains a mapping from PageId to Params for the output
    filter, allowing different pages to have different output settings.
    """

    # Default settings
    DEFAULT_DPI: int = 600
    DEFAULT_COLOR_MODE: ColorMode = ColorMode.BLACK_AND_WHITE
    DEFAULT_DESPECKLE: DespeckleLevel = DespeckleLevel.NORMAL

    def __init__(self) -> None:
        """Initialize empty settings."""
        self._params: dict[PageId, Params] = {}
        self._default_dpi: Dpi = Dpi.uniform(self.DEFAULT_DPI)
        self._default_color_mode: ColorMode = self.DEFAULT_COLOR_MODE
        self._default_despeckle: DespeckleLevel = self.DEFAULT_DESPECKLE

    def get_params(self, page_id: "PageId") -> Params:
        """Get parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters or default if not set.
        """
        if page_id in self._params:
            return self._params[page_id]
        return Params(
            output_dpi=self._default_dpi,
            color_mode=self._default_color_mode,
            despeckle_level=self._default_despeckle,
        )

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

    def get_default_dpi(self) -> Dpi:
        """Get the default output DPI."""
        return self._default_dpi

    def set_default_dpi(self, dpi: Dpi) -> None:
        """Set the default output DPI.

        Args:
            dpi: The new default DPI.
        """
        self._default_dpi = dpi

    def get_default_color_mode(self) -> ColorMode:
        """Get the default color mode."""
        return self._default_color_mode

    def set_default_color_mode(self, mode: ColorMode) -> None:
        """Set the default color mode.

        Args:
            mode: The new default color mode.
        """
        self._default_color_mode = mode

    def get_default_despeckle_level(self) -> DespeckleLevel:
        """Get the default despeckle level."""
        return self._default_despeckle

    def set_default_despeckle_level(self, level: DespeckleLevel) -> None:
        """Set the default despeckle level.

        Args:
            level: The new default despeckle level.
        """
        self._default_despeckle = level

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
