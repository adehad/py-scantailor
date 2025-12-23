"""Settings for the Deskew filter.

This module provides storage for per-page deskew parameters.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from scantailor.core.models import PageId
from scantailor.filters.deskew.params import Params

if TYPE_CHECKING:
    from collections.abc import Iterable


class Settings(BaseModel):
    """Thread-safe storage for per-page deskew parameters.

    Each page can have associated deskew parameters (angle and mode).
    Unlike fix_orientation which applies to images, deskew applies to
    individual pages since different pages from the same image can have
    different skew angles.

    This class is thread-safe for concurrent access.
    """

    params: dict[PageId, Params] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    # Use PrivateAttr to exclude lock from serialization
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

    def set_angle(self, page_id: PageId, angle: float, mode: str | None = None) -> None:
        """Set just the deskew angle for a page.

        Args:
            page_id: The page to set angle for.
            angle: The deskew angle in degrees.
            mode: "auto" or "manual". If None, defaults to "manual" since
                this is typically called from user interaction.
        """
        from scantailor.filters.deskew.params import AutoManualMode

        actual_mode = AutoManualMode(mode) if mode else AutoManualMode.MANUAL
        with self._lock:
            self.params[page_id] = Params(deskew_angle_deg=angle, mode=actual_mode)

    def apply_params_to_pages(self, page_ids: Iterable[PageId], params: Params) -> None:
        """Set the parameters for multiple pages.

        Args:
            page_ids: The pages to set parameters for.
            params: The parameters to apply.
        """
        with self._lock:
            for page_id in page_ids:
                self.params[page_id] = params

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

    def get_all_angles(self) -> dict[PageId, float]:
        """Get all stored deskew angles.

        Returns:
            Dictionary mapping page IDs to their deskew angles.
        """
        with self._lock:
            return {
                page_id: params.deskew_angle_deg
                for page_id, params in self.params.items()
            }
