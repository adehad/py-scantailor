"""Settings for the Deskew filter.

This module provides storage for per-page deskew parameters.
"""

import threading
from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, Field, PrivateAttr

from scantailor.core.models import PageId
from scantailor.filters.deskew.dependencies import Dependencies
from scantailor.filters.deskew.params import AutoManualMode, Params


class Detection(str, Enum):
    """A part of the page's parameters that detection fills in."""

    ROTATION = "rotation"
    OBLIQUE = "oblique"


_AUTO_ROTATION = {
    "deskew_angle_deg": 0.0,
    "mode": AutoManualMode.AUTO,
    "redetect_rotation": True,
}
_AUTO_OBLIQUE = {
    "oblique_angle_deg": 0.0,
    "oblique_mode": AutoManualMode.AUTO,
    "redetect_oblique": True,
}


class Settings(BaseModel):
    """Thread-safe storage for per-page deskew parameters.

    Each page can have associated deskew parameters (angle and mode).
    Unlike fix_orientation which applies to images, deskew applies to
    individual pages since different pages from the same image can have
    different skew angles.

    Attributes:
        params: Parameters set explicitly or detected, per page.
        auto_oblique_default: Whether pages without parameters detect their
            oblique correction. Off keeps a fresh page unsheared.
        content_based: True to find the skew from the page content, False to
            find it from the top page edge.

    This class is thread-safe for concurrent access.
    """

    params: dict[PageId, Params] = Field(default_factory=dict)
    auto_oblique_default: bool = False
    content_based: bool = True

    model_config = {"arbitrary_types_allowed": True}

    # Use PrivateAttr to exclude lock from serialization
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _fresh_params(self, redetect: bool = False) -> Params:
        """Build the parameters of a page that has none.

        Args:
            redetect: Mark the automatic parts for detection, for a page that
                gets an entry before detection ever ran.
        """
        oblique_auto = self.auto_oblique_default
        return Params(
            oblique_mode=(
                AutoManualMode.AUTO if oblique_auto else AutoManualMode.MANUAL
            ),
            redetect_rotation=redetect,
            redetect_oblique=redetect and oblique_auto,
        )

    def get_params(self, page_id: PageId) -> Params:
        """Get the parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters, or fresh ones if not set: automatic rotation,
            and an oblique mode that follows ``auto_oblique_default``.
        """
        with self._lock:
            return self.params.get(page_id) or self._fresh_params()

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

    def set_auto_oblique_default(self, enabled: bool) -> None:
        """Choose whether pages without parameters detect their oblique angle.

        Args:
            enabled: True to detect, False to leave new pages unsheared.
        """
        with self._lock:
            self.auto_oblique_default = enabled

    def get_auto_oblique_default(self) -> bool:
        """Return whether pages without parameters detect their oblique angle."""
        with self._lock:
            return self.auto_oblique_default

    def set_content_based(self, enabled: bool) -> None:
        """Choose how the skew is found.

        Args:
            enabled: True to use the page content, False to use the top page edge.
        """
        with self._lock:
            self.content_based = enabled

    def get_content_based(self) -> bool:
        """Return whether the skew is found from the page content."""
        with self._lock:
            return self.content_based

    def request_detection(self, page_id: PageId, part: Detection) -> None:
        """Make one part of a page automatic and have it detected again.

        The other part keeps its mode and value.

        Args:
            page_id: The page.
            part: The part to detect again.
        """
        with self._lock:
            current = self.params.get(page_id) or self._fresh_params(redetect=True)
            update = _AUTO_ROTATION if part is Detection.ROTATION else _AUTO_OBLIQUE
            self.params[page_id] = current.model_copy(update=update)

    def store_detection(
        self,
        page_id: PageId,
        dependencies: Dependencies,
        rotation: float | None = None,
        oblique: float | None = None,
    ) -> bool:
        """Store detected angles, keeping any part the user made manual.

        Detection runs on worker threads, so the user can set a manual angle
        between the start of a detection and its result; checking and storing
        under one lock keeps that manual angle.

        Args:
            page_id: The page the detection ran for.
            dependencies: The upstream geometry the detection ran under.
            rotation: The detected deskew angle, or None if not detected.
            oblique: The detected oblique angle, or None if not detected.

        Returns:
            True if anything was stored, False if every detected part was kept
            manual.
        """
        with self._lock:
            current = self.params.get(page_id)
            base = current or self._fresh_params()
            update: dict[str, object] = {}
            if rotation is not None and base.is_auto():
                update["deskew_angle_deg"] = rotation
                update["redetect_rotation"] = False
            if oblique is not None and base.is_oblique_auto():
                update["oblique_angle_deg"] = oblique
                update["redetect_oblique"] = False
            if not update and current is not None:
                return False
            update["dependencies"] = dependencies
            self.params[page_id] = base.model_copy(update=update)
            return True

    def set_angle(self, page_id: PageId, angle: float, mode: str | None = None) -> None:
        """Set just the deskew angle for a page, keeping its oblique correction.

        Args:
            page_id: The page to set angle for.
            angle: The deskew angle in degrees.
            mode: "auto" or "manual". If None, defaults to "manual" since
                this is typically called from user interaction.
        """
        actual_mode = AutoManualMode(mode) if mode else AutoManualMode.MANUAL
        with self._lock:
            current = self.params.get(page_id) or self._fresh_params()
            self.params[page_id] = current.with_angle(angle, actual_mode)

    def apply_params_to_pages(self, page_ids: Iterable[PageId], params: Params) -> None:
        """Set the parameters for multiple pages.

        Args:
            page_ids: The pages to set parameters for.
            params: The parameters to apply.
        """
        with self._lock:
            for page_id in page_ids:
                self.params[page_id] = params

    def apply_parts_to_pages(
        self,
        source: Params,
        page_ids: Iterable[PageId],
        *,
        rotation: bool = True,
        oblique: bool = True,
    ) -> None:
        """Copy the chosen parts of a page's parameters to other pages.

        A manual part is copied as it is. An automatic one is not: each target
        detects its own. A part that is not chosen stays as the target has it.

        Args:
            source: The parameters being copied.
            page_ids: The pages that receive them.
            rotation: Copy the deskew angle and mode.
            oblique: Copy the oblique angle and mode.
        """
        with self._lock:
            for page_id in page_ids:
                target = self.params.get(page_id) or self._fresh_params(redetect=True)
                if rotation:
                    target = (
                        target.with_angle(
                            source.deskew_angle_deg, AutoManualMode.MANUAL
                        )
                        if source.is_manual()
                        else target.model_copy(update=_AUTO_ROTATION)
                    )
                if oblique:
                    target = (
                        target.with_oblique(
                            source.oblique_angle_deg, AutoManualMode.MANUAL
                        )
                        if source.is_oblique_manual()
                        else target.model_copy(update=_AUTO_OBLIQUE)
                    )
                self.params[page_id] = target

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
