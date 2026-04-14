"""Select Content filter implementation."""

import numpy as np
from numpy.typing import NDArray

from scantailor.core import PageId

from .content_box import ContentBox, PageBox, PhysicalSize
from .content_finder import ContentDetectionResult, find_content_box, find_page_edges
from .detection import ContentDetectionMode, PageDetectionMode
from .params import Params
from .settings import Settings


class Filter:
    """Select Content filter for detecting content and page boundaries.

    This filter detects the actual content area within a scanned page,
    distinguishing it from margins, shadows, and other artifacts.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the filter.

        Args:
            settings: Optional settings instance. Creates new if not provided.
        """
        self._settings = settings if settings is not None else Settings()

    @property
    def name(self) -> str:
        """Return the filter name."""
        return "Select Content"

    @property
    def settings(self) -> Settings:
        """Return the settings instance."""
        return self._settings

    def get_params(self, page_id: PageId) -> Params:
        """Get parameters for a page."""
        return self._settings.get_params(page_id)

    def set_params(self, page_id: PageId, params: Params) -> None:
        """Set parameters for a page."""
        self._settings.set_params(page_id, params)

    def is_params_set(self, page_id: PageId) -> bool:
        """Check if parameters have been set for a page."""
        return self._settings.is_params_set(page_id)

    def detect_content(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: float = 300.0,
        store_result: bool = True,
    ) -> ContentDetectionResult:
        """Detect content boundaries in an image.

        Args:
            image: Grayscale or color image.
            page_id: Page identifier.
            dpi: Image resolution.
            store_result: Whether to store the result in settings.

        Returns:
            ContentDetectionResult with detected box and confidence.
        """
        result = find_content_box(image, dpi=dpi)

        if store_result:
            params = self.get_params(page_id)
            new_params = params.with_content_box(
                result.content_box, ContentDetectionMode.AUTO
            )
            self.set_params(page_id, new_params)

        return result

    def detect_page(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: float = 300.0,
        store_result: bool = True,
    ) -> PageBox | None:
        """Detect page boundaries in an image.

        Args:
            image: Grayscale or color image.
            page_id: Page identifier.
            dpi: Image resolution.
            store_result: Whether to store the result in settings.

        Returns:
            PageBox if detected, None otherwise.
        """
        params = self.get_params(page_id)
        tolerance = self._settings.get_page_detection_tolerance()

        content_box = find_page_edges(
            image, dpi=dpi, tolerance=tolerance, fine_tune=params.fine_tune_corners
        )

        if content_box is None:
            # Use full image as page
            h, w = image.shape[:2]
            page_box = PageBox.from_image_size(w, h)
        else:
            page_box = PageBox(
                x=content_box.x,
                y=content_box.y,
                width=content_box.width,
                height=content_box.height,
            )

        if store_result:
            new_params = params.with_page_box(page_box, PageDetectionMode.AUTO)
            self.set_params(page_id, new_params)

        return page_box

    def process(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: float = 300.0,
    ) -> Params:
        """Process an image to detect content and page boundaries.

        Respects detection modes - only runs auto-detection when mode is AUTO.

        Args:
            image: Grayscale or color image.
            page_id: Page identifier.
            dpi: Image resolution.

        Returns:
            Updated parameters with detected/stored boxes.
        """
        params = self.get_params(page_id)
        h, w = image.shape[:2]

        # Handle page detection
        if params.is_page_auto():
            self.detect_page(image, page_id, dpi=dpi)
            params = self.get_params(page_id)
        elif params.is_page_disabled():
            page_box = PageBox.from_image_size(w, h)
            params = params.with_page_box(page_box, PageDetectionMode.DISABLED)
            self.set_params(page_id, params)

        # Handle content detection
        if params.is_content_auto():
            self.detect_content(image, page_id, dpi=dpi)
            params = self.get_params(page_id)
        elif params.is_content_disabled():
            # Use page box as content
            content_box = params.page_box.to_content_box()
            params = params.with_content_box(content_box, ContentDetectionMode.DISABLED)
            self.set_params(page_id, params)

        # Ensure content is within page
        params = params.ensure_content_within_page()

        # Calculate physical size
        content_size_mm = PhysicalSize.from_pixels(
            params.content_box.width, params.content_box.height, dpi
        )
        params = params.model_copy(update={"content_size_mm": content_size_mm})

        self.set_params(page_id, params)
        return params

    def set_manual_content(self, page_id: PageId, content_box: ContentBox) -> Params:
        """Set content box manually.

        Args:
            page_id: Page identifier.
            content_box: Manual content box.

        Returns:
            Updated parameters.
        """
        params = self.get_params(page_id).with_manual_content(content_box)
        self.set_params(page_id, params)
        return params

    def set_manual_page(self, page_id: PageId, page_box: PageBox) -> Params:
        """Set page box manually.

        Args:
            page_id: Page identifier.
            page_box: Manual page box.

        Returns:
            Updated parameters.
        """
        params = self.get_params(page_id).with_manual_page(page_box)
        self.set_params(page_id, params)
        return params

    def reset_content_to_auto(self, page_id: PageId) -> None:
        """Reset content detection to automatic mode."""
        params = self.get_params(page_id).with_auto_content()
        self.set_params(page_id, params)

    def reset_page_to_auto(self, page_id: PageId) -> None:
        """Reset page detection to automatic mode."""
        params = self.get_params(page_id).with_auto_page()
        self.set_params(page_id, params)

    def disable_content_detection(self, page_id: PageId) -> None:
        """Disable content detection (use full page as content)."""
        params = self.get_params(page_id).with_disabled_content()
        self.set_params(page_id, params)

    def disable_page_detection(self, page_id: PageId) -> None:
        """Disable page detection (use full image as page)."""
        params = self.get_params(page_id).with_disabled_page()
        self.set_params(page_id, params)

    def get_content_box(self, page_id: PageId) -> ContentBox:
        """Get the content box for a page."""
        return self.get_params(page_id).content_box

    def get_page_box(self, page_id: PageId) -> PageBox:
        """Get the page box for a page."""
        return self.get_params(page_id).page_box

    def apply_to_pages(self, page_ids: list[PageId], params: Params) -> None:
        """Apply parameters to multiple pages."""
        self._settings.apply_params_to_pages(page_ids, params)

    def set_fine_tune(self, page_id: PageId, enabled: bool) -> None:
        """Set fine-tune corners option."""
        params = self.get_params(page_id).with_fine_tune(enabled)
        self.set_params(page_id, params)
