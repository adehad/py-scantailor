"""Dependencies tracking for cache invalidation."""

from pydantic import BaseModel

from .detection import ContentDetectionMode, PageDetectionMode


class Dependencies(BaseModel):
    """Tracks dependencies that affect content/page detection.

    When any of these change, the cached results need to be recomputed.
    """

    content_detection_mode: ContentDetectionMode = ContentDetectionMode.AUTO
    page_detection_mode: PageDetectionMode = PageDetectionMode.DISABLED
    fine_tune_corners: bool = False
    rotated_page_outline: list[tuple[float, float]] = []

    def needs_content_update(self, other: Dependencies) -> bool:
        """Check if content box needs to be recomputed.

        Content needs update if:
        - Content detection mode changed
        - Rotated page outline changed (affects what's considered content area)
        """
        if self.content_detection_mode != other.content_detection_mode:
            return True
        if self.rotated_page_outline != other.rotated_page_outline:
            return True
        return False

    def needs_page_update(self, other: Dependencies) -> bool:
        """Check if page box needs to be recomputed.

        Page box needs update if:
        - Page detection mode changed
        - Rotated page outline changed
        - Fine-tune corners changed (only matters if page detection is AUTO)
        """
        if self.page_detection_mode != other.page_detection_mode:
            return True
        if self.rotated_page_outline != other.rotated_page_outline:
            return True
        if (
            self.page_detection_mode == PageDetectionMode.AUTO
            and self.fine_tune_corners != other.fine_tune_corners
        ):
            return True
        return False

    def is_compatible_with(self, other: Dependencies) -> bool:
        """Check if cached results are still valid.

        Returns True if neither content nor page boxes need updating.
        """
        return not self.needs_content_update(other) and not self.needs_page_update(
            other
        )
