"""Dependencies tracking for cache invalidation."""

import numpy as np
from pydantic import BaseModel

from .detection import ContentDetectionMode, PageDetectionMode

# Outline coordinates are pixels; anything below this is float noise.
_OUTLINE_TOLERANCE_PX = 1e-3


class Dependencies(BaseModel):
    """Tracks dependencies that affect content/page detection.

    When any of these change, the cached results need to be recomputed.

    ``view_transform`` is not a dependency: it is the 3x3 matrix from the
    source image's pixels to the view the boxes are expressed in, kept so that
    a hand-drawn box can be carried into a changed view. Empty when unknown.
    """

    content_detection_mode: ContentDetectionMode = ContentDetectionMode.AUTO
    page_detection_mode: PageDetectionMode = PageDetectionMode.DISABLED
    fine_tune_corners: bool = False
    rotated_page_outline: list[tuple[float, float]] = []
    view_transform: list[list[float]] = []

    def same_outline(self, other: Dependencies) -> bool:
        """Tell whether both outlines describe the same polygon, up to float noise."""
        if len(self.rotated_page_outline) != len(other.rotated_page_outline):
            return False
        if not self.rotated_page_outline:
            return True
        return bool(
            np.allclose(
                np.asarray(self.rotated_page_outline, dtype=np.float64),
                np.asarray(other.rotated_page_outline, dtype=np.float64),
                rtol=0.0,
                atol=_OUTLINE_TOLERANCE_PX,
            )
        )

    def needs_content_update(self, other: Dependencies) -> bool:
        """Check if content box needs to be recomputed.

        Content needs update if:
        - Content detection mode changed
        - Rotated page outline changed (affects what's considered content area)
        """
        if self.content_detection_mode != other.content_detection_mode:
            return True
        return not self.same_outline(other)

    def needs_page_update(self, other: Dependencies) -> bool:
        """Check if page box needs to be recomputed.

        Page box needs update if:
        - Page detection mode changed
        - Rotated page outline changed
        - Fine-tune corners changed (only matters if page detection is AUTO)
        """
        if self.page_detection_mode != other.page_detection_mode:
            return True
        if not self.same_outline(other):
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
