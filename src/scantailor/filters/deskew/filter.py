"""Deskew filter implementation.

This filter detects and corrects page skew by rotating the image to align
horizontal text lines. It can operate in automatic mode (detecting skew)
or manual mode (using user-specified angles).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.filters.deskew.params import AutoManualMode, Params
from scantailor.filters.deskew.settings import Settings
from scantailor.imageproc import SkewResult, binarize_otsu, find_skew

if TYPE_CHECKING:
    from scantailor.core import PageId


class Filter:
    """Deskew filter.

    This filter is the third in the processing pipeline (after Fix Orientation
    and Page Split). It detects and corrects page skew to align horizontal
    text lines.

    The filter can operate in two modes:
    - AUTO: Automatically detect skew using image analysis
    - MANUAL: Use a user-specified angle

    The deskew angle is stored as metadata and can be applied to images
    during final output generation.

    Example:
        >>> from scantailor.filters.deskew import Filter, Settings
        >>> from scantailor.core import ImageId, PageId, SubPage
        >>> from pathlib import Path
        >>> import numpy as np
        >>>
        >>> settings = Settings()
        >>> filter = Filter(settings)
        >>>
        >>> # Auto-detect skew for an image
        >>> image = np.zeros((100, 100), dtype=np.uint8)
        >>> page_id = PageId(
        ...     image_id=ImageId(file_path=Path("/scan.tiff")),
        ...     sub_page=SubPage.SINGLE_PAGE,
        ... )
        >>> result = filter.detect_skew(image, page_id)
    """

    name: str = "Deskew"

    # Default parameters for skew detection
    DEFAULT_MAX_ANGLE: float = 7.0
    DEFAULT_ACCURACY: float = 0.1
    DEFAULT_MIN_ANGLE: float = 0.1

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the filter.

        Args:
            settings: Settings storage for per-page parameters.
                     If None, a new Settings instance is created.
        """
        self._settings = settings if settings is not None else Settings()

    @property
    def settings(self) -> Settings:
        """Get the settings storage."""
        return self._settings

    def get_params(self, page_id: PageId) -> Params:
        """Get the current parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters (angle 0.0, AUTO mode if not set).
        """
        return self._settings.get_params(page_id)

    def get_deskew_angle(self, page_id: PageId) -> float:
        """Get the deskew angle for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The deskew angle in degrees.
        """
        return self.get_params(page_id).deskew_angle_deg

    def set_params(self, page_id: PageId, params: Params) -> None:
        """Set the parameters for a page.

        Args:
            page_id: The page to set parameters for.
            params: The parameters to store.
        """
        self._settings.set_params(page_id, params)

    def set_manual_angle(self, page_id: PageId, angle: float) -> Params:
        """Set a manual deskew angle for a page.

        Args:
            page_id: The page to set angle for.
            angle: The deskew angle in degrees (-45 to +45).

        Returns:
            The new Params instance.
        """
        params = Params(deskew_angle_deg=angle, mode=AutoManualMode.MANUAL)
        self._settings.set_params(page_id, params)
        return params

    def detect_skew(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        max_angle: float | None = None,
        accuracy: float | None = None,
        store_result: bool = True,
    ) -> SkewResult:
        """Detect the skew angle for an image.

        This performs automatic skew detection and optionally stores the
        result in the settings.

        Args:
            image: The image to analyze (grayscale or color).
            page_id: The page ID to associate with the result.
            max_angle: Maximum angle to search (default 7.0 degrees).
            accuracy: Target accuracy (default 0.1 degrees).
            store_result: If True, store the detected angle in settings.

        Returns:
            SkewResult with detected angle and confidence.
        """
        # Use defaults if not specified
        if max_angle is None:
            max_angle = self.DEFAULT_MAX_ANGLE
        if accuracy is None:
            accuracy = self.DEFAULT_ACCURACY

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Binarize for better skew detection
        binary = binarize_otsu(gray)

        # Detect skew
        result = find_skew(
            binary,
            max_angle=max_angle,
            accuracy=accuracy,
            min_angle=self.DEFAULT_MIN_ANGLE,
        )

        # Store result if confident and requested
        if store_result:
            # Use detected angle if confident, otherwise 0
            angle = result.angle if result.is_confident() else 0.0
            # Negate the skew angle to get the correction angle
            deskew_angle = -angle
            params = Params(deskew_angle_deg=deskew_angle, mode=AutoManualMode.AUTO)
            self._settings.set_params(page_id, params)

        return result

    def reset_to_auto(self, page_id: PageId) -> None:
        """Reset a page to auto mode with 0 degree angle.

        This clears any manual angle and resets to default auto detection.

        Args:
            page_id: The page to reset.
        """
        self._settings.set_params(page_id, Params())

    def apply_to_pages(
        self, page_ids: list[PageId], params: Params
    ) -> None:
        """Apply parameters to multiple pages.

        Args:
            page_ids: List of pages to set parameters for.
            params: The parameters to apply.
        """
        self._settings.apply_params_to_pages(page_ids, params)

    def process_image(
        self, image: NDArray[np.uint8], page_id: PageId
    ) -> NDArray[np.uint8]:
        """Apply the stored deskew rotation to an image.

        This method actually rotates the image pixels. It's typically used
        during final output generation or preview rendering.

        Args:
            image: The input image as a numpy array.
            page_id: The page ID to look up the deskew angle.

        Returns:
            The rotated image.
        """
        params = self.get_params(page_id)
        angle = params.deskew_angle_deg

        if abs(angle) < 0.001:
            return image.copy()

        return self._rotate_image(image, angle)

    def _rotate_image(
        self, image: NDArray[np.uint8], angle: float
    ) -> NDArray[np.uint8]:
        """Rotate an image by a given angle.

        Args:
            image: The input image.
            angle: Rotation angle in degrees (positive = clockwise).

        Returns:
            The rotated image.
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Get rotation matrix
        # Note: cv2.getRotationMatrix2D uses counter-clockwise positive,
        # so we negate the angle for clockwise positive convention
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Calculate new bounding box size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust the rotation matrix for the new center
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Determine background color (white for document images)
        if len(image.shape) == 3:
            border_value = (255, 255, 255)
        else:
            border_value = 255  # type: ignore[assignment]

        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        return rotated
