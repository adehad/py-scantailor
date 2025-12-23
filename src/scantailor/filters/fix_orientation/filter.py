"""Fix Orientation filter implementation.

This filter applies orthogonal rotations (0, 90, 180, 270 degrees) to images.
The rotation is stored as metadata and applied during rendering/output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import ImageId, OrthogonalRotation
from scantailor.filters.fix_orientation.settings import Settings
from scantailor.imageproc import rotate_orthogonal

if TYPE_CHECKING:
    from scantailor.core import PageId


class Filter:
    """Fix Orientation filter.

    This is the first filter in the processing pipeline. It stores and applies
    orthogonal rotations to images.

    The rotation can be:
    - 0 degrees (no rotation)
    - 90 degrees clockwise
    - 180 degrees
    - 270 degrees clockwise (= 90 counter-clockwise)

    Example:
        >>> from scantailor.filters.fix_orientation import Filter, Settings
        >>> from scantailor.core import ImageId
        >>> from pathlib import Path
        >>>
        >>> settings = Settings()
        >>> filter = Filter(settings)
        >>>
        >>> image_id = ImageId(file_path=Path("/scan.tiff"))
        >>> filter.rotate_clockwise(image_id)  # Now rotated 90°
        >>> filter.rotate_clockwise(image_id)  # Now rotated 180°
    """

    name: str = "Fix Orientation"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the filter.

        Args:
            settings: Settings storage for per-image rotations.
                     If None, a new Settings instance is created.
        """
        self._settings = settings if settings is not None else Settings()

    @property
    def settings(self) -> Settings:
        """Get the settings storage."""
        return self._settings

    def get_rotation(self, image_id: ImageId) -> OrthogonalRotation:
        """Get the current rotation for an image.

        Args:
            image_id: The image to look up.

        Returns:
            The stored rotation (0 degrees if not set).
        """
        return self._settings.get_rotation_for(image_id)

    def set_rotation(
        self, image_id: ImageId, rotation: OrthogonalRotation
    ) -> None:
        """Set the rotation for an image.

        Args:
            image_id: The image to set rotation for.
            rotation: The rotation to apply.
        """
        self._settings.apply_rotation(image_id, rotation)

    def rotate_clockwise(self, image_id: ImageId) -> OrthogonalRotation:
        """Rotate the image 90 degrees clockwise.

        Args:
            image_id: The image to rotate.

        Returns:
            The new rotation value.
        """
        current = self.get_rotation(image_id)
        new_rotation = current.rotate_clockwise()
        self.set_rotation(image_id, new_rotation)
        return new_rotation

    def rotate_counter_clockwise(self, image_id: ImageId) -> OrthogonalRotation:
        """Rotate the image 90 degrees counter-clockwise.

        Args:
            image_id: The image to rotate.

        Returns:
            The new rotation value.
        """
        current = self.get_rotation(image_id)
        new_rotation = current.rotate_counter_clockwise()
        self.set_rotation(image_id, new_rotation)
        return new_rotation

    def reset_rotation(self, image_id: ImageId) -> None:
        """Reset the rotation to 0 degrees.

        Args:
            image_id: The image to reset.
        """
        self.set_rotation(image_id, OrthogonalRotation(degrees=0))

    def apply_to_pages(
        self, page_ids: list[PageId], rotation: OrthogonalRotation
    ) -> None:
        """Apply a rotation to multiple pages.

        Since rotation is per-image, all pages from the same image will
        share the same rotation.

        Args:
            page_ids: List of pages to set rotation for.
            rotation: The rotation to apply.
        """
        self._settings.apply_rotation_to_pages(page_ids, rotation)

    def process_image(
        self, image: NDArray[np.uint8], image_id: ImageId
    ) -> NDArray[np.uint8]:
        """Apply the stored rotation to an image.

        This method actually rotates the image pixels. It's typically used
        during final output generation or preview rendering.

        Args:
            image: The input image as a numpy array.
            image_id: The image ID to look up the rotation.

        Returns:
            The rotated image.
        """
        rotation = self.get_rotation(image_id)
        if rotation.degrees == 0:
            return image.copy()
        return rotate_orthogonal(image, rotation.degrees)
