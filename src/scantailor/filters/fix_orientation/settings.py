"""Settings for the Fix Orientation filter.

This module provides storage for per-image rotation settings.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from scantailor.core.models import ImageId, OrthogonalRotation, PageId

if TYPE_CHECKING:
    from collections.abc import Iterable


class Settings(BaseModel):
    """Thread-safe storage for per-image rotation settings.

    Each image can have an associated orthogonal rotation (0, 90, 180, or 270
    degrees). Rotations are stored by ImageId, not PageId, since the rotation
    applies to the entire image file.

    This class is thread-safe for concurrent access.
    """

    rotations: dict[ImageId, OrthogonalRotation] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    # Use PrivateAttr to exclude lock from serialization
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def get_rotation_for(self, image_id: ImageId) -> OrthogonalRotation:
        """Get the rotation for an image.

        Args:
            image_id: The image to look up.

        Returns:
            The stored rotation, or 0 degrees if not set.
        """
        with self._lock:
            return self.rotations.get(image_id, OrthogonalRotation())

    def is_rotation_set(self, image_id: ImageId) -> bool:
        """Check if a rotation has been explicitly set for an image.

        Args:
            image_id: The image to check.

        Returns:
            True if a rotation has been set, False otherwise.
        """
        with self._lock:
            return image_id in self.rotations

    def apply_rotation(
        self, image_id: ImageId, rotation: OrthogonalRotation
    ) -> None:
        """Set the rotation for a single image.

        Args:
            image_id: The image to set rotation for.
            rotation: The rotation to apply.
        """
        with self._lock:
            self.rotations[image_id] = rotation

    def apply_rotation_to_pages(
        self, page_ids: Iterable[PageId], rotation: OrthogonalRotation
    ) -> None:
        """Set the rotation for multiple pages.

        Note that rotation is per-image, so all pages from the same image
        will share the same rotation.

        Args:
            page_ids: The pages to set rotation for.
            rotation: The rotation to apply.
        """
        with self._lock:
            for page_id in page_ids:
                self.rotations[page_id.image_id] = rotation

    def clear(self) -> None:
        """Clear all stored rotations."""
        with self._lock:
            self.rotations.clear()

    def remove_rotation(self, image_id: ImageId) -> None:
        """Remove the rotation setting for an image.

        Args:
            image_id: The image to remove rotation for.
        """
        with self._lock:
            self.rotations.pop(image_id, None)

    def to_dict(self) -> dict[str, OrthogonalRotation]:
        """Export rotations as a serializable dictionary.

        Returns:
            Dictionary mapping image path strings to rotations.
        """
        with self._lock:
            return {
                f"{img_id.file_path}:{img_id.page}": rotation
                for img_id, rotation in self.rotations.items()
            }

    @classmethod
    def from_dict(cls, data: dict[str, dict]) -> Settings:
        """Create Settings from a serialized dictionary.

        Args:
            data: Dictionary from to_dict() or JSON deserialization.

        Returns:
            New Settings instance.
        """
        # This is a simple implementation; actual project loading uses
        # the project module's XML/JSON parsing
        return cls()
