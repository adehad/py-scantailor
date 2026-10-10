"""Settings for the Fix Orientation filter.

This module provides storage for per-image rotation settings.
"""

import threading
from collections.abc import Iterable
from typing import Any, Self

from pydantic import BaseModel, Field, PrivateAttr

from scantailor.core.models import ImageId, OrthogonalRotation, PageId
from scantailor.core.project import StageParamsEntry


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

    def apply_rotation(self, image_id: ImageId, rotation: OrthogonalRotation) -> None:
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

    def to_dict(self) -> list[dict[str, Any]]:
        """Export the stored rotations in the project file's stage-params shape.

        Returns:
            list[dict[str, Any]]: One ``{"page": ..., "params": ...}`` entry per
                image, JSON-compatible. The page is the image's single-page id.
        """
        with self._lock:
            rotations = dict(self.rotations)
        return [
            StageParamsEntry(
                page=PageId(image_id=image_id),
                params=rotation.model_dump(mode="json"),
            ).model_dump(mode="json")
            for image_id, rotation in rotations.items()
        ]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> Self:
        """Create Settings from the output of :meth:`to_dict`.

        Args:
            data (list[dict[str, Any]]): Entries as produced by ``to_dict`` or
                read from a project file's ``fix_orientation`` stage params.

        Returns:
            Self: New Settings holding every entry's rotation.

        Raises:
            pydantic.ValidationError: If an entry is malformed or holds a
                rotation that is not a multiple of 90 degrees.
        """
        rotations: dict[ImageId, OrthogonalRotation] = {}
        for raw in data:
            entry = StageParamsEntry.model_validate(raw)
            rotations[entry.page.image_id] = OrthogonalRotation.model_validate(
                entry.params
            )
        return cls(rotations=rotations)
