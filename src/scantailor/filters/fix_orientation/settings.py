"""Settings for the Fix Orientation filter.

This module provides storage for per-image rotation and trim settings.
"""

import threading
from collections.abc import Iterable
from typing import Any, Self

from pydantic import BaseModel, Field, PrivateAttr

from scantailor.core.models import ImageId, OrthogonalRotation, PageId
from scantailor.core.project import StageParamsEntry
from scantailor.filters.fix_orientation.image_trim import ImageTrim


class Settings(BaseModel):
    """Thread-safe storage for per-image rotation and trim settings.

    Each image can have an associated orthogonal rotation (0, 90, 180, or 270
    degrees) and a manual trim. Both are stored by ImageId, not PageId, since
    they apply to the entire image file. Only enabled trims are stored.

    This class is thread-safe for concurrent access.
    """

    rotations: dict[ImageId, OrthogonalRotation] = Field(default_factory=dict)
    trims: dict[ImageId, ImageTrim] = Field(default_factory=dict)

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

    def get_trim(self, image_id: ImageId) -> ImageTrim:
        """Get the trim of an image.

        Args:
            image_id (ImageId): The image to look up.

        Returns:
            ImageTrim: The stored trim, or a disabled one if none is stored.
        """
        with self._lock:
            return self.trims.get(image_id, ImageTrim())

    def set_trim(self, image_id: ImageId, trim: ImageTrim) -> None:
        """Set the trim of an image; a disabled trim removes it.

        Args:
            image_id (ImageId): The image to trim.
            trim (ImageTrim): The trim to store.
        """
        with self._lock:
            self._set_trim_locked(image_id, trim)

    def clear_trim(self, image_id: ImageId) -> None:
        """Remove the trim of an image.

        Args:
            image_id (ImageId): The image whose trim is removed.
        """
        with self._lock:
            self.trims.pop(image_id, None)

    def apply_trim_to_pages(self, page_ids: Iterable[PageId], trim: ImageTrim) -> None:
        """Set the trim of the images of several pages.

        Args:
            page_ids (Iterable[PageId]): The pages; pages of one image share its trim.
            trim (ImageTrim): The trim to store; a disabled one removes the trims.
        """
        with self._lock:
            for page_id in page_ids:
                self._set_trim_locked(page_id.image_id, trim)

    def _set_trim_locked(self, image_id: ImageId, trim: ImageTrim) -> None:
        if trim.enabled:
            self.trims[image_id] = trim
        else:
            self.trims.pop(image_id, None)

    def has_params(self, image_id: ImageId) -> bool:
        """Tell whether a rotation or an enabled trim is stored for an image.

        Args:
            image_id (ImageId): The image to check.

        Returns:
            bool: True if anything is stored for the image.
        """
        with self._lock:
            return image_id in self.rotations or image_id in self.trims

    def dump_params(self, image_id: ImageId) -> dict[str, Any]:
        """Export what is stored for an image in the project file's shape.

        Args:
            image_id (ImageId): The image to export.

        Returns:
            dict[str, Any]: JSON-compatible ``degrees`` (when a rotation is
            stored) and ``trim`` (when a trim is enabled).
        """
        with self._lock:
            rotation = self.rotations.get(image_id)
            trim = self.trims.get(image_id)
        params: dict[str, Any] = {}
        if rotation is not None:
            params.update(rotation.model_dump(mode="json"))
        if trim is not None:
            params["trim"] = trim.model_dump(mode="json")
        return params

    def load_params(self, image_id: ImageId, params: dict[str, Any]) -> None:
        """Store what :meth:`dump_params` exported for an image.

        Args:
            image_id (ImageId): The image the parameters belong to.
            params (dict[str, Any]): Output of ``dump_params``; the keys that are
                absent leave that setting as it is.

        Raises:
            pydantic.ValidationError: If the rotation is not a multiple of 90
                degrees or the trim is malformed.
        """
        rotation = (
            OrthogonalRotation.model_validate(params) if "degrees" in params else None
        )
        trim = ImageTrim.model_validate(params["trim"]) if "trim" in params else None
        with self._lock:
            if rotation is not None:
                self.rotations[image_id] = rotation
            if trim is not None:
                self._set_trim_locked(image_id, trim)

    def clear(self) -> None:
        """Clear all stored rotations and trims."""
        with self._lock:
            self.rotations.clear()
            self.trims.clear()

    def remove_rotation(self, image_id: ImageId) -> None:
        """Remove the rotation setting for an image.

        Args:
            image_id: The image to remove rotation for.
        """
        with self._lock:
            self.rotations.pop(image_id, None)

    def to_dict(self) -> list[dict[str, Any]]:
        """Export the stored rotations and trims in the project file's shape.

        Returns:
            list[dict[str, Any]]: One ``{"page": ..., "params": ...}`` entry per
                image, JSON-compatible. The page is the image's single-page id.
        """
        with self._lock:
            image_ids = list(dict.fromkeys([*self.rotations, *self.trims]))
        return [
            StageParamsEntry(
                page=PageId(image_id=image_id),
                params=self.dump_params(image_id),
            ).model_dump(mode="json")
            for image_id in image_ids
        ]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> Self:
        """Create Settings from the output of :meth:`to_dict`.

        Args:
            data (list[dict[str, Any]]): Entries as produced by ``to_dict`` or
                read from a project file's ``fix_orientation`` stage params.

        Returns:
            Self: New Settings holding every entry's rotation and trim.

        Raises:
            pydantic.ValidationError: If an entry is malformed or holds a
                rotation that is not a multiple of 90 degrees.
        """
        settings = cls()
        for raw in data:
            entry = StageParamsEntry.model_validate(raw)
            settings.load_params(entry.page.image_id, entry.params)
        return settings
