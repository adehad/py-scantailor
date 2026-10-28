"""Fix Orientation filter implementation.

This filter applies orthogonal rotations (0, 90, 180, 270 degrees) to images and
can trim the edges of each image by hand. Both are stored as metadata; the
rotation is applied to the pixels and the trim is the first crop of the
transformation that later stages build on.
"""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import ImageId, OrthogonalRotation, PageId
from scantailor.filters.fix_orientation.image_trim import ImageTrim
from scantailor.filters.fix_orientation.settings import Settings
from scantailor.imageproc.geometry import rotate_orthogonal


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

    def set_rotation(self, image_id: ImageId, rotation: OrthogonalRotation) -> None:
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

    def get_trim(self, image_id: ImageId) -> ImageTrim:
        """Get the manual trim of an image.

        Args:
            image_id (ImageId): The image to look up.

        Returns:
            ImageTrim: The stored trim; disabled if none is set.
        """
        return self._settings.get_trim(image_id)

    def set_trim(self, image_id: ImageId, trim: ImageTrim) -> None:
        """Set the manual trim of an image; a disabled trim removes it.

        Args:
            image_id (ImageId): The image to trim.
            trim (ImageTrim): The trim to store.
        """
        self._settings.set_trim(image_id, trim)

    def trim_crop_area(
        self, image_id: ImageId, oriented_size: tuple[int, int]
    ) -> NDArray[np.float64] | None:
        """Get the trimmed region of an image in oriented pixels.

        Args:
            image_id (ImageId): The image to look up.
            oriented_size (tuple[int, int]): Width and height of the image
                after Fix Orientation's rotation.

        Returns:
            NDArray[np.float64] | None: Four corners, shape (4, 2); None when the
            image has no enabled trim.
        """
        return self.get_trim(image_id).crop_area(
            self.get_rotation(image_id), oriented_size
        )

    def apply_current_to_pages(
        self, source: ImageId, targets: Iterable[PageId]
    ) -> None:
        """Copy the rotation and the trim of one image to the images of pages.

        Args:
            source (ImageId): The image whose settings are copied.
            targets (Iterable[PageId]): The pages that receive them.
        """
        pages = list(targets)
        self._settings.apply_rotation_to_pages(pages, self.get_rotation(source))
        self._settings.apply_trim_to_pages(pages, self.get_trim(source))

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
            The rotated image, new; a read-only view of ``image`` when no
            rotation is stored.
        """
        rotation = self.get_rotation(image_id)
        if rotation.degrees == 0:
            unturned = image.view()
            unturned.flags.writeable = False
            return unturned
        return rotate_orthogonal(image, rotation.degrees)
