"""Deskew filter implementation.

This filter detects and corrects page skew by rotating the image to align
horizontal text lines, and optionally an oblique (shear) distortion. Each can
operate in automatic mode (detecting it) or manual mode (using user-specified
angles).
"""

from dataclasses import replace

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.models import Dpi, PageId
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.deskew.dependencies import Dependencies
from scantailor.filters.deskew.oblique import find_oblique_degrees, rotate_mask
from scantailor.filters.deskew.params import MAX_OBLIQUE_DEG, Params
from scantailor.filters.deskew.settings import Detection, Settings
from scantailor.filters.deskew.shadow_cleanup import remove_horizontal_shadows
from scantailor.filters.deskew.top_edge import find_skew_from_top_edge
from scantailor.imageproc.analysis import SkewResult, find_skew
from scantailor.imageproc.binarize import binarize_otsu


class Filter:
    """Deskew filter.

    This filter is the third in the processing pipeline (after Fix Orientation
    and Page Split). It detects and corrects page skew to align horizontal
    text lines.

    The rotation and the oblique correction each run in one of two modes:
    - AUTO: Automatically detect it using image analysis
    - MANUAL: Use a user-specified angle

    The angles are stored as metadata and can be applied to images
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
        """Set a manual deskew angle for a page, keeping its oblique correction.

        Args:
            page_id: The page to set angle for.
            angle: The deskew angle in degrees (-45 to +45).

        Returns:
            The new Params instance.
        """
        params = self.get_params(page_id).with_manual_angle(angle)
        self._settings.set_params(page_id, params)
        return params

    def set_manual_oblique(self, page_id: PageId, angle: float) -> Params:
        """Set a manual oblique angle for a page, keeping its rotation.

        Args:
            page_id: The page to set the angle for.
            angle: The oblique angle in degrees, clamped to the oblique limit.

        Returns:
            The new Params instance.
        """
        clamped = max(-MAX_OBLIQUE_DEG, min(MAX_OBLIQUE_DEG, angle))
        params = self.get_params(page_id).with_manual_oblique(clamped)
        self._settings.set_params(page_id, params)
        return params

    @property
    def auto_oblique_default(self) -> bool:
        """Whether pages without parameters detect their oblique angle."""
        return self._settings.get_auto_oblique_default()

    @auto_oblique_default.setter
    def auto_oblique_default(self, enabled: bool) -> None:
        self._settings.set_auto_oblique_default(enabled)

    @property
    def content_based(self) -> bool:
        """Whether the skew is found from the page content, not the top page edge."""
        return self._settings.get_content_based()

    @content_based.setter
    def content_based(self, enabled: bool) -> None:
        self._settings.set_content_based(enabled)

    def detect_skew(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        max_angle: float | None = None,
        accuracy: float | None = None,
        store_result: bool = True,
        dependencies: Dependencies | None = None,
        dpi: Dpi | None = None,
        dbg: DebugImages | None = None,
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
            dependencies: Upstream geometry to record with the angle, so a
                change of it invalidates the angle.
            dpi: Resolution of ``image``. When given, horizontal scanner
                shadows are removed from the binarised page before the content
                based skew and the oblique angle are measured; the top edge is
                always found on the page as it is. None measures the page
                uncleaned throughout.
            dbg (DebugImages | None): Receives the binarised page and, with
                ``dpi``, the page after the shadow cleanup.

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
            gray = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        else:
            gray = image

        # Binarize for better skew detection
        binary = binarize_otsu(gray)
        if dbg is not None:
            dbg.add(binary, "bw_rotated")

        result = None
        if not self.content_based:
            # The dark surround along the top is the edge to find; the shadow
            # cleanup erases it.
            result = replace(
                find_skew_from_top_edge(
                    binary, max_angle=max_angle, min_angle=self.DEFAULT_MIN_ANGLE
                ),
                method="top edge",
            )
        if dpi is not None:
            binary = remove_horizontal_shadows(binary, dpi)
            if dbg is not None:
                dbg.add(binary, "after_cleanup")
        if result is None or not result.is_confident():
            if result is not None:
                logger.debug(
                    "top edge not confident ({:.2f}); using content skew",
                    result.confidence,
                )
            # Pages without a dark surround reaching the top have no usable edge.
            result = find_skew(
                binary,
                max_angle=max_angle,
                accuracy=accuracy,
                min_angle=self.DEFAULT_MIN_ANGLE,
            )

        if store_result:
            self._store_detection(binary, result, page_id, dependencies)

        return result

    def _automatic_parts(self, page_id: PageId) -> tuple[bool, bool]:
        """Tell which of rotation and oblique are automatic for a page.

        Returns:
            Whether the rotation and whether the oblique angle are automatic;
            a manual part is never replaced by detection.
        """
        if not self._settings.is_params_set(page_id):
            return True, self._settings.get_auto_oblique_default()
        params = self._settings.get_params(page_id)
        return params.is_auto(), params.is_oblique_auto()

    def _store_detection(
        self,
        binary: NDArray[np.uint8],
        skew: SkewResult,
        page_id: PageId,
        dependencies: Dependencies | None,
    ) -> None:
        """Store the parts of a detection that the page's modes allow."""
        detect_rotation, detect_oblique = self._automatic_parts(page_id)
        # The correction is the negated skew; a weak skew counts as none.
        rotation = (0.0 - skew.angle) if skew.is_confident() else 0.0
        oblique: float | None = None
        if detect_oblique:
            kept = self._settings.get_params(page_id).deskew_angle_deg
            level = rotate_mask(binary, rotation if detect_rotation else kept)
            found = find_oblique_degrees(level)
            oblique = 0.0 if found is None else 0.0 - found
        self._settings.store_detection(
            page_id,
            dependencies or Dependencies(),
            rotation=rotation if detect_rotation else None,
            oblique=oblique,
        )

    def needs_detection(self, page_id: PageId, dependencies: Dependencies) -> bool:
        """Tell whether an angle is missing or detected under other geometry.

        Args:
            page_id: The page to check.
            dependencies: The upstream geometry as it stands now.

        Returns:
            True if the rotation or the oblique angle must be detected. Manual
            angles never are.
        """
        if not self._settings.is_params_set(page_id):
            return True
        params = self._settings.get_params(page_id)
        if params.needs_redetection():
            return True
        if not (params.is_auto() or params.is_oblique_auto()):
            return False
        return not (
            params.dependencies.is_null() or params.dependencies.matches(dependencies)
        )

    def is_rotation_known(self, page_id: PageId) -> bool:
        """Tell whether a page has a rotation to show, set by hand or detected.

        Args:
            page_id: The page to check.

        Returns:
            False for a page that detection has not reached yet, or whose
            automatic rotation waits to be detected again.
        """
        if not self._settings.is_params_set(page_id):
            return False
        params = self._settings.get_params(page_id)
        return not (params.is_auto() and params.redetect_rotation)

    def redetect_rotation(self, page_id: PageId) -> None:
        """Make the rotation automatic again; the oblique correction is kept.

        Args:
            page_id: The page to reset.
        """
        self._settings.request_detection(page_id, Detection.ROTATION)

    def redetect_oblique(self, page_id: PageId) -> None:
        """Make the oblique correction automatic again; the rotation is kept.

        Args:
            page_id: The page to reset.
        """
        self._settings.request_detection(page_id, Detection.OBLIQUE)

    def reset_to_auto(self, page_id: PageId) -> None:
        """Reset a page to its fresh state, forgetting manual angles.

        Rotation is automatic with 0 degrees, and the oblique correction starts
        as the default says.

        Args:
            page_id: The page to reset.
        """
        self._settings.clear_page(page_id)

    def apply_geometry(
        self, transformation: ImageTransformation, params: Params
    ) -> None:
        """Set the rotation and the shear of a transformation from parameters.

        Args:
            transformation: The transformation to update; later steps reset.
            params: The page's parameters.
        """
        transformation.set_post_rotation(params.deskew_angle_deg)
        transformation.set_post_oblique(params.oblique_angle_deg)

    def apply_to_pages(self, page_ids: list[PageId], params: Params) -> None:
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

        return np.asarray(rotated, dtype=np.uint8)
