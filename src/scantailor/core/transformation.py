"""Image transformation pipeline.

This module provides coordinate transformation management through a 6-step
pipeline, matching the C++ ImageTransformation class:

1. Pre-scale: Equalize DPI (make pixels square)
2. Pre-rotate: Orthogonal rotation (0, 90, 180, 270 degrees)
3. Pre-crop: Crop to page boundaries (after split)
4. Post-rotate: Deskew rotation (arbitrary angle), then oblique (shear)
5. Post-crop: Adjust margins
6. Post-scale: Scale to output DPI
"""

import math
from dataclasses import dataclass, field
from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi, OrthogonalRotation


@dataclass(frozen=True)
class Rect:
    """A rectangle defined by position and size."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def left(self) -> float:
        """Left edge."""
        return self.x

    @property
    def top(self) -> float:
        """Top edge."""
        return self.y

    @property
    def right(self) -> float:
        """Right edge."""
        return self.x + self.width

    @property
    def bottom(self) -> float:
        """Bottom edge."""
        return self.y + self.height

    def center(self) -> tuple[float, float]:
        """Return center point."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def size(self) -> tuple[float, float]:
        """Return (width, height)."""
        return (self.width, self.height)

    def is_empty(self) -> bool:
        """Return True if the rectangle has no area."""
        return self.width <= 0 or self.height <= 0

    def to_polygon(self) -> NDArray[np.float64]:
        """Return corners as a (4, 2) array, clockwise from the top-left."""
        return _rect_to_polygon(self)

    @classmethod
    def from_size(cls, width: float, height: float) -> Rect:
        """Create rect from size at origin."""
        return cls(x=0, y=0, width=width, height=height)


def _polygon_bounding_rect(polygon: NDArray[np.float64]) -> Rect:
    """Get the bounding rectangle of a polygon (Nx2 array of points)."""
    if len(polygon) == 0:
        return Rect()
    min_x = float(np.min(polygon[:, 0]))
    min_y = float(np.min(polygon[:, 1]))
    max_x = float(np.max(polygon[:, 0]))
    max_y = float(np.max(polygon[:, 1]))
    return Rect(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)


def _rect_to_polygon(rect: Rect) -> NDArray[np.float64]:
    """Convert a Rect to a polygon (4x2 array)."""
    return np.array(
        [
            [rect.left, rect.top],
            [rect.right, rect.top],
            [rect.right, rect.bottom],
            [rect.left, rect.bottom],
        ],
        dtype=np.float64,
    )


def _transform_polygon(
    polygon: NDArray[np.float64], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply a 3x3 transformation matrix to a polygon.

    Args:
        polygon: Nx2 array of points.
        matrix: 3x3 transformation matrix.

    Returns:
        Transformed Nx2 array.
    """
    if len(polygon) == 0:
        return polygon.copy()

    # Convert to homogeneous coordinates
    ones = np.ones((len(polygon), 1), dtype=np.float64)
    homogeneous = np.hstack([polygon, ones])

    # Transform
    transformed = homogeneous @ matrix.T

    # Convert back to 2D (divide by w)
    w = transformed[:, 2:3]
    w = np.where(np.abs(w) < 1e-10, 1.0, w)
    return transformed[:, :2] / w


def _identity_matrix() -> NDArray[np.float64]:
    """Return a 3x3 identity matrix."""
    return np.eye(3, dtype=np.float64)


def _translation_matrix(dx: float, dy: float) -> NDArray[np.float64]:
    """Create a translation matrix."""
    m = np.eye(3, dtype=np.float64)
    m[0, 2] = dx
    m[1, 2] = dy
    return m


def _scale_matrix(sx: float, sy: float) -> NDArray[np.float64]:
    """Create a scale matrix."""
    m = np.eye(3, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    return m


def _rotation_matrix(degrees: float) -> NDArray[np.float64]:
    """Create a rotation matrix around origin."""
    rad = math.radians(degrees)
    c = math.cos(rad)
    s = math.sin(rad)
    m = np.eye(3, dtype=np.float64)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def _shear_matrix(oblique_degrees: float) -> NDArray[np.float64]:
    """Create the horizontal shear x' = x - tan(oblique) * y about the origin.

    A vertical line leaning right by ``oblique`` degrees going down comes out
    vertical; heights and horizontal distances are kept.
    """
    m = np.eye(3, dtype=np.float64)
    m[0, 1] = -math.tan(math.radians(oblique_degrees))
    return m


def _then(*matrices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compose column-vector transforms so the first argument is applied first.

    Args:
        *matrices: 3x3 transforms in application order.

    Returns:
        The single 3x3 matrix equivalent to applying them in sequence.
    """
    combined = _identity_matrix()
    for matrix in matrices:
        combined = matrix @ combined
    return combined


def _orthogonal_rotation_matrix(
    rotation: OrthogonalRotation, width: float, height: float
) -> NDArray[np.float64]:
    """Create the clockwise orthogonal rotation of a width x height image.

    The result maps onto a rectangle whose top-left corner is the origin.
    """
    degrees = rotation.degrees
    if degrees == 0:
        return _identity_matrix()

    if degrees == 90:
        # (x, y) -> (height - y, x)
        return np.array([[0, -1, height], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if degrees == 180:
        # (x, y) -> (width - x, height - y)
        return np.array([[-1, 0, width], [0, -1, height], [0, 0, 1]], dtype=np.float64)
    # 270: (x, y) -> (y, width - x)
    return np.array([[0, 1, 0], [-1, 0, width], [0, 0, 1]], dtype=np.float64)


@dataclass
class ImageTransformation:
    """Provides a transformed view of an image through a 6-step pipeline.

    The transformation pipeline consists of:
    1. Pre-scale: Scale to equalize DPI (make pixels square)
    2. Pre-rotate: Orthogonal rotation (0, 90, 180, 270 degrees)
    3. Pre-crop: Crop to page boundaries
    4. Post-rotate: Deskew rotation (arbitrary angle), then oblique (shear)
    5. Post-crop: Margin adjustment
    6. Post-scale: Scale to output DPI

    Each step builds on the previous ones. Setting an earlier step
    resets all subsequent steps.

    Attributes:
        orig_rect: Original image rectangle.
        orig_dpi: Original image DPI.
    """

    orig_rect: Rect
    orig_dpi: Dpi

    # Step 1: Pre-scale
    _pre_scale_xform: NDArray[np.float64] = field(
        default_factory=_identity_matrix, repr=False
    )
    _pre_scaled_dpi: Dpi = field(default_factory=Dpi)

    # Step 2: Pre-rotate
    _pre_rotation: OrthogonalRotation = field(default_factory=OrthogonalRotation)
    _pre_rotate_xform: NDArray[np.float64] = field(
        default_factory=_identity_matrix, repr=False
    )

    # Step 3: Pre-crop
    _pre_crop_area: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64).reshape(0, 2), repr=False
    )
    _pre_crop_xform: NDArray[np.float64] = field(
        default_factory=_identity_matrix, repr=False
    )

    # Step 4: Post-rotate
    _post_rotation: float = 0.0
    _post_oblique: float = 0.0
    _post_rotate_xform: NDArray[np.float64] = field(
        default_factory=_identity_matrix, repr=False
    )

    # Step 5: Post-crop
    _post_crop_area: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64).reshape(0, 2), repr=False
    )
    _post_crop_xform: NDArray[np.float64] = field(
        default_factory=_identity_matrix, repr=False
    )

    # Step 6: Post-scale
    _post_scaled_dpi: Dpi = field(default_factory=Dpi)
    _post_scale_xform: NDArray[np.float64] = field(
        default_factory=_identity_matrix, repr=False
    )

    # Computed transforms
    _transform: NDArray[np.float64] = field(
        default_factory=_identity_matrix, repr=False
    )
    _inv_transform: NDArray[np.float64] = field(
        default_factory=_identity_matrix, repr=False
    )

    # Computed areas
    _resulting_rect: Rect = field(default_factory=Rect)
    _resulting_pre_crop_area: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64).reshape(0, 2), repr=False
    )
    _resulting_post_crop_area: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64).reshape(0, 2), repr=False
    )

    def __post_init__(self) -> None:
        """Initialize by equalizing DPI."""
        self._resulting_rect = Rect(
            x=self.orig_rect.x,
            y=self.orig_rect.y,
            width=self.orig_rect.width,
            height=self.orig_rect.height,
        )
        self.pre_scale_to_equalize_dpi()

    @classmethod
    def from_size(cls, width: float, height: float, dpi: Dpi | None = None) -> Self:
        """Create transformation from image dimensions."""
        rect = Rect.from_size(width, height)
        return cls(orig_rect=rect, orig_dpi=dpi or Dpi())

    # Properties for read access

    @property
    def pre_scaled_dpi(self) -> Dpi:
        """Get DPI after pre-scaling."""
        return self._pre_scaled_dpi

    @property
    def pre_rotation(self) -> OrthogonalRotation:
        """Get orthogonal pre-rotation."""
        return self._pre_rotation

    @property
    def pre_crop_area(self) -> NDArray[np.float64]:
        """Get pre-crop area polygon."""
        return self._pre_crop_area.copy()

    @property
    def post_rotation(self) -> float:
        """Get deskew rotation in degrees."""
        return self._post_rotation

    @property
    def post_oblique(self) -> float:
        """Get the oblique (shear) correction in degrees."""
        return self._post_oblique

    @property
    def transform(self) -> NDArray[np.float64]:
        """Get full transformation matrix (original to resulting)."""
        return self._transform.copy()

    @property
    def transform_back(self) -> NDArray[np.float64]:
        """Get inverse transformation matrix (resulting to original)."""
        return self._inv_transform.copy()

    @property
    def resulting_rect(self) -> Rect:
        """Get the resulting image rectangle after all transformations."""
        return self._resulting_rect

    @property
    def resulting_pre_crop_area(self) -> NDArray[np.float64]:
        """Get pre-crop area in resulting coordinates."""
        return self._resulting_pre_crop_area.copy()

    @property
    def resulting_post_crop_area(self) -> NDArray[np.float64]:
        """Get post-crop area in resulting coordinates."""
        return self._resulting_post_crop_area.copy()

    # Step 1: Pre-scale

    def pre_scale_to_dpi(self, dpi: Dpi) -> None:
        """Set pre-scale to achieve specific DPI."""
        if self.orig_dpi.is_null() or dpi.is_null():
            return

        self._pre_scaled_dpi = dpi

        xscale = dpi.horizontal / self.orig_dpi.horizontal
        yscale = dpi.vertical / self.orig_dpi.vertical

        new_width = self.orig_rect.width * xscale
        new_height = self.orig_rect.height * yscale

        # Save old transforms for recalculation
        old_pre_scale_rotate = _then(self._pre_scale_xform, self._pre_rotate_xform)
        old_post_rotate_chain = _then(
            old_pre_scale_rotate, self._pre_crop_xform, self._post_rotate_xform
        )

        # Update pre-scale
        self._pre_scale_xform = _scale_matrix(xscale, yscale)

        # Update pre-rotate for new size
        self._pre_rotate_xform = _orthogonal_rotation_matrix(
            self._pre_rotation, new_width, new_height
        )

        # Recalculate pre-crop area in new coordinates
        new_pre_scale_rotate = _then(self._pre_scale_xform, self._pre_rotate_xform)
        if len(self._pre_crop_area) > 0:
            # Transform through: old_inv -> new
            old_inv = np.asarray(np.linalg.inv(old_pre_scale_rotate), dtype=np.float64)
            self._pre_crop_area = _transform_polygon(
                _transform_polygon(self._pre_crop_area, old_inv),
                new_pre_scale_rotate,
            )
            self._pre_crop_xform = self._calc_crop_xform(self._pre_crop_area)

        # Recalculate post-rotate
        self._post_rotate_xform = self._calc_post_rotate_xform(
            self._post_rotation, self._post_oblique
        )

        # Recalculate post-crop area
        if len(self._post_crop_area) > 0:
            new_post_rotate_chain = _then(
                new_pre_scale_rotate, self._pre_crop_xform, self._post_rotate_xform
            )
            old_inv = np.asarray(np.linalg.inv(old_post_rotate_chain), dtype=np.float64)
            self._post_crop_area = _transform_polygon(
                _transform_polygon(self._post_crop_area, old_inv),
                new_post_rotate_chain,
            )
            self._post_crop_xform = self._calc_crop_xform(self._post_crop_area)

        # Recalculate post-scale
        self._post_scale_xform = self._calc_post_scale_xform(self._post_scaled_dpi)

        self._update()

    def pre_scale_to_equalize_dpi(self) -> None:
        """Scale to make pixels square (equalize horizontal/vertical DPI)."""
        if self.orig_dpi.is_null():
            self._pre_scaled_dpi = self.orig_dpi
            self._update()
            return

        min_dpi = int(min(self.orig_dpi.horizontal, self.orig_dpi.vertical))
        self.pre_scale_to_dpi(Dpi(horizontal=min_dpi, vertical=min_dpi))

    # Step 2: Pre-rotate

    def set_pre_rotation(self, rotation: OrthogonalRotation) -> None:
        """Set orthogonal pre-rotation, resetting subsequent steps."""
        self._pre_rotation = rotation

        # Calculate pre-rotated size
        scaled_width = self.orig_rect.width
        scaled_height = self.orig_rect.height
        if not self.orig_dpi.is_null() and not self._pre_scaled_dpi.is_null():
            scaled_width *= self._pre_scaled_dpi.horizontal / self.orig_dpi.horizontal
            scaled_height *= self._pre_scaled_dpi.vertical / self.orig_dpi.vertical

        self._pre_rotate_xform = _orthogonal_rotation_matrix(
            rotation, scaled_width, scaled_height
        )

        self._reset_pre_crop_area()
        self._reset_post_rotation()
        self._reset_post_crop()
        self._reset_post_scale()
        self._update()

    # Step 3: Pre-crop

    def set_pre_crop_area(self, area: NDArray[np.float64]) -> None:
        """Set pre-crop area polygon, resetting subsequent steps."""
        self._pre_crop_area = area.copy()
        self._pre_crop_xform = self._calc_crop_xform(area)
        self._reset_post_rotation()
        self._reset_post_crop()
        self._reset_post_scale()
        self._update()

    def orig_rect_to_pre_crop_space(self, rect: Rect) -> NDArray[np.float64]:
        """Map a rectangle of the original image to the pre-crop area's space.

        Args:
            rect (Rect): Rectangle in original image coordinates.

        Returns:
            numpy.ndarray of numpy.float64:
                Its four corners after pre-scale and pre-rotate,
            shape (4, 2); the coordinates ``set_pre_crop_area`` takes.
        """
        return self.orig_polygon_to_pre_crop_space(_rect_to_polygon(rect))

    def orig_polygon_to_pre_crop_space(
        self, polygon: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map a polygon of the original image to the pre-crop area's space.

        Args:
            polygon (numpy.ndarray of numpy.float64): ``(N, 2)`` points in
                original image coordinates.

        Returns:
            numpy.ndarray of numpy.float64: The points after pre-scale and
            pre-rotate; the coordinates ``set_pre_crop_area`` takes.
        """
        return _transform_polygon(
            np.asarray(polygon, dtype=np.float64),
            _then(self._pre_scale_xform, self._pre_rotate_xform),
        )

    # Step 4: Post-rotate

    def set_post_rotation(self, degrees: float) -> None:
        """Set deskew rotation in degrees, resetting subsequent steps."""
        self._post_rotation = degrees
        self._post_rotate_xform = self._calc_post_rotate_xform(
            degrees, self._post_oblique
        )
        self._reset_post_crop()
        self._reset_post_scale()
        self._update()

    def set_post_oblique(self, degrees: float) -> None:
        """Set the oblique (shear) correction, resetting subsequent steps.

        The shear is applied after the deskew rotation: x' = x - tan(degrees) * y.

        Args:
            degrees (float): Lean of the page's vertical axis in degrees, positive
                when it drifts right going down; the shear removes exactly that
                lean. Zero means no shear.
        """
        self._post_oblique = degrees
        self._post_rotate_xform = self._calc_post_rotate_xform(
            self._post_rotation, degrees
        )
        self._reset_post_crop()
        self._reset_post_scale()
        self._update()

    # Step 5: Post-crop

    def set_post_crop_area(self, area: NDArray[np.float64]) -> None:
        """Set post-crop area polygon, resetting subsequent steps."""
        self._post_crop_area = area.copy()
        self._post_crop_xform = self._calc_crop_xform(area)
        self._reset_post_scale()
        self._update()

    # Step 6: Post-scale

    def post_scale_to_dpi(self, dpi: Dpi) -> None:
        """Set post-scale to achieve specific output DPI."""
        self._post_scaled_dpi = dpi
        self._post_scale_xform = self._calc_post_scale_xform(dpi)
        self._update()

    # Transform methods

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transform a point from original to resulting coordinates."""
        pt = np.array([[x, y, 1]], dtype=np.float64)
        result = pt @ self._transform.T
        w = result[0, 2] if abs(result[0, 2]) > 1e-10 else 1.0
        return (float(result[0, 0] / w), float(result[0, 1] / w))

    def transform_point_back(self, x: float, y: float) -> tuple[float, float]:
        """Transform a point from resulting to original coordinates."""
        pt = np.array([[x, y, 1]], dtype=np.float64)
        result = pt @ self._inv_transform.T
        w = result[0, 2] if abs(result[0, 2]) > 1e-10 else 1.0
        return (float(result[0, 0] / w), float(result[0, 1] / w))

    def map_polygon(self, polygon: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map an Nx2 polygon from original to resulting coordinates."""
        return _transform_polygon(polygon, self._transform)

    def map_polygon_back(self, polygon: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map an Nx2 polygon from resulting to original coordinates."""
        return _transform_polygon(polygon, self._inv_transform)

    # Private methods

    def _calc_crop_xform(self, area: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate crop transformation (translate to put bounds at origin)."""
        if len(area) == 0:
            return _identity_matrix()
        bounds = _polygon_bounding_rect(area)
        return _translation_matrix(-bounds.x, -bounds.y)

    def _calc_post_rotate_xform(
        self, degrees: float, oblique_degrees: float = 0.0
    ) -> NDArray[np.float64]:
        """Calculate the rotation, then shear, around the pre-crop area center.

        The result is shifted so the top-left corner of the pre-crop area's
        bounds stays where it was.
        """
        if abs(degrees) < 1e-10 and abs(oblique_degrees) < 1e-10:
            return _identity_matrix()

        # Get center of pre-crop area
        if len(self._pre_crop_area) == 0:
            # Use original rect transformed through pre-scale and pre-rotate
            pre_scale_rotate = _then(self._pre_scale_xform, self._pre_rotate_xform)
            poly = _transform_polygon(
                _rect_to_polygon(self.orig_rect), pre_scale_rotate
            )
        else:
            poly = self._pre_crop_area

        bounds = _polygon_bounding_rect(poly)
        cx, cy = bounds.center()

        xform = _then(
            _translation_matrix(-cx, -cy),
            _rotation_matrix(degrees),
            _shear_matrix(oblique_degrees),
            _translation_matrix(cx, cy),
        )

        # Calculate size change and adjust
        pre_rotate_poly = _transform_polygon(poly, self._pre_crop_xform)
        pre_rotate_bounds = _polygon_bounding_rect(pre_rotate_poly)

        post_rotate_poly = _transform_polygon(pre_rotate_poly, xform)
        post_rotate_bounds = _polygon_bounding_rect(post_rotate_poly)

        # Translate to align top-left corners
        dx = pre_rotate_bounds.left - post_rotate_bounds.left
        dy = pre_rotate_bounds.top - post_rotate_bounds.top
        return _then(xform, _translation_matrix(dx, dy))

    def _calc_post_scale_xform(self, target_dpi: Dpi) -> NDArray[np.float64]:
        """Calculate the post-scale that brings the result to ``target_dpi``.

        A unit step along each result axis is measured after the pre-scale,
        where pixels are square at ``pre_scaled_dpi``, so the length is in the
        same units whatever the rotations in between. In the original pixels
        of an image with unequal resolutions, a step that a rotation turns onto
        the other axis would be divided by the wrong axis's resolution.
        """
        pre_dpi = self._pre_scaled_dpi
        if target_dpi.is_null() or pre_dpi.is_null():
            return _identity_matrix()

        after_pre_scale = _then(
            self._pre_rotate_xform,
            self._pre_crop_xform,
            self._post_rotate_xform,
            self._post_crop_xform,
        )
        back = np.asarray(np.linalg.inv(after_pre_scale), dtype=np.float64)
        units = _transform_polygon(
            np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64), back
        )
        h_len = float(np.linalg.norm(units[1] - units[0]))
        v_len = float(np.linalg.norm(units[2] - units[0]))
        if h_len < 1e-10 or v_len < 1e-10:
            return _identity_matrix()

        xscale = float(target_dpi.horizontal * h_len / pre_dpi.horizontal)
        yscale = float(target_dpi.vertical * v_len / pre_dpi.vertical)
        return _scale_matrix(xscale, yscale)

    def _reset_pre_crop_area(self) -> None:
        """Reset pre-crop area."""
        self._pre_crop_area = np.array([], dtype=np.float64).reshape(0, 2)
        self._pre_crop_xform = _identity_matrix()

    def _reset_post_rotation(self) -> None:
        """Reset post-rotation and oblique."""
        self._post_rotation = 0.0
        self._post_oblique = 0.0
        self._post_rotate_xform = _identity_matrix()

    def _reset_post_crop(self) -> None:
        """Reset post-crop area."""
        self._post_crop_area = np.array([], dtype=np.float64).reshape(0, 2)
        self._post_crop_xform = _identity_matrix()

    def _reset_post_scale(self) -> None:
        """Reset post-scale."""
        self._post_scaled_dpi = Dpi()
        self._post_scale_xform = _identity_matrix()

    def _update(self) -> None:
        """Recalculate the combined transformation and resulting areas."""
        # Combine transforms: 1*2, 3*4, 5*6
        pre_scale_then_pre_rotate = _then(self._pre_scale_xform, self._pre_rotate_xform)
        pre_crop_then_post_rotate = _then(self._pre_crop_xform, self._post_rotate_xform)
        post_crop_then_post_scale = _then(self._post_crop_xform, self._post_scale_xform)

        pre_crop_and_further = _then(
            pre_crop_then_post_rotate, post_crop_then_post_scale
        )
        self._transform = _then(pre_scale_then_pre_rotate, pre_crop_and_further)

        # Inverse transform
        try:
            self._inv_transform = np.asarray(
                np.linalg.inv(self._transform), dtype=np.float64
            )
        except np.linalg.LinAlgError:
            self._inv_transform = _identity_matrix()

        # Default pre-crop area to full image if not set
        if len(self._pre_crop_area) == 0:
            self._pre_crop_area = _transform_polygon(
                _rect_to_polygon(self.orig_rect), pre_scale_then_pre_rotate
            )

        # Default post-crop area to pre-crop area after rotation if not set
        if len(self._post_crop_area) == 0:
            self._post_crop_area = _transform_polygon(
                self._pre_crop_area, pre_crop_then_post_rotate
            )

        # Calculate resulting areas
        self._resulting_pre_crop_area = _transform_polygon(
            self._pre_crop_area, pre_crop_and_further
        )
        self._resulting_post_crop_area = _transform_polygon(
            self._post_crop_area, post_crop_then_post_scale
        )
        self._resulting_rect = _polygon_bounding_rect(self._resulting_post_crop_area)


_WHITE = 255
_INTEGER_SHIFT_TOLERANCE = 1e-6
_FILL_SHIFT_BITS = 4


def render_transformed(
    image: NDArray[np.uint8],
    transformation: ImageTransformation,
    size: tuple[int, int] | None = None,
    *,
    invert: bool = False,
    source_scale: tuple[float, float] = (1.0, 1.0),
) -> NDArray[np.uint8]:
    """Resample an image through a transformation in a single pass.

    Every step of ImageTransformation is affine, so one warp covers pre-crop,
    deskew rotation, post-crop and the scale to output DPI; see
    ``warp_area_averaged`` for how a reduction averages source pixels. Areas of
    the result with no source pixels, or outside the pre-crop area, are white.
    Translation-only transforms by whole pixels are served by slicing, which
    keeps pixel values exact.

    Args:
        image (numpy.ndarray of numpy.uint8): Source image, (H, W) or (H, W, C) with at
            most 4 channels. Its size must match transformation.orig_rect
            times ``source_scale``.
        transformation (ImageTransformation): Maps source to resulting
            coordinates.
        size (tuple[int, int] | None): Width and height of the result. It
            covers the same area as the full-size result, so each of its pixels
            spans ``full / size`` full-size pixels along each axis and averages
            the source under them. None gives the full size.
        invert (bool): Render ``255 - image`` without building it: the pixels
            are those of the inverted image resampled, and the area with no
            source pixel is still white.
        source_scale (tuple[float, float]): Width and height of ``image`` over
            those of the source the transformation was made for, when
            ``image`` is a shrunk copy of it.

    Returns:
        numpy.ndarray of numpy.uint8:
            New array with the same channel layout as the input
        and ``size``, or the size of transformation.resulting_rect rounded to
        whole pixels and at least 1x1. The input is not modified.

    Raises:
        TypeError: If the image is not uint8.
    """
    if image.dtype != np.uint8:
        raise TypeError(f"render_transformed needs a uint8 image, got {image.dtype}")
    rect = transformation.resulting_rect
    full_width = max(1, round(rect.width))
    full_height = max(1, round(rect.height))
    out_width, out_height = (full_width, full_height) if size is None else size
    scale = (full_width / out_width, full_height / out_height)

    # OpenCV addresses pixel centres at integer coordinates while the
    # transformation addresses pixel corners, hence the half pixel shifts.
    dst_to_src = _then(
        _translation_matrix(0.5, 0.5),
        _scale_matrix(*scale),
        _translation_matrix(rect.x, rect.y),
        transformation.transform_back,
        _scale_matrix(*source_scale),
        _translation_matrix(-0.5, -0.5),
    )

    shift = _whole_pixel_shift(dst_to_src)
    window = _crop_window(image, shift, out_width, out_height) if shift else None
    if window is not None:
        result = np.subtract(_WHITE, window, out=window) if invert else window
    else:
        result = warp_area_averaged(
            image, dst_to_src, (out_width, out_height), invert=invert
        )

    if result.ndim < image.ndim:
        # cv2.warpAffine drops a trailing singleton channel axis.
        result = result.reshape(out_height, out_width, 1)

    crop_area = (transformation.resulting_pre_crop_area - [rect.x, rect.y]) / scale
    _whiten_outside(result, crop_area)
    return result


def warp_area_averaged(
    image: NDArray[np.uint8],
    dst_to_src: NDArray[np.float64],
    size: tuple[int, int],
    *,
    invert: bool = False,
) -> NDArray[np.uint8]:
    """Warp an image bilinearly, averaging every source pixel an output pixel covers.

    This is upstream's transformGeneric in two OpenCV calls. A separable filter
    sized to the source area of one output pixel runs over the part of the
    source the warp reads, then one bilinear ``cv2.warpAffine`` samples the
    filtered image. ``_box_taps`` sizes the filter so the two steps together
    average as much as upstream's box does.

    An output pixel that covers at most one source pixel along both axes skips
    the filter, so 1:1 and enlarging warps are plain ``cv2.warpAffine``, unless
    ``invert`` asks for the one-tap filter that inverts the window.

    Args:
        image (numpy.ndarray of numpy.uint8): Source, (H, W) or (H, W, C) with
            at most 4 channels. Only read.
        dst_to_src (numpy.ndarray of numpy.float64): 3x3 affine matrix taking
            output pixel centres to source pixel centres, both with OpenCV's
            convention that pixel (0, 0) is centred on (0, 0).
        size (tuple[int, int]): Output width and height.
        invert (bool): Warp ``255 - image``. The filter inverts the window it
            reads as it filters, by negating its taps and adding 255, so no
            inverted copy of the whole image is made.

    Returns:
        numpy.ndarray of numpy.uint8: The warped image. Points that map outside
        the source are white. cv2.warpAffine drops a trailing singleton channel
        axis, so an (H, W, 1) input comes back as (H, W).
    """
    taps_x, taps_y = (_box_taps(width) for width in _source_footprint(dst_to_src))
    if taps_x is None and taps_y is None and not invert:
        return _warp_bilinear(image, dst_to_src, size)
    taps_x = taps_x or _SINGLE_TAP
    taps_y = taps_y or _SINGLE_TAP
    margin = max(len(taps_x.weights), len(taps_y.weights)) + 2
    window = _sampled_window(image.shape[:2], dst_to_src, size, margin)
    if window is None:
        # Every output pixel lies outside the source, so it is white either way.
        return _warp_bilinear(image, dst_to_src, size)
    left, top, right, bottom = window
    # ddepth -1 makes OpenCV round and saturate to uint8 itself, so the
    # conversion only states the dtype and copies nothing.
    filtered = np.asarray(
        cv2.sepFilter2D(
            image[top:bottom, left:right],
            -1,
            taps_x.weights,
            -taps_y.weights if invert else taps_y.weights,
            anchor=(taps_x.anchor, taps_y.anchor),
            delta=_WHITE if invert else 0,
            borderType=cv2.BORDER_REPLICATE,
        ),
        dtype=np.uint8,
    )
    in_filtered = _then(
        dst_to_src,
        _translation_matrix(-left - taps_x.centre, -top - taps_y.centre),
    )
    return _warp_bilinear(filtered, in_filtered, size)


@dataclass(frozen=True)
class _BoxTaps:
    """One axis of a box filter sampled on the pixel grid.

    Filtered pixel ``j`` holds the box centred on source position
    ``j + centre``; ``anchor`` is the index of the tap at offset zero.
    """

    weights: NDArray[np.float32]
    anchor: int
    centre: float


_SINGLE_TAP = _BoxTaps(np.ones(1, dtype=np.float32), 0, 0.0)
# Upstream measures the box in 1/32 pixel units and rounds it to a whole unit.
_BOX_UNITS_PER_PIXEL = 32
_NEGLIGIBLE_WEIGHT = 1e-9


def _source_footprint(dst_to_src: NDArray[np.float64]) -> tuple[float, float]:
    """Return the width and height of the source box one output pixel covers.

    As upstream's calcSrcUnitSize does, this maps the midpoints of an output
    pixel's edges rather than its corners, so a rotation without scaling keeps
    the box within one source pixel.
    """
    edge_midpoints = np.array([[0.5, 1.0, 0.5, 0.0], [0.0, 0.5, 1.0, 0.5]])
    mapped = dst_to_src[:2, :2] @ edge_midpoints
    width, height = np.ptp(mapped, axis=1)
    return float(width), float(height)


def _box_taps(width: float) -> _BoxTaps | None:
    """Return the filter taps that stand in for a box ``width`` source pixels wide.

    The taps are matched by variance, not shape, because the bilinear warp that
    follows them blurs too. Upstream's box over pixel cells has the variance of
    a ``width`` box convolved with a one pixel box. The bilinear step adds a one
    pixel tent. So the taps supply the difference, ``(width**2 - 1) / 12``.

    The taps sit centred on a pixel when the nearest whole width is odd and on
    a pixel edge when it is even, rounding halves down, so a whole width gives
    exactly the plain box.

    Args:
        width (float): Source pixels one output pixel covers along this axis.

    Returns:
        _BoxTaps | None: None when the box covers at most one pixel.
    """
    units = round(width * _BOX_UNITS_PER_PIXEL)
    if units <= _BOX_UNITS_PER_PIXEL:
        return None
    width = units / _BOX_UNITS_PER_PIXEL
    centre = 0.5 if math.ceil(width - 0.5) % 2 == 0 else 0.0
    return _variance_matched_taps((width * width - 1) / 12, centre)


def _variance_matched_taps(variance: float, centre: float) -> _BoxTaps:
    """Return symmetric taps of whole weights plus one fractional outer pair.

    The taps grow outwards from the centre one pair at a time, each pair with
    weight 1, until a further whole pair would overshoot ``variance``. The last
    pair then gets the fractional weight that lands on ``variance`` exactly.

    Args:
        variance (float): Variance to reach, in square source pixels. A
            ``centre`` of 0.5 cannot go below 0.25, its two middle taps alone.
        centre (float): 0.0 for taps centred on a pixel, 0.5 for taps centred
            on the edge between two.

    Returns:
        _BoxTaps: Normalised taps, as close to ``variance`` as the lattice allows.
    """
    distances = [0.5] if centre else [0.0]
    weights = [1.0]
    total, moment = (2.0, 0.5) if centre else (1.0, 0.0)
    while variance * total - moment > _NEGLIGIBLE_WEIGHT:
        distance = distances[-1] + 1
        if moment + 2 * distance**2 <= variance * (total + 2):
            weight = 1.0
        else:
            weight = (variance * total - moment) / (2 * distance**2 - 2 * variance)
        distances.append(distance)
        weights.append(weight)
        total += 2 * weight
        moment += 2 * weight * distance**2
    outer_first = np.array(weights[::-1], dtype=np.float64)
    mirrored = outer_first[::-1] if centre else outer_first[-2::-1]
    taps = np.concatenate([outer_first, mirrored])
    return _BoxTaps(
        (taps / taps.sum()).astype(np.float32), len(outer_first) - 1, centre
    )


def _sampled_window(
    shape: tuple[int, ...],
    dst_to_src: NDArray[np.float64],
    size: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int] | None:
    """Return the (left, top, right, bottom) source window a warp reads.

    Filtering only this window skips source pixels no output pixel reaches,
    such as the facing page of a split scan. ``margin`` must cover the filter's
    reach plus the bilinear neighbour, so that the clamping at a window edge
    inside the image never shows in the result.

    Args:
        shape (tuple[int, ...]): Source height and width.
        dst_to_src (numpy.ndarray of numpy.float64): Output pixel centres to
            source pixel centres.
        size (tuple[int, int]): Output width and height.
        margin (int): Source pixels added on every side.

    Returns:
        tuple[int, int, int, int] | None: The window clipped to the image, or
        None when nothing of it lies inside.
    """
    width, height = size
    corners = np.array(
        [[0, width - 1, width - 1, 0], [0, 0, height - 1, height - 1], [1, 1, 1, 1]],
        dtype=np.float64,
    )
    mapped = (dst_to_src @ corners)[:2]
    low = np.floor(mapped.min(axis=1)).astype(int) - margin
    high = np.ceil(mapped.max(axis=1)).astype(int) + margin + 1
    image_height, image_width = shape[0], shape[1]
    left, top = max(int(low[0]), 0), max(int(low[1]), 0)
    right, bottom = min(int(high[0]), image_width), min(int(high[1]), image_height)
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def _warp_bilinear(
    image: NDArray[np.uint8], dst_to_src: NDArray[np.float64], size: tuple[int, int]
) -> NDArray[np.uint8]:
    """Sample ``image`` bilinearly where each output pixel maps; white outside."""
    return np.asarray(
        cv2.warpAffine(
            image,
            dst_to_src[:2],
            size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(_WHITE,) * 4,
        ),
        dtype=np.uint8,
    )


def _whiten_outside(image: NDArray[np.uint8], area: NDArray[np.float64]) -> None:
    """Paint white everything outside a polygon, in place.

    Margins can reach past the pre-crop area, where the source holds pixels of
    a neighbouring page that must not show through.

    Args:
        image: Array to modify; the polygon is in its pixel-corner coordinates.
        area: (N, 2) polygon, convex or not.
    """
    height, width = image.shape[:2]
    lo, hi = area.min(axis=0), area.max(axis=0)
    axis_aligned = bool(
        np.all(np.isclose(area[:, 0], lo[0]) | np.isclose(area[:, 0], hi[0]))
        and np.all(np.isclose(area[:, 1], lo[1]) | np.isclose(area[:, 1], hi[1]))
    )
    if axis_aligned:
        # A pixel is inside when its centre is.
        first_col, first_row = np.ceil(lo - 0.5 - _INTEGER_SHIFT_TOLERANCE).astype(int)
        end_col, end_row = np.floor(hi - 0.5 + _INTEGER_SHIFT_TOLERANCE).astype(int) + 1
        image[:, : max(first_col, 0)] = _WHITE
        image[:, max(min(end_col, width), 0) :] = _WHITE
        image[: max(first_row, 0), :] = _WHITE
        image[max(min(end_row, height), 0) :, :] = _WHITE
        return

    inside = np.zeros((height, width), dtype=np.uint8)
    # fillPoly takes pixel-centre coordinates in fixed point.
    centre_coords = np.round((area - 0.5) * (1 << _FILL_SHIFT_BITS)).astype(np.int32)
    cv2.fillPoly(inside, [centre_coords], 1, shift=_FILL_SHIFT_BITS)
    image[inside == 0] = _WHITE


def _whole_pixel_shift(dst_to_src: NDArray[np.float64]) -> tuple[int, int] | None:
    """Return (dx, dy) when the matrix is a translation by whole pixels."""
    if not np.allclose(dst_to_src[:2, :2], np.eye(2), atol=_INTEGER_SHIFT_TOLERANCE):
        return None
    dx, dy = float(dst_to_src[0, 2]), float(dst_to_src[1, 2])
    if (
        abs(dx - round(dx)) > _INTEGER_SHIFT_TOLERANCE
        or abs(dy - round(dy)) > _INTEGER_SHIFT_TOLERANCE
    ):
        return None
    return round(dx), round(dy)


def _crop_window(
    image: NDArray[np.uint8], shift: tuple[int, int], width: int, height: int
) -> NDArray[np.uint8] | None:
    """Copy the window at the given offset, or None if it leaves the image."""
    dx, dy = shift
    if dx < 0 or dy < 0 or dx + width > image.shape[1] or dy + height > image.shape[0]:
        return None
    return image[dy : dy + height, dx : dx + width].copy()
