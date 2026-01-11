"""Image transformation pipeline.

This module provides coordinate transformation management through a 6-step
pipeline, matching the C++ ImageTransformation class:

1. Pre-scale: Equalize DPI (make pixels square)
2. Pre-rotate: Orthogonal rotation (0, 90, 180, 270 degrees)
3. Pre-crop: Crop to page boundaries (after split)
4. Post-rotate: Deskew rotation (arbitrary angle)
5. Post-crop: Adjust margins
6. Post-scale: Scale to output DPI
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi, OrthogonalRotation


@dataclass
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


def _orthogonal_rotation_matrix(
    rotation: OrthogonalRotation, width: float, height: float
) -> NDArray[np.float64]:
    """Create transformation matrix for orthogonal rotation.

    The rotation is around the image center, with size adjustment.
    """
    degrees = rotation.degrees
    if degrees == 0:
        return _identity_matrix()

    if degrees == 90:
        # 90 CW: (x, y) -> (height - y - 1, x)
        m = np.array(
            [[0, 1, 0], [-1, 0, height], [0, 0, 1]],
            dtype=np.float64,
        )
    elif degrees == 180:
        # 180: (x, y) -> (width - x - 1, height - y - 1)
        m = np.array(
            [[-1, 0, width], [0, -1, height], [0, 0, 1]],
            dtype=np.float64,
        )
    else:  # 270
        # 270 CW (90 CCW): (x, y) -> (y, width - x - 1)
        m = np.array(
            [[0, -1, width], [1, 0, 0], [0, 0, 1]],
            dtype=np.float64,
        )
    return m


@dataclass
class ImageTransformation:
    """Provides a transformed view of an image through a 6-step pipeline.

    The transformation pipeline consists of:
    1. Pre-scale: Scale to equalize DPI (make pixels square)
    2. Pre-rotate: Orthogonal rotation (0, 90, 180, 270 degrees)
    3. Pre-crop: Crop to page boundaries
    4. Post-rotate: Deskew rotation (arbitrary angle)
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
    def from_size(
        cls, width: float, height: float, dpi: Dpi | None = None
    ) -> ImageTransformation:
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
        old_pre_scale_rotate = self._pre_scale_xform @ self._pre_rotate_xform

        # Update pre-scale
        self._pre_scale_xform = _scale_matrix(xscale, yscale)

        # Update pre-rotate for new size
        self._pre_rotate_xform = _orthogonal_rotation_matrix(
            self._pre_rotation, new_width, new_height
        )

        # Recalculate pre-crop area in new coordinates
        new_pre_scale_rotate = self._pre_scale_xform @ self._pre_rotate_xform
        if len(self._pre_crop_area) > 0:
            # Transform through: old_inv -> new
            old_inv = np.asarray(np.linalg.inv(old_pre_scale_rotate), dtype=np.float64)
            self._pre_crop_area = _transform_polygon(
                _transform_polygon(self._pre_crop_area, old_inv),
                new_pre_scale_rotate,
            )
            self._pre_crop_xform = self._calc_crop_xform(self._pre_crop_area)

        # Recalculate post-rotate
        self._post_rotate_xform = self._calc_post_rotate_xform(self._post_rotation)

        # Recalculate post-crop area
        if len(self._post_crop_area) > 0:
            old_pre_crop_post_rotate = (
                old_pre_scale_rotate @ self._pre_crop_xform @ self._post_rotate_xform
            )
            new_pre_crop_post_rotate = (
                new_pre_scale_rotate @ self._pre_crop_xform @ self._post_rotate_xform
            )
            old_inv = np.asarray(
                np.linalg.inv(old_pre_crop_post_rotate), dtype=np.float64
            )
            self._post_crop_area = _transform_polygon(
                _transform_polygon(self._post_crop_area, old_inv),
                new_pre_crop_post_rotate,
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

    # Step 4: Post-rotate

    def set_post_rotation(self, degrees: float) -> None:
        """Set deskew rotation in degrees, resetting subsequent steps."""
        self._post_rotation = degrees
        self._post_rotate_xform = self._calc_post_rotate_xform(degrees)
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

    # Private methods

    def _calc_crop_xform(self, area: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate crop transformation (translate to put bounds at origin)."""
        if len(area) == 0:
            return _identity_matrix()
        bounds = _polygon_bounding_rect(area)
        return _translation_matrix(-bounds.x, -bounds.y)

    def _calc_post_rotate_xform(self, degrees: float) -> NDArray[np.float64]:
        """Calculate post-rotation transformation around pre-crop area center."""
        if abs(degrees) < 1e-10:
            return _identity_matrix()

        # Get center of pre-crop area
        if len(self._pre_crop_area) == 0:
            # Use original rect transformed through pre-scale and pre-rotate
            pre_scale_rotate = self._pre_scale_xform @ self._pre_rotate_xform
            poly = _transform_polygon(
                _rect_to_polygon(self.orig_rect), pre_scale_rotate
            )
        else:
            poly = self._pre_crop_area

        bounds = _polygon_bounding_rect(poly)
        cx, cy = bounds.center()

        # Rotate around center
        t1 = _translation_matrix(-cx, -cy)
        r = _rotation_matrix(degrees)
        t2 = _translation_matrix(cx, cy)
        xform = t1 @ r @ t2

        # Calculate size change and adjust
        pre_rotate_poly = _transform_polygon(poly, self._pre_crop_xform)
        pre_rotate_bounds = _polygon_bounding_rect(pre_rotate_poly)

        post_rotate_poly = _transform_polygon(pre_rotate_poly, xform)
        post_rotate_bounds = _polygon_bounding_rect(post_rotate_poly)

        # Translate to align top-left corners
        dx = pre_rotate_bounds.left - post_rotate_bounds.left
        dy = pre_rotate_bounds.top - post_rotate_bounds.top
        adjust = _translation_matrix(dx, dy)

        return xform @ adjust

    def _calc_post_scale_xform(self, target_dpi: Dpi) -> NDArray[np.float64]:
        """Calculate post-scale transformation to achieve target DPI."""
        if target_dpi.is_null():
            return _identity_matrix()

        # Calculate effective DPI at this point in the pipeline
        # We need to measure how a unit length in resulting coords
        # maps back to original coords
        current_transform = (
            self._pre_scale_xform
            @ self._pre_rotate_xform
            @ self._pre_crop_xform
            @ self._post_rotate_xform
            @ self._post_crop_xform
        )
        inv_current = np.asarray(np.linalg.inv(current_transform), dtype=np.float64)

        # Transform unit vectors back to original
        origin = np.array([[0, 0]], dtype=np.float64)
        h_unit = np.array([[1, 0]], dtype=np.float64)
        v_unit = np.array([[0, 1]], dtype=np.float64)

        orig_origin = _transform_polygon(origin, inv_current)
        orig_h = _transform_polygon(h_unit, inv_current)
        orig_v = _transform_polygon(v_unit, inv_current)

        h_len = float(np.linalg.norm(orig_h - orig_origin))
        v_len = float(np.linalg.norm(orig_v - orig_origin))

        if self.orig_dpi.is_null() or h_len < 1e-10 or v_len < 1e-10:
            return _identity_matrix()

        xscale = float(target_dpi.horizontal * h_len / self.orig_dpi.horizontal)
        yscale = float(target_dpi.vertical * v_len / self.orig_dpi.vertical)

        return _scale_matrix(xscale, yscale)

    def _reset_pre_crop_area(self) -> None:
        """Reset pre-crop area."""
        self._pre_crop_area = np.array([], dtype=np.float64).reshape(0, 2)
        self._pre_crop_xform = _identity_matrix()

    def _reset_post_rotation(self) -> None:
        """Reset post-rotation."""
        self._post_rotation = 0.0
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
        pre_scale_then_pre_rotate = self._pre_scale_xform @ self._pre_rotate_xform
        pre_crop_then_post_rotate = self._pre_crop_xform @ self._post_rotate_xform
        post_crop_then_post_scale = self._post_crop_xform @ self._post_scale_xform

        # Full pipeline: 12 * 34 * 56
        pre_crop_and_further = pre_crop_then_post_rotate @ post_crop_then_post_scale
        self._transform = pre_scale_then_pre_rotate @ pre_crop_and_further

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
