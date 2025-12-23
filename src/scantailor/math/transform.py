"""Affine transform utilities.

This module provides the AffineTransform class for 2D affine transformations
including translation, rotation, scaling, and shearing.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator


class AffineTransform(BaseModel):
    """Represents a 2D affine transformation.

    The transformation is stored as a 2x3 matrix that can be applied
    to 2D points via cv2.transform() or cv2.warpAffine().

    The matrix form is:
        [a, b, tx]
        [c, d, ty]

    Applied to point (x, y):
        x' = a*x + b*y + tx
        y' = c*x + d*y + ty

    Attributes:
        matrix: 2x3 transformation matrix.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    matrix: NDArray[np.floating]

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, v: NDArray[np.floating]) -> NDArray[np.floating]:
        """Validate matrix shape."""
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (2, 3):
            msg = "Matrix must have shape (2, 3)"
            raise ValueError(msg)
        return v

    @classmethod
    def identity(cls) -> "AffineTransform":
        """Create an identity transformation.

        Returns:
            AffineTransform: Identity transformation that leaves points unchanged.
        """
        return cls(matrix=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64))

    @classmethod
    def translation(cls, tx: float, ty: float) -> "AffineTransform":
        """Create a translation transformation.

        Args:
            tx: Translation in x direction.
            ty: Translation in y direction.

        Returns:
            AffineTransform: Translation transformation.
        """
        return cls(matrix=np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float64))

    @classmethod
    def rotation(
        cls, angle: float, center: tuple[float, float] = (0, 0)
    ) -> "AffineTransform":
        """Create a rotation transformation.

        Args:
            angle: Rotation angle in degrees (counter-clockwise positive).
            center: Center of rotation as (x, y). Default is origin.

        Returns:
            AffineTransform: Rotation transformation.
        """
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cls(matrix=matrix.astype(np.float64))

    @classmethod
    def scaling(
        cls, sx: float, sy: float | None = None, center: tuple[float, float] = (0, 0)
    ) -> "AffineTransform":
        """Create a scaling transformation.

        Args:
            sx: Scale factor in x direction.
            sy: Scale factor in y direction. If None, uses sx.
            center: Center of scaling as (x, y). Default is origin.

        Returns:
            AffineTransform: Scaling transformation.
        """
        if sy is None:
            sy = sx

        cx, cy = center
        # Scale around center: translate to origin, scale, translate back
        matrix = np.array(
            [[sx, 0, cx * (1 - sx)], [0, sy, cy * (1 - sy)]], dtype=np.float64
        )
        return cls(matrix=matrix)

    @classmethod
    def shear(cls, shx: float = 0, shy: float = 0) -> "AffineTransform":
        """Create a shearing transformation.

        Args:
            shx: Shear factor in x direction (horizontal shear).
            shy: Shear factor in y direction (vertical shear).

        Returns:
            AffineTransform: Shearing transformation.
        """
        matrix = np.array([[1, shx, 0], [shy, 1, 0]], dtype=np.float64)
        return cls(matrix=matrix)

    def apply_to_point(self, point: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply transformation to a single point.

        Args:
            point: Point as array [x, y].

        Returns:
            NDArray[np.floating]: Transformed point as array [x', y'].
        """
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (2,):
            msg = "Point must be a 2D array with shape (2,)"
            raise ValueError(msg)

        # Add homogeneous coordinate
        point_h = np.array([point[0], point[1], 1], dtype=np.float64)
        result = self.matrix @ point_h
        return result

    def apply_to_points(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply transformation to multiple points.

        Args:
            points: Array of points with shape (N, 2).

        Returns:
            NDArray[np.floating]: Transformed points with shape (N, 2).
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            msg = "Points must be a 2D array with shape (N, 2)"
            raise ValueError(msg)

        # Use cv2.transform for efficiency
        points_reshaped = points.reshape((-1, 1, 2)).astype(np.float32)
        result = cv2.transform(points_reshaped, self.matrix.astype(np.float32))
        return result.reshape((-1, 2)).astype(np.float64)

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Compose this transformation with another.

        The resulting transformation applies 'other' first, then 'self'.
        I.e., result(p) = self(other(p))

        Args:
            other: Transformation to compose with.

        Returns:
            AffineTransform: Composed transformation.
        """
        # Convert to 3x3 homogeneous matrices for composition
        m1 = np.vstack([self.matrix, [0, 0, 1]])
        m2 = np.vstack([other.matrix, [0, 0, 1]])
        composed = m1 @ m2
        return AffineTransform(matrix=composed[:2, :])

    def inverse(self) -> "AffineTransform":
        """Compute the inverse transformation.

        Returns:
            AffineTransform: Inverse transformation.

        Raises:
            ValueError: If transformation is singular (non-invertible).
        """
        # Convert to 3x3 for inversion
        m = np.vstack([self.matrix, [0, 0, 1]])
        det = np.linalg.det(m[:2, :2])

        if abs(det) < 1e-10:
            msg = "Transformation is singular and cannot be inverted"
            raise ValueError(msg)

        inv = np.linalg.inv(m)
        return AffineTransform(matrix=inv[:2, :])

    def apply_to_image(
        self,
        image: np.ndarray,
        output_size: tuple[int, int] | None = None,
        flags: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: float | tuple[float, ...] = 0,
    ) -> np.ndarray:
        """Apply transformation to an image.

        Args:
            image: Input image.
            output_size: Output size as (width, height). If None, uses input size.
            flags: Interpolation flags (cv2.INTER_*).
            border_mode: Border handling mode (cv2.BORDER_*).
            border_value: Value for border pixels.

        Returns:
            np.ndarray: Transformed image.
        """
        if output_size is None:
            output_size = (image.shape[1], image.shape[0])

        return cv2.warpAffine(
            image,
            self.matrix.astype(np.float32),
            output_size,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value,
        )


def compose_transforms(*transforms: AffineTransform) -> AffineTransform:
    """Compose multiple transformations.

    Transformations are applied right-to-left, i.e., the last transform
    in the list is applied first.

    Args:
        *transforms: Transformations to compose.

    Returns:
        AffineTransform: Composed transformation.

    Raises:
        ValueError: If no transforms provided.
    """
    if not transforms:
        msg = "At least one transform must be provided"
        raise ValueError(msg)

    result = transforms[0]
    for t in transforms[1:]:
        result = result.compose(t)
    return result
