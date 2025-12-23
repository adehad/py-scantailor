"""Homography (perspective transform) utilities.

This module provides the Homography class for perspective transformations,
which can map any quadrilateral to another quadrilateral.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator


class Homography(BaseModel):
    """Represents a 2D homography (perspective transformation).

    The transformation is stored as a 3x3 matrix that can be applied
    to 2D points via cv2.perspectiveTransform() or cv2.warpPerspective().

    Attributes:
        matrix: 3x3 homography matrix.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    matrix: NDArray[np.floating]

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, v: NDArray[np.floating]) -> NDArray[np.floating]:
        """Validate matrix shape."""
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (3, 3):
            msg = "Matrix must have shape (3, 3)"
            raise ValueError(msg)
        return v

    @classmethod
    def identity(cls) -> "Homography":
        """Create an identity homography.

        Returns:
            Homography: Identity transformation.
        """
        return cls(matrix=np.eye(3, dtype=np.float64))

    @classmethod
    def from_four_points(
        cls,
        src_points: NDArray[np.floating],
        dst_points: NDArray[np.floating],
    ) -> "Homography":
        """Create a homography from four point correspondences.

        Args:
            src_points: Source points with shape (4, 2).
            dst_points: Destination points with shape (4, 2).

        Returns:
            Homography: Transformation mapping src_points to dst_points.

        Raises:
            ValueError: If points don't have correct shape.
        """
        src = np.asarray(src_points, dtype=np.float32)
        dst = np.asarray(dst_points, dtype=np.float32)

        if src.shape != (4, 2) or dst.shape != (4, 2):
            msg = "Both src_points and dst_points must have shape (4, 2)"
            raise ValueError(msg)

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cls(matrix=matrix.astype(np.float64))

    @classmethod
    def from_points(
        cls,
        src_points: NDArray[np.floating],
        dst_points: NDArray[np.floating],
        method: int = cv2.RANSAC,
        ransac_reproj_threshold: float = 5.0,
    ) -> "Homography":
        """Create a homography from multiple point correspondences.

        Uses cv2.findHomography which can handle more than 4 points
        and supports robust estimation methods.

        Args:
            src_points: Source points with shape (N, 2) where N >= 4.
            dst_points: Destination points with shape (N, 2) where N >= 4.
            method: Computation method (0, cv2.RANSAC, cv2.LMEDS, cv2.RHO).
            ransac_reproj_threshold: Maximum reprojection error for RANSAC.

        Returns:
            Homography: Transformation mapping src_points to dst_points.

        Raises:
            ValueError: If homography cannot be computed.
        """
        src = np.asarray(src_points, dtype=np.float32)
        dst = np.asarray(dst_points, dtype=np.float32)

        if src.shape[0] < 4 or dst.shape[0] < 4:
            msg = "At least 4 point correspondences required"
            raise ValueError(msg)

        if src.shape != dst.shape:
            msg = "src_points and dst_points must have same shape"
            raise ValueError(msg)

        matrix, _mask = cv2.findHomography(src, dst, method, ransac_reproj_threshold)

        if matrix is None:
            msg = "Could not compute homography from given points"
            raise ValueError(msg)

        return cls(matrix=matrix.astype(np.float64))

    @classmethod
    def from_rectangle_to_quad(
        cls,
        rect: tuple[float, float, float, float],
        quad: NDArray[np.floating],
    ) -> "Homography":
        """Create a homography mapping a rectangle to a quadrilateral.

        Args:
            rect: Rectangle as (x, y, width, height).
            quad: Destination quadrilateral as array with shape (4, 2),
                  vertices in order: top-left, top-right, bottom-right, bottom-left.

        Returns:
            Homography: Transformation mapping rectangle to quad.
        """
        x, y, w, h = rect
        src_points = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        )
        dst_points = np.asarray(quad, dtype=np.float32)

        return cls.from_four_points(src_points, dst_points)

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

        # Apply homography: (x', y', w') = H * (x, y, 1)
        point_h = np.array([point[0], point[1], 1], dtype=np.float64)
        result_h = self.matrix @ point_h

        # Normalize by w'
        if abs(result_h[2]) < 1e-10:
            msg = "Point maps to infinity"
            raise ValueError(msg)

        return np.array([result_h[0] / result_h[2], result_h[1] / result_h[2]])

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

        # Use cv2.perspectiveTransform
        points_reshaped = points.reshape((-1, 1, 2)).astype(np.float32)
        result = cv2.perspectiveTransform(
            points_reshaped, self.matrix.astype(np.float32)
        )
        return result.reshape((-1, 2)).astype(np.float64)

    def compose(self, other: "Homography") -> "Homography":
        """Compose this homography with another.

        The resulting transformation applies 'other' first, then 'self'.
        I.e., result(p) = self(other(p))

        Args:
            other: Homography to compose with.

        Returns:
            Homography: Composed homography.
        """
        composed = self.matrix @ other.matrix
        return Homography(matrix=composed)

    def inverse(self) -> "Homography":
        """Compute the inverse homography.

        Returns:
            Homography: Inverse transformation.

        Raises:
            ValueError: If homography is singular.
        """
        det = np.linalg.det(self.matrix)
        if abs(det) < 1e-10:
            msg = "Homography is singular and cannot be inverted"
            raise ValueError(msg)

        inv = np.linalg.inv(self.matrix)
        return Homography(matrix=inv)

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

        return cv2.warpPerspective(
            image,
            self.matrix.astype(np.float32),
            output_size,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value,
        )


def warp_perspective(
    image: np.ndarray,
    src_points: NDArray[np.floating],
    dst_points: NDArray[np.floating],
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Warp an image using perspective transformation.

    Convenience function that computes homography and applies it in one step.

    Args:
        image: Input image.
        src_points: Source points with shape (4, 2).
        dst_points: Destination points with shape (4, 2).
        output_size: Output size as (width, height). If None, uses input size.

    Returns:
        np.ndarray: Warped image.
    """
    homography = Homography.from_four_points(src_points, dst_points)
    return homography.apply_to_image(image, output_size)
