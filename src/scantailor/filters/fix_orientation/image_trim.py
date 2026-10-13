"""Manual per-image trim of the Fix Orientation stage."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from scantailor.core.models import OrthogonalRotation
from scantailor.core.transformation import ImageTransformation, Rect

# Smallest width or height, in source pixels, a trim leaves.
MIN_TRIM_INNER_SIDE = 32
# An intersection thinner than this is treated as empty.
_MIN_OVERLAP_AREA = 1e-6


class ImageTrim(BaseModel):
    """Inward crop from each edge of the source image, in source pixels.

    The edges are those of the image as stored, before any rotation. The other
    fields are ignored while the trim is disabled.

    Attributes:
        enabled (bool): Whether the trim is applied.
        left (int): Pixels cut from the left edge.
        top (int): Pixels cut from the top edge.
        right (int): Pixels cut from the right edge.
        bottom (int): Pixels cut from the bottom edge.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def to_inner_rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return the part of an image that survives the trim.

        Negative margins count as zero. Margins that would leave less than
        ``MIN_TRIM_INNER_SIDE`` pixels are reduced, the leading edge first.

        Args:
            width (int): Width of the stored image in pixels.
            height (int): Height of the stored image in pixels.

        Returns:
            tuple[int, int, int, int]: ``(x, y, width, height)`` in source pixels;
            the whole image when the trim is disabled or the image is empty.
        """
        if not self.enabled or width <= 0 or height <= 0:
            return 0, 0, width, height

        left, top = max(0, self.left), max(0, self.top)
        right, bottom = max(0, self.right), max(0, self.bottom)
        if left + right + MIN_TRIM_INNER_SIDE > width:
            left, right = _redistribute_excess(
                left, right, left + right + MIN_TRIM_INNER_SIDE - width
            )
        if top + bottom + MIN_TRIM_INNER_SIDE > height:
            top, bottom = _redistribute_excess(
                top, bottom, top + bottom + MIN_TRIM_INNER_SIDE - height
            )

        inner_width = max(MIN_TRIM_INNER_SIDE, width - left - right)
        inner_height = max(MIN_TRIM_INNER_SIDE, height - top - bottom)
        if left + inner_width > width:
            left = max(0, width - inner_width)
        if top + inner_height > height:
            top = max(0, height - inner_height)
        return left, top, inner_width, inner_height

    def crop_area(
        self, rotation: OrthogonalRotation, oriented_size: tuple[int, int]
    ) -> NDArray[np.float64] | None:
        """Return the trimmed region as a polygon of the oriented image.

        Args:
            rotation (OrthogonalRotation): Rotation Fix Orientation applies.
            oriented_size (tuple[int, int]): Width and height in pixels of the
                image after the rotation.

        Returns:
            NDArray[np.float64] | None: Four corners, shape (4, 2), in oriented
            pixels; None when the trim is disabled.
        """
        if not self.enabled:
            return None
        oriented_width, oriented_height = oriented_size
        width, height = rotation.unrotate_dimensions(oriented_width, oriented_height)
        x, y, inner_width, inner_height = self.to_inner_rect(
            round(width), round(height)
        )
        xform = ImageTransformation.from_size(width, height)
        xform.set_pre_rotation(rotation)
        return xform.orig_rect_to_pre_crop_space(
            Rect(x=x, y=y, width=inner_width, height=inner_height)
        )


def _redistribute_excess(leading: int, trailing: int, excess: int) -> tuple[int, int]:
    """Take ``excess`` pixels back from a pair of opposite margins."""
    from_leading = min(leading, (excess + 1) // 2)
    leading -= from_leading
    trailing -= excess - from_leading
    return max(0, leading), max(0, trailing)


def merge_with_trim(
    page_outline: NDArray[np.float64], trim_area: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Restrict a page outline to the trimmed region.

    Args:
        page_outline (NDArray[np.float64]): Convex polygon of the page, shape (N, 2).
        trim_area (NDArray[np.float64] | None): Convex polygon of the trimmed
            region in the same coordinates, or None when nothing is trimmed.

    Returns:
        NDArray[np.float64]: The intersection of both; the page outline itself
        when there is no trim or the two do not overlap.
    """
    if trim_area is None:
        return page_outline
    clipped = _clip_convex(page_outline, trim_area)
    if len(clipped) < 3 or abs(_polygon_area(clipped)) < _MIN_OVERLAP_AREA:
        return page_outline
    return clipped


def _polygon_area(polygon: NDArray[np.float64]) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_convex(
    subject: NDArray[np.float64], clip: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sutherland-Hodgman clipping of a polygon by a convex polygon."""
    sign = 1.0 if _polygon_area(clip) > 0 else -1.0
    points = [tuple(point) for point in subject]
    for index in range(len(clip)):
        edge_start = clip[index]
        edge_end = clip[(index + 1) % len(clip)]

        def side(point: tuple[float, float]) -> float:
            return sign * (
                (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
                - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
            )

        clipped: list[tuple[float, float]] = []
        for current, following in zip(points, points[1:] + points[:1], strict=True):
            current_side, following_side = side(current), side(following)
            if current_side >= 0:
                clipped.append(current)
            if (current_side >= 0) != (following_side >= 0):
                t = current_side / (current_side - following_side)
                clipped.append(
                    (
                        current[0] + t * (following[0] - current[0]),
                        current[1] + t * (following[1] - current[1]),
                    )
                )
        points = clipped
        if not points:
            break
    return np.array(points, dtype=np.float64).reshape(-1, 2)
