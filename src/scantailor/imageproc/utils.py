"""Image processing utilities.

Provides utility functions for drawing, polygon operations,
and color manipulation.
"""

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def fill_polygon(
    image: NDArray[np.uint8],
    points: NDArray | list[tuple[int, int]],
    color: int | tuple[int, int, int] = 255,
) -> NDArray[np.uint8]:
    """Fill a polygon on an image.

    Args:
        image: Image to draw on (modified in place).
        points: Polygon vertices as Nx2 array or list of (x, y) tuples.
        color: Fill color (grayscale value or BGR tuple).

    Returns:
        The modified image.
    """
    pts = np.asarray(points, dtype=np.int32)
    if pts.ndim == 2:
        pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], color)
    return image


def draw_polygon(
    image: NDArray[np.uint8],
    points: NDArray | list[tuple[int, int]],
    color: int | tuple[int, int, int] = 255,
    thickness: int = 1,
    closed: bool = True,
) -> NDArray[np.uint8]:
    """Draw a polygon outline on an image.

    Args:
        image: Image to draw on (modified in place).
        points: Polygon vertices as Nx2 array or list of (x, y) tuples.
        color: Line color (grayscale value or BGR tuple).
        thickness: Line thickness in pixels.
        closed: Whether to close the polygon.

    Returns:
        The modified image.
    """
    pts = np.asarray(points, dtype=np.int32)
    if pts.ndim == 2:
        pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], closed, color, thickness)
    return image


def contour_area(contour: NDArray[np.int32]) -> float:
    """Compute the area of a contour.

    Args:
        contour: Contour points as Nx1x2 or Nx2 array.

    Returns:
        Signed area (positive for counter-clockwise, negative for clockwise).
    """
    return cv2.contourArea(contour)


def contour_perimeter(contour: NDArray[np.int32], closed: bool = True) -> float:
    """Compute the perimeter (arc length) of a contour.

    Args:
        contour: Contour points as Nx1x2 or Nx2 array.
        closed: Whether to treat the contour as closed.

    Returns:
        Perimeter length.
    """
    return cv2.arcLength(contour, closed)


def bounding_rect(contour: NDArray[np.int32]) -> tuple[int, int, int, int]:
    """Compute the axis-aligned bounding rectangle of a contour.

    Args:
        contour: Contour points as Nx1x2 or Nx2 array.

    Returns:
        Tuple (x, y, width, height).
    """
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, w, h)


def min_area_rect(
    contour: NDArray[np.int32],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Compute the minimum area rotated rectangle enclosing a contour.

    Args:
        contour: Contour points as Nx1x2 or Nx2 array.

    Returns:
        Tuple ((center_x, center_y), (width, height), angle).
    """
    center, size, angle = cv2.minAreaRect(contour)
    return (
        (float(center[0]), float(center[1])),
        (float(size[0]), float(size[1])),
        float(angle),
    )


def convex_hull(contour: NDArray[np.int32]) -> NDArray[np.int32]:
    """Compute the convex hull of a contour.

    Args:
        contour: Contour points as Nx1x2 or Nx2 array.

    Returns:
        Convex hull points.
    """
    return np.asarray(cv2.convexHull(contour), dtype=np.int32)


def draw_over(
    background: NDArray[np.uint8],
    foreground: NDArray[np.uint8],
    x: int,
    y: int,
) -> NDArray[np.uint8]:
    """Draw one image over another at a specified position.

    Args:
        background: Background image (modified in place).
        foreground: Foreground image to draw.
        x: X coordinate for top-left corner of foreground.
        y: Y coordinate for top-left corner of foreground.

    Returns:
        The modified background image.
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    # Calculate visible region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + fg_w)
    y2 = min(bg_h, y + fg_h)

    if x1 >= x2 or y1 >= y2:
        return background  # No overlap

    # Calculate foreground region to copy
    fx1 = x1 - x
    fy1 = y1 - y
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    background[y1:y2, x1:x2] = foreground[fy1:fy2, fx1:fx2]
    return background


def blend(
    image1: NDArray[np.uint8],
    image2: NDArray[np.uint8],
    alpha: float,
) -> NDArray[np.uint8]:
    """Blend two images together.

    Result = alpha * image1 + (1 - alpha) * image2

    Args:
        image1: First image.
        image2: Second image (must be same size as image1).
        alpha: Blend factor (0.0 = all image2, 1.0 = all image1).

    Returns:
        Blended image.
    """
    result = cv2.addWeighted(image1, alpha, image2, 1.0 - alpha, 0)
    return np.asarray(result, dtype=np.uint8)


def color_interpolate(
    color1: tuple[int, int, int],
    color2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Interpolate between two colors.

    Args:
        color1: First color (BGR or RGB tuple).
        color2: Second color (BGR or RGB tuple).
        t: Interpolation factor (0.0 = color1, 1.0 = color2).

    Returns:
        Interpolated color.
    """
    t = max(0.0, min(1.0, t))
    r = int(color1[0] * (1 - t) + color2[0] * t)
    g = int(color1[1] * (1 - t) + color2[1] * t)
    b = int(color1[2] * (1 - t) + color2[2] * t)
    return (r, g, b)


def to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert an image to grayscale.

    Args:
        image: Input image (grayscale or BGR color).

    Returns:
        Grayscale image.
    """
    if len(image.shape) == 2:
        return image.copy()
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def to_color(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a grayscale image to BGR color.

    Args:
        image: Input grayscale image.

    Returns:
        BGR color image.
    """
    if len(image.shape) == 3:
        return image.copy()
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
