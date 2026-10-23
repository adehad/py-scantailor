"""Drawing helpers for the stages' debug images.

They draw what upstream's ``visualize*`` functions draw, with OpenCV on BGR
copies. Only debug mode calls them, so none of this runs on a normal render.
Colours are BGR tuples, and points are ``(x, y)`` in pixels of the canvas.
"""

from collections.abc import Iterable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

Color = tuple[int, int, int]
Point = tuple[float, float]
Line = tuple[Point, Point]

BLUE: Color = (255, 0, 0)
GREEN: Color = (0, 255, 0)
RED: Color = (0, 0, 255)
MAGENTA: Color = (255, 0, 255)
YELLOW: Color = (0, 255, 255)

# Fixed-point bits for anti-aliased drawing at sub-pixel positions.
_SHIFT = 4
_SCALE = 1 << _SHIFT


def to_bgr(image: NDArray) -> NDArray[np.uint8]:
    """Return a new BGR canvas of a gray, BGR or bool image.

    Args:
        image (NDArray): ``(H, W)`` gray, ``(H, W, 3)`` BGR, or a bool mask
            drawn black where True on white.

    Returns:
        NDArray[np.uint8]: ``(H, W, 3)`` copy the caller may draw on.
    """
    if image.dtype == np.bool_:
        image = np.where(image, np.uint8(0), np.uint8(255))
    if image.ndim == 2:
        return np.asarray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    return np.array(image[:, :, :3], dtype=np.uint8)


def blend(base: NDArray[np.uint8], drawn: NDArray[np.uint8], opacity: float) -> None:
    """Mix ``drawn`` over ``base`` in place, as QPainter's opacity does."""
    cv2.addWeighted(drawn, opacity, base, 1.0 - opacity, 0.0, dst=base)


def _fixed(points: NDArray[np.float64]) -> NDArray[np.int32]:
    return np.round(np.asarray(points, dtype=np.float64) * _SCALE).astype(np.int32)


def _thickness(width: float) -> int:
    return max(1, round(width))


def draw_polylines(
    canvas: NDArray[np.uint8],
    polylines: Iterable[NDArray[np.float64] | Sequence[tuple[float, float]]],
    color: Color,
    width: float,
) -> None:
    """Draw open polylines in place; ones with fewer than two points are skipped."""
    shapes = [_fixed(np.asarray(p, dtype=np.float64).reshape(-1, 2)) for p in polylines]
    shapes = [shape for shape in shapes if len(shape) >= 2]
    if shapes:
        cv2.polylines(
            canvas, shapes, False, color, _thickness(width), cv2.LINE_AA, _SHIFT
        )


def draw_lines(
    canvas: NDArray[np.uint8], lines: Iterable[Line], color: Color, width: float
) -> None:
    """Draw line segments in place, from their first to their second point."""
    draw_polylines(
        canvas, [np.array(line, dtype=np.float64) for line in lines], color, width
    )


def draw_knots(
    canvas: NDArray[np.uint8],
    points: Iterable[Sequence[float]] | NDArray[np.float64],
    color: Color,
    diameter: float,
) -> None:
    """Draw filled dots of ``diameter`` pixels centred on ``points``, in place."""
    radius = max(1, round(diameter * 0.5 * _SCALE))
    for x, y in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        center = (round(x * _SCALE), round(y * _SCALE))
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA, _SHIFT)


def signed_overlay(
    background: NDArray[np.uint8] | None,
    values: NDArray[np.floating],
    positive: Color = RED,
    negative: Color = BLUE,
) -> NDArray[np.uint8]:
    """Tint a canvas by signed values, as upstream's ``visualizeGradient`` does.

    The values are scaled so the largest magnitude reaches 255; a pixel's tint
    has that magnitude as both its colour and its opacity.

    Args:
        background (NDArray[np.uint8] | None): Gray or BGR image under the
            tint; None paints on white.
        values (NDArray[np.floating]): ``(H, W)`` values, same size as the canvas.
        positive (Color): Tint of positive values.
        negative (Color): Tint of zero and negative values.

    Returns:
        NDArray[np.uint8]: New ``(H, W, 3)`` canvas.
    """
    values = np.asarray(values, dtype=np.float32)
    if background is None:
        canvas = np.full((*values.shape, 3), 255, dtype=np.float32)
    else:
        canvas = to_bgr(background).astype(np.float32)
    largest = float(np.abs(values).max(initial=0.0))
    scale = 255.0 / largest if largest > np.finfo(np.float32).eps else 1.0
    magnitude = np.clip(np.round(np.abs(values) * scale), 0, 255)
    alpha = (magnitude / 255.0)[:, :, None]
    tint = np.where(
        (values > 0)[:, :, None],
        np.array(positive, dtype=np.float32),
        np.array(negative, dtype=np.float32),
    )
    mixed = tint * alpha + canvas * (1.0 - alpha)
    return np.clip(np.round(mixed), 0, 255).astype(np.uint8)


def tint(
    canvas: NDArray[np.uint8],
    where: NDArray[np.bool_] | tuple[slice, slice],
    color: Color,
    alpha: float,
) -> None:
    """Blend ``color`` at ``alpha`` opacity over part of a canvas, in place.

    Args:
        canvas (NDArray[np.uint8]): ``(H, W, 3)`` BGR canvas.
        where (NDArray[np.bool_] | tuple[slice, slice]): A mask of the canvas,
            or the rows and columns of a rectangle.
        color (Color): The tint.
        alpha (float): Its opacity, 0 to 1.
    """
    part = canvas[where].astype(np.float32)
    mixed = part * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    canvas[where] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
