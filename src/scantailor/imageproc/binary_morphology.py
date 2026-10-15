"""Morphology and region growing on black-and-white images (True is black)."""

import cv2
import numpy as np
from numpy.typing import NDArray


def open_brick(
    black: NDArray[np.bool_],
    width: int,
    height: int,
    *,
    black_surroundings: bool = False,
) -> NDArray[np.bool_]:
    """Keep the black pixels covered by a ``width`` x ``height`` black rectangle.

    Args:
        black (NDArray[np.bool_]): (H, W) image, True for black.
        width (int): Rectangle width in pixels.
        height (int): Rectangle height in pixels.
        black_surroundings (bool): Treat everything outside the image as black,
            so a rectangle may extend past the border. By default the outside
            is white and the rectangle must fit inside the image.

    Returns:
        NDArray[np.bool_]: New image of the same size.
    """
    kernel = np.ones((height, width), dtype=np.uint8)
    erode_anchor = (width >> 1, height >> 1)
    # Opening does not depend on where the rectangle's origin sits, provided the
    # dilation uses the mirrored origin.
    dilate_anchor = (width - 1 - erode_anchor[0], height - 1 - erode_anchor[1])
    fill = 1 if black_surroundings else 0
    margin_x = 2 * width if black_surroundings else 0
    margin_y = 2 * height if black_surroundings else 0

    work = black.astype(np.uint8)
    if black_surroundings:
        work = cv2.copyMakeBorder(
            work,
            margin_y,
            margin_y,
            margin_x,
            margin_x,
            cv2.BORDER_CONSTANT,
            value=fill,
        )
    eroded = cv2.erode(
        work,
        kernel,
        anchor=erode_anchor,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    opened = cv2.dilate(
        eroded,
        kernel,
        anchor=dilate_anchor,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    if black_surroundings:
        h, w = black.shape
        opened = opened[margin_y : margin_y + h, margin_x : margin_x + w]
    return opened.astype(np.bool_)


def seed_fill(seed: NDArray[np.bool_], mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return the 8-connected components of ``mask`` that contain a ``seed`` pixel.

    Args:
        seed (NDArray[np.bool_]): Pixels to grow from; any outside ``mask`` are ignored.
        mask (NDArray[np.bool_]): Region the result is confined to.

    Returns:
        NDArray[np.bool_]: New image, a subset of ``mask``.
    """
    if mask.size == 0:
        return np.zeros(mask.shape, dtype=np.bool_)
    count, labels = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    hit = np.zeros(count, dtype=np.bool_)
    hit[labels[seed & mask]] = True
    hit[0] = False
    return hit[labels]
