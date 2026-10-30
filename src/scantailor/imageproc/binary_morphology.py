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


# The replaced cell and its two nearest neighbours already narrow a page down to
# pixels on an edge, under 1 % of it, so only they are tested across the whole
# image and the other cells at the placements left. More or fewer is slower.
_WHOLE_IMAGE_CELLS = 3


def hit_miss_replace(
    black: NDArray[np.bool_], pattern: tuple[str, ...]
) -> NDArray[np.bool_]:
    """Recolour the pixels a hit-miss pattern marks, wherever it matches.

    Each row of ``pattern`` is a string of cells: ``X`` must be black, a space
    must be white and ``?`` may be either. ``-`` must be black and turns white
    on a match; ``+`` must be white and turns black. A match may hang over the
    edge of the image, where everything counts as white. All matches are found
    before any pixel changes.

    Args:
        black (NDArray[np.bool_]): (H, W) image, True for black. Not modified.
        pattern (tuple[str, ...]): Rows of the pattern, top to bottom, of equal length.

    Returns:
        NDArray[np.bool_]: New image of the same size.
    """
    rows, cols = len(pattern), len(pattern[0])
    cells = [
        (y, x, cell)
        for y, row in enumerate(pattern)
        for x, cell in enumerate(row)
        if cell != "?"
    ]
    replaced = [(y, x, cell) for y, x, cell in cells if cell in "+-"]
    if not replaced:
        return black.copy()
    first_y, first_x, _ = replaced[0]
    cells.sort(key=lambda c: abs(c[0] - first_y) + abs(c[1] - first_x))

    h, w = black.shape
    padded = np.zeros((h + 2 * rows, w + 2 * cols), dtype=np.bool_)
    padded[rows : rows + h, cols : cols + w] = black
    # A placement is known by the pixel under its top-left cell; the first
    # ``span`` of ``padded`` holds every placement that overlaps the image.
    span = (h + rows, w + cols)
    fits = np.ones(span, dtype=np.bool_)
    for y, x, cell in cells[:_WHOLE_IMAGE_CELLS]:
        under = padded[y : y + span[0], x : x + span[1]]
        fits &= under if cell in "X-" else ~under

    stride = padded.shape[1]
    top, left = np.divmod(np.flatnonzero(fits), span[1])
    placements = top * stride + left
    flat = padded.ravel()
    for y, x, cell in cells[_WHOLE_IMAGE_CELLS:]:
        under = flat[placements + y * stride + x]
        placements = placements[under if cell in "X-" else ~under]

    # Upstream turns pixels black before it turns any white.
    for y, x, cell in sorted(replaced, key=lambda c: c[2] != "+"):
        flat[placements + y * stride + x] = cell == "+"
    return padded[rows : rows + h, cols : cols + w]


def hit_miss_replace_all_directions(
    black: NDArray[np.bool_], pattern: tuple[str, ...]
) -> NDArray[np.bool_]:
    """Apply ``hit_miss_replace`` with ``pattern`` as given and turned three times.

    The turns are clockwise, upside down and counter-clockwise, in that order,
    each on the result of the one before. The order matters where two matches
    share pixels.

    Args:
        black (NDArray[np.bool_]): (H, W) image, True for black. Not modified.
        pattern (tuple[str, ...]): Rows of the pattern, top to bottom, of equal length.

    Returns:
        NDArray[np.bool_]: New image of the same size.
    """
    grid = np.array([list(row) for row in pattern])
    for quarter_turns in (0, -1, 2, 1):
        turned = np.rot90(grid, quarter_turns)
        black = hit_miss_replace(black, tuple("".join(row) for row in turned))
    return black
