"""Colour reduction: split RGB space into a grid and keep one colour per cell."""

import numpy as np
from numpy.typing import NDArray

_MIN_LEVEL = 2
_MAX_LEVEL = 255
# A level is kept when at least this share of the pixels has it in some channel,
# which discards sensor noise at the ends of the range. Must stay below 1/256.
_NOISE_SHARE = 0.0005
# Saturation-times-value below this (scaled by how strong the colour is) marks a
# colour as gray.
_GRAY_LIMIT = 0.1
_GRAY_KNEE = 0.28
_BLACK_LIGHTNESS = 0.5
_WHITE_LIGHTNESS = 0.8


def posterize(
    image: NDArray[np.uint8],
    level: int,
    *,
    normalize: bool = False,
    force_black_and_white: bool = False,
    normalize_black_level: int = 0,
    normalize_white_level: int = 255,
) -> NDArray[np.uint8]:
    """Reduce an image to a few colours.

    Every channel range is cut into ``level`` equal cells, so RGB space becomes
    a grid. The most frequent colour of each occupied cell replaces all colours
    in it. A channel value of 255 lands in a cell of its own, so one channel can
    take up to ``level + 1`` distinct values.

    Args:
        image (NDArray[np.uint8]): 2-D gray or (H, W, 3|4) BGR(A) image. Not
            modified; an alpha channel is passed through.
        level (int): Cells per channel, from 2 to 255. 255 with no other option
            returns an unchanged copy.
        normalize (bool): Stretch the colour range to the full range before
            grouping, and write the stretched colours. Pure black and pure white
            stay as they are.
        force_black_and_white (bool): Turn grayish representative colours that are
            dark into black and those that are light into white.
        normalize_black_level (int): Level counted for pure black pixels when
            finding the range to stretch.
        normalize_white_level (int): Level counted for pure white pixels when
            finding the range to stretch; usually the page background level.

    Returns:
        NDArray[np.uint8]: New image of the same shape.

    Raises:
        ValueError: If ``level`` is outside 2..255.
    """
    if not _MIN_LEVEL <= level <= _MAX_LEVEL:
        msg = f"level must be between {_MIN_LEVEL} and {_MAX_LEVEL}, got {level}"
        raise ValueError(msg)
    if level == _MAX_LEVEL and not normalize and not force_black_and_white:
        return image.copy()

    rgb = _as_rgb(image)
    packed = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    )
    colors, inverse, counts = np.unique(
        packed.ravel(), return_inverse=True, return_counts=True
    )
    palette = np.stack(
        [(colors >> 16) & 0xFF, (colors >> 8) & 0xFF, colors & 0xFF], axis=1
    ).astype(np.int64)

    normalized = _normalize_palette(
        palette,
        counts,
        packed.size,
        normalize_black_level,
        normalize_white_level,
    )
    cells = (normalized / (255.0 / level)).astype(np.int64)
    cell_id = (cells[:, 0] << 16) | (cells[:, 1] << 8) | cells[:, 2]

    # Within each cell the most frequent colour wins; ties go to the lowest colour.
    _, cell = np.unique(cell_id, return_inverse=True)
    order = np.lexsort((np.arange(len(counts)), -counts, cell))
    sorted_cell = cell[order]
    first_in_cell = np.ones(len(order), dtype=bool)
    first_in_cell[1:] = sorted_cell[1:] != sorted_cell[:-1]
    winner = order[first_in_cell][cell]

    new_palette = normalized[winner] if normalize else palette[winner]
    if force_black_and_white:
        gray_winner = _is_gray(normalized)[winner]
        lightness = _gray_lightness(normalized[winner])
        to_black = gray_winner & (lightness <= _BLACK_LIGHTNESS)
        to_white = gray_winner & (lightness >= _WHITE_LIGHTNESS) & ~to_black
        new_palette = new_palette.copy()
        new_palette[to_black] = 0
        new_palette[to_white] = 255

    result_rgb = new_palette.astype(np.uint8)[inverse.reshape(packed.shape)]
    return _from_rgb(image, result_rgb)


def _as_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return an (H, W, 3) RGB view or copy of a gray or BGR(A) image."""
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    return image[..., 2::-1]


def _from_rgb(image: NDArray[np.uint8], rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Write RGB values back into the layout of ``image``."""
    if image.ndim == 2:
        return np.ascontiguousarray(rgb[..., 0])
    result = image.copy()
    result[..., :3] = rgb[..., ::-1]
    return result


def _normalize_palette(
    palette: NDArray[np.int64],
    counts: NDArray[np.int64],
    pixel_count: int,
    black_level: int,
    white_level: int,
) -> NDArray[np.int64]:
    """Stretch the channel levels that carry at least a noise share of pixels.

    Pure black and pure white are excluded from the stretch and kept as they are,
    but they count towards the range at the levels given for them.
    """
    is_black = ~palette.any(axis=1)
    is_white = (palette == 255).all(axis=1)
    special = is_black | is_white

    histogram = np.zeros((3, 256), dtype=np.float64)
    for channel in range(3):
        histogram[channel] = np.bincount(
            palette[~special, channel], weights=counts[~special], minlength=256
        )
    histogram[:, black_level] += counts[is_black].sum()
    histogram[:, white_level] += counts[is_white].sum()
    occupied = np.flatnonzero((histogram / pixel_count >= _NOISE_SHARE).any(axis=0))
    low, high = int(occupied[0]), int(occupied[-1])

    if high <= low:
        return palette.copy()
    stretched = np.floor((palette - low) / (high - low) * 255 + 0.5).astype(np.int64)
    stretched = np.clip(stretched, 0, 255)
    stretched[special] = palette[special]
    return stretched


def _hsv_saturation_value(rgb: NDArray[np.int64]) -> tuple[NDArray, NDArray]:
    """HSV saturation and value, both from 0 to 1."""
    peak = rgb.max(axis=1).astype(np.float64)
    low = rgb.min(axis=1).astype(np.float64)
    saturation = np.divide(peak - low, peak, out=np.zeros_like(peak), where=peak > 0)
    return saturation, peak / 255.0


def _is_gray(rgb: NDArray[np.int64]) -> NDArray[np.bool_]:
    """True for colours with too little saturation or brightness to read as a hue."""
    saturation, value = _hsv_saturation_value(rgb)
    strength = np.maximum(
        0.0, (np.maximum(saturation, value) - _GRAY_KNEE) / (1 - _GRAY_KNEE)
    )
    return saturation * value < _GRAY_LIMIT * (strength + 1)


def _gray_lightness(rgb: NDArray[np.int64]) -> NDArray[np.float64]:
    """HSL lightness of the gray level of each colour (11/16/5 luma), 0 to 1."""
    luma = (11 * rgb[:, 0] + 16 * rgb[:, 1] + 5 * rgb[:, 2]) // 32
    return luma / 255.0
