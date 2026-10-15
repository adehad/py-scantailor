"""Area-averaging downscale of gray images with the upstream fixed-point rules.

Source positions are tracked in 1/32 pixel units and every destination pixel is
the integer-rounded weighted mean of the source pixels its window overlaps. The
two axes are independent, so each axis becomes a few taps per destination
pixel, a source index and an integer weight each, applied to rows then columns.
"""

import numpy as np
from numpy.typing import NDArray

_UNIT = 32
_UNIT_SHIFT = 5

# Per axis: source index and weight of every tap, both (taps, destination
# length), and the total weight of every destination pixel. Taps a pixel does
# not use have weight 0.
Axis = tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]


def transform_scale_to_gray(
    src: NDArray[np.uint8],
    scale: float,
    dst_size: tuple[int, int],
    min_window: float = 5.0,
) -> NDArray[np.uint8]:
    """Scale a gray image by averaging windows of at least ``min_window`` pixels.

    A destination pixel averages a box centred on where its centre maps in the
    source. The box is as wide as the pixel's footprint but never narrower than
    ``min_window`` source pixels, so a mild downscale still smooths. Parts of
    the box outside the image do not count.

    Args:
        src (NDArray[np.uint8]): (H, W) gray image.
        scale (float): Destination pixels per source pixel, the same on both axes.
        dst_size (tuple[int, int]): (width, height) of the result.
        min_window (float): Smallest box edge in source pixels.

    Returns:
        NDArray[np.uint8]: (height, width) gray image; a pixel whose box misses the
        source entirely is 0.
    """
    src_h, src_w = src.shape
    dst_w, dst_h = dst_size
    step32 = _UNIT / scale
    window = max(1, _q_round(max(min_window * _UNIT, step32)))
    return _apply(
        src,
        _window_axis(src_w, dst_w, step32, window),
        _window_axis(src_h, dst_h, step32, window),
    )


def scale_to_gray(
    src: NDArray[np.uint8], dst_size: tuple[int, int]
) -> NDArray[np.uint8]:
    """Resize a gray image the way upstream's ``scaleToGray`` does.

    Whole-number reductions average exact blocks, whole-number enlargements
    repeat pixels, other enlargements interpolate bilinearly and everything
    else averages the overlapped source area.

    Args:
        src (NDArray[np.uint8]): (H, W) gray image.
        dst_size (tuple[int, int]): (width, height) of the result, both positive.

    Returns:
        NDArray[np.uint8]: (height, width) gray image.
    """
    src_h, src_w = src.shape
    dst_w, dst_h = dst_size
    if (src_w, src_h) == (dst_w, dst_h):
        return src
    if src_w % dst_w == 0 and src_h % dst_h == 0:
        return _apply(src, _block_axis(src_w, dst_w), _block_axis(src_h, dst_h))
    if dst_w % src_w == 0 and dst_h % src_h == 0:
        return np.repeat(np.repeat(src, dst_h // src_h, axis=0), dst_w // src_w, axis=1)
    if dst_w > src_w and dst_h > src_h:
        return _apply(src, _bilinear_axis(src_w, dst_w), _bilinear_axis(src_h, dst_h))
    return _apply(src, _area_axis(src_w, dst_w), _area_axis(src_h, dst_h))


def _q_round(value: float) -> int:
    """Round half away from zero like Qt's ``qRound``."""
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def _apply(src: NDArray[np.uint8], x_axis: Axis, y_axis: Axis) -> NDArray[np.uint8]:
    """Weight and sum ``src`` along both axes and divide by the window areas."""
    x_sources, x_weights, x_area = x_axis
    y_sources, y_weights, y_area = y_axis
    # Integer sums keep the result exact; int32 is about twice as fast as int64
    # and suffices unless the windows are very large.
    largest_sum = 255 * int(x_area.max(initial=0)) * int(y_area.max(initial=0))
    total = np.int32 if largest_sum <= np.iinfo(np.int32).max else np.int64
    rows = _weighted_rows(src, y_sources, y_weights, total)
    sums = _weighted_rows(np.ascontiguousarray(rows.T), x_sources, x_weights, total)
    sums = sums.T.astype(np.int64)
    area = np.outer(y_area, x_area)
    inside = area > 0
    out = np.zeros(sums.shape, dtype=np.uint8)
    safe_area = np.where(inside, area, 1)
    out[inside] = ((sums + (safe_area >> 1)) // safe_area)[inside]
    return out


def _weighted_rows(
    image: NDArray,
    sources: NDArray[np.int64],
    weights: NDArray[np.int64],
    total: type[np.signedinteger],
) -> NDArray:
    """Sum the rows of ``image`` that every destination row takes, by weight.

    Args:
        image (NDArray): (H, W) integer image.
        sources (NDArray[np.int64]): (taps, destination rows) source row indices.
        weights (NDArray[np.int64]): (taps, destination rows) weights.
        total (type[np.signedinteger]): Integer type wide enough for the sums.

    Returns:
        NDArray: (destination rows, W) weighted sums of type ``total``.
    """
    sums = np.zeros((sources.shape[1], image.shape[1]), dtype=total)
    term = np.empty_like(sums)
    for source, weight in zip(sources, weights.astype(total), strict=True):
        np.multiply(image[source], weight[:, None], out=term)
        sums += term
    return sums


def _overlap_axis(src_len: int, lo: NDArray[np.int64], hi: NDArray[np.int64]) -> Axis:
    """Taps of the source pixels under windows [lo, hi) given in 1/32 pixels.

    Windows are clipped to the image; a window left empty by the clip gets no
    weights and an area of 0.
    """
    lo = np.clip(lo, 0, src_len * _UNIT)
    hi = np.clip(hi, 0, src_len * _UNIT)
    valid = hi > lo
    first = lo >> _UNIT_SHIFT
    last = (hi - 1) >> _UNIT_SHIFT
    taps = int((last - first)[valid].max()) + 1 if valid.any() else 1

    pixel = first + np.arange(taps)[:, None]
    member = valid & (pixel <= last)
    left = np.maximum(lo, pixel * _UNIT)
    right = np.minimum(hi, (pixel + 1) * _UNIT)
    sources = np.where(member, pixel, 0)
    weights = np.where(member, right - left, 0)
    return sources, weights, np.where(valid, hi - lo, 0)


def _window_axis(src_len: int, dst_len: int, step32: float, window: int) -> Axis:
    centres = ((np.arange(dst_len) + 0.5) * step32).astype(np.int64)
    lo = centres - (window >> 1)
    return _overlap_axis(src_len, lo, lo + window)


def _block_axis(src_len: int, dst_len: int) -> Axis:
    factor = (src_len // dst_len) * _UNIT
    lo = np.arange(dst_len, dtype=np.int64) * factor
    return _overlap_axis(src_len, lo, lo + factor)


def _ratio_to_fit(dst: int, src: int, upscaling: bool) -> float:
    """Source units (1/32 pixel) per destination pixel, kept inside the source."""
    src32 = src << _UNIT_SHIFT
    ratio = src32 / dst
    if upscaling:
        while (int(ratio * (dst - 1)) >> _UNIT_SHIFT) + 1 >= src:
            src32 -= 1
            ratio = src32 / dst
    else:
        while (int(ratio * dst) - 1) >> _UNIT_SHIFT >= src:
            src32 -= 1
            ratio = src32 / dst
    return ratio


def _area_axis(src_len: int, dst_len: int) -> Axis:
    ratio = _ratio_to_fit(dst_len, src_len, upscaling=False)
    edges = (np.arange(dst_len + 1) * ratio).astype(np.int64)
    return _overlap_axis(src_len, edges[:-1], edges[1:])


def _bilinear_axis(src_len: int, dst_len: int) -> Axis:
    ratio = _ratio_to_fit(dst_len, src_len, upscaling=True)
    pos32 = (np.arange(dst_len) * ratio).astype(np.int64)
    pixel = pos32 >> _UNIT_SHIFT
    right = pos32 & (_UNIT - 1)
    sources = np.stack([pixel, pixel + 1])
    weights = np.stack([_UNIT - right, right])
    return sources, weights, np.full(dst_len, _UNIT, dtype=np.int64)
