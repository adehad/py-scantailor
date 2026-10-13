"""Area-averaging downscale of gray images with the upstream fixed-point rules.

Source positions are tracked in 1/32 pixel units and every destination pixel is
the integer-rounded weighted mean of the source pixels its window overlaps. The
two axes are independent, so each axis becomes a sparse matrix of integer
weights and the image is multiplied by both.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

_UNIT = 32
_UNIT_SHIFT = 5

Axis = tuple[sparse.csr_array, NDArray[np.int64]]


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
    """Multiply ``src`` by the two axis matrices and divide by the window areas."""
    x_matrix, x_area = x_axis
    y_matrix, y_area = y_axis
    rows = y_matrix @ src.astype(np.int64)
    sums = (x_matrix @ rows.T).T
    area = np.outer(y_area, x_area)
    inside = area > 0
    out = np.zeros(sums.shape, dtype=np.uint8)
    safe_area = np.where(inside, area, 1)
    out[inside] = ((sums + (safe_area >> 1)) // safe_area)[inside]
    return out


def _weight_matrix(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    weights: NDArray[np.int64],
    shape: tuple[int, int],
) -> sparse.csr_array:
    """Build a sparse matrix holding ``weights`` at ``(rows, cols)``."""
    order = np.argsort(rows, kind="stable")
    row_ends = np.cumsum(np.bincount(rows, minlength=shape[0]))
    indptr = np.concatenate([[0], row_ends])
    return sparse.csr_array((weights[order], cols[order], indptr), shape=shape)


def _overlap_axis(src_len: int, lo: NDArray[np.int64], hi: NDArray[np.int64]) -> Axis:
    """Weights of the source pixels under windows [lo, hi) given in 1/32 pixels.

    Windows are clipped to the image; a window left empty by the clip gets no
    weights and an area of 0.
    """
    lo = np.clip(lo, 0, src_len * _UNIT)
    hi = np.clip(hi, 0, src_len * _UNIT)
    valid = hi > lo
    first = lo >> _UNIT_SHIFT
    last = (hi - 1) >> _UNIT_SHIFT
    taps = int((last - first)[valid].max()) + 1 if valid.any() else 1

    rows, cols, weights = [], [], []
    index = np.arange(len(lo))
    for k in range(taps):
        pixel = first + k
        member = valid & (pixel <= last)
        left = np.maximum(lo, pixel * _UNIT)
        right = np.minimum(hi, (pixel + 1) * _UNIT)
        rows.append(index[member])
        cols.append(pixel[member])
        weights.append((right - left)[member])
    matrix = _weight_matrix(
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(weights),
        (len(lo), src_len),
    )
    return matrix, np.where(valid, hi - lo, 0)


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
    index = np.arange(dst_len)
    matrix = _weight_matrix(
        np.concatenate([index, index]),
        np.concatenate([pixel, pixel + 1]),
        np.concatenate([_UNIT - right, right]),
        (dst_len, src_len),
    )
    return matrix, np.full(dst_len, _UNIT, dtype=np.int64)
