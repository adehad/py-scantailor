"""Generic region encoding with arithmetic coding (T.88 6.2, MMR = 0).

Only GBTEMPLATE 0 with the nominal adaptive-template pixels is produced. With
those AT positions the 16-pixel context is a solid stencil, which lets the
context be maintained with three shift registers instead of 16 lookups:

    row y-2:  x-2 .. x+2   (5 bits, A4 and A3 at the ends)
    row y-1:  x-3 .. x+3   (7 bits, A2 and A1 at the ends)
    row y:    x-4 .. x-1   (4 bits)

concatenated most significant first in raster order, which is the bit order
of CONTEXT in T.88 6.2.5.3 and the order the TPGDON context 0x9B25 assumes.
"""

import numpy as np
from numba import njit

from jbig2_numba.mq_coder import STATE_SIZE, mq_encode, mq_flush, mq_init, output_capacity

# T.88 6.2.5.3, nominal AT pixels for GBTEMPLATE 0 as (dx, dy): A1, A2, A3, A4.
TEMPLATE0_AT_PIXELS: tuple[tuple[int, int], ...] = ((3, -1), (-3, -1), (2, -2), (-2, -2))

# T.88 6.2.5.7, context used to code SLTP for GBTEMPLATE 0.
_SLTP_CONTEXT_TEMPLATE0 = 0x9B25

_PAD_TOP = 2
_PAD_LEFT = 4
_PAD_RIGHT = 3


def as_bilevel(image: np.ndarray) -> np.ndarray:
    """Validate an input image and return its black mask.

    Args:
        image (np.ndarray): 2-D ``bool`` or ``uint8`` array; ``True`` or nonzero is black.

    Returns:
        np.ndarray: New ``bool`` array, ``True`` where the pixel is black.

    Raises:
        TypeError: If ``image`` is not a NumPy array of dtype ``bool`` or ``uint8``.
        ValueError: If ``image`` is not 2-D or has a zero-length axis.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype not in (np.bool_, np.uint8):
        raise TypeError(f"image dtype must be bool or uint8, got {image.dtype}")
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got {image.ndim} dimensions")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty, got shape {image.shape}")
    if image.shape[0] >= 2**32 or image.shape[1] >= 2**32:
        raise ValueError("image dimensions must fit in 32 bits")
    return image != 0


@njit(cache=True)
def _encode_template0(padded: np.ndarray, typical_rows: np.ndarray, tpgdon: bool) -> np.ndarray:
    height = padded.shape[0] - _PAD_TOP
    width = padded.shape[1] - _PAD_LEFT - _PAD_RIGHT
    state = np.empty(STATE_SIZE, dtype=np.int64)
    mq_init(state)
    index = np.zeros(65536, dtype=np.uint8)
    mps = np.zeros(65536, dtype=np.uint8)
    out = np.empty(output_capacity(height * width + height), dtype=np.uint8)

    ltp = 0
    for y in range(height):
        row = y + _PAD_TOP
        if tpgdon:
            typical = 1 if typical_rows[y] else 0
            mq_encode(state, out, index, mps, _SLTP_CONTEXT_TEMPLATE0, typical ^ ltp)
            ltp = typical
            if typical:
                continue

        above2 = padded[row - 2]
        above1 = padded[row - 1]
        current = padded[row]
        # Preload so that the first shift in the loop lands on x = 0.
        r2 = 0
        for col in range(1, 5 + 1):
            r2 = (r2 << 1) | above2[col]
        r1 = 0
        for col in range(0, 6 + 1):
            r1 = (r1 << 1) | above1[col]
        r0 = 0
        for x in range(width):
            col = x + _PAD_LEFT
            r2 = ((r2 << 1) | above2[col + 2]) & 0x1F
            r1 = ((r1 << 1) | above1[col + 3]) & 0x7F
            pixel = current[col]
            mq_encode(state, out, index, mps, (r2 << 11) | (r1 << 4) | r0, pixel)
            r0 = ((r0 << 1) | pixel) & 0xF

    n = mq_flush(state, out)
    return out[:n].copy()


def encode_generic(image: np.ndarray, *, tpgdon: bool = True) -> bytes:
    """Arithmetic-code a bilevel image as generic region data (T.88 6.2).

    The result is the coded bitmap only, as it appears after the generic region
    segment data header (7.4.6). It uses GBTEMPLATE 0, the nominal AT pixels in
    ``TEMPLATE0_AT_PIXELS`` and is terminated by the 0xFF 0xAC marker.

    Args:
        image (np.ndarray): 2-D ``bool`` or ``uint8`` array; ``True`` or nonzero is black.
            Not modified.
        tpgdon (bool): Use typical prediction (6.2.5.7), which codes a row equal to the
            one above it as a single decision. Must match the TPGDON flag in the segment.

    Returns:
        bytes: Arithmetic-coded generic region data.
    """
    black = as_bilevel(image)
    height, width = black.shape
    padded = np.zeros((height + _PAD_TOP, width + _PAD_LEFT + _PAD_RIGHT), dtype=np.uint8)
    padded[_PAD_TOP:, _PAD_LEFT : _PAD_LEFT + width] = black
    # The row above the first one is all white (6.2.5.7 step 3b uses the
    # out-of-bitmap value 0).
    typical_rows = np.empty(height, dtype=np.bool_)
    typical_rows[0] = not black[0].any()
    typical_rows[1:] = (black[1:] == black[:-1]).all(axis=1)
    return _encode_template0(padded, typical_rows, bool(tpgdon)).tobytes()
