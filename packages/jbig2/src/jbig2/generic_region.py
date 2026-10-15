"""Generic region encoding with arithmetic coding (T.88 6.2, MMR = 0).

Only GBTEMPLATE 0 with the nominal adaptive-template pixels is produced. With
those AT positions the 16-pixel context is a solid stencil:

    row y-2:  x-2 .. x+2   (5 bits, A4 and A3 at the ends)
    row y-1:  x-3 .. x+3   (7 bits, A2 and A1 at the ends)
    row y:    x-4 .. x-1   (4 bits)

concatenated most significant first in raster order, which is the bit order
of CONTEXT in T.88 6.2.5.3 and the order the TPGDON context 0x9B25 assumes.
The context of every pixel is therefore one correlation of the bitmap with a
kernel holding the bit weights.
"""

import cv2
import numpy as np

from jbig2.mq_coder import MQEncoder

# T.88 6.2.5.3, nominal AT pixels for GBTEMPLATE 0 as (dx, dy): A1, A2, A3, A4.
TEMPLATE0_AT_PIXELS: tuple[tuple[int, int], ...] = ((3, -1), (-3, -1), (2, -2), (-2, -2))

# T.88 6.2.5.7, context used to code SLTP for GBTEMPLATE 0.
_SLTP_CONTEXT_TEMPLATE0 = 0x9B25


def _template0_weights() -> np.ndarray:
    """Kernel whose correlation with the bitmap gives each pixel's context.

    Returns:
        np.ndarray: ``float32`` array of shape (3, 8); rows are y-2, y-1 and y,
        columns x-4 to x+3, so the pixel being coded sits at row 2, column 4.
    """
    weights = np.zeros((3, 8), dtype=np.float32)
    bit = 15
    for row, columns in ((0, range(2, 7)), (1, range(1, 8)), (2, range(0, 4))):
        for column in columns:
            weights[row, column] = 1 << bit
            bit -= 1
    return weights


_TEMPLATE0_WEIGHTS = _template0_weights()
_TEMPLATE0_ANCHOR = (4, 2)  # (x, y) in the kernel, as OpenCV expects it.

# Rows are coded in bands of about this many pixels, which bounds the memory
# for contexts and runs on large pages without changing the output.
_BAND_PIXELS = 1 << 21


def as_bilevel(image: np.ndarray) -> np.ndarray:
    """Validate an input image and return its black mask.

    Args:
        image (np.ndarray): 2-D ``bool`` or ``uint8`` array; ``True`` or nonzero is black.

    Returns:
        np.ndarray: New ``bool`` array, ``True`` where the pixel is black.

    Raises:
        TypeError: If ``image`` is not a NumPy array of dtype ``bool`` or ``uint8``,
            or is a masked array, whose mask would otherwise be ignored.
        ValueError: If ``image`` is not 2-D or has a zero-length axis.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if isinstance(image, np.ma.MaskedArray):
        raise TypeError("masked arrays are not supported; fill the mask first")
    if image.dtype not in (np.bool_, np.uint8):
        raise TypeError(f"image dtype must be bool or uint8, got {image.dtype}")
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got {image.ndim} dimensions")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty, got shape {image.shape}")
    if image.shape[0] >= 2**32 or image.shape[1] >= 2**32:
        raise ValueError("image dimensions must fit in 32 bits")
    return image != 0


def _contexts(black: np.ndarray, top: int, bottom: int) -> np.ndarray:
    """Template 0 context of every pixel in rows ``top`` to ``bottom - 1``.

    Args:
        black (np.ndarray): ``bool`` image, ``True`` where black.
        top (int): First row.
        bottom (int): Row after the last one.

    Returns:
        np.ndarray: ``int32`` array of shape (bottom - top, width).
    """
    # Two rows above the band feed its contexts. Above row 0 and left and right
    # of the image the template reads white, the out-of-bitmap value in T.88.
    first = max(0, top - 2)
    contexts = cv2.filter2D(
        black[first:bottom].view(np.uint8),
        cv2.CV_32F,
        _TEMPLATE0_WEIGHTS,
        anchor=_TEMPLATE0_ANCHOR,
        borderType=cv2.BORDER_CONSTANT,
    )
    # Sums of distinct powers of two below 2**16 are exact in float32.
    return contexts[top - first :].astype(np.int32)


def _typical_rows(black: np.ndarray) -> np.ndarray:
    """Rows that TPGDON can skip because they equal the row above (T.88 6.2.5.7).

    Args:
        black (np.ndarray): ``bool`` image, ``True`` where black.

    Returns:
        np.ndarray: ``bool`` array with one entry per row.
    """
    typical = np.empty(black.shape[0], dtype=np.bool_)
    # The row above the first one is all white (6.2.5.7 step 3b uses the
    # out-of-bitmap value 0).
    typical[0] = not black[0].any()
    typical[1:] = (black[1:] == black[:-1]).all(axis=1)
    return typical


def _band_decisions(
    black: np.ndarray, top: int, bottom: int, sltp: np.ndarray | None, coded: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Contexts and decisions for rows ``top`` to ``bottom - 1``, in coding order.

    Each row starts with its SLTP decision when typical prediction is on,
    followed by every pixel of the row if the row is coded.

    Args:
        black (np.ndarray): ``bool`` image, ``True`` where black.
        top (int): First row of the band.
        bottom (int): Row after the last one.
        sltp (np.ndarray | None): ``uint8`` SLTP decision for every row of the
            image, or ``None`` without typical prediction.
        coded (np.ndarray): ``bool`` per row of the image, ``True`` where its pixels are coded.

    Returns:
        tuple[np.ndarray, np.ndarray]: 1-D ``int32`` contexts and 1-D ``uint8`` decisions.
    """
    width = black.shape[1]
    band_coded = coded[top:bottom]
    if band_coded.any():
        contexts = _contexts(black, top, bottom)[band_coded].ravel()
    else:
        contexts = np.empty(0, dtype=np.int32)
    pixels = black[top:bottom][band_coded].ravel().view(np.uint8)
    if sltp is None:
        return contexts, pixels

    decisions_per_row = 1 + width * band_coded
    row_starts = np.cumsum(decisions_per_row) - decisions_per_row
    is_sltp = np.zeros(int(decisions_per_row.sum()), dtype=np.bool_)
    is_sltp[row_starts] = True
    stream_contexts = np.empty(is_sltp.size, dtype=np.int32)
    stream_bits = np.empty(is_sltp.size, dtype=np.uint8)
    stream_contexts[is_sltp] = _SLTP_CONTEXT_TEMPLATE0
    stream_bits[is_sltp] = sltp[top:bottom]
    stream_contexts[~is_sltp] = contexts
    stream_bits[~is_sltp] = pixels
    return stream_contexts, stream_bits


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
    if tpgdon:
        typical = _typical_rows(black)
        # SLTP is 1 where a row's typical flag differs from the previous row's.
        previous = np.concatenate(([False], typical[:-1]))
        sltp = (typical != previous).view(np.uint8)
        coded = ~typical
    else:
        sltp = None
        coded = np.ones(height, dtype=np.bool_)

    encoder = MQEncoder(1 << 16)
    band_rows = max(1, _BAND_PIXELS // width)
    for top in range(0, height, band_rows):
        bottom = min(height, top + band_rows)
        encoder.encode(*_band_decisions(black, top, bottom, sltp, coded))
    return encoder.finish()
