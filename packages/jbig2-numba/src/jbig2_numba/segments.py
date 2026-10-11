"""JBIG2 segment headers, segment data and file framing (T.88 clause 7 and Annex D).

Only the segments a single-page, single-region lossless encoding needs are
built: page information, immediate lossless generic region, end of page and
end of file.
"""

import struct
from enum import IntEnum

import numpy as np

from jbig2_numba.generic_region import TEMPLATE0_AT_PIXELS, as_bilevel, encode_generic

# T.88 D.4.1 file header ID string.
FILE_ID = bytes((0x97, 0x4A, 0x42, 0x32, 0x0D, 0x0A, 0x1A, 0x0A))

_METRES_PER_INCH = 0.0254


class SegmentType(IntEnum):
    """Segment type numbers from T.88 7.3."""

    IMMEDIATE_LOSSLESS_GENERIC_REGION = 39
    PAGE_INFORMATION = 48
    END_OF_PAGE = 49
    END_OF_FILE = 51


def segment_header(number: int, segment_type: SegmentType, page: int, data_length: int) -> bytes:
    """Build a segment header with no referred-to segments (T.88 7.2).

    Args:
        number (int): Segment number, unique within the file.
        segment_type (SegmentType): Segment type.
        page (int): Page association; 0 for segments not tied to a page.
        data_length (int): Length in bytes of the segment data that follows.

    Returns:
        bytes: Encoded segment header.
    """
    # 7.2.3: bit 6 clear selects a one-byte page association field.
    flags = int(segment_type) & 0x3F
    if page > 0xFF:
        flags |= 0x40
    # 7.2.4: referred-to count 0 in bits 5-7 and no retain bits.
    referred = 0
    page_field = struct.pack(">I", page) if page > 0xFF else struct.pack(">B", page)
    return (
        struct.pack(">IBB", number, flags, referred) + page_field + struct.pack(">I", data_length)
    )


def _pixels_per_metre(dpi: int) -> int:
    return round(dpi / _METRES_PER_INCH)


def page_information(width: int, height: int, dpi: tuple[int, int] | None) -> bytes:
    """Page information segment data (T.88 7.4.8).

    Args:
        width (int): Page width in pixels.
        height (int): Page height in pixels.
        dpi (tuple[int, int] | None): Horizontal and vertical resolution in dots
            per inch, or ``None`` to record the resolution as unknown.

    Returns:
        bytes: The 19-byte segment data.

    Raises:
        ValueError: If a resolution is not positive.
    """
    if dpi is None:
        x_res = y_res = 0
    else:
        if dpi[0] <= 0 or dpi[1] <= 0:
            raise ValueError(f"dpi must be positive, got {dpi}")
        x_res, y_res = _pixels_per_metre(dpi[0]), _pixels_per_metre(dpi[1])
    # 7.4.8.5: bit 0 "page is eventually lossless"; default pixel 0 (white)
    # and default combination operator OR.
    flags = 0x01
    striping = 0
    return struct.pack(">IIIIBH", width, height, x_res, y_res, flags, striping)


def generic_region(image: np.ndarray, *, tpgdon: bool) -> bytes:
    """Immediate lossless generic region segment data (T.88 7.4.6).

    Args:
        image (np.ndarray): 2-D ``bool`` or ``uint8`` array; ``True`` or nonzero is black.
        tpgdon (bool): Whether typical prediction is used.

    Returns:
        bytes: Region segment information field, generic region flags, AT
        pixels and the arithmetic-coded bitmap.
    """
    height, width = image.shape
    # 7.4.1: region at the page origin, external combination operator OR.
    region_info = struct.pack(">IIIIB", width, height, 0, 0, 0)
    # 7.4.6.2: MMR = 0, GBTEMPLATE = 0, TPGDON in bit 3.
    region_flags = struct.pack(">B", 0x08 if tpgdon else 0x00)
    at_pixels = struct.pack(">8b", *(c for pixel in TEMPLATE0_AT_PIXELS for c in pixel))
    return region_info + region_flags + at_pixels + encode_generic(image, tpgdon=tpgdon)


def _segment(number: int, segment_type: SegmentType, page: int, data: bytes) -> bytes:
    return segment_header(number, segment_type, page, len(data)) + data


def _page_segments(black: np.ndarray, dpi: tuple[int, int] | None, tpgdon: bool) -> bytes:
    height, width = black.shape
    page_info = page_information(width, height, dpi)
    region = generic_region(black, tpgdon=tpgdon)
    return _segment(0, SegmentType.PAGE_INFORMATION, 1, page_info) + _segment(
        1, SegmentType.IMMEDIATE_LOSSLESS_GENERIC_REGION, 1, region
    )


def encode_pdf_stream(
    image: np.ndarray, *, dpi: tuple[int, int] | None = None, tpgdon: bool = True
) -> bytes:
    """Encode an image as the body of a PDF ``/JBIG2Decode`` image stream.

    ISO 32000-1 7.4.7 takes the embedded organisation (T.88 Annex D.2) and
    excludes the file header, end-of-page and end-of-file segments, so this is
    a page information segment followed by one region segment. No
    ``/JBIG2Globals`` stream is needed. The image dictionary must declare
    ``/BitsPerComponent 1`` and the same ``/Width`` and ``/Height``; with
    ``/DeviceGray`` the decoded samples are 0 for black.

    Args:
        image (np.ndarray): 2-D ``bool`` or ``uint8`` array; ``True`` or nonzero is black.
            Not modified.
        dpi (tuple[int, int] | None): Resolution recorded in the page information
            segment, or ``None`` for unknown. The size on the PDF page comes
            from the content stream, not from this value.
        tpgdon (bool): Use typical prediction, which codes a row identical to
            the one above as a single decision.

    Returns:
        bytes: Stream data for the image XObject.
    """
    return _page_segments(as_bilevel(image), dpi, tpgdon)


def encode_file(
    image: np.ndarray, *, dpi: tuple[int, int] | None = None, tpgdon: bool = True
) -> bytes:
    """Encode an image as a standalone single-page ``.jb2`` file.

    Uses the sequential organisation (T.88 D.1): file header, then each
    segment header immediately followed by its data, ending with end-of-page
    and end-of-file segments.

    Args:
        image (np.ndarray): 2-D ``bool`` or ``uint8`` array; ``True`` or nonzero is black.
            Not modified.
        dpi (tuple[int, int] | None): Resolution recorded in the page information
            segment, or ``None`` for unknown.
        tpgdon (bool): Use typical prediction.

    Returns:
        bytes: Complete JBIG2 file.
    """
    black = as_bilevel(image)
    # D.4.2: bit 0 sequential organisation, bit 1 clear means the page count is known.
    header = FILE_ID + struct.pack(">BI", 0x01, 1)
    return (
        header
        + _page_segments(black, dpi, tpgdon)
        + _segment(2, SegmentType.END_OF_PAGE, 1, b"")
        + _segment(3, SegmentType.END_OF_FILE, 0, b"")
    )
