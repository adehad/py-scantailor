"""Input validation, polarity and segment framing of the public API."""

import struct

import numpy as np
import pytest
from sample_images import noise

from jbig2 import encode_file, encode_generic, encode_pdf_stream

ENCODERS = [encode_generic, encode_pdf_stream, encode_file]


def _parse_segments(data: bytes) -> list[tuple[int, int, int, bytes]]:
    """Split sequential-organisation segments into (number, type, page, data)."""
    segments = []
    pos = 0
    while pos < len(data):
        number, flags, referred = struct.unpack_from(">IBB", data, pos)
        assert referred == 0
        pos += 6
        page_size = 4 if flags & 0x40 else 1
        page = int.from_bytes(data[pos : pos + page_size], "big")
        pos += page_size
        (length,) = struct.unpack_from(">I", data, pos)
        pos += 4
        segments.append((number, flags & 0x3F, page, data[pos : pos + length]))
        pos += length
    return segments


@pytest.mark.parametrize("encode", ENCODERS)
def test_rejects_unsupported_dtypes(encode) -> None:
    with pytest.raises(TypeError, match="dtype"):
        encode(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(TypeError, match="dtype"):
        encode(np.zeros((4, 4), dtype=np.uint16))


@pytest.mark.parametrize("encode", ENCODERS)
def test_rejects_non_arrays(encode) -> None:
    with pytest.raises(TypeError, match="numpy array"):
        encode([[0, 1], [1, 0]])


@pytest.mark.parametrize("encode", ENCODERS)
@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
def test_rejects_wrong_dimensions(encode, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="2-D"):
        encode(np.zeros(shape, dtype=bool))


@pytest.mark.parametrize("encode", ENCODERS)
@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_rejects_empty_images(encode, shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="empty"):
        encode(np.zeros(shape, dtype=bool))


@pytest.mark.parametrize("encode", ENCODERS)
def test_does_not_mutate_input(encode) -> None:
    image = (noise(20, 30, 0.5, seed=0) * np.uint8(200)).astype(np.uint8)
    original = image.copy()

    encode(image)

    np.testing.assert_array_equal(image, original)


def test_any_nonzero_uint8_is_black() -> None:
    mask = noise(20, 30, 0.5, seed=1)
    as_ones = mask.astype(np.uint8)
    as_mixed = np.where(mask, np.uint8(1) + np.arange(600, dtype=np.uint8).reshape(20, 30) % 255, 0)

    assert encode_generic(as_ones) == encode_generic(mask)
    assert encode_generic(as_mixed.astype(np.uint8)) == encode_generic(mask)


def test_accepts_non_contiguous_views() -> None:
    image = noise(40, 60, 0.5, seed=2)

    assert encode_generic(image[::2, ::3]) == encode_generic(np.ascontiguousarray(image[::2, ::3]))


def test_pdf_stream_is_page_information_then_lossless_generic_region() -> None:
    image = noise(21, 34, 0.3, seed=3)

    segments = _parse_segments(encode_pdf_stream(image, dpi=(300, 600)))

    assert [(n, t, p) for n, t, p, _ in segments] == [(0, 48, 1), (1, 39, 1)]
    width, height, x_res, y_res, flags, striping = struct.unpack(">IIIIBH", segments[0][3])
    assert (width, height) == (34, 21)
    assert (x_res, y_res) == (11811, 23622)
    assert flags & 0x01, "page must be flagged eventually lossless"
    assert striping == 0
    region = segments[1][3]
    assert struct.unpack_from(">IIIIB", region) == (34, 21, 0, 0, 0)
    assert region[17] == 0x08, "MMR 0, GBTEMPLATE 0, TPGDON 1"
    assert struct.unpack_from(">8b", region, 18) == (3, -1, -3, -1, 2, -2, -2, -2)
    assert region[26:] == encode_generic(image)


def test_pdf_stream_records_unknown_resolution_as_zero() -> None:
    segments = _parse_segments(encode_pdf_stream(np.ones((2, 2), dtype=bool)))

    assert struct.unpack_from(">II", segments[0][3], 8) == (0, 0)


def test_tpgdon_flag_follows_argument() -> None:
    segments = _parse_segments(encode_pdf_stream(np.ones((2, 2), dtype=bool), tpgdon=False))

    assert segments[1][3][17] == 0x00


def test_rejects_non_positive_dpi() -> None:
    with pytest.raises(ValueError, match="dpi"):
        encode_pdf_stream(np.ones((2, 2), dtype=bool), dpi=(0, 300))


def test_file_has_header_and_ends_with_end_of_page_and_end_of_file() -> None:
    image = noise(9, 10, 0.5, seed=4)

    data = encode_file(image)

    assert data[:8] == bytes((0x97, 0x4A, 0x42, 0x32, 0x0D, 0x0A, 0x1A, 0x0A))
    assert data[8] == 0x01, "sequential organisation, page count known"
    assert struct.unpack_from(">I", data, 9) == (1,)
    segments = _parse_segments(data[13:])
    assert [(n, t, p, len(d)) for n, t, p, d in segments][2:] == [(2, 49, 1, 0), (3, 51, 0, 0)]
    assert data[13:].startswith(encode_pdf_stream(image))


def test_masked_array_is_rejected_instead_of_encoding_its_raw_data() -> None:
    masked = np.ma.masked_array(np.ones((4, 4), dtype=bool), mask=np.eye(4, dtype=bool))

    with pytest.raises(TypeError, match="masked"):
        encode_pdf_stream(masked)
    with pytest.raises(TypeError, match="masked"):
        encode_file(masked)


def test_segment_longer_than_a_32_bit_length_field_is_refused() -> None:
    from jbig2.segments import SegmentType, segment_header

    with pytest.raises(ValueError, match="32-bit"):
        segment_header(0, SegmentType.PAGE_INFORMATION, 1, 2**32)
    with pytest.raises(ValueError, match="32-bit"):
        segment_header(0, SegmentType.PAGE_INFORMATION, 1, 0xFFFFFFFF)


def test_segment_header_accepts_the_largest_known_length() -> None:
    from jbig2.segments import SegmentType, segment_header

    header = segment_header(0, SegmentType.PAGE_INFORMATION, 1, 0xFFFFFFFE)

    assert header.endswith(b"\xff\xff\xff\xfe")
