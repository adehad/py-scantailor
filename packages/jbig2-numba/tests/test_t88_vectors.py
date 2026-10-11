"""Conformance checks against published T.88 data and exact field layouts."""

import struct

import numpy as np

from jbig2_numba import encode_file, encode_pdf_stream
from jbig2_numba.mq_coder import NLPS, NMPS, QE, SWITCH, encode_decisions

# T.88 Annex H.2 test sequence for the arithmetic coder (single context).
_H2_INPUT = bytes.fromhex("00020051000000C00352872AAAAAAAAA82C02000FCD79EF6BF7FED904F46A3BF")
_H2_CODED = bytes.fromhex("84C73BFCE1A1430402200000410DBB86F4317FFF88FF37471ADB6ADFFFAC")


def test_mq_coder_reproduces_the_annex_h2_test_sequence() -> None:
    bits = np.unpackbits(np.frombuffer(_H2_INPUT, dtype=np.uint8))
    coded = encode_decisions(np.zeros(bits.size, dtype=np.int64), bits, 1)
    assert coded == _H2_CODED


def test_probability_table_matches_t88_table_e1_spot_values() -> None:
    assert len(QE) == 47
    assert (QE[0], NMPS[0], NLPS[0], SWITCH[0]) == (0x5601, 1, 1, 1)
    assert (QE[13], NMPS[13], NLPS[13], SWITCH[13]) == (0x1601, 29, 21, 0)
    assert (QE[45], NMPS[45], NLPS[45], SWITCH[45]) == (0x0001, 45, 43, 0)
    assert (QE[46], NMPS[46], NLPS[46], SWITCH[46]) == (0x5601, 46, 46, 0)
    assert [i for i in range(47) if SWITCH[i]] == [0, 6, 14]


def _parse_segments(data: bytes) -> list[tuple[int, int, int, bytes]]:
    segments = []
    pos = 0
    while pos < len(data):
        number, flags, refs = struct.unpack_from(">IBB", data, pos)
        pos += 6
        assert refs >> 5 == 0
        page_size = 4 if flags & 0x40 else 1
        page = int.from_bytes(data[pos : pos + page_size], "big")
        pos += page_size
        (length,) = struct.unpack_from(">I", data, pos)
        pos += 4
        segments.append((number, flags & 0x3F, page, data[pos : pos + length]))
        pos += length
    return segments


def test_pdf_stream_fields_follow_t88_clause_7() -> None:
    image = np.zeros((5, 9), dtype=bool)
    image[2, 3] = True
    segments = _parse_segments(encode_pdf_stream(image, dpi=(300, 600)))
    assert [(s[0], s[1], s[2]) for s in segments] == [(0, 48, 1), (1, 39, 1)]
    width, height, xres, yres, flags, striping = struct.unpack(">IIIIBH", segments[0][3])
    assert (width, height, xres, yres, striping) == (9, 5, 11811, 23622, 0)
    assert flags & 0b01111110 == 0  # no refinement, default pixel white, OR
    region = segments[1][3]
    assert struct.unpack_from(">IIIIB", region) == (9, 5, 0, 0, 0)
    assert region[17] == 0x08
    assert struct.unpack_from(">8b", region, 18) == (3, -1, -3, -1, 2, -2, -2, -2)
    assert region.endswith(b"\xff\xac")


def test_file_follows_t88_annex_d_sequential_layout() -> None:
    data = encode_file(np.ones((3, 3), dtype=bool))
    assert data[:8] == bytes.fromhex("974A42320D0A1A0A")
    assert data[8] == 0x01  # sequential, page count known
    assert struct.unpack_from(">I", data, 9) == (1,)
    segments = _parse_segments(data[13:])
    assert [(s[1], s[2]) for s in segments] == [(48, 1), (39, 1), (49, 1), (51, 0)]
    assert [s[0] for s in segments] == [0, 1, 2, 3]
    assert segments[2][3] == b"" and segments[3][3] == b""
