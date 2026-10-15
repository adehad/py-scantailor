"""MQ binary arithmetic encoder from ITU-T T.88 Annex E.

The coder runs in plain Python over runs of equal decisions. Inside a run of
more probable symbols the interval register A only shrinks by Qe per decision
until it drops below 0x8000, so all the decisions before that point are coded
with one subtraction; this is what keeps a mostly white page fast.
"""

import numpy as np

# T.88 Table E.1: Qe, NMPS, NLPS, SWITCH for each probability state.
_TABLE_E1 = np.array(
    [
        (0x5601, 1, 1, 1),
        (0x3401, 2, 6, 0),
        (0x1801, 3, 9, 0),
        (0x0AC1, 4, 12, 0),
        (0x0521, 5, 29, 0),
        (0x0221, 38, 33, 0),
        (0x5601, 7, 6, 1),
        (0x5401, 8, 14, 0),
        (0x4801, 9, 14, 0),
        (0x3801, 10, 14, 0),
        (0x3001, 11, 17, 0),
        (0x2401, 12, 18, 0),
        (0x1C01, 13, 20, 0),
        (0x1601, 29, 21, 0),
        (0x5601, 15, 14, 1),
        (0x5401, 16, 14, 0),
        (0x5101, 17, 15, 0),
        (0x4801, 18, 16, 0),
        (0x3801, 19, 17, 0),
        (0x3401, 20, 18, 0),
        (0x3001, 21, 19, 0),
        (0x2801, 22, 19, 0),
        (0x2401, 23, 20, 0),
        (0x2201, 24, 21, 0),
        (0x1C01, 25, 22, 0),
        (0x1801, 26, 23, 0),
        (0x1601, 27, 24, 0),
        (0x1401, 28, 25, 0),
        (0x1201, 29, 26, 0),
        (0x1101, 30, 27, 0),
        (0x0AC1, 31, 28, 0),
        (0x09C1, 32, 29, 0),
        (0x08A1, 33, 30, 0),
        (0x0521, 34, 31, 0),
        (0x0441, 35, 32, 0),
        (0x02A1, 36, 33, 0),
        (0x0221, 37, 34, 0),
        (0x0141, 38, 35, 0),
        (0x0111, 39, 36, 0),
        (0x0085, 40, 37, 0),
        (0x0049, 41, 38, 0),
        (0x0025, 42, 39, 0),
        (0x0015, 43, 40, 0),
        (0x0009, 44, 41, 0),
        (0x0005, 45, 42, 0),
        (0x0001, 45, 43, 0),
        (0x5601, 46, 46, 0),
    ],
    dtype=np.int64,
)
QE = _TABLE_E1[:, 0].copy()
NMPS = _TABLE_E1[:, 1].astype(np.uint8)
NLPS = _TABLE_E1[:, 2].astype(np.uint8)
SWITCH = _TABLE_E1[:, 3].astype(np.uint8)

# The coder keeps one number per context, 2 * I(CX) + MPS(CX), so one list
# lookup gives both. These tables are indexed by that number.
_QE_BY_STATE = [int(QE[i]) for i in range(len(QE)) for _mps in (0, 1)]
_NEXT_STATE_AFTER_MPS = [2 * int(NMPS[i]) + mps for i in range(len(QE)) for mps in (0, 1)]
_NEXT_STATE_AFTER_LPS = [
    2 * int(NLPS[i]) + (mps ^ int(SWITCH[i])) for i in range(len(QE)) for mps in (0, 1)
]


def _byte_out(out: bytearray, b: int, c: int) -> tuple[int, int, int]:
    """BYTEOUT (T.88 E.2.6) with 0xFF bit stuffing and carry propagation.

    Args:
        out (bytearray): Output so far; B is appended to it.
        b (int): Byte waiting to be written.
        c (int): Code register.

    Returns:
        tuple[int, int, int]: The new B, C and CT.
    """
    if b != 0xFF and c >= 0x8000000:
        b += 1
        c &= 0x7FFFFFF
    out.append(b)
    if b == 0xFF:
        # After a 0xFF only seven bits go in the next byte so a carry can never
        # turn it into a marker.
        return c >> 20, c & 0xFFFFF, 7
    return c >> 19, c & 0x7FFFF, 8


def _runs(contexts: np.ndarray, bits: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    """Split a decision sequence into runs that share both context and decision.

    Args:
        contexts (np.ndarray): 1-D integer context per decision.
        bits (np.ndarray): 1-D decision per entry, same length as ``contexts``.

    Returns:
        tuple[list[int], list[int], list[int]]: Context, decision and length of each run.
    """
    changes = (contexts[1:] != contexts[:-1]) | (bits[1:] != bits[:-1])
    starts = np.flatnonzero(np.concatenate(([True], changes)))
    lengths = np.diff(starts, append=contexts.size)
    return contexts[starts].tolist(), bits[starts].tolist(), lengths.tolist()


class MQEncoder:
    """MQ encoder over a fixed set of contexts that all start in state 0 with MPS 0.

    Decisions can be fed in several calls to ``encode``; the result only
    depends on the whole sequence, not on how it was split.
    """

    def __init__(self, context_count: int) -> None:
        """INITENC (T.88 E.2.8).

        Args:
            context_count (int): Number of distinct contexts.
        """
        self._states = [0] * context_count
        self._a = 0x8000
        self._c = 0
        self._ct = 12
        # The first byte BYTEOUT writes is a placeholder that ``finish`` drops,
        # which stands for the BP = BPST - 1 start in INITENC.
        self._b = 0
        self._out = bytearray()

    def encode(self, contexts: np.ndarray, bits: np.ndarray) -> None:
        """ENCODE each decision in its context, in order (T.88 E.2.2 to E.2.5).

        Args:
            contexts (np.ndarray): 1-D integer context per decision, each below
                ``context_count``.
            bits (np.ndarray): 1-D decision per entry, 0 or 1, same length as ``contexts``.
        """
        if contexts.size == 0:
            return
        states, out = self._states, self._out
        a, c, ct, b = self._a, self._c, self._ct, self._b
        for cx, bit, count in zip(*_runs(contexts, bits), strict=True):
            while count:
                state = states[cx]
                qe = _QE_BY_STATE[state]
                if bit == state & 1:
                    # CODEMPS (E.2.4). Decisions that leave A >= 0x8000 only
                    # move Qe from A to C, so code them all at once.
                    without_renorm = (a - 0x8000) // qe
                    if count <= without_renorm:
                        a -= count * qe
                        c += count * qe
                        break
                    a -= without_renorm * qe
                    c += without_renorm * qe
                    count -= without_renorm + 1
                    a -= qe
                    if a < qe:
                        a = qe
                    else:
                        c += qe
                    states[cx] = _NEXT_STATE_AFTER_MPS[state]
                else:
                    # CODELPS (E.2.3), including the conditional exchange.
                    count -= 1
                    a -= qe
                    if a < qe:
                        c += qe
                    else:
                        a = qe
                    states[cx] = _NEXT_STATE_AFTER_LPS[state]
                # RENORME (E.2.5): double A until it reaches 0x8000, shifting C
                # with it and writing a byte each time CT runs out.
                shift = 16 - a.bit_length()
                a <<= shift
                while shift >= ct:
                    c <<= ct
                    shift -= ct
                    b, c, ct = _byte_out(out, b, c)
                c <<= shift
                ct -= shift
        self._a, self._c, self._ct, self._b = a, c, ct, b

    def finish(self) -> bytes:
        """FLUSH (T.88 E.2.9) and append the 0xFF 0xAC end marker.

        The encoder is unusable afterwards.

        Returns:
            bytes: Coded data terminated by 0xFF 0xAC.
        """
        out, a, c, ct, b = self._out, self._a, self._c, self._ct, self._b
        # SETBITS (E.2.9): pick the value in [C, C + A) with most trailing ones.
        temp = c + a
        c |= 0xFFFF
        if c >= temp:
            c -= 0x8000
        c <<= ct
        b, c, ct = _byte_out(out, b, c)
        c <<= ct
        b, c, ct = _byte_out(out, b, c)
        out.append(b)
        if b != 0xFF:
            out.append(0xFF)
        out.append(0xAC)
        return bytes(out[1:])


def encode_decisions(contexts: np.ndarray, bits: np.ndarray, context_count: int) -> bytes:
    """Code a sequence of decisions, each in its own context, with a fresh coder.

    Intended for testing the coder in isolation and for callers that build
    their own context model.

    Args:
        contexts (np.ndarray): Integer context number per decision, each in ``[0, context_count)``.
        bits (np.ndarray): Decision per entry, 0 or 1, same length as ``contexts``.
        context_count (int): Number of distinct contexts; all start in state 0 with MPS 0.

    Returns:
        bytes: Coded data terminated by 0xFF 0xAC.

    Raises:
        ValueError: If the arrays differ in length or hold out-of-range values.
    """
    contexts = np.asarray(contexts, dtype=np.int64).ravel()
    bits = np.asarray(bits).ravel()
    if contexts.shape != bits.shape:
        raise ValueError("contexts and bits must have the same length")
    if contexts.size and (contexts.min() < 0 or contexts.max() >= context_count):
        raise ValueError("context out of range")
    if bits.size and not np.isin(bits, (0, 1)).all():
        raise ValueError("bits must be 0 or 1")
    encoder = MQEncoder(context_count)
    encoder.encode(contexts, bits.astype(np.uint8))
    return encoder.finish()
