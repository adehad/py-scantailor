"""MQ binary arithmetic encoder from ITU-T T.88 Annex E.

The coder state lives in a small ``int64`` array so the procedures can be
composed inside other ``njit`` functions without allocating per symbol:

====  =====================================================
slot  meaning
====  =====================================================
0     A, the interval register
1     C, the code register
2     CT, the shift counter until the next BYTEOUT
3     B, the byte waiting to be written
4     BP, index in the output of B (-1 before the first byte)
====  =====================================================

Context state is two ``uint8`` arrays indexed by context: ``index`` (I(CX))
and ``mps`` (MPS(CX)).
"""

import numpy as np
from numba import njit

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
QE = np.ascontiguousarray(_TABLE_E1[:, 0])
NMPS = _TABLE_E1[:, 1].astype(np.uint8)
NLPS = _TABLE_E1[:, 2].astype(np.uint8)
SWITCH = _TABLE_E1[:, 3].astype(np.uint8)

_A, _C, _CT, _B, _BP = 0, 1, 2, 3, 4
STATE_SIZE = 5


@njit(cache=True)
def output_capacity(symbol_count: int) -> int:
    """Upper bound on the bytes the encoder can emit for a number of symbols.

    Each symbol renormalises at most 15 times because A stays at least Qe >= 1,
    and every output byte absorbs at least 7 of those shifts (7 after a stuffed
    0xFF, 8 otherwise). FLUSH adds at most a few bytes plus the 0xFF 0xAC marker.

    Args:
        symbol_count (int): Number of binary decisions that will be coded.

    Returns:
        int: Buffer size in bytes that cannot overflow.
    """
    return symbol_count * 15 // 7 + 16


@njit(cache=True)
def mq_init(state: np.ndarray) -> None:
    """INITENC (T.88 E.2.8).

    Args:
        state (np.ndarray): ``int64`` coder state of length ``STATE_SIZE``, overwritten.
    """
    state[_A] = 0x8000
    state[_C] = 0
    state[_CT] = 12
    state[_B] = 0
    state[_BP] = -1


@njit(cache=True)
def _byte_out(state: np.ndarray, out: np.ndarray) -> None:
    """BYTEOUT (T.88 E.2.6) with 0xFF bit stuffing and carry propagation."""
    b = state[_B]
    c = state[_C]
    if b != 0xFF:
        if c >= 0x8000000:
            b += 1
            c &= 0x7FFFFFF
        if b != 0xFF:
            if state[_BP] >= 0:
                out[state[_BP]] = b
            state[_BP] += 1
            state[_B] = c >> 19
            state[_C] = c & 0x7FFFF
            state[_CT] = 8
            return
    # After a 0xFF only seven bits go in the next byte so a carry can never
    # turn it into a marker.
    if state[_BP] >= 0:
        out[state[_BP]] = b
    state[_BP] += 1
    state[_B] = c >> 20
    state[_C] = c & 0xFFFFF
    state[_CT] = 7


@njit(cache=True)
def _renorm(state: np.ndarray, out: np.ndarray) -> None:
    """RENORME (T.88 E.2.5)."""
    a = state[_A]
    while True:
        a <<= 1
        state[_C] <<= 1
        state[_CT] -= 1
        if state[_CT] == 0:
            _byte_out(state, out)
        if a & 0x8000:
            break
    state[_A] = a


@njit(cache=True)
def mq_encode(
    state: np.ndarray,
    out: np.ndarray,
    index: np.ndarray,
    mps: np.ndarray,
    cx: int,
    bit: int,
) -> None:
    """ENCODE one decision in context ``cx`` (T.88 E.2.2 to E.2.4).

    Args:
        state (np.ndarray): ``int64`` coder state from ``mq_init``.
        out (np.ndarray): ``uint8`` output buffer sized with ``output_capacity``.
        index (np.ndarray): ``uint8`` probability state I(CX) per context, updated in place.
        mps (np.ndarray): ``uint8`` more probable symbol MPS(CX) per context, updated in place.
        cx (int): Context number.
        bit (int): Decision to code, 0 or 1.
    """
    i = index[cx]
    qe = QE[i]
    a = state[_A] - qe
    if bit == mps[cx]:
        # CODEMPS (E.2.4)
        if a & 0x8000:
            state[_A] = a
            state[_C] += qe
            return
        if a < qe:
            a = qe
        else:
            state[_C] += qe
        index[cx] = NMPS[i]
        state[_A] = a
        _renorm(state, out)
    else:
        # CODELPS (E.2.3), including the conditional exchange.
        if a < qe:
            state[_C] += qe
        else:
            a = qe
        if SWITCH[i]:
            mps[cx] = 1 - mps[cx]
        index[cx] = NLPS[i]
        state[_A] = a
        _renorm(state, out)


@njit(cache=True)
def mq_flush(state: np.ndarray, out: np.ndarray) -> int:
    """FLUSH (T.88 E.2.9) then append the 0xFF 0xAC end marker.

    Args:
        state (np.ndarray): ``int64`` coder state; unusable afterwards.
        out (np.ndarray): ``uint8`` output buffer.

    Returns:
        int: Number of valid bytes in ``out``.
    """
    # SETBITS (E.2.9): pick the value in [C, C + A) with most trailing ones.
    temp = state[_C] + state[_A]
    state[_C] |= 0xFFFF
    if state[_C] >= temp:
        state[_C] -= 0x8000
    state[_C] <<= state[_CT]
    _byte_out(state, out)
    state[_C] <<= state[_CT]
    _byte_out(state, out)
    bp = state[_BP]
    out[bp] = state[_B]
    bp += 1
    if state[_B] != 0xFF:
        out[bp] = 0xFF
        bp += 1
    out[bp] = 0xAC
    return bp + 1


@njit(cache=True)
def _encode_decisions(contexts: np.ndarray, bits: np.ndarray, context_count: int) -> np.ndarray:
    state = np.empty(STATE_SIZE, dtype=np.int64)
    mq_init(state)
    index = np.zeros(context_count, dtype=np.uint8)
    mps = np.zeros(context_count, dtype=np.uint8)
    out = np.empty(output_capacity(contexts.size), dtype=np.uint8)
    for k in range(contexts.size):
        mq_encode(state, out, index, mps, contexts[k], bits[k])
    n = mq_flush(state, out)
    return out[:n].copy()


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
    return _encode_decisions(contexts, bits.astype(np.uint8), context_count).tobytes()
