"""The random numbers Qt's ``QRandomGenerator(seed)`` produces.

Qt seeds a 32-bit Mersenne Twister with ``std::seed_seq``. The RANSAC steps of
the distortion model builder pick their samples from it, so the same page has
to see the same sequence to give the same model.
"""

import numpy as np
from numpy.typing import NDArray

_WORD_MASK = 0xFFFFFFFF
_STATE_SIZE = 624


def seed_sequence_words(seed_values: list[int], count: int) -> NDArray[np.uint32]:
    """Run ``std::seed_seq(seed_values).generate`` for ``count`` words.

    Args:
        seed_values (list[int]): The 32-bit seed values.
        count (int): Number of words to generate.

    Returns:
        NDArray[np.uint32]: The generated words.
    """
    n = count
    s = len(seed_values)
    m = max(s + 1, n)
    if n >= 623:
        t = 11
    elif n >= 68:
        t = 7
    elif n >= 39:
        t = 5
    elif n >= 7:
        t = 3
    else:
        t = (n - 1) // 2
    p = (n - t) // 2
    q = p + t

    words = [0x8B8B8B8B] * n

    def mix(x: int) -> int:
        return x ^ (x >> 27)

    for k in range(m):
        r1 = (
            1664525 * mix(words[k % n] ^ words[(k + p) % n] ^ words[(k - 1) % n])
            & _WORD_MASK
        )
        if k == 0:
            addend = s
        elif k <= s:
            addend = k % n + seed_values[k - 1]
        else:
            addend = k % n
        r2 = (r1 + addend) & _WORD_MASK
        words[(k + p) % n] = (words[(k + p) % n] + r1) & _WORD_MASK
        words[(k + q) % n] = (words[(k + q) % n] + r2) & _WORD_MASK
        words[k % n] = r2
    for k in range(m, m + n):
        r3 = (
            1566083941
            * mix((words[k % n] + words[(k + p) % n] + words[(k - 1) % n]) & _WORD_MASK)
            & _WORD_MASK
        )
        r4 = (r3 - k % n) & _WORD_MASK
        words[(k + p) % n] ^= r3
        words[(k + q) % n] ^= r4
        words[k % n] = r4
    return np.array(words, dtype=np.uint32)


class QtRandomGenerator:
    """``QRandomGenerator(seed)``: a Mersenne Twister seeded by ``std::seed_seq``."""

    def __init__(self, seed: int = 1) -> None:
        """Seed the generator.

        Args:
            seed (int): The 32-bit seed, as passed to ``QRandomGenerator``.
        """
        state = seed_sequence_words([seed & _WORD_MASK], _STATE_SIZE)
        if not state[1:].any() and not state[0] & 0x80000000:
            state[0] = 0x80000000
        engine = np.random.MT19937()
        engine.state = {
            "bit_generator": "MT19937",
            "state": {"key": state, "pos": _STATE_SIZE},
        }
        self._engine = engine

    def generate(self) -> int:
        """Return the next 32-bit unsigned value."""
        return int(self._engine.random_raw())
