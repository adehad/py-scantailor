"""Straightforward pure-Python decoders written from T.88, used as test oracles.

They are deliberately written differently from the encoder (a 16-pixel
template lookup instead of shift registers, a split C register) so a shared
misreading of the spec is less likely to cancel out.
"""

import numpy as np

from jbig2.mq_coder import NLPS, NMPS, QE, SWITCH


class MQDecoder:
    """MQ arithmetic decoder (T.88 E.3)."""

    def __init__(self, data: bytes, context_count: int) -> None:
        """Run INITDEC.

        Args:
            data (bytes): Coded data.
            context_count (int): Number of contexts; all start in state 0 with MPS 0.
        """
        self.data = data
        self.index = [0] * context_count
        self.mps = [0] * context_count
        self.bp = 0
        self.c_high = self._byte(0)
        self.c_low = 0
        self.ct = 0
        self._byte_in()
        self.c_high = ((self.c_high << 7) & 0xFFFF) | ((self.c_low >> 9) & 0x7F)
        self.c_low = (self.c_low << 7) & 0xFFFF
        self.ct -= 7
        self.a = 0x8000

    def _byte(self, position: int) -> int:
        # E.3.4: reading past the end behaves as an endless run of 0xFF.
        return self.data[position] if position < len(self.data) else 0xFF

    def _byte_in(self) -> None:
        if self._byte(self.bp) == 0xFF:
            if self._byte(self.bp + 1) > 0x8F:
                self.c_low += 0xFF00
                self.ct = 8
            else:
                self.bp += 1
                self.c_low += self._byte(self.bp) << 9
                self.ct = 7
        else:
            self.bp += 1
            self.c_low += self._byte(self.bp) << 8
            self.ct = 8
        if self.c_low > 0xFFFF:
            self.c_high += self.c_low >> 16
            self.c_low &= 0xFFFF

    def decode(self, cx: int) -> int:
        """Decode one decision.

        Args:
            cx (int): Context number.

        Returns:
            int: The decoded bit.
        """
        i = self.index[cx]
        qe = int(QE[i])
        self.a -= qe
        if self.c_high < qe:
            if self.a < qe:
                self.a = qe
                bit = self.mps[cx]
                self.index[cx] = int(NMPS[i])
            else:
                self.a = qe
                bit = 1 - self.mps[cx]
                if SWITCH[i]:
                    self.mps[cx] = bit
                self.index[cx] = int(NLPS[i])
        else:
            self.c_high -= qe
            if self.a & 0x8000:
                return self.mps[cx]
            if self.a < qe:
                bit = 1 - self.mps[cx]
                if SWITCH[i]:
                    self.mps[cx] = bit
                self.index[cx] = int(NLPS[i])
            else:
                bit = self.mps[cx]
                self.index[cx] = int(NMPS[i])
        while True:
            if self.ct == 0:
                self._byte_in()
            self.a <<= 1
            self.c_high = ((self.c_high << 1) & 0xFFFF) | ((self.c_low >> 15) & 1)
            self.c_low = (self.c_low << 1) & 0xFFFF
            self.ct -= 1
            if self.a & 0x8000:
                break
        return bit


# T.88 Figure 3 (GBTEMPLATE 0), most significant context bit first, with the
# nominal AT pixels A4, A3, A2, A1 in their places.
_TEMPLATE0 = (
    (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
    (-3, -1), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (3, -1),
    (-4, 0), (-3, 0), (-2, 0), (-1, 0),
)  # fmt: skip


def decode_generic(data: bytes, width: int, height: int, *, tpgdon: bool) -> np.ndarray:
    """Decode GBTEMPLATE 0 generic region data (T.88 6.2.5).

    Args:
        data (bytes): Arithmetic-coded data from ``encode_generic``.
        width (int): Region width in pixels.
        height (int): Region height in pixels.
        tpgdon (bool): Whether the data was coded with typical prediction.

    Returns:
        np.ndarray: ``bool`` image, ``True`` for black.
    """
    decoder = MQDecoder(data, 65536)
    bitmap = [[0] * width for _ in range(height)]

    def pixel(x: int, y: int) -> int:
        return bitmap[y][x] if 0 <= x < width and 0 <= y < height else 0

    ltp = 0
    for y in range(height):
        if tpgdon:
            ltp ^= decoder.decode(0x9B25)
            if ltp:
                bitmap[y] = list(bitmap[y - 1]) if y > 0 else [0] * width
                continue
        for x in range(width):
            cx = 0
            for dx, dy in _TEMPLATE0:
                cx = (cx << 1) | pixel(x + dx, y + dy)
            bitmap[y][x] = decoder.decode(cx)
    return np.array(bitmap, dtype=bool).reshape(height, width)
