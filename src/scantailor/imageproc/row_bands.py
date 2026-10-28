"""Running a kernel over horizontal bands of an image.

A kernel that only looks a fixed number of rows up and down (its halo) gives
the same pixels whether it sees the whole image or a band of rows plus that
halo on each side. Cutting the image into bands bounds the memory of the
kernel's float or label temporaries to one band per thread instead of the whole
page.

Band edges depend only on the image height, the halo and the kernel's bytes per
row, never on the thread count, so a kernel run on one thread and on eight
cuts the same bands and gives the same pixels.
"""

import itertools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2

# Target working memory of one band's temporaries; a band may use up to twice
# this, since the band count is rounded down.
_BAND_BYTES = 32 << 20
# Bands are at least this many halos tall, so the rows read twice stay a small
# share of the work.
_MIN_HALOS_PER_BAND = 8
# Most threads that work on the bands of one image.
_MAX_THREADS = 8


@dataclass(frozen=True)
class RowBand:
    """One band of rows and the halo rows its kernel reads around it.

    Attributes:
        first (int): First row the band writes.
        stop (int): Row after the last row the band writes.
        top (int): First row the kernel reads, ``first - halo`` clipped at 0.
        bottom (int): Row after the last row the kernel reads, ``stop + halo``
            clipped at the image height.
    """

    first: int
    stop: int
    top: int
    bottom: int

    @property
    def rows(self) -> slice:
        """Rows of the image that the band writes."""
        return slice(self.first, self.stop)

    @property
    def padded(self) -> slice:
        """Rows of the image that the kernel reads."""
        return slice(self.top, self.bottom)

    @property
    def inside(self) -> slice:
        """Rows of the padded block that belong to the band."""
        return slice(self.first - self.top, self.stop - self.top)


def row_bands(rows: int, halo: int, bytes_per_row: int) -> list[RowBand]:
    """Cut ``rows`` into bands sized by memory, each at least eight halos tall.

    Args:
        rows (int): Image height.
        halo (int): Rows the kernel reads beyond each side of a band; 0 or more.
        bytes_per_row (int): Bytes of temporaries the kernel makes per image row.

    Returns:
        list[RowBand]: Consecutive bands covering ``0..rows`` in order, of
            near-equal height between the target and twice it; one empty band
            when ``rows`` is 0.
    """
    height = max(1, _MIN_HALOS_PER_BAND * halo, _BAND_BYTES // max(1, bytes_per_row))
    # Rounding the count down keeps every band at least ``height`` rows tall.
    count = max(1, rows // height)
    edges = [rows * index // count for index in range(count + 1)]
    return [
        RowBand(first, stop, max(0, first - halo), min(rows, stop + halo))
        for first, stop in itertools.pairwise(edges)
    ]


def map_row_bands[T](
    rows: int,
    halo: int,
    bytes_per_row: int,
    kernel: Callable[[RowBand], T],
) -> list[T]:
    """Call ``kernel`` on every band of ``row_bands``, on up to eight threads.

    The thread count follows OpenCV's, so a batch worker limited to one OpenCV
    thread runs the bands one after another and holds one band's temporaries
    at a time.

    Args:
        rows (int): Image height.
        halo (int): Rows the kernel reads beyond each side of a band.
        bytes_per_row (int): Bytes of temporaries the kernel makes per image row.
        kernel (Callable[[RowBand], T]): Handles one band. Bands are disjoint,
            so it may write its own rows of a shared output array.

    Returns:
        list[T]: The kernel's results in band order.
    """
    bands = row_bands(rows, halo, bytes_per_row)
    workers = min(len(bands), max(1, cv2.getNumThreads()), _MAX_THREADS)
    if workers == 1:
        return [kernel(band) for band in bands]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(kernel, bands))
