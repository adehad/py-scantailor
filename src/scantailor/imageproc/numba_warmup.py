"""Compile the numba kernels ahead of the first page that needs them.

A kernel compiles on its first call, which costs about a second each, and the
result is cached on disk. A source checkout caches next to the sources. A
packaged build cannot ship that cache: numba keys it by the path of the source
file and, when frozen, by a hash of the executable, and it writes into the
user's cache directory (``%LOCALAPPDATA%/numba`` on Windows, ``~/.cache/numba``
elsewhere) once per installed build. Calling every kernel once on tiny input
fills that cache without waiting for the first real page.

Run it by hand with ``python -m scantailor.imageproc.numba_warmup``, or start it
in the background when the application opens with ``start_warmup``.
"""

import threading

import cv2
import numpy as np
from loguru import logger

__all__ = ["start_warmup", "warm_up_kernels"]

_SIDE = 160
_INK = 30
_PAPER = 235
_BLOCK_DPI = 150
_DESPECKLE_DPI = 300
_DESPECKLE_LEVEL = 2.0
_NOISE_THRESHOLD = 1

_started = threading.Lock()
_thread: threading.Thread | None = None


def _text_like_page() -> np.ndarray:
    """A small gray page with text-sized marks inside wide margins."""
    page = np.full((_SIDE * 2, _SIDE * 2), _PAPER, dtype=np.uint8)
    for row in range(60, 260, 14):
        for column in range(60, 260, 9):
            page[row : row + 8, column : column + 5] = _INK
    return page


def warm_up_kernels() -> None:
    """Call every numba kernel once, with the argument types real pages use.

    Safe to call from any thread and as often as wanted; calls after the first
    only run the kernels on tiny input.
    """
    from scantailor.core.estimate_background import normalize_illumination_gray
    from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper
    from scantailor.dewarping.raster_dewarper import compute_dewarp_maps
    from scantailor.filters.select_content.content_box_finder import find_content_rect
    from scantailor.imageproc.color_segmenter import segment_colors
    from scantailor.imageproc.despeckle import despeckle
    from scantailor.imageproc.reconstruction import reconstruction

    page = _text_like_page()
    find_content_rect(page, None)
    normalize_illumination_gray(page)

    binary = np.where(page < _PAPER // 2, 0, 255).astype(np.uint8)
    despeckle(binary, _DESPECKLE_DPI, _DESPECKLE_LEVEL)

    ink = binary == 0
    segment_colors(
        ink, page, (_DESPECKLE_DPI, _DESPECKLE_DPI), _NOISE_THRESHOLD, (0, 0, 0)
    )
    segment_colors(
        ink,
        np.asarray(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR), dtype=np.uint8),
        (_DESPECKLE_DPI, _DESPECKLE_DPI),
        _NOISE_THRESHOLD,
        (0, 0, 0),
    )
    reconstruction(binary, np.full_like(binary, 255))

    top = np.array([[0, 20], [100, 5], [200, 20]], dtype=np.float64)
    bottom = np.array([[0, 280], [100, 295], [200, 280]], dtype=np.float64)
    dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom)
    compute_dewarp_maps(dewarper, (_BLOCK_DPI, _BLOCK_DPI), (0.0, 0.0, 1.0, 1.0))


def start_warmup() -> threading.Thread:
    """Warm the kernels on a background thread, once per process.

    Returns:
        threading.Thread: The warm-up thread; later calls return the same one.
    """
    global _thread
    with _started:
        if _thread is None:
            _thread = threading.Thread(
                target=_run_logged, name="numba-warmup", daemon=True
            )
            _thread.start()
        return _thread


def _run_logged() -> None:
    try:
        warm_up_kernels()
    except Exception as error:
        logger.opt(exception=True).warning(f"Kernel warm-up failed: {error}")


if __name__ == "__main__":
    warm_up_kernels()
