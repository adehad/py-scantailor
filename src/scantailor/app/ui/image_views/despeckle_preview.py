"""The despeckling tab: the page with the specks a level removes marked.

Specks are drawn red and the area around them is tinted red, fading out with
distance, so that they can be found on a page of mostly white. Pictures of a
mixed page are shown as they are. Despeckling a full page takes a noticeable
fraction of a second, so the preview is computed on the global thread pool and
only the newest request is shown.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.imageproc.despeckle import despeckle

SPECKLE_RGB = (255, 0, 0)
_INK = 0
_BACKGROUND = 255
# The tint reaches this far from a speck at 600 dpi, which scales with the
# resolution, and is this opaque next to it.
_HALO_RADIUS_AT_600_DPI = 45.0
_REFERENCE_DPI = 600
_HALO_PEAK_ALPHA = 0.7
# Without a single speck the whole page gets this tint.
_NO_SPECKS_ALPHA = 0.3


def despeckle_preview(
    binary: NDArray[np.uint8],
    dpi: int,
    level: DespeckleLevel,
    page: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Draw a page with the ink that despeckling would remove in red.

    Args:
        binary (NDArray[np.uint8]): Bilevel page before despeckling, 0 is ink.
            Not modified.
        dpi (int): Resolution of ``binary``; the smaller of both axes.
        level (DespeckleLevel): The level to preview.
        page (NDArray[np.uint8] | None): The output page, gray or RGB, of the
            same height and width. Its pictures show through and the ink of
            ``binary`` is drawn black over it. A white page when None. Not
            modified.

    Returns:
        NDArray[np.uint8]: RGB image of the same height and width: ink that stays
            black, ink that goes red with a red tint that fades with distance,
            and the rest as on the page.
    """
    ink = binary == _INK
    shown = _rgb_copy(page, binary.shape)
    shown[ink] = 0
    if not level.is_enabled():
        return shown
    kept = despeckle(binary, dpi, level.to_factor())
    specks = ink & (kept != _INK)
    _tint_around(shown, specks, dpi)
    shown[specks] = SPECKLE_RGB
    return shown


def _rgb_copy(
    page: NDArray[np.uint8] | None, shape: tuple[int, ...]
) -> NDArray[np.uint8]:
    if page is None:
        return np.full((*shape, 3), _BACKGROUND, dtype=np.uint8)
    if page.ndim == 2:
        return np.repeat(page[:, :, None], 3, axis=2)
    return np.ascontiguousarray(page[:, :, :3]).copy()


def _tint_around(image: NDArray[np.uint8], specks: NDArray[np.bool_], dpi: int) -> None:
    """Blend red into the pixels near specks, leaving black pixels alone."""
    if specks.any():
        radius = _HALO_RADIUS_AT_600_DPI * dpi / _REFERENCE_DPI
        away_from_specks = np.where(specks, 0, 1).astype(np.uint8)
        distance = cv2.distanceTransform(
            away_from_specks, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        alpha = _HALO_PEAK_ALPHA * (1.0 - np.square(distance) / (radius * radius))
    else:
        alpha = np.full(specks.shape, _NO_SPECKS_ALPHA, dtype=np.float32)
    black = ~image.any(axis=2)
    tinted = (alpha > 0) & ~specks & ~black
    weight = alpha[tinted][:, None]
    red = np.array(SPECKLE_RGB, dtype=np.float32)
    image[tinted] = (image[tinted] * (1.0 - weight) + red * weight).astype(np.uint8)


class _Signals(QObject):
    done = Signal(int, object)


class _Job(QRunnable):
    def __init__(
        self,
        token: int,
        binary: NDArray[np.uint8],
        dpi: int,
        level: DespeckleLevel,
        page: NDArray[np.uint8] | None,
        signals: _Signals,
    ) -> None:
        super().__init__()
        self._token = token
        self._binary = binary
        self._dpi = dpi
        self._level = level
        self._page = page
        self._signals = signals

    def run(self) -> None:
        preview = despeckle_preview(self._binary, self._dpi, self._level, self._page)
        try:
            self._signals.done.emit(self._token, preview)
        except RuntimeError:
            pass  # The previewer was deleted while the job ran.


class DespecklePreviewer(QObject):
    """Computes despeckle previews in the background.

    Signals:
        preview_ready: The newest requested preview is done (RGB ndarray).
    """

    preview_ready = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        """Create the previewer.

        Args:
            parent (QObject | None): Parent object.
        """
        super().__init__(parent)
        self._latest = 0
        self._signals = _Signals(self)
        self._signals.done.connect(self._on_done)

    def request(
        self,
        binary: NDArray[np.uint8],
        dpi: int,
        level: DespeckleLevel,
        page: NDArray[np.uint8] | None = None,
    ) -> None:
        """Start a preview; any earlier one that is still running is dropped.

        Args:
            binary (NDArray[np.uint8]): Page before despeckling; the caller must
                not change it while the job runs.
            dpi (int): Resolution of the page.
            level (DespeckleLevel): The level to preview.
            page (NDArray[np.uint8] | None): The output page to show the pictures
                of; the caller must not change it while the job runs.
        """
        self._latest += 1
        QThreadPool.globalInstance().start(
            _Job(self._latest, binary, dpi, level, page, self._signals)
        )

    def _on_done(self, token: int, preview: NDArray[np.uint8]) -> None:
        if token == self._latest:
            self.preview_ready.emit(preview)
