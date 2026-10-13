"""The despeckling tab: the page with the specks a level removes marked.

Despeckling a full page takes a noticeable fraction of a second, so the preview
is computed on the global thread pool and only the newest request is shown.
"""

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.imageproc.despeckle import despeckle

SPECKLE_RGB = (220, 40, 40)
_INK = 0
_BACKGROUND = 255


def despeckle_preview(
    binary: NDArray[np.uint8], dpi: int, level: DespeckleLevel
) -> NDArray[np.uint8]:
    """Draw a page with the ink that despeckling would remove in red.

    Args:
        binary (NDArray[np.uint8]): Bilevel page before despeckling, 0 is ink.
            Not modified.
        dpi (int): Resolution of ``binary``; the smaller of both axes.
        level (DespeckleLevel): The level to preview.

    Returns:
        NDArray[np.uint8]: RGB image of the same height and width: black ink that
            stays, red ink that goes, white elsewhere.
    """
    kept = despeckle(binary, dpi, level.to_factor()) if level.is_enabled() else binary
    ink = binary == _INK
    removed = ink & (kept != _INK)
    preview = np.full((*binary.shape, 3), _BACKGROUND, dtype=np.uint8)
    preview[ink] = 0
    preview[removed] = SPECKLE_RGB
    return preview


class _Signals(QObject):
    done = Signal(int, object)


class _Job(QRunnable):
    def __init__(
        self,
        token: int,
        binary: NDArray[np.uint8],
        dpi: int,
        level: DespeckleLevel,
        signals: _Signals,
    ) -> None:
        super().__init__()
        self._token = token
        self._binary = binary
        self._dpi = dpi
        self._level = level
        self._signals = signals

    def run(self) -> None:
        preview = despeckle_preview(self._binary, self._dpi, self._level)
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
        self, binary: NDArray[np.uint8], dpi: int, level: DespeckleLevel
    ) -> None:
        """Start a preview; any earlier one that is still running is dropped.

        Args:
            binary (NDArray[np.uint8]): Page before despeckling; the caller must
                not change it while the job runs.
            dpi (int): Resolution of the page.
            level (DespeckleLevel): The level to preview.
        """
        self._latest += 1
        QThreadPool.globalInstance().start(
            _Job(self._latest, binary, dpi, level, self._signals)
        )

    def _on_done(self, token: int, preview: NDArray[np.uint8]) -> None:
        if token == self._latest:
            self.preview_ready.emit(preview)
