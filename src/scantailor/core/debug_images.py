"""Intermediate images a render records while debug mode is on.

Pipeline code takes a ``dbg: DebugImages | None`` argument, None by default,
and builds an image only inside an ``if dbg is not None:`` branch, so a render
without debug mode makes nothing and pays one comparison per recording point.

Each image goes to a PNG file in a private temporary directory as soon as it is
added, so a render that records dozens of full-page images holds at most one of
them in memory. The files belong to the collector: ``close`` deletes the
directory, and so does the collector's garbage collection or interpreter exit
when nobody closed it.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

# PNG compression trades file size for time; level 1 writes a 4000 x 6000 page
# several times faster than the default 3, and the files only live for a render.
_PNG_COMPRESSION = 1
_DIRECTORY_PREFIX = "scantailor-dbg-"


@dataclass(frozen=True)
class DebugImage:
    """One recorded image.

    Attributes:
        label (str): Upstream's name for the step, shown as the tab title.
        path (pathlib.Path): PNG file holding the image, gray or BGR.
    """

    label: str
    path: Path


def as_displayable(image: NDArray) -> NDArray[np.uint8]:
    """Convert an intermediate image to 8-bit pixels a PNG can hold.

    Args:
        image (numpy.ndarray): ``(H, W)`` or ``(H, W, 3)`` BGR array. Bool masks
            become black where True on white, as upstream draws binary images;
            uint8 is kept; any other dtype is stretched to 0..255.

    Returns:
        NDArray[np.uint8]: A new or the given array, never modified in place.
    """
    if image.dtype == np.bool_:
        return np.where(image, np.uint8(0), np.uint8(255))
    if image.dtype == np.uint8:
        return image
    values = np.asarray(image, dtype=np.float64)
    if values.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.round((values - low) * (255.0 / (high - low))).astype(np.uint8)


class DebugImages:
    """Collects the intermediate images of one render, in recording order.

    Recording may happen on any thread; the lock only guards the list. Hand the
    collector to the GUI thread once the render has finished with it.
    """

    def __init__(self) -> None:
        """Create an empty collector; the directory is made on the first image."""
        self._lock = threading.Lock()
        self._images: list[DebugImage] = []
        self._directory: Path | None = None
        self._cleanup: weakref.finalize | None = None

    def add(self, image: NDArray, label: str) -> None:
        """Write ``image`` to a temporary PNG and record it under ``label``.

        An image that cannot be written is logged and skipped. Upstream skips
        it too, because debug output must never fail a render.

        Args:
            image (numpy.ndarray): Gray, BGR or bool mask; see ``as_displayable``.
            label (str): Upstream's name for the step, such as ``"bw150"``.
        """
        pixels = as_displayable(np.asarray(image))
        entry: DebugImage | None = None
        try:
            with self._lock:
                directory = self._ensure_directory()
                entry = DebugImage(label, directory / f"{len(self._images):03d}.png")
                self._images.append(entry)
            if cv2.imwrite(
                str(entry.path),
                pixels,
                [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION],
            ):
                return
            logger.debug("Debug image {!r} not written", label)
        except (OSError, cv2.error) as error:
            logger.debug("Debug image {!r} not written: {}", label, error)
        if entry is not None:
            with self._lock:
                self._images = [kept for kept in self._images if kept is not entry]

    @property
    def images(self) -> tuple[DebugImage, ...]:
        """The recorded images, oldest first."""
        with self._lock:
            return tuple(self._images)

    @property
    def directory(self) -> Path | None:
        """The directory holding the files, or None before the first image."""
        return self._directory

    def close(self) -> None:
        """Delete the files. The collector is empty afterwards and may be reused."""
        with self._lock:
            cleanup, self._cleanup = self._cleanup, None
            self._directory = None
            self._images = []
        if cleanup is not None:
            cleanup()

    def __enter__(self) -> Self:
        """Return the collector; leaving the block closes it."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Delete the files."""
        self.close()

    def _ensure_directory(self) -> Path:
        if self._directory is None:
            self._directory = Path(tempfile.mkdtemp(prefix=_DIRECTORY_PREFIX))
            self._cleanup = weakref.finalize(
                self, shutil.rmtree, self._directory, ignore_errors=True
            )
        return self._directory
