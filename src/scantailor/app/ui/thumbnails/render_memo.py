"""What thumbnail renders keep so that a stage switch decodes as little as it can."""

import threading
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage

__all__ = ["ReducedSource", "ThumbnailMemo", "reduced_copy"]

_MIB = 1 << 20
# A 600 dpi A4 page shrunk for 160 px thumbnails takes about 0.5 MiB.
_SOURCE_BUDGET = 192 * _MIB
_IMAGE_BUDGET = 64 * _MIB
_SETTLED_CAPACITY = 100_000


@dataclass(frozen=True)
class ReducedSource:
    """A decoded source shrunk evenly, with what is needed to render from it.

    Attributes:
        pixels (NDArray[np.uint8]): Read-only ``(H, W)`` gray or ``(H, W, 3)``
            BGR pixels, before Fix Orientation.
        full_size (tuple[int, int]): Width and height of the full-size decode.
        dpi (float): Horizontal resolution of the full-size decode, as the
            pipeline is given it.
    """

    pixels: NDArray[np.uint8]
    full_size: tuple[int, int]
    dpi: float

    @property
    def scale(self) -> float:
        """The smaller of the two ratios of reduced to full size."""
        height, width = self.pixels.shape[:2]
        return min(width / self.full_size[0], height / self.full_size[1])


class _SizedLru[Value]:
    """Least-recently-used map that evicts by total size rather than count."""

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._used = 0
        self._items: OrderedDict[Hashable, tuple[Value, int]] = OrderedDict()

    def get(self, key: Hashable) -> Value | None:
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item[0]

    def put(self, key: Hashable, value: Value, size: int) -> None:
        if size > self._budget:
            return
        previous = self._items.pop(key, None)
        if previous is not None:
            self._used -= previous[1]
        self._items[key] = (value, size)
        self._used += size
        while self._used > self._budget:
            _key, (_value, evicted) = self._items.popitem(last=False)
            self._used -= evicted

    def clear(self) -> None:
        self._items.clear()
        self._used = 0


class ThumbnailMemo:
    """Shrunk sources, finished thumbnails and parameters known to be settled.

    Every key names its content: a source key includes the file's state, a
    thumbnail key the digest of every parameter its render reads. One memo can
    therefore serve any number of projects. Safe to use from worker threads.

    Args:
        source_budget (int): Bytes of shrunk sources kept.
        image_budget (int): Bytes of finished thumbnails kept.
        settled_capacity (int): Settled parameter digests remembered.
    """

    def __init__(
        self,
        source_budget: int = _SOURCE_BUDGET,
        image_budget: int = _IMAGE_BUDGET,
        settled_capacity: int = _SETTLED_CAPACITY,
    ) -> None:
        """Create an empty memo; arguments are described on the class."""
        self._lock = threading.Lock()
        self._sources = _SizedLru[ReducedSource](source_budget)
        self._images = _SizedLru[QImage](image_budget)
        self._settled: OrderedDict[Hashable, None] = OrderedDict()
        self._settled_capacity = settled_capacity

    def reduced_source(self, key: Hashable) -> ReducedSource | None:
        """Return the shrunk source stored under a key, if any.

        Args:
            key (Hashable): Names the file, its state and its resolution.

        Returns:
            ReducedSource | None: The source, or None when it is not kept.
        """
        with self._lock:
            return self._sources.get(key)

    def keep_reduced_source(self, key: Hashable, source: ReducedSource) -> None:
        """Remember a shrunk source; its pixels are made read-only.

        Args:
            key (Hashable): Names the file, its state and its resolution.
            source (ReducedSource): The source.
        """
        source.pixels.flags.writeable = False
        with self._lock:
            self._sources.put(key, source, source.pixels.nbytes)

    def image(self, key: Hashable) -> QImage | None:
        """Return the thumbnail stored under a key, if any.

        Args:
            key (Hashable): Names the page, stage, parameters and size.

        Returns:
            QImage | None: The thumbnail, shared; or None.
        """
        with self._lock:
            return self._images.get(key)

    def keep_image(self, key: Hashable, image: QImage) -> None:
        """Remember a finished thumbnail.

        Args:
            key (Hashable): Names the page, stage, parameters and size.
            image (QImage): The thumbnail; shared, not copied.
        """
        with self._lock:
            self._images.put(key, image, image.sizeInBytes())

    def is_settled(self, digest: Hashable) -> bool:
        """Tell whether a full render left these parameters as it found them.

        Args:
            digest (Hashable): Digest of the parameters of the stages that can
                detect, up to some stage.

        Returns:
            bool: True when no stage up to there would detect again.
        """
        with self._lock:
            if digest not in self._settled:
                return False
            self._settled.move_to_end(digest)
            return True

    def settle(self, digest: Hashable) -> None:
        """Record that a full render left these parameters as it found them.

        Args:
            digest (Hashable): As for ``is_settled``.
        """
        with self._lock:
            self._settled[digest] = None
            self._settled.move_to_end(digest)
            while len(self._settled) > self._settled_capacity:
                self._settled.popitem(last=False)

    def clear(self) -> None:
        """Forget everything."""
        with self._lock:
            self._sources.clear()
            self._images.clear()
            self._settled.clear()


def reduced_copy(
    pixels: NDArray[np.uint8], dpi: float, longest_side: int
) -> ReducedSource | None:
    """Shrink a decoded source so that its longest side is ``longest_side``.

    Args:
        pixels (NDArray[np.uint8]): Full-size ``(H, W)`` or ``(H, W, 3)`` pixels.
        dpi (float): Horizontal resolution of ``pixels``, for the pipeline.
        longest_side (int): Wanted length of the longer side, in pixels.

    Returns:
        ReducedSource | None: The copy, or None when it would not be less than
            half the size, which saves too little to be worth keeping.
    """
    height, width = pixels.shape[:2]
    scale = longest_side / max(width, height)
    if scale >= 0.5:
        return None
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = np.asarray(
        cv2.resize(pixels, size, interpolation=cv2.INTER_AREA), dtype=np.uint8
    )
    return ReducedSource(small, (width, height), dpi)
