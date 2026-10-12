"""Thumbnail cache: thread-safe in-memory LRU with an optional on-disk layer."""

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PySide6.QtGui import QImage

from scantailor.core.models import PageId


@dataclass(frozen=True)
class ThumbnailKey:
    """Identity of one rendered thumbnail.

    Args:
        page (PageId): The page shown.
        stage (str): Filter stage the preview belongs to.
        params_hash (Hashable): Digest of the stage parameters that affect the
            preview; a change produces a new key and therefore a re-render.
        render_size (tuple[int, int]): Width and height of the box the image
            was scaled to; a different size is a different thumbnail.
    """

    page: PageId
    stage: str
    params_hash: Hashable = 0
    render_size: tuple[int, int] = (0, 0)

    def file_stem(self) -> str:
        """Return a filesystem-safe name that is unique to this key."""
        image_id = self.page.image_id
        text = (
            f"{image_id.file_path}|{image_id.page}|{int(self.page.sub_page)}"
            f"|{self.stage}|{self.params_hash!r}|{self.render_size}"
        )
        return hashlib.sha1(text.encode("utf-8")).hexdigest()


class ThumbnailCache:
    """Least-recently-used cache of thumbnail images.

    Args:
        capacity (int): Maximum number of images kept in memory.
        disk_directory (Path | None): Directory for persisted PNG thumbnails, or
            None to keep thumbnails in memory only. Created on first write.
            Files are never evicted; a changed ``params_hash`` leaves the old
            file behind.
    """

    def __init__(self, capacity: int = 256, disk_directory: Path | None = None) -> None:
        """Initialise; arguments are described on the class."""
        if capacity < 1:
            msg = "capacity must be at least 1"
            raise ValueError(msg)
        self._capacity = capacity
        self._disk_directory = disk_directory
        self._items: OrderedDict[ThumbnailKey, QImage] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Return the number of images held in memory."""
        with self._lock:
            return len(self._items)

    def __contains__(self, key: object) -> bool:
        """Return True if the key is in memory, without refreshing its recency."""
        with self._lock:
            return key in self._items

    def get(self, key: ThumbnailKey) -> QImage | None:
        """Return the in-memory image and mark it most recently used.

        Args:
            key (ThumbnailKey): The thumbnail identity.

        Returns:
            QImage | None: The cached image, or None on a miss.
        """
        with self._lock:
            image = self._items.get(key)
            if image is not None:
                self._items.move_to_end(key)
            return image

    def put(self, key: ThumbnailKey, image: QImage) -> None:
        """Store an image in memory, evicting the least recently used on overflow.

        Args:
            key (ThumbnailKey): The thumbnail identity.
            image (QImage): The thumbnail; shared, not copied.
        """
        with self._lock:
            self._items[key] = image
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def clear(self) -> None:
        """Drop all in-memory images. Disk files are kept."""
        with self._lock:
            self._items.clear()

    def load_from_disk(self, key: ThumbnailKey) -> QImage | None:
        """Read a persisted thumbnail. Safe to call from worker threads.

        Args:
            key (ThumbnailKey): The thumbnail identity.

        Returns:
            QImage | None: The image, or None if disk caching is off or the file
                is missing or unreadable.
        """
        path = self._disk_path(key)
        if path is None or not path.is_file():
            return None
        image = QImage(str(path))
        return None if image.isNull() else image

    def store_to_disk(self, key: ThumbnailKey, image: QImage) -> None:
        """Persist a thumbnail. Safe to call from worker threads.

        Args:
            key (ThumbnailKey): The thumbnail identity.
            image (QImage): The image to write as PNG.
        """
        path = self._disk_path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Cannot create thumbnail cache directory {}", path.parent)
            return
        # Write-then-rename so a reader never sees a partial file.
        temporary = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp.png")
        if image.save(str(temporary)):
            temporary.replace(path)
        else:
            logger.warning("Cannot write thumbnail {}", temporary)

    def _disk_path(self, key: ThumbnailKey) -> Path | None:
        if self._disk_directory is None:
            return None
        return self._disk_directory / f"{key.file_stem()}.png"
