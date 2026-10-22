"""Thumbnail cache: thread-safe in-memory LRU with an optional on-disk layer."""

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PySide6.QtGui import QImage

from scantailor.core.models import PageId

# PNG text chunk holding the digest a persisted thumbnail was rendered from.
_DIGEST_TEXT_KEY = "scantailor-digest"


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
        """Return a filesystem-safe name for this page, stage and size.

        ``params_hash`` is left out: it only lives as long as the session, and
        one file per page, stage and size means a new render replaces the
        thumbnail it supersedes. The name starts with ``page_file_prefix``, so
        every file of a page can be found without knowing its stages and sizes.
        """
        view = f"{self.stage}|{self.render_size}"
        return page_file_prefix(self.page) + _sha1(view)[:16]


def page_file_prefix(page: PageId) -> str:
    """Return how the name of every thumbnail file of a page starts.

    Args:
        page (PageId): The page.

    Returns:
        str: A filesystem-safe prefix, unique to the page.
    """
    image_id = page.image_id
    return f"{_sha1(f'{image_id.file_path}|{image_id.page}|{int(page.sub_page)}')}_"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class ThumbnailCache:
    """Least-recently-used cache of thumbnail images.

    Args:
        capacity (int): Maximum number of images kept in memory.
        disk_directory (Path | None): Directory for persisted PNG thumbnails, or
            None to keep thumbnails in memory only. Created on first write.
            Each file holds the digest it was rendered with and is overwritten
            by the next render of the same page, stage and size.
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

    @property
    def disk_directory(self) -> Path | None:
        """Directory of the persisted thumbnails, or None when kept in memory only."""
        with self._lock:
            return self._disk_directory

    def set_disk_directory(self, directory: Path | None) -> None:
        """Persist thumbnails to another directory from now on.

        Args:
            directory (Path | None): The new directory, or None to stop
                persisting. Files in the old one are left where they are.
        """
        with self._lock:
            self._disk_directory = directory

    def load_from_disk(self, key: ThumbnailKey, digest: str) -> QImage | None:
        """Read a persisted thumbnail rendered with ``digest``.

        Safe to call from worker threads.

        Args:
            key (ThumbnailKey): The thumbnail identity.
            digest (str): What the thumbnail must have been rendered from; a
                file stored with another digest is stale and counts as a miss.

        Returns:
            QImage | None: The image, or None if disk caching is off, or the
                file is missing, unreadable or stale.
        """
        path = self._disk_path(key)
        if path is None or not path.is_file():
            return None
        image = QImage(str(path))
        if image.isNull() or image.text(_DIGEST_TEXT_KEY) != digest:
            return None
        return image

    def store_to_disk(self, key: ThumbnailKey, digest: str, image: QImage) -> None:
        """Persist a thumbnail, replacing the file of the same page, stage and size.

        Safe to call from worker threads.

        Args:
            key (ThumbnailKey): The thumbnail identity.
            digest (str): What the thumbnail was rendered from, written into
                the PNG for ``load_from_disk`` to compare.
            image (QImage): The image to write; not modified.
        """
        path = self._disk_path(key)
        if path is None:
            return
        image = QImage(image)
        image.setText(_DIGEST_TEXT_KEY, digest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Cannot create thumbnail cache directory {}", path.parent)
            return
        # Write-then-rename so a reader never sees a partial file.
        temporary = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp.png")
        if not image.save(str(temporary)):
            logger.warning("Cannot write thumbnail {}", temporary)
            return
        # Windows refuses the rename while another process, such as a virus
        # scanner or a second window, has the old file open.
        try:
            temporary.replace(path)
        except OSError as error:
            logger.warning("Cannot replace thumbnail {}: {}", path, error)
            temporary.unlink(missing_ok=True)

    def remove_from_disk(self, pages: Iterable[PageId]) -> None:
        """Delete every persisted thumbnail of some pages, at all stages and sizes.

        Args:
            pages (Iterable[PageId]): Pages that left the project.
        """
        directory = self.disk_directory
        if directory is None or not directory.is_dir():
            return
        for page in pages:
            for path in directory.glob(f"{page_file_prefix(page)}*.png"):
                try:
                    path.unlink()
                except OSError as error:
                    logger.warning("Cannot delete thumbnail {}: {}", path, error)

    def _disk_path(self, key: ThumbnailKey) -> Path | None:
        directory = self.disk_directory
        if directory is None:
            return None
        return directory / f"{key.file_stem()}.png"
