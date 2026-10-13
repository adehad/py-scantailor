"""Decoded source images shared by everything that shows or renders a page."""

from collections.abc import Hashable

from scantailor.core.models import Dpi, ImageId
from scantailor.core.page_source import DpiOverrides, LoadedPage, load_page
from scantailor.core.result_memo import ResultMemo

__all__ = ["SourceCache"]

# Pages kept decoded: the one on screen and the one shown before it.
_CAPACITY = 2


class SourceCache:
    """Remembers the decoded pixels of the pages shown most recently.

    A page is decoded once however many stages, renders and threads ask for it;
    threads that ask while it is being decoded wait for that decode. The pixels
    are shared, so they are read-only.

    Args:
        capacity (int): Most pages kept.
    """

    def __init__(self, capacity: int = _CAPACITY) -> None:
        """Create an empty cache; arguments are described on the class."""
        self._memo = ResultMemo[LoadedPage](capacity)

    def load(
        self,
        image_id: ImageId,
        dpi_overrides: DpiOverrides | None = None,
        *,
        keep: bool = True,
    ) -> LoadedPage:
        """Return a page's decoded source, decoding it unless it is cached.

        Args:
            image_id (ImageId): File and page to decode.
            dpi_overrides (DpiOverrides | None): Project resolutions that win over
                the file's own, as for ``load_page``.
            keep (bool): Whether a page that had to be decoded is remembered.
                Pass False for pages that are only passing through, such as
                thumbnails, so that they do not push out the page on screen.

        Returns:
            LoadedPage: Read-only pixels and the effective resolution.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImageLoadError: If the file cannot be decoded.
        """
        key = _key(image_id, dpi_overrides)
        if keep:
            return self._memo.get_or_compute(
                key, lambda: _decode(image_id, dpi_overrides)
            )
        cached = self._memo.peek(key)
        return cached if cached is not None else _decode(image_id, dpi_overrides)

    def peek(
        self, image_id: ImageId, dpi_overrides: DpiOverrides | None = None
    ) -> LoadedPage | None:
        """Return a page's decoded source if it is cached, without decoding.

        Args:
            image_id (ImageId): File and page.
            dpi_overrides (DpiOverrides | None): Project resolutions, as for ``load``.

        Returns:
            LoadedPage | None: The cached source, or None.
        """
        return self._memo.peek(_key(image_id, dpi_overrides))

    def clear(self) -> None:
        """Forget every cached page."""
        self._memo.clear()


def _decode(image_id: ImageId, dpi_overrides: DpiOverrides | None) -> LoadedPage:
    page = load_page(image_id, dpi_overrides)
    page.pixels.flags.writeable = False
    return page


def _key(image_id: ImageId, dpi_overrides: DpiOverrides | None) -> Hashable:
    """Identify a decode by the image, its project resolution and the file's state."""
    override: Dpi | None = dpi_overrides.get(image_id) if dpi_overrides else None
    try:
        stat = image_id.file_path.stat()
        version: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        version = None
    return (image_id, override, version)
