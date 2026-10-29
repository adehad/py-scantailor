"""Decoded source images shared by everything that shows or renders a page."""

import threading
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi, ImageId, OrthogonalDegrees
from scantailor.core.page_source import (
    DpiOverrides,
    LoadedPage,
    load_page,
    shrink_page,
)
from scantailor.core.pipeline import DETECTION_DPI
from scantailor.core.result_memo import ResultMemo
from scantailor.imageproc.geometry import rotate_orthogonal

__all__ = ["SourceCache"]

# Pages kept decoded: the one on screen and the one shown before it.
_CAPACITY = 2


class SourceCache:
    """Remembers the decoded pixels of the pages shown most recently.

    A page is decoded once however many stages, renders and threads ask for it;
    threads that ask while it is being decoded wait for that decode. The pixels
    are shared, so they are read-only.

    Each page is also kept as ``for_detection`` shrinks it, at a quarter of
    the pixels or less.

    The page last asked for through ``oriented`` is also kept turned by Fix
    Orientation's rotation, so that it is turned once per page and rotation
    rather than once per render and view.

    Args:
        capacity (int): Most pages kept.
    """

    def __init__(self, capacity: int = _CAPACITY) -> None:
        """Create an empty cache; arguments are described on the class."""
        self._memo = ResultMemo[LoadedPage](capacity)
        self._shrunk = ResultMemo[LoadedPage](capacity)
        self._turn_lock = threading.Lock()
        self._turned: _Turned | None = None

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

    def for_detection(
        self,
        image_id: ImageId,
        decoded: LoadedPage,
        dpi_overrides: DpiOverrides | None = None,
        *,
        keep: bool = True,
    ) -> LoadedPage:
        """Return a page as the batch geometry pass decodes it, at most 300 dpi.

        Args:
            image_id (ImageId): File and page.
            decoded (LoadedPage): The page from ``load``, shrunk as
                ``shrink_page`` does.
            dpi_overrides (DpiOverrides | None): Project resolutions, as for ``load``.
            keep (bool): Whether a page shrunk here is remembered, as for ``load``.

        Returns:
            LoadedPage: Read-only pixels, ``decoded`` itself when there is
                nothing to shrink.

        Raises:
            ImageLoadError: If the file cannot be decoded.
        """
        key = _key(image_id, dpi_overrides)

        def shrink() -> LoadedPage:
            page = shrink_page(decoded, image_id, dpi_overrides, DETECTION_DPI)
            page.pixels.flags.writeable = False
            return page

        if keep:
            return self._shrunk.get_or_compute(key, shrink)
        cached = self._shrunk.peek(key)
        return cached if cached is not None else shrink()

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

    def oriented(
        self, pixels: NDArray[np.uint8], degrees: OrthogonalDegrees
    ) -> NDArray[np.uint8]:
        """Return pixels turned clockwise by ``degrees``, turning them only once.

        One turned copy is kept, for the pixels and rotation asked for last;
        asking for other pixels, at any rotation, lets it go. The copy is keyed
        by the pixels' buffer, and the cache holds those pixels while it keeps
        the copy, so no other image can take over the buffer and be served a
        stale copy. A thread asking while another turns the same pixels waits
        for that copy.

        Args:
            pixels (NDArray[np.uint8]): A decoded page, or a view of one; only
                read.
            degrees (OrthogonalDegrees): Clockwise rotation.

        Returns:
            NDArray[np.uint8]: Read-only pixels; a view of ``pixels`` when
                ``degrees`` is 0.
        """
        key = _buffer_key(pixels)
        with self._turn_lock:
            turned = self._turned
            if turned is not None and turned.key == key:
                if turned.degrees == degrees:
                    return turned.pixels
            else:
                # Keep the copy only for the page being worked on.
                self._turned = None
            if degrees == 0:
                unturned = pixels.view()
                unturned.flags.writeable = False
                return unturned
            result = rotate_orthogonal(pixels, degrees)
            result.flags.writeable = False
            self._turned = _Turned(key, pixels, degrees, result)
            return result

    def clear(self) -> None:
        """Forget every cached page."""
        self._memo.clear()
        self._shrunk.clear()
        with self._turn_lock:
            self._turned = None


@dataclass(frozen=True)
class _Turned:
    """A turned copy and the pixels it was turned from, which it keeps alive."""

    key: Hashable
    source: NDArray[np.uint8]
    degrees: OrthogonalDegrees
    pixels: NDArray[np.uint8]


def _buffer_key(pixels: NDArray[np.uint8]) -> Hashable:
    """Identify the memory an array reads, so views of one page share a key."""
    address = pixels.__array_interface__["data"][0]
    return (address, pixels.shape, pixels.strides, pixels.dtype.str)


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
