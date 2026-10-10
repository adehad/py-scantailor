"""List model that renders page thumbnails lazily on a worker pool."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QImage

from scantailor.app.ui.thumbnails.cache import ThumbnailCache, ThumbnailKey
from scantailor.core.models import PageId, SubPage
from scantailor.core.project import Project

ThumbnailRenderer = Callable[[PageId, str], QImage]
"""Renders a small preview of a page at a stage. Called on worker threads."""

PageRole = Qt.ItemDataRole.UserRole
IsCurrentRole = Qt.ItemDataRole.UserRole + 1


@dataclass(frozen=True)
class PageEntry:
    """What an order option may inspect about a page.

    Args:
        page (PageId): The page.
        width (int): Source width in pixels; half the image for one page of a
            two-page image, 0 when the image size is unknown.
        height (int): Source height in pixels, 0 when unknown.
        position (int): Index in the project's natural order.
    """

    page: PageId
    width: int
    height: int
    position: int


@dataclass(frozen=True)
class PageOrderOption:
    """A named way of ordering the thumbnails.

    Orders that need stage results (C++ "decreasing deviation", "split type",
    completeness) are not built in; callers can supply them as extra options.

    Args:
        name (str): Text shown in the sort combo box.
        key (Callable[[PageEntry], Any]): Sort key. Ties keep natural order.
    """

    name: str
    key: Callable[[PageEntry], Any]


def _unknown_last(size: Callable[[PageEntry], int]) -> Callable[[PageEntry], Any]:
    """Sort by size, with unknown (zero) sizes after all known ones."""
    return lambda entry: (size(entry) == 0, size(entry))


def natural_order() -> PageOrderOption:
    """Return the project's own page order."""
    return PageOrderOption("Natural order", lambda entry: entry.position)


def order_by_width() -> PageOrderOption:
    """Return the order by increasing page width."""
    return PageOrderOption(
        "Order by increasing width", _unknown_last(lambda entry: entry.width)
    )


def order_by_height() -> PageOrderOption:
    """Return the order by increasing page height."""
    return PageOrderOption(
        "Order by increasing height", _unknown_last(lambda entry: entry.height)
    )


class _RenderSignals(QObject):
    """Carries worker results to the GUI thread through a queued connection."""

    finished = Signal(object, object)  # ThumbnailKey, QImage | None


class _RenderTask(QRunnable):
    def __init__(
        self,
        key: ThumbnailKey,
        renderer: ThumbnailRenderer,
        cache: ThumbnailCache,
        max_size: QSize,
        signals: _RenderSignals,
    ) -> None:
        super().__init__()
        self._key = key
        self._renderer = renderer
        self._cache = cache
        self._max_size = max_size
        self._signals = signals

    def run(self) -> None:
        image = self._cache.load_from_disk(self._key)
        if image is None:
            try:
                rendered = self._renderer(self._key.page, self._key.stage)
            except Exception:
                logger.exception("Thumbnail render failed for {}", self._key.page)
                self._signals.finished.emit(self._key, None)
                return
            if rendered.isNull():
                self._signals.finished.emit(self._key, None)
                return
            image = rendered.scaled(
                self._max_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._cache.store_to_disk(self._key, image)
        self._signals.finished.emit(self._key, image)


class ThumbnailModel(QAbstractListModel):
    """One row per page; thumbnails are produced on demand off the GUI thread.

    Args:
        renderer (ThumbnailRenderer): Renders a preview for (page, stage).
        cache (ThumbnailCache | None): Cache to use; a default in-memory one
            when None.
        params_hash (Callable[[PageId, str], Hashable] | None): Digest of the
            parameters that affect a page's preview at a stage. Evaluated on
            the GUI thread whenever a thumbnail is requested.
        max_thumbnail_size (QSize | None): Rendered images are scaled to fit
            this; defaults to 160x160.
        thread_count (int): Worker threads.
        parent (QObject | None): Parent object.
    """

    def __init__(
        self,
        renderer: ThumbnailRenderer,
        cache: ThumbnailCache | None = None,
        params_hash: Callable[[PageId, str], Hashable] | None = None,
        max_thumbnail_size: QSize | None = None,
        thread_count: int = 2,
        parent: QObject | None = None,
    ) -> None:
        """Initialise; arguments are described on the class."""
        super().__init__(parent)
        self._renderer = renderer
        self._cache = cache if cache is not None else ThumbnailCache()
        self._params_hash = params_hash
        self._max_size = max_thumbnail_size or QSize(160, 160)
        self._entries: list[PageEntry] = []
        self._rows: dict[PageId, int] = {}
        self._order = natural_order()
        self._stage = ""
        self._current: PageId | None = None
        self._in_flight: set[ThumbnailKey] = set()
        self._failed: set[ThumbnailKey] = set()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(thread_count)
        self._signals = _RenderSignals(self)
        self._signals.finished.connect(self._on_render_finished)

    @property
    def cache(self) -> ThumbnailCache:
        """The thumbnail cache."""
        return self._cache

    @property
    def stage(self) -> str:
        """The stage whose previews are shown."""
        return self._stage

    @property
    def current_page(self) -> PageId | None:
        """The page highlighted as current."""
        return self._current

    @property
    def max_thumbnail_size(self) -> QSize:
        """Bounding size of rendered thumbnails."""
        return self._max_size

    def pages(self) -> list[PageId]:
        """Return the pages in display order."""
        return [entry.page for entry in self._entries]

    def row_of(self, page: PageId) -> int:
        """Return the row showing a page, or -1."""
        return self._rows.get(page, -1)

    def set_project(self, project: Project | None) -> None:
        """Replace the pages with those of a project.

        Args:
            project (Project | None): The project, or None to empty the list.
        """
        self.beginResetModel()
        self._entries = _entries_for(project) if project is not None else []
        self._current = None
        self._failed.clear()
        self._sort_entries()
        self.endResetModel()

    def set_order(self, order: PageOrderOption) -> None:
        """Reorder the rows.

        Args:
            order (PageOrderOption): The order to apply.
        """
        self.beginResetModel()
        self._order = order
        self._sort_entries()
        self.endResetModel()

    def set_stage(self, stage: str) -> None:
        """Switch the stage whose previews are shown.

        Args:
            stage (str): Stage key passed to the renderer.
        """
        if stage == self._stage:
            return
        self._stage = stage
        self.invalidate()

    def set_current_page(self, page: PageId | None) -> None:
        """Highlight a page as current.

        Args:
            page (PageId | None): The page, or None for no highlight.
        """
        previous, self._current = self._current, page
        for changed in (previous, page):
            self._emit_changed(changed, [IsCurrentRole])

    def invalidate(self, page: PageId | None = None) -> None:
        """Ask views to re-request thumbnails, for one page or all.

        Call after stage parameters change. The new parameter digest yields a new
        cache key, so the preview is re-rendered.

        Args:
            page (PageId | None): The page to refresh, or None for all.
        """
        self._failed.clear()
        if page is not None:
            self._emit_changed(page, [Qt.ItemDataRole.DecorationRole])
        elif self._entries:
            self.dataChanged.emit(
                self.index(0),
                self.index(len(self._entries) - 1),
                [Qt.ItemDataRole.DecorationRole],
            )

    def wait_for_idle(self, timeout_ms: int = -1) -> bool:
        """Block until queued renders finish.

        Results reach the model through the event loop afterwards.

        Args:
            timeout_ms (int): Maximum wait, or -1 for no limit.

        Returns:
            bool: True if the pool went idle.
        """
        return self._pool.waitForDone(timeout_ms)

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        """Return the number of pages."""
        if parent is not None and parent.isValid():
            return 0
        return len(self._entries)

    def data(  # noqa: D102
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._entries):
            return None
        page = self._entries[index.row()].page
        if role == Qt.ItemDataRole.DisplayRole:
            return page_label(page)
        if role == PageRole:
            return page
        if role == IsCurrentRole:
            return page == self._current
        if role == Qt.ItemDataRole.DecorationRole:
            return self._thumbnail(page)
        return None

    def _thumbnail(self, page: PageId) -> QImage | None:
        key = self._key_for(page)
        image = self._cache.get(key)
        if image is None and key not in self._in_flight and key not in self._failed:
            self._in_flight.add(key)
            self._pool.start(
                _RenderTask(
                    key, self._renderer, self._cache, self._max_size, self._signals
                )
            )
        return image

    def _key_for(self, page: PageId) -> ThumbnailKey:
        digest = self._params_hash(page, self._stage) if self._params_hash else 0
        return ThumbnailKey(page, self._stage, digest)

    def _on_render_finished(self, key: ThumbnailKey, image: QImage | None) -> None:
        self._in_flight.discard(key)
        if image is None:
            self._failed.add(key)
            return
        self._cache.put(key, image)
        self._emit_changed(key.page, [Qt.ItemDataRole.DecorationRole])

    def _emit_changed(self, page: PageId | None, roles: list[int]) -> None:
        row = self._rows.get(page, -1) if page is not None else -1
        if row >= 0:
            self.dataChanged.emit(self.index(row), self.index(row), roles)

    def _sort_entries(self) -> None:
        self._entries.sort(key=self._order.key)
        self._rows = {entry.page: row for row, entry in enumerate(self._entries)}


def page_label(page: PageId) -> str:
    """Return the caption shown under a thumbnail."""
    name = page.image_id.file_path.name
    if page.image_id.page > 0:
        name = f"{name} [{page.image_id.page}]"
    if page.sub_page == SubPage.LEFT_PAGE:
        return f"{name} (L)"
    if page.sub_page == SubPage.RIGHT_PAGE:
        return f"{name} (R)"
    return name


def _entries_for(project: Project) -> list[PageEntry]:
    sizes: dict[PageId, tuple[int, int]] = {}
    for info in project.images:
        width, height = info.metadata.width, info.metadata.height
        if info.num_sub_pages == 2:
            width //= 2
        for sub_page in SubPage:
            sizes[PageId(image_id=info.id, sub_page=sub_page)] = (width, height)
    return [
        PageEntry(page, *sizes.get(page, (0, 0)), position)
        for position, page in enumerate(project.get_pages())
    ]
