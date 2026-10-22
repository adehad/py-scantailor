"""List model that renders page thumbnails lazily on a worker pool."""

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, replace
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
from scantailor.core.models import ImageId, PageId, SubPage, page_name
from scantailor.core.project import Project

ThumbnailRenderer = Callable[[PageId, str], QImage]
"""Renders a small preview of a page at a stage. Called on worker threads."""

PageRole = Qt.ItemDataRole.UserRole
IsCurrentRole = Qt.ItemDataRole.UserRole + 1
IncompleteRole = Qt.ItemDataRole.UserRole + 2

IncompletenessProvider = Callable[[PageId, str], bool]
"""Tells whether a page has no usable result at a stage yet."""


@dataclass(frozen=True)
class PageEntry:
    """What an order option may inspect about a page.

    Args:
        page (PageId): The page; when the list shows images, the image's first
            page, standing for the whole image.
        width (int): Source width in pixels; half the image for one page of a
            two-page image, 0 when the image size is unknown.
        height (int): Source height in pixels, 0 when unknown.
        position (int): Index in the project's natural order.
        incomplete (bool): The page has no usable result at the shown stage; the
            list marks it with a question mark.
    """

    page: PageId
    width: int
    height: int
    position: int
    incomplete: bool = False


@dataclass(frozen=True)
class PageOrderOption:
    """A named way of ordering the thumbnails.

    Orders that need stage results (deviation, split type, completeness) are
    built by ``stage_orders`` for the shown stage.

    Args:
        name (str): Text shown in the sort combo box.
        key (Callable[[PageEntry], Any]): Sort key. Ties keep natural order.
        prepare (Callable[[Sequence[PageEntry]], None] | None): Called with every
            entry, in natural order, before the keys are read, for orders whose
            key depends on all pages such as a deviation from the mean.
    """

    name: str
    key: Callable[[PageEntry], Any]
    prepare: Callable[[Sequence[PageEntry]], None] | None = None


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
    skipped = Signal(object)  # ThumbnailKey


class _RenderTask(QRunnable):
    def __init__(
        self,
        key: ThumbnailKey,
        renderer: ThumbnailRenderer,
        cache: ThumbnailCache,
        max_size: QSize,
        signals: _RenderSignals,
        is_wanted: Callable[[ThumbnailKey], bool],
    ) -> None:
        super().__init__()
        self._is_wanted = is_wanted
        self._key = key
        self._renderer = renderer
        self._cache = cache
        self._max_size = max_size
        self._signals = signals

    def run(self) -> None:
        with logger.contextualize(page=page_name(self._key.page)):
            self._render()

    def _render(self) -> None:
        if not self._is_wanted(self._key):
            self._signals.skipped.emit(self._key)
            return
        image = self._cache.load_from_disk(self._key)
        if image is None:
            try:
                rendered = self._renderer(self._key.page, self._key.stage)
            except Exception:
                logger.exception("Thumbnail render failed")
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
    """One row per page, or per image; thumbnails are rendered off the GUI thread.

    At stages that work on whole images the model lists images: each row's page
    is the image's first page, and every method that takes a page finds the row
    of any page of that image.

    Args:
        renderer (ThumbnailRenderer): Renders a preview for (page, stage).
        cache (ThumbnailCache | None): Cache to use; a default in-memory one
            when None.
        params_hash (Callable[[PageId, str], Hashable] | None): Digest of the
            parameters that affect a page's preview at a stage. Evaluated on
            the GUI thread whenever a thumbnail is requested.
        is_incomplete (IncompletenessProvider | None): Tells whether a page has
            no usable result at a stage. Evaluated on the GUI thread.
        render_size (QSize | None): Rendered images are scaled to fit this, the
            resolution thumbnails are kept at; defaults to 160x160.
        thread_count (int): Worker threads.
        parent (QObject | None): Parent object.
    """

    def __init__(
        self,
        renderer: ThumbnailRenderer,
        cache: ThumbnailCache | None = None,
        params_hash: Callable[[PageId, str], Hashable] | None = None,
        render_size: QSize | None = None,
        thread_count: int = 2,
        parent: QObject | None = None,
        is_incomplete: IncompletenessProvider | None = None,
    ) -> None:
        """Initialise; arguments are described on the class."""
        super().__init__(parent)
        self._renderer = renderer
        self._cache = cache if cache is not None else ThumbnailCache()
        self._params_hash = params_hash
        self._is_incomplete = is_incomplete
        self._reversed = False
        self._render_size = render_size or QSize(160, 160)
        self._project: Project | None = None
        self._lists_images = False
        self._entries: list[PageEntry] = []
        self._rows: dict[PageId, int] = {}
        self._image_pages: dict[ImageId, list[PageId]] = {}
        self._order = natural_order()
        self._stage = ""
        self._current: PageId | None = None
        self._in_flight: set[ThumbnailKey] = set()
        self._failed: set[ThumbnailKey] = set()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(thread_count)
        self._signals = _RenderSignals(self)
        self._signals.finished.connect(self._on_render_finished)
        self._signals.skipped.connect(self._on_render_skipped)

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
    def render_size(self) -> QSize:
        """Bounding size of rendered thumbnails."""
        return self._render_size

    def set_render_size(self, size: QSize) -> None:
        """Keep thumbnails at another resolution and render them again.

        Args:
            size (QSize): The new bounding size of rendered thumbnails.
        """
        if size == self._render_size:
            return
        self._render_size = QSize(size)
        self.invalidate()

    @property
    def lists_images(self) -> bool:
        """Whether each row stands for a whole image rather than one page."""
        return self._lists_images

    def pages(self) -> list[PageId]:
        """Return the page of each row, in display order."""
        return [entry.page for entry in self._entries]

    def row_of(self, page: PageId) -> int:
        """Return the row showing a page, or -1."""
        return self._rows.get(self._row_page(page), -1)

    def pages_of_row(self, page: PageId) -> list[PageId]:
        """Return every page the row showing ``page`` stands for.

        Args:
            page (PageId): Any page of the project.

        Returns:
            list[PageId]: All pages of the image when the list shows images, in
                the project's order; otherwise just ``page``.
        """
        if not self._lists_images:
            return [page]
        return list(self._image_pages.get(page.image_id, [page]))

    def _row_page(self, page: PageId) -> PageId:
        """Return the page that identifies the row showing ``page``."""
        if not self._lists_images:
            return page
        return self._image_pages.get(page.image_id, [page])[0]

    def set_project(self, project: Project | None) -> None:
        """Replace the pages with those of a project.

        Args:
            project (Project | None): The project, or None to empty the list.
        """
        self.beginResetModel()
        self._project = project
        self._current = None
        self._failed.clear()
        self._fill_entries()
        self.endResetModel()

    def _fill_entries(self) -> None:
        """List the project's pages or images, marked and in the chosen order."""
        project = self._project
        pages = project.get_pages() if project is not None else []
        self._image_pages = {}
        for page in pages:
            self._image_pages.setdefault(page.image_id, []).append(page)
        self._entries = (
            _entries_for(project, self._lists_images) if project is not None else []
        )
        self._mark_incomplete_pages()
        self._sort_entries()

    def set_order(self, order: PageOrderOption, reverse: bool | None = None) -> None:
        """Reorder the rows.

        Args:
            order (PageOrderOption): The order to apply.
            reverse (bool | None): Put the rows in the opposite order; keeps the
                current direction when None.
        """
        self.beginResetModel()
        self._order = order
        if reverse is not None:
            self._reversed = reverse
        self._sort_entries()
        self.endResetModel()

    @property
    def reversed_order(self) -> bool:
        """Whether the rows are in the opposite of the chosen order."""
        return self._reversed

    def set_incompleteness_provider(
        self, provider: IncompletenessProvider | None
    ) -> None:
        """Choose how pages without a result at the shown stage are found.

        Args:
            provider (IncompletenessProvider | None): The check, or None to
                treat every page as complete.
        """
        self._is_incomplete = provider
        self.refresh()

    def refresh(self, page: PageId | None = None) -> None:
        """Look again at which pages are incomplete and keep the rows in order.

        Rows move without a reset, so selection and current page stay.

        Args:
            page (PageId | None): The only page to look at again, or None for all.
        """
        changed = self._mark_incomplete_pages(page)
        for entry_page in changed:
            self._emit_changed(entry_page, [IncompleteRole])
        self._resort()

    def set_stage(self, stage: str, lists_images: bool = False) -> None:
        """Switch the stage whose previews are shown.

        Switching between listing pages and listing images resets the model, so
        views lose their selection and current index.

        Args:
            stage (str): Stage key passed to the renderer.
            lists_images (bool): Show one row per image instead of one per page.
        """
        if stage == self._stage and lists_images == self._lists_images:
            return
        self._stage = stage
        if lists_images != self._lists_images:
            self.beginResetModel()
            self._lists_images = lists_images
            self._fill_entries()
            self.endResetModel()
        else:
            self._mark_incomplete_pages()
            self._resort()
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
        self.refresh(page)
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
            return image_label(page) if self._lists_images else page_label(page)
        if role == PageRole:
            return page
        if role == IsCurrentRole:
            return self._current is not None and page == self._row_page(self._current)
        if role == IncompleteRole:
            return self._entries[index.row()].incomplete
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
                    key,
                    self._renderer,
                    self._cache,
                    self._render_size,
                    self._signals,
                    self._is_wanted,
                )
            )
        return image

    def _key_for(self, page: PageId) -> ThumbnailKey:
        digest = self._params_hash(page, self._stage) if self._params_hash else 0
        size = self._render_size
        return ThumbnailKey(page, self._stage, digest, (size.width(), size.height()))

    def _is_wanted(self, key: ThumbnailKey) -> bool:
        """Tell a queued task whether its stage is still the one being shown."""
        return key.stage == self._stage

    def _on_render_skipped(self, key: ThumbnailKey) -> None:
        self._in_flight.discard(key)

    def _on_render_finished(self, key: ThumbnailKey, image: QImage | None) -> None:
        self._in_flight.discard(key)
        if image is None:
            self._failed.add(key)
            return
        self._cache.put(key, image)
        self._emit_changed(key.page, [Qt.ItemDataRole.DecorationRole])

    def _emit_changed(self, page: PageId | None, roles: list[int]) -> None:
        row = self.row_of(page) if page is not None else -1
        if row >= 0:
            self.dataChanged.emit(self.index(row), self.index(row), roles)

    def _mark_incomplete_pages(self, page: PageId | None = None) -> list[PageId]:
        """Update the incomplete flag of the entries; return the pages that changed."""
        provider, stage = self._is_incomplete, self._stage
        wanted = self._row_page(page) if page is not None else None
        changed: list[PageId] = []
        for row, entry in enumerate(self._entries):
            if wanted is not None and entry.page != wanted:
                continue
            incomplete = provider(entry.page, stage) if provider is not None else False
            if incomplete != entry.incomplete:
                self._entries[row] = replace(entry, incomplete=incomplete)
                changed.append(entry.page)
        return changed

    def _ordered(self) -> list[PageEntry]:
        """Return the entries in the chosen order; ties keep natural order."""
        ordered = sorted(self._entries, key=lambda entry: entry.position)
        if self._order.prepare is not None:
            self._order.prepare(ordered)
        ordered.sort(key=self._order.key, reverse=self._reversed)
        return ordered

    def _sort_entries(self) -> None:
        self._entries = self._ordered()
        self._rows = {entry.page: row for row, entry in enumerate(self._entries)}

    def _resort(self) -> None:
        """Move rows to where the chosen order puts them, keeping selections."""
        ordered = self._ordered()
        if [entry.page for entry in ordered] == [e.page for e in self._entries]:
            self._entries = ordered
            return
        self.layoutAboutToBeChanged.emit()
        old_pages = [
            (index, self._entries[index.row()].page)
            for index in self.persistentIndexList()
        ]
        self._entries = ordered
        self._rows = {entry.page: row for row, entry in enumerate(self._entries)}
        self.changePersistentIndexList(
            [index for index, _page in old_pages],
            [self.index(self._rows[page]) for _index, page in old_pages],
        )
        self.layoutChanged.emit()


def image_label(page: PageId) -> str:
    """Return the caption shown under the thumbnail of the image a page is on."""
    name = page.image_id.file_path.name
    if page.image_id.page > 0:
        name = f"{name} [{page.image_id.page}]"
    return name


def page_label(page: PageId) -> str:
    """Return the caption shown under a thumbnail."""
    name = image_label(page)
    if page.sub_page == SubPage.LEFT_PAGE:
        return f"{name} (L)"
    if page.sub_page == SubPage.RIGHT_PAGE:
        return f"{name} (R)"
    return name


def _entries_for(project: Project, lists_images: bool) -> list[PageEntry]:
    sizes: dict[PageId, tuple[int, int]] = {}
    for info in project.images:
        width, height = info.metadata.width, info.metadata.height
        if info.num_sub_pages == 2 and not lists_images:
            width //= 2
        for sub_page in SubPage:
            sizes[PageId(image_id=info.id, sub_page=sub_page)] = (width, height)
    pages = project.get_image_pages() if lists_images else project.get_pages()
    return [
        PageEntry(page, *sizes.get(page, (0, 0)), position)
        for position, page in enumerate(pages)
    ]
