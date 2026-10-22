"""Tests for the standalone page thumbnail list."""

import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox
from pytestqt.qtbot import QtBot

from scantailor.app.ui.thumbnails import (
    PageOrderOption,
    ThumbnailCache,
    ThumbnailKey,
    ThumbnailListWidget,
)
from scantailor.app.ui.thumbnails.model import IsCurrentRole
from scantailor.core.models import ImageId, PageId
from scantailor.core.project import ImageMetadata, Project
from scantailor.core.settings import ApplicationSettings

WIDTHS = [900, 300, 600, 450]


class CountingRenderer:
    """Renderer that records its calls and the threads they ran on."""

    def __init__(self) -> None:
        self.calls: list[tuple[PageId, str]] = []
        self.threads: set[int] = set()
        self._lock = threading.Lock()

    def __call__(self, page: PageId, stage: str) -> QImage:
        with self._lock:
            self.calls.append((page, stage))
            self.threads.add(threading.get_ident())
        image = QImage(40, 60, QImage.Format.Format_RGB32)
        image.fill(QColor("steelblue"))
        return image


@pytest.fixture
def project() -> Project:
    """Four single pages with distinct widths and heights."""
    project = Project()
    for number, width in enumerate(WIDTHS, start=1):
        project.add_image(
            ImageId(file_path=Path(f"page_{number}.png")),
            ImageMetadata(width=width, height=1000 - 100 * number),
        )
    return project


@pytest.fixture
def renderer() -> CountingRenderer:
    return CountingRenderer()


@pytest.fixture
def widget(
    qtbot: QtBot, project: Project, renderer: CountingRenderer
) -> ThumbnailListWidget:
    thumbnails = ThumbnailListWidget(renderer, project, stage="page_split")
    qtbot.addWidget(thumbnails)
    thumbnails.resize(240, 700)
    thumbnails.show()
    qtbot.wait_exposed(thumbnails)
    return thumbnails


def click(
    qtbot: QtBot,
    widget: ThumbnailListWidget,
    page: PageId,
    modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    qtbot.mouseClick(
        widget.view.viewport(),
        Qt.MouseButton.LeftButton,
        modifier,
        widget.item_rect(page).center(),
    )


def wait_for_renders(qtbot: QtBot, widget: ThumbnailListWidget, count: int) -> None:
    qtbot.waitUntil(lambda: len(widget.model.cache) >= count, timeout=5000)


def test_one_item_per_page(widget: ThumbnailListWidget, project: Project) -> None:
    assert widget.model.rowCount() == len(WIDTHS)
    assert widget.pages() == project.get_pages()


def test_empty_project_yields_no_items(
    qtbot: QtBot, renderer: CountingRenderer
) -> None:
    thumbnails = ThumbnailListWidget(renderer, Project())
    qtbot.addWidget(thumbnails)
    assert thumbnails.model.rowCount() == 0
    assert thumbnails.selected_pages() == []


def test_click_selects_page_and_emits_page_selected(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    target = widget.pages()[2]
    with qtbot.waitSignal(widget.page_selected) as page_signal:
        click(qtbot, widget, target)
    assert page_signal.args == [target]
    assert widget.selected_pages() == [target]
    assert widget.current_page() == target


def test_selection_changed_reports_selected_pages(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    target = widget.pages()[1]
    with qtbot.waitSignal(widget.selection_changed) as selection_signal:
        click(qtbot, widget, target)
    assert selection_signal.args == [[target]]


def test_shift_click_selects_range(qtbot: QtBot, widget: ThumbnailListWidget) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    click(qtbot, widget, pages[2], Qt.KeyboardModifier.ShiftModifier)
    assert widget.selected_pages() == pages[:3]


def test_ctrl_click_adds_single_pages(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    click(qtbot, widget, pages[2], Qt.KeyboardModifier.ControlModifier)
    assert widget.selected_pages() == [pages[0], pages[2]]


def test_ctrl_click_adds_pages_with_selection_mode_off(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    widget.set_selection_mode_enabled(False)
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    click(qtbot, widget, pages[2], Qt.KeyboardModifier.ControlModifier)
    assert widget.selected_pages() == [pages[0], pages[2]]


def test_plain_click_selects_only_the_clicked_page_with_selection_mode_off(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    ApplicationSettings.get_instance().show_canceling_selection_question = False
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    click(qtbot, widget, pages[2], Qt.KeyboardModifier.ControlModifier)
    with qtbot.waitSignal(widget.selection_reset):
        click(qtbot, widget, pages[3])
    assert widget.selected_pages() == [pages[3]]


def test_plain_click_adds_a_page_in_selection_mode(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    widget.set_selection_mode_enabled(True)
    with qtbot.assertNotEmitted(widget.selection_reset):
        click(qtbot, widget, pages[2])
    assert widget.selected_pages() == [pages[0], pages[2]]
    assert widget.current_page() == pages[2]


def test_plain_click_removes_a_selected_page_in_selection_mode(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    widget.set_selection_mode_enabled(True)
    click(qtbot, widget, pages[1])
    click(qtbot, widget, pages[2])
    click(qtbot, widget, pages[1])
    assert widget.selected_pages() == [pages[0], pages[2]]
    assert widget.current_page() == pages[2]


def test_removing_the_current_page_makes_the_nearest_selected_page_current(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    widget.set_selection_mode_enabled(True)
    click(qtbot, widget, pages[3])
    click(qtbot, widget, pages[1])
    with qtbot.waitSignal(widget.page_selected) as announced:
        click(qtbot, widget, pages[1])
    assert announced.args == [pages[0]]
    assert widget.selected_pages() == [pages[0], pages[3]]


def test_removing_the_current_page_prefers_the_selected_page_before_it_on_a_tie(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[1])
    widget.set_selection_mode_enabled(True)
    click(qtbot, widget, pages[3])
    click(qtbot, widget, pages[2])
    click(qtbot, widget, pages[2])
    assert widget.current_page() == pages[1]


def test_the_last_selected_page_stays_selected_when_clicked_again(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[1])
    widget.set_selection_mode_enabled(True)
    click(qtbot, widget, pages[1])
    click(qtbot, widget, pages[1], Qt.KeyboardModifier.ControlModifier)
    assert widget.selected_pages() == [pages[1]]
    assert widget.current_page() == pages[1]


def test_shift_click_extends_from_the_current_page_keeping_other_pages(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    click(qtbot, widget, pages[2], Qt.KeyboardModifier.ControlModifier)
    click(qtbot, widget, pages[3], Qt.KeyboardModifier.ShiftModifier)
    assert widget.selected_pages() == [pages[0], pages[2], pages[3]]
    assert widget.current_page() == pages[3]


def press_and_drag(
    widget: ThumbnailListWidget,
    start: PageId,
    stops: list[PageId],
    modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Press on one thumbnail, move over others in turn, then release.

    Each point is taken just before its event, as pressing may scroll the list.
    """
    viewport = widget.view.viewport()
    left = Qt.MouseButton.LeftButton
    last = len(stops)
    for step, page in enumerate([start, *stops]):
        point = widget.item_rect(page).center()
        kind = QEvent.Type.MouseButtonPress if step == 0 else QEvent.Type.MouseMove
        events = [(kind, left)]
        if step == last:
            events.append((QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton))
        for event_kind, buttons in events:
            QApplication.sendEvent(
                viewport,
                QMouseEvent(
                    event_kind,
                    QPointF(point),
                    QPointF(viewport.mapToGlobal(point)),
                    Qt.MouseButton.NoButton
                    if event_kind == QEvent.Type.MouseMove
                    else left,
                    buttons,
                    modifier,
                ),
            )


def test_press_and_drag_selects_the_pages_in_between_with_selection_mode_off(
    widget: ThumbnailListWidget,
) -> None:
    pages = widget.pages()
    press_and_drag(widget, pages[0], [pages[2]])
    assert widget.selected_pages() == pages[:3]
    assert widget.current_page() == pages[0]


def test_dragging_back_shrinks_the_dragged_range(
    widget: ThumbnailListWidget,
) -> None:
    pages = widget.pages()
    press_and_drag(widget, pages[0], [pages[3], pages[1]])
    assert widget.selected_pages() == pages[:2]


def test_press_and_drag_in_selection_mode_adds_to_the_selection(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[3])
    widget.set_selection_mode_enabled(True)
    press_and_drag(widget, pages[0], [pages[1]])
    assert widget.selected_pages() == [pages[0], pages[1], pages[3]]


def test_dragging_from_a_removed_page_removes_the_range_but_keeps_one_page(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    click(qtbot, widget, pages[0])
    for page in pages[1:]:
        click(qtbot, widget, page, Qt.KeyboardModifier.ControlModifier)
    press_and_drag(widget, pages[1], [pages[2]], Qt.KeyboardModifier.ControlModifier)
    assert widget.selected_pages() == [pages[0], pages[3]]
    press_and_drag(widget, pages[0], [pages[3]], Qt.KeyboardModifier.ControlModifier)
    assert widget.selected_pages() == [pages[3]]


def test_set_current_page_announces_a_selection_reset(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    with qtbot.waitSignal(widget.selection_reset):
        widget.set_current_page(widget.pages()[2])
    assert widget.selected_pages() == [widget.pages()[2]]


def test_sort_by_width_reorders_increasing(
    widget: ThumbnailListWidget, project: Project
) -> None:
    natural = project.get_pages()
    widget.set_order(1)
    expected = [
        page
        for _, page in sorted(zip(WIDTHS, natural, strict=True), key=lambda p: p[0])
    ]
    assert widget.pages() == expected


def test_sort_by_height_reorders_increasing(
    widget: ThumbnailListWidget, project: Project
) -> None:
    widget.set_order(2)
    assert widget.pages() == list(reversed(project.get_pages()))


@pytest.mark.parametrize("order", [1, 2])
def test_sort_puts_pages_of_unknown_size_last(
    qtbot: QtBot, order: int, renderer: CountingRenderer
) -> None:
    project = Project()
    project.add_image(ImageId(file_path=Path("unknown.png")), ImageMetadata())
    project.add_image(
        ImageId(file_path=Path("known.png")), ImageMetadata(width=500, height=700)
    )
    thumbnails = ThumbnailListWidget(renderer, project, stage="page_split")
    qtbot.addWidget(thumbnails)
    thumbnails.set_order(order)
    assert thumbnails.pages() == list(reversed(project.get_pages()))


def test_returning_to_natural_order_restores_project_order(
    widget: ThumbnailListWidget, project: Project
) -> None:
    widget.set_order(1)
    widget.set_order(0)
    assert widget.pages() == project.get_pages()


def test_sort_keeps_selection_and_does_not_emit_page_selected(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    target = widget.pages()[0]
    click(qtbot, widget, target)
    emitted: list[PageId] = []
    widget.page_selected.connect(emitted.append)
    widget.set_order(1)
    assert widget.selected_pages() == [target]
    assert widget.current_page() == target
    assert emitted == []


def test_extra_orders_appear_after_built_ins(
    qtbot: QtBot, project: Project, renderer: CountingRenderer
) -> None:
    reverse = PageOrderOption("Reverse", lambda entry: -entry.position)
    thumbnails = ThumbnailListWidget(renderer, project, extra_orders=[reverse])
    qtbot.addWidget(thumbnails)
    assert thumbnails.order_combo.itemText(3) == "Reverse"
    thumbnails.order_combo.setCurrentIndex(3)
    assert thumbnails.pages() == list(reversed(project.get_pages()))


def test_current_page_highlight_follows_set_current_page(
    widget: ThumbnailListWidget,
) -> None:
    target = widget.pages()[3]
    widget.set_current_page(target)
    assert widget.current_page() == target
    assert widget.selected_pages() == [target]
    flags = [
        widget.model.data(widget.model.index(row), IsCurrentRole)
        for row in range(widget.model.rowCount())
    ]
    assert flags == [False, False, False, True]


def test_set_current_page_emits_page_selected(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    target = widget.pages()[1]
    with qtbot.waitSignal(widget.page_selected) as page_signal:
        widget.set_current_page(target)
    assert page_signal.args == [target]


def test_scroll_to_current_brings_last_page_into_view(
    qtbot: QtBot, project: Project, renderer: CountingRenderer
) -> None:
    thumbnails = _long_list(qtbot, project, renderer)
    last = thumbnails.pages()[-1]
    thumbnails.set_current_page(last)
    viewport = thumbnails.view.viewport().rect()
    assert viewport.contains(thumbnails.item_rect(last).center())


def _long_list(
    qtbot: QtBot, project: Project, renderer: CountingRenderer
) -> ThumbnailListWidget:
    for number in range(5, 30):
        project.add_image(
            ImageId(file_path=Path(f"page_{number}.png")),
            ImageMetadata(width=500, height=700),
        )
    thumbnails = ThumbnailListWidget(renderer, project)
    qtbot.addWidget(thumbnails)
    thumbnails.resize(240, 400)
    thumbnails.show()
    qtbot.wait_exposed(thumbnails)
    return thumbnails


def test_programmatic_navigation_does_not_scroll_when_following_is_off(
    qtbot: QtBot, project: Project, renderer: CountingRenderer
) -> None:
    thumbnails = _long_list(qtbot, project, renderer)
    thumbnails.set_follow_current(False)

    thumbnails.set_current_page(thumbnails.pages()[-1])

    assert thumbnails.current_page() == thumbnails.pages()[-1]
    assert thumbnails.view.verticalScrollBar().value() == 0


def test_programmatic_navigation_scrolls_when_following_is_on(
    qtbot: QtBot, project: Project, renderer: CountingRenderer
) -> None:
    thumbnails = _long_list(qtbot, project, renderer)
    thumbnails.set_follow_current(True)
    last = thumbnails.pages()[-1]

    thumbnails.set_current_page(last)

    viewport = thumbnails.view.viewport().rect()
    assert viewport.contains(thumbnails.item_rect(last).center())


def test_unknown_page_is_ignored(widget: ThumbnailListWidget) -> None:
    widget.set_current_page(PageId(image_id=ImageId(file_path=Path("missing.png"))))
    assert widget.current_page() is None


def test_renders_happen_off_gui_thread_and_fill_items(
    qtbot: QtBot, widget: ThumbnailListWidget, renderer: CountingRenderer
) -> None:
    with qtbot.waitSignal(widget.model.dataChanged, timeout=5000):
        widget.view.viewport().repaint()
    wait_for_renders(qtbot, widget, len(WIDTHS))
    assert renderer.threads
    assert threading.get_ident() not in renderer.threads
    image = widget.model.data(widget.model.index(0), Qt.ItemDataRole.DecorationRole)
    assert isinstance(image, QImage)
    assert not image.isNull()


def test_cache_hit_avoids_second_render(
    qtbot: QtBot, widget: ThumbnailListWidget, renderer: CountingRenderer
) -> None:
    widget.view.viewport().repaint()
    wait_for_renders(qtbot, widget, len(WIDTHS))
    rendered = len(renderer.calls)
    assert rendered == len(WIDTHS)

    widget.model.wait_for_idle()
    widget.view.viewport().repaint()
    widget.invalidate()
    widget.view.viewport().repaint()
    widget.model.wait_for_idle()
    qtbot.wait(50)
    assert len(renderer.calls) == rendered


def test_changed_params_hash_triggers_rerender(
    qtbot: QtBot, project: Project, renderer: CountingRenderer
) -> None:
    digest = {"value": 1}
    thumbnails = ThumbnailListWidget(
        renderer, project, stage="deskew", params_hash=lambda _p, _s: digest["value"]
    )
    qtbot.addWidget(thumbnails)
    thumbnails.resize(240, 700)
    thumbnails.show()
    qtbot.wait_exposed(thumbnails)
    thumbnails.view.viewport().repaint()
    wait_for_renders(qtbot, thumbnails, len(WIDTHS))

    digest["value"] = 2
    thumbnails.invalidate()
    thumbnails.view.viewport().repaint()
    qtbot.waitUntil(lambda: len(renderer.calls) >= 2 * len(WIDTHS), timeout=5000)


def test_stage_change_renders_that_stage(
    qtbot: QtBot, widget: ThumbnailListWidget, renderer: CountingRenderer
) -> None:
    widget.view.viewport().repaint()
    wait_for_renders(qtbot, widget, len(WIDTHS))
    widget.set_stage("output")
    widget.view.viewport().repaint()
    qtbot.waitUntil(
        lambda: any(stage == "output" for _p, stage in renderer.calls), timeout=5000
    )


def test_failed_render_is_not_retried_in_a_loop(qtbot: QtBot, project: Project) -> None:
    attempts: list[PageId] = []

    def failing(page: PageId, stage: str) -> QImage:
        attempts.append(page)
        raise RuntimeError("boom")

    thumbnails = ThumbnailListWidget(failing, project)
    qtbot.addWidget(thumbnails)
    thumbnails.resize(240, 700)
    thumbnails.show()
    qtbot.wait_exposed(thumbnails)
    thumbnails.view.viewport().repaint()
    thumbnails.model.wait_for_idle()
    qtbot.wait(100)
    thumbnails.view.viewport().repaint()
    thumbnails.model.wait_for_idle()
    assert len(attempts) == len(WIDTHS)


def test_disk_cache_survives_new_widget(
    qtbot: QtBot, project: Project, tmp_path: Path
) -> None:
    first = CountingRenderer()
    one = ThumbnailListWidget(
        first, project, stage="s", cache_directory=tmp_path / "cache"
    )
    qtbot.addWidget(one)
    one.resize(240, 700)
    one.show()
    qtbot.wait_exposed(one)
    one.view.viewport().repaint()
    wait_for_renders(qtbot, one, len(WIDTHS))
    assert len(list((tmp_path / "cache").glob("*.png"))) == len(WIDTHS)

    second = CountingRenderer()
    two = ThumbnailListWidget(
        second, project, stage="s", cache_directory=tmp_path / "cache"
    )
    qtbot.addWidget(two)
    two.resize(240, 700)
    two.show()
    qtbot.wait_exposed(two)
    two.view.viewport().repaint()
    wait_for_renders(qtbot, two, len(WIDTHS))
    assert second.calls == []


class TestThumbnailCache:
    """Unit tests for the LRU cache."""

    @staticmethod
    def key(number: int) -> ThumbnailKey:
        return ThumbnailKey(
            PageId(image_id=ImageId(file_path=Path(f"{number}.png"))), "s"
        )

    def test_evicts_least_recently_used(self) -> None:
        cache = ThumbnailCache(capacity=2)
        cache.put(self.key(1), QImage(1, 1, QImage.Format.Format_RGB32))
        cache.put(self.key(2), QImage(1, 1, QImage.Format.Format_RGB32))
        assert cache.get(self.key(1)) is not None
        cache.put(self.key(3), QImage(1, 1, QImage.Format.Format_RGB32))
        assert self.key(2) not in cache
        assert self.key(1) in cache
        assert self.key(3) in cache

    def test_rejects_zero_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            ThumbnailCache(capacity=0)

    def test_different_params_hash_is_a_different_key(self) -> None:
        page = PageId(image_id=ImageId(file_path=Path("a.png")))
        assert ThumbnailKey(page, "s", 1) != ThumbnailKey(page, "s", 2)
        assert (
            ThumbnailKey(page, "s", 1).file_stem()
            != ThumbnailKey(page, "s", 2).file_stem()
        )

    def test_disk_round_trip(self, tmp_path: Path) -> None:
        cache = ThumbnailCache(disk_directory=tmp_path / "d")
        image = QImage(8, 4, QImage.Format.Format_RGB32)
        image.fill(QColor("red"))
        cache.store_to_disk(self.key(1), image)
        loaded = cache.load_from_disk(self.key(1))
        assert loaded is not None
        assert (loaded.width(), loaded.height()) == (8, 4)

    def test_disk_miss_without_directory(self) -> None:
        cache = ThumbnailCache()
        assert cache.load_from_disk(self.key(1)) is None


def test_queued_renders_of_a_stage_that_is_no_longer_shown_are_skipped(
    qtbot: QtBot, project: Project
) -> None:
    from scantailor.app.ui.thumbnails.model import ThumbnailModel

    started = threading.Event()
    release = threading.Event()
    calls: list[tuple[PageId, str]] = []

    def blocking(page: PageId, stage: str) -> QImage:
        calls.append((page, stage))
        started.set()
        release.wait(5)
        image = QImage(40, 60, QImage.Format.Format_RGB32)
        image.fill(QColor("steelblue"))
        return image

    model = ThumbnailModel(blocking, thread_count=1)
    model.set_project(project)
    model.set_stage("page_split")
    for row in range(model.rowCount()):
        model.data(model.index(row), Qt.ItemDataRole.DecorationRole)
    assert started.wait(5)

    model.set_stage("output")
    release.set()
    model.wait_for_idle()
    qtbot.wait(50)

    assert [stage for _page, stage in calls] == ["page_split"]


class TestImageList:
    """At the stages that work on whole images, a spread has one thumbnail."""

    @pytest.fixture
    def spread_project(self) -> Project:
        """A two-page spread between two single pages."""
        project = Project()
        for name in ("first.png", "spread.png", "last.png"):
            project.add_image(
                ImageId(file_path=Path(name)), ImageMetadata(width=800, height=600)
            )
        project.images[1].num_sub_pages = 2
        return project

    @pytest.fixture
    def thumbnails(
        self, qtbot: QtBot, spread_project: Project, renderer: CountingRenderer
    ) -> ThumbnailListWidget:
        thumbnails = ThumbnailListWidget(renderer, spread_project, stage="2")
        qtbot.addWidget(thumbnails)
        return thumbnails

    @staticmethod
    def _halves(project: Project) -> tuple[PageId, PageId]:
        left, right = project.get_pages()[1:3]
        return left, right

    def test_a_spread_has_one_thumbnail_captioned_with_its_file_name(
        self, thumbnails: ThumbnailListWidget, spread_project: Project
    ) -> None:
        thumbnails.set_stage("1", lists_images=True)

        assert thumbnails.pages() == spread_project.get_image_pages()
        model = thumbnails.model
        assert model.data(model.index(1)) == "spread.png"

    def test_leaving_the_image_stages_lists_both_halves_again(
        self, thumbnails: ThumbnailListWidget, spread_project: Project
    ) -> None:
        thumbnails.set_stage("1", lists_images=True)

        thumbnails.set_stage("2")

        assert thumbnails.pages() == spread_project.get_pages()
        assert thumbnails.model.data(thumbnails.model.index(1)) == "spread.png (L)"

    def test_either_half_finds_the_spread_thumbnail(
        self, thumbnails: ThumbnailListWidget, spread_project: Project
    ) -> None:
        left, right = self._halves(spread_project)

        thumbnails.set_stage("1", lists_images=True)

        assert thumbnails.model.row_of(left) == thumbnails.model.row_of(right) == 1
        assert thumbnails.model.pages_of_row(right) == [left, right]

    def test_an_image_thumbnail_is_as_wide_as_the_whole_spread(
        self, thumbnails: ThumbnailListWidget
    ) -> None:
        thumbnails.set_stage("1", lists_images=True)

        thumbnails.set_order(1)

        assert [entry.width for entry in thumbnails.model._entries] == [800] * 3

    def test_the_right_half_stays_current_through_the_image_stages(
        self, thumbnails: ThumbnailListWidget, spread_project: Project
    ) -> None:
        _left, right = self._halves(spread_project)
        thumbnails.set_current_page(right)
        emitted: list[PageId] = []
        thumbnails.page_selected.connect(emitted.append)

        thumbnails.set_stage("1", lists_images=True)
        on_images = thumbnails.current_page()
        highlighted = thumbnails.model.data(thumbnails.model.index(1), IsCurrentRole)
        thumbnails.set_stage("2")

        assert on_images == right
        assert highlighted
        assert thumbnails.current_page() == right
        assert thumbnails.view.currentIndex().row() == 2
        assert emitted == []

    def test_selecting_one_half_selects_the_spread_and_then_both_halves(
        self, thumbnails: ThumbnailListWidget, spread_project: Project
    ) -> None:
        left, right = self._halves(spread_project)
        thumbnails.select_pages([right])

        thumbnails.set_stage("1", lists_images=True)
        on_images = thumbnails.selected_pages()
        thumbnails.set_stage("2")

        assert on_images == [left]
        assert thumbnails.selected_pages() == [left, right]

    def test_removing_a_spread_thumbnail_removes_both_halves(
        self, thumbnails: ThumbnailListWidget, spread_project: Project
    ) -> None:
        left, right = self._halves(spread_project)
        thumbnails.set_stage("1", lists_images=True)
        thumbnails.set_current_page(left)
        requested: list[list[PageId]] = []
        thumbnails.remove_pages_requested.connect(requested.append)

        thumbnails._request_removal()

        assert requested == [[left, right]]


class TestCancelingSelectionQuestion:
    @pytest.fixture
    def answers(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Answer the question with the next queued reply and record its text."""
        replies: list[tuple[QMessageBox.StandardButton, bool]] = []
        asked: list[str] = []

        def answer(box: QMessageBox) -> int:
            asked.append(box.text())
            button, dont_ask_again = replies.pop(0)
            box.checkBox().setChecked(dont_ask_again)
            return int(button.value)

        monkeypatch.setattr(QMessageBox, "exec", answer)
        self.replies = replies
        return asked

    @staticmethod
    def _select_three(qtbot: QtBot, widget: ThumbnailListWidget) -> list[PageId]:
        pages = widget.pages()
        click(qtbot, widget, pages[0])
        for page in pages[1:3]:
            click(qtbot, widget, page, Qt.KeyboardModifier.ControlModifier)
        return pages

    def test_cancelling_the_question_keeps_the_selection_and_current_page(
        self, qtbot: QtBot, widget: ThumbnailListWidget, answers: list[str]
    ) -> None:
        pages = self._select_three(qtbot, widget)
        self.replies.append((QMessageBox.StandardButton.Cancel, False))

        click(qtbot, widget, pages[3])

        assert answers == ["3 pages selection are going to be canceled. Continue?"]
        assert widget.selected_pages() == pages[:3]
        assert widget.current_page() == pages[2]

    def test_accepting_the_question_selects_only_the_clicked_page(
        self, qtbot: QtBot, widget: ThumbnailListWidget, answers: list[str]
    ) -> None:
        pages = self._select_three(qtbot, widget)
        self.replies.append((QMessageBox.StandardButton.Ok, False))

        click(qtbot, widget, pages[1])

        assert widget.selected_pages() == [pages[1]]
        assert ApplicationSettings.get_instance().show_canceling_selection_question

    def test_dont_show_again_with_ok_turns_the_question_off(
        self, qtbot: QtBot, widget: ThumbnailListWidget, answers: list[str]
    ) -> None:
        pages = self._select_three(qtbot, widget)
        self.replies.append((QMessageBox.StandardButton.Ok, True))
        click(qtbot, widget, pages[3])
        for page in pages[:2]:
            click(qtbot, widget, page, Qt.KeyboardModifier.ControlModifier)

        click(qtbot, widget, pages[2])

        assert len(answers) == 1
        assert widget.selected_pages() == [pages[2]]
        assert not ApplicationSettings.get_instance().show_canceling_selection_question

    def test_dont_show_again_with_cancel_keeps_asking(
        self, qtbot: QtBot, widget: ThumbnailListWidget, answers: list[str]
    ) -> None:
        pages = self._select_three(qtbot, widget)
        self.replies.append((QMessageBox.StandardButton.Cancel, True))

        click(qtbot, widget, pages[3])

        assert ApplicationSettings.get_instance().show_canceling_selection_question

    def test_nothing_is_asked_when_the_setting_is_off(
        self, qtbot: QtBot, widget: ThumbnailListWidget, answers: list[str]
    ) -> None:
        ApplicationSettings.get_instance().show_canceling_selection_question = False
        pages = self._select_three(qtbot, widget)

        click(qtbot, widget, pages[3])

        assert answers == []
        assert widget.selected_pages() == [pages[3]]

    def test_nothing_is_asked_for_one_selected_page(
        self, qtbot: QtBot, widget: ThumbnailListWidget, answers: list[str]
    ) -> None:
        pages = widget.pages()
        click(qtbot, widget, pages[0])

        click(qtbot, widget, pages[1])

        assert answers == []

    def test_nothing_is_asked_for_the_two_halves_of_one_image(
        self, qtbot: QtBot, renderer: CountingRenderer, answers: list[str]
    ) -> None:
        image = ImageId(file_path=Path("spread.png"))
        project = Project()
        project.add_image(image, ImageMetadata(width=2000, height=1400))
        project.images[0].num_sub_pages = 2
        thumbnails = ThumbnailListWidget(renderer, project, stage="3")
        qtbot.addWidget(thumbnails)
        left, right = thumbnails.pages()
        thumbnails.select_pages([left, right])

        assert thumbnails.canceling_selection_accepted()
        assert answers == []

    def test_ctrl_click_never_asks(
        self, qtbot: QtBot, widget: ThumbnailListWidget, answers: list[str]
    ) -> None:
        pages = self._select_three(qtbot, widget)

        click(qtbot, widget, pages[3], Qt.KeyboardModifier.ControlModifier)

        assert answers == []
        assert widget.selected_pages() == pages
