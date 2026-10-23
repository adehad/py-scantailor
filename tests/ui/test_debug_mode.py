"""Tools > Debug Mode: the stage view's debug tabs and the files behind them."""

from pathlib import Path

import numpy as np
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import page_renderer
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.widgets.debug_image_tabs import DebugImageTab, DebugImageTabs
from scantailor.core.debug_images import DebugImages
from scantailor.core.stage_sequence import FilterStage

RENDER_TIMEOUT_MS = 60_000
DESKEW_LABELS = ["Main", "bw_rotated", "after_cleanup"]


def _switch_stage(window: MainWindowManager, stage: FilterStage) -> None:
    getattr(window, f"actionSwitchFilter{int(stage) + 1}").trigger()


def _tab_labels(tabs: QtWidgets.QTabWidget) -> list[str]:
    return [tabs.tabText(index) for index in range(tabs.count())]


def _debug_files(tabs: QtWidgets.QTabWidget) -> list[Path]:
    return [
        tab.path
        for index in range(tabs.count())
        if isinstance(tab := tabs.widget(index), DebugImageTab)
    ]


def _turn_debug_on_at_deskew(qtbot: QtBot, window: MainWindowManager) -> None:
    _switch_stage(window, FilterStage.DESKEW)
    with qtbot.waitSignal(
        window.renderer.debug_images_found, timeout=RENDER_TIMEOUT_MS
    ):
        window.actionDebug.trigger()
    qtbot.waitUntil(
        lambda: _tab_labels(window.debug_image_tabs) == DESKEW_LABELS,
        timeout=RENDER_TIMEOUT_MS,
    )


def _collector_with(count: int) -> DebugImages:
    dbg = DebugImages()
    for index in range(count):
        dbg.add(np.full((8, 8), index, dtype=np.uint8), f"step{index}")
    return dbg


def test_debug_mode_starts_off_and_toggling_it_stores_no_setting(
    main_window: MainWindowManager,
):
    assert not main_window.actionDebug.isChecked()
    assert not main_window.debug

    main_window.actionDebug.trigger()

    assert main_window.debug
    assert not [key for key in main_window.settings.allKeys() if "debug" in key.lower()]


def test_turning_debug_mode_on_shows_the_stage_view_first_then_its_debug_tabs(
    qtbot: QtBot, project_window: MainWindowManager
):
    _turn_debug_on_at_deskew(qtbot, project_window)

    tabs = project_window.debug_image_tabs
    assert project_window.imageFrameLayout.currentWidget() is tabs
    assert tabs.widget(0) is project_window._panels.view
    assert all(path.is_file() for path in _debug_files(tabs))


def test_selecting_a_debug_tab_shows_its_image(
    qtbot: QtBot, project_window: MainWindowManager
):
    _turn_debug_on_at_deskew(qtbot, project_window)
    tabs = project_window.debug_image_tabs

    tabs.setCurrentIndex(1)

    tab = tabs.widget(1)
    assert isinstance(tab, DebugImageTab)
    assert tab.view is not None
    assert tab.view.image is not None and not tab.view.image.isNull()


def test_turning_debug_mode_off_puts_the_view_back_and_deletes_the_files(
    qtbot: QtBot, project_window: MainWindowManager
):
    _turn_debug_on_at_deskew(qtbot, project_window)
    files = _debug_files(project_window.debug_image_tabs)

    project_window.actionDebug.trigger()

    view = project_window._panels.view
    assert project_window.imageFrameLayout.currentWidget() is view
    assert project_window.debug_image_tabs.count() == 0
    assert files and not any(path.exists() for path in files)


def test_moving_to_another_page_deletes_the_previous_pages_debug_files(
    qtbot: QtBot, project_window: MainWindowManager
):
    _turn_debug_on_at_deskew(qtbot, project_window)
    files = _debug_files(project_window.debug_image_tabs)

    with qtbot.waitSignal(
        project_window.renderer.debug_images_found, timeout=RENDER_TIMEOUT_MS
    ):
        project_window.goNextPage()

    assert files and not any(path.exists() for path in files)
    qtbot.waitUntil(
        lambda: _tab_labels(project_window.debug_image_tabs) == DESKEW_LABELS,
        timeout=RENDER_TIMEOUT_MS,
    )


def test_leaving_the_stage_deletes_its_debug_files(
    qtbot: QtBot, project_window: MainWindowManager
):
    _turn_debug_on_at_deskew(qtbot, project_window)
    files = _debug_files(project_window.debug_image_tabs)

    _switch_stage(project_window, FilterStage.PAGE_LAYOUT)

    assert files and not any(path.exists() for path in files)
    assert project_window.debug_image_tabs.main is None


def test_render_without_debug_mode_reports_no_debug_images(
    qtbot: QtBot, project_window: MainWindowManager
):
    _switch_stage(project_window, FilterStage.DESKEW)
    found: list[object] = []
    project_window.renderer.debug_images_found.connect(
        lambda _page, images: found.append(images)
    )

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        project_window._request_render()
    QtWidgets.QApplication.processEvents()

    assert found == []
    assert project_window.debug_image_tabs.count() == 0


def test_superseded_debug_renders_leave_no_files_behind(
    qtbot: QtBot, project_window: MainWindowManager, monkeypatch
):
    made: list[DebugImages] = []

    class Recorded(DebugImages):
        def __init__(self) -> None:
            super().__init__()
            made.append(self)

    monkeypatch.setattr(page_renderer, "DebugImages", Recorded)
    _switch_stage(project_window, FilterStage.DESKEW)
    project_window.debug = True
    for _ in range(3):
        project_window._request_render()
    assert project_window.renderer.wait_for_done(RENDER_TIMEOUT_MS)
    qtbot.waitUntil(lambda: project_window.debug_image_tabs.count() > 0)

    shown = set(_debug_files(project_window.debug_image_tabs))
    left = {item.path for dbg in made for item in dbg.images if item.path.exists()}
    assert left == shown


def test_debug_tabs_decode_at_most_three_images_at_a_time(qtbot: QtBot):
    tabs = DebugImageTabs()
    qtbot.addWidget(tabs)
    main = QtWidgets.QLabel("view")
    tabs.show_images(main, _collector_with(5))

    for index in range(1, 6):
        tabs.setCurrentIndex(index)

    live = [
        tabs.widget(index).view is not None  # type: ignore[union-attr]
        for index in range(1, 6)
    ]
    assert live == [False, False, True, True, True]


def test_new_debug_images_replace_the_old_ones_and_delete_their_files(qtbot: QtBot):
    tabs = DebugImageTabs()
    qtbot.addWidget(tabs)
    main = QtWidgets.QLabel("view")
    old = _collector_with(2)
    old_files = [item.path for item in old.images]
    tabs.show_images(main, old)

    tabs.show_images(main, _collector_with(1))

    assert _tab_labels(tabs) == ["Main", "step0"]
    assert not any(path.exists() for path in old_files)


def test_rerender_keeps_the_debug_tab_the_user_was_looking_at(qtbot: QtBot):
    tabs = DebugImageTabs()
    qtbot.addWidget(tabs)
    main = QtWidgets.QLabel("view")
    tabs.show_images(main, _collector_with(3))
    tabs.setCurrentIndex(2)

    tabs.show_images(main, _collector_with(3))

    assert tabs.currentIndex() == 2


def test_taking_the_view_back_leaves_no_tabs_and_no_files(qtbot: QtBot):
    tabs = DebugImageTabs()
    qtbot.addWidget(tabs)
    main = QtWidgets.QLabel("view")
    images = _collector_with(2)
    directory = images.directory
    tabs.show_images(main, images)

    taken = tabs.take_main()
    qtbot.addWidget(main)

    assert taken is main
    assert tabs.count() == 0
    assert directory is not None and not directory.exists()


def test_failed_debug_render_deletes_its_files(
    qtbot: QtBot, project_window: MainWindowManager, monkeypatch
):
    made: list[DebugImages] = []

    def failing_render(*_args, debug_images: DebugImages | None = None, **_kwargs):
        assert debug_images is not None
        made.append(debug_images)
        debug_images.add(np.zeros((4, 4), dtype=np.uint8), "partial")
        raise ValueError("broken page")

    _switch_stage(project_window, FilterStage.DESKEW)
    assert project_window.renderer.wait_for_done(RENDER_TIMEOUT_MS)
    monkeypatch.setattr(page_renderer, "render_page_with_model", failing_render)
    project_window.debug = True

    with qtbot.waitSignal(
        project_window.renderer.render_failed, timeout=RENDER_TIMEOUT_MS
    ):
        project_window._request_render()

    assert made and made[0].images == ()
    assert project_window.debug_image_tabs.count() == 0


def test_closing_the_project_deletes_the_shown_debug_files(
    qtbot: QtBot, project_window: MainWindowManager
):
    _turn_debug_on_at_deskew(qtbot, project_window)
    files = _debug_files(project_window.debug_image_tabs)
    project_window.modified = False

    project_window.closeProject()

    assert files and not any(path.exists() for path in files)
    assert project_window.debug_image_tabs.count() == 0


def test_releasing_the_renderer_closes_debug_images_still_in_flight(
    qtbot: QtBot, project_window: MainWindowManager
):
    _switch_stage(project_window, FilterStage.DESKEW)
    project_window.debug = True
    project_window._request_render()
    assert project_window.renderer.wait_for_done(RENDER_TIMEOUT_MS)
    in_flight = set(project_window.renderer._signals.debug_in_flight)
    directories = [images.directory for images in in_flight]

    project_window.renderer.release_debug_images()
    QtWidgets.QApplication.processEvents()

    assert in_flight
    assert not any(d is not None and d.exists() for d in directories)
    assert project_window.debug_image_tabs.count() == 0
