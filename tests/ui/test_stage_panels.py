"""The options widget and image view that MainWindow shows for the current stage."""

import threading

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import page_renderer
from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.options_widgets import (
    deskew,
    fix_orientation,
    output,
    page_layout,
    page_split,
    select_content,
)
from scantailor.core.stage_sequence import FilterStage
from scantailor.filters.page_layout.params import ContentRect, ContentSize

RENDER_TIMEOUT_MS = 60_000

OPTIONS_WIDGET_TYPES = {
    FilterStage.FIX_ORIENTATION: fix_orientation.OptionsWidget,
    FilterStage.PAGE_SPLIT: page_split.OptionsWidget,
    FilterStage.DESKEW: deskew.OptionsWidget,
    FilterStage.SELECT_CONTENT: select_content.OptionsWidget,
    FilterStage.PAGE_LAYOUT: page_layout.OptionsWidget,
    FilterStage.OUTPUT: output.OptionsWidget,
}


def _switch_stage(window: MainWindowManager, stage: FilterStage) -> None:
    getattr(window, f"actionSwitchFilter{int(stage) + 1}").trigger()


def _render_current_page(qtbot: QtBot, window: MainWindowManager) -> None:
    with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
        window._request_render()


def _options_container(window: MainWindowManager) -> QtWidgets.QWidget:
    container = window.ui.findChild(QtWidgets.QWidget, "filterOptions")
    assert container is not None
    return container


@pytest.mark.parametrize("stage", list(FilterStage))
def test_switching_stage_shows_that_stages_options_widget(
    main_window: MainWindowManager, stage: FilterStage
):
    _switch_stage(main_window, stage)

    container = _options_container(main_window)
    expected = OPTIONS_WIDGET_TYPES[stage]
    assert container.findChildren(expected)
    for other in set(OPTIONS_WIDGET_TYPES.values()) - {expected}:
        assert not container.findChildren(other)


def test_switching_stage_replaces_the_image_view(main_window: MainWindowManager):
    first_view = main_window._panels.view

    _switch_stage(main_window, FilterStage.DESKEW)

    new_view = main_window._panels.view
    assert new_view is not first_view
    assert main_window.imageViewFrame.findChildren(ImageViewBase) == [new_view]


@pytest.mark.parametrize(
    "stage",
    [
        pytest.param(stage, marks=pytest.mark.slow)
        if stage is FilterStage.OUTPUT
        else stage
        for stage in FilterStage
    ],
)
def test_image_view_receives_the_rendered_first_page_at_each_stage(
    qtbot: QtBot, project_window: MainWindowManager, stage: FilterStage
):
    _switch_stage(project_window, stage)
    if stage is FilterStage.OUTPUT:
        # Output only renders once every page's size is known.
        layout = project_window.stages.page_layout_filter.settings
        for page in project_window.pages():
            layout.update_content_size_and_get_aggregate(
                page,
                ContentRect(width=600, height=800),
                ContentRect(width=500, height=700),
                ContentSize(width_mm=100, height_mm=150),
            )

    _render_current_page(qtbot, project_window)

    image = project_window._panels.view.image
    assert image is not None
    assert not image.isNull()
    assert image.width() > 0
    assert image.height() > 0


def test_changing_the_deskew_angle_updates_params_and_marks_project_modified(
    qtbot: QtBot, project_window: MainWindowManager
):
    _switch_stage(project_window, FilterStage.DESKEW)
    page = project_window.current_page
    project_window.modified = False
    spin_box = _options_container(project_window).findChild(QtWidgets.QDoubleSpinBox)

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        spin_box.setValue(1.5)

    params = project_window.stages.deskew_filter.get_params(page)
    assert params.deskew_angle_deg == pytest.approx(1.5)
    assert params.is_manual()
    assert project_window.modified


def test_editing_an_option_rerenders_the_current_page(
    qtbot: QtBot,
    project_window: MainWindowManager,
    monkeypatch: pytest.MonkeyPatch,
):
    _switch_stage(project_window, FilterStage.DESKEW)
    _render_current_page(qtbot, project_window)
    stages_run: list[FilterStage] = []
    real_render = page_renderer.render_page

    def spy(page_id, sequence, end_stage):
        stages_run.append(end_stage)
        return real_render(page_id, sequence, end_stage)

    monkeypatch.setattr(page_renderer, "render_page", spy)
    spin_box = _options_container(project_window).findChild(QtWidgets.QDoubleSpinBox)

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        spin_box.setValue(-2.0)

    # The edited page's thumbnail is rendered again as well, to the same stage.
    assert stages_run
    assert set(stages_run) == {FilterStage.DESKEW}
    project_window.thumbnails.model.wait_for_idle()


def test_options_widget_follows_the_current_page(
    qtbot: QtBot, project_window: MainWindowManager
):
    _switch_stage(project_window, FilterStage.DESKEW)
    pages = project_window.pages()
    project_window.stages.deskew_filter.settings.set_params(
        pages[1],
        project_window.stages.deskew_filter.get_params(pages[1]).with_manual_angle(3.0),
    )
    spin_box = _options_container(project_window).findChild(QtWidgets.QDoubleSpinBox)

    project_window.goNextPage()

    assert spin_box.value() == pytest.approx(3.0)
    project_window.renderer.wait_for_done()


def test_changing_page_cancels_a_slow_render(
    qtbot: QtBot,
    project_window: MainWindowManager,
    monkeypatch: pytest.MonkeyPatch,
):
    entered = threading.Event()
    release = threading.Event()
    real_process_page = page_renderer.pipeline.process_page
    first_page = project_window.pages()[0]
    delivered = []

    def slow_process_page(image, page_id, *args, **kwargs):
        if page_id == first_page:
            entered.set()
            assert release.wait(timeout=30)
        return real_process_page(image, page_id, *args, **kwargs)

    project_window.renderer.wait_for_done()
    monkeypatch.setattr(page_renderer.pipeline, "process_page", slow_process_page)
    project_window.renderer.render_done.connect(
        lambda page_id, _image: delivered.append(page_id)
    )
    qtbot.wait(50)
    delivered.clear()

    project_window._request_render()
    assert entered.wait(timeout=10)
    second_page = project_window.pages()[1]
    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        project_window.goNextPage()
    release.set()
    project_window.renderer.wait_for_done()
    qtbot.wait(50)

    assert delivered == [second_page]
    assert project_window.current_page == second_page


def test_switching_stage_cancels_a_pending_render(
    qtbot: QtBot,
    project_window: MainWindowManager,
    monkeypatch: pytest.MonkeyPatch,
):
    entered = threading.Event()
    release = threading.Event()
    real_process_page = page_renderer.pipeline.process_page
    delivered = []

    def slow_process_page(image, page_id, sequence, options):
        if options.end_stage == FilterStage.FIX_ORIENTATION:
            entered.set()
            assert release.wait(timeout=30)
        return real_process_page(image, page_id, sequence, options)

    project_window.renderer.wait_for_done()
    monkeypatch.setattr(page_renderer.pipeline, "process_page", slow_process_page)
    project_window.renderer.render_done.connect(
        lambda _page, image: delivered.append(image)
    )
    project_window._request_render()
    assert entered.wait(timeout=10)

    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        _switch_stage(project_window, FilterStage.PAGE_SPLIT)
    release.set()
    project_window.renderer.wait_for_done()
    qtbot.wait(50)

    assert len(delivered) == 1


def test_a_page_that_cannot_be_decoded_does_not_break_the_window(
    qtbot: QtBot, project_window: MainWindowManager, log_messages: list[str]
):
    project_window.pages()[0].image_id.file_path.write_bytes(b"not an image")

    with qtbot.waitSignal(
        project_window.renderer.render_failed, timeout=RENDER_TIMEOUT_MS
    ):
        project_window._request_render()

    assert any("could not be shown" in message for message in log_messages)


def test_an_unexpected_render_error_is_reported_as_a_failure(
    qtbot: QtBot,
    project_window: MainWindowManager,
    monkeypatch: pytest.MonkeyPatch,
):
    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(page_renderer, "render_page", explode)

    with qtbot.waitSignal(
        project_window.renderer.render_failed, timeout=RENDER_TIMEOUT_MS
    ) as blocker:
        project_window._request_render()

    assert blocker.args[1] == "boom"


def test_set_thumbnail_widget_replaces_the_placeholder_view(
    main_window: MainWindowManager,
):
    thumbnails = QtWidgets.QListWidget()

    main_window.set_thumbnail_widget(thumbnails)

    assert thumbnails.parent() is main_window.thumbView.parent()
    assert main_window.thumbView.isHidden()
    assert thumbnails.isVisible()


def test_set_thumbnail_widget_can_be_called_again(main_window: MainWindowManager):
    first = QtWidgets.QListWidget()
    second = QtWidgets.QListWidget()

    main_window.set_thumbnail_widget(first)
    main_window.set_thumbnail_widget(second)

    assert second.isVisible()
    assert first.parent() is None


def test_page_selected_makes_that_page_current(project_window: MainWindowManager):
    target = project_window.pages()[2]

    project_window.page_selected(target)

    assert project_window.current_page == target
    project_window.renderer.wait_for_done()


def test_page_selected_ignores_pages_of_other_projects(
    project_window: MainWindowManager,
):
    other = project_window.pages()[0].model_copy(
        update={
            "image_id": project_window.pages()[0].image_id.model_copy(
                update={
                    "file_path": project_window.pages()[0].image_id.file_path.with_name(
                        "x.png"
                    )
                }
            )
        }
    )

    project_window.page_selected(other)

    assert project_window.current_page == project_window.pages()[0]


@pytest.mark.parametrize("stage", list(FilterStage))
def test_showing_another_page_does_not_mark_the_project_modified(
    project_window: MainWindowManager, stage: FilterStage
):
    _switch_stage(project_window, stage)
    project_window.modified = False

    project_window.page_selected(project_window.pages()[1])
    project_window.renderer.wait_for_done()

    assert not project_window.modified


def test_selecting_a_thumbnail_changes_the_current_page_and_image_view(
    qtbot: QtBot, project_window: MainWindowManager
):
    pages = project_window.pages()
    _render_current_page(qtbot, project_window)
    first_image = project_window._panels.view.image
    thumbnails = project_window.thumbnails
    assert thumbnails.pages() == pages

    target = pages[2]
    index = thumbnails.model.index(thumbnails.model.row_of(target))
    with qtbot.waitSignal(
        project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
    ):
        thumbnails.view.setCurrentIndex(index)

    assert project_window.current_page == target
    view = project_window._panels.view
    qtbot.waitUntil(
        lambda: view.image is not None and view.image is not first_image,
        timeout=RENDER_TIMEOUT_MS,
    )


def test_thumbnails_are_rendered_for_the_pages_of_the_project(
    qtbot: QtBot, project_window: MainWindowManager
):
    model = project_window.thumbnails.model
    first = model.index(0)

    def thumbnail_arrived() -> None:
        icon = model.data(first, QtCore.Qt.ItemDataRole.DecorationRole)
        assert icon is not None
        assert not icon.isNull()

    qtbot.waitUntil(thumbnail_arrived, timeout=RENDER_TIMEOUT_MS)


def test_changing_the_current_page_selects_its_thumbnail(
    project_window: MainWindowManager,
):
    target = project_window.pages()[1]

    project_window.current_page = target

    assert project_window.thumbnails.current_page() == target
    project_window.renderer.wait_for_done()
