from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import APP_TITLE, MainWindowManager, SaveChoice
from scantailor.core.image_io import list_image_pages

if TYPE_CHECKING:
    from conftest import DialogFactory, PageImages, SavePromptScript


def test_new_project_action_is_enabled_with_ctrl_n_shortcut(
    main_window: MainWindowManager,
):
    action = main_window.actionNewProject

    assert action.isEnabled()
    assert action.shortcut().toString() == "Ctrl+N"


def test_new_project_shortcut_dispatches_to_the_action(
    qtbot: QtBot,
    main_window: MainWindowManager,
    dialog_factory: DialogFactory,
    respond_to_modal: Callable[..., None],
):
    main_window.project_files_dialog_factory = dialog_factory
    respond_to_modal(accept=False)

    with qtbot.waitSignal(main_window.actionNewProject.triggered, timeout=2000):
        qtbot.keyClick(
            main_window.ui,
            QtCore.Qt.Key.Key_N,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

    assert len(dialog_factory.created) == 1


def test_window_without_project_shows_plain_title_and_disabled_project_actions(
    main_window: MainWindowManager,
):
    assert main_window.project is None
    assert main_window.ui.windowTitle() == APP_TITLE
    assert not main_window.actionSaveProject.isEnabled()
    assert not main_window.actionSaveProjectAs.isEnabled()
    assert not main_window.actionCloseProject.isEnabled()
    assert not main_window.actionAddFiles.isEnabled()


def test_new_project_opens_the_files_dialog_and_adds_one_page_per_chosen_file(
    project_window: MainWindowManager,
    dialog_factory: DialogFactory,
    page_images: PageImages,
):
    assert len(dialog_factory.created) == 1
    assert project_window.project is not None
    assert [info.id.file_path for info in project_window.project.images] == (
        page_images.paths
    )
    assert len(project_window.pages()) == 3
    assert project_window.current_page == project_window.pages()[0]


def test_new_project_records_image_size_and_output_directory(
    project_window: MainWindowManager,
    tmp_path: pathlib.Path,
    page_images: PageImages,
):
    assert project_window.project is not None
    first = project_window.project.images[0]

    assert (first.metadata.width, first.metadata.height) == (600, 800)
    assert first.metadata == list_image_pages(page_images.paths[0])[0][1]
    assert project_window.project.output_directory == tmp_path / "out"


def test_new_project_honours_rtl_layout_checkbox(
    main_window: MainWindowManager,
    dialog_factory: DialogFactory,
    respond_to_modal: Callable[..., None],
    qtbot: QtBot,
):
    def factory(parent: QtWidgets.QWidget):
        dialog = dialog_factory(parent)
        dialog._rtl_layout_cb.setChecked(True)
        return dialog

    main_window.project_files_dialog_factory = factory
    respond_to_modal(accept=True)

    main_window.actionNewProject.trigger()

    assert main_window.project is not None
    assert main_window.project.layout_direction == "RTL"


def test_cancelled_new_project_dialog_leaves_no_project(
    main_window: MainWindowManager,
    dialog_factory: DialogFactory,
    respond_to_modal: Callable[..., None],
):
    main_window.project_files_dialog_factory = dialog_factory
    respond_to_modal(accept=False)

    main_window.actionNewProject.trigger()

    assert main_window.project is None
    assert main_window.ui.windowTitle() == APP_TITLE


def test_new_project_with_unreadable_file_reports_it_and_keeps_good_pages(
    main_window: MainWindowManager,
    page_images: PageImages,
    dialog_factory: DialogFactory,
    respond_to_modal: Callable[..., None],
):
    (page_images.directory / "broken.png").write_bytes(b"not a png")
    dialog_factory.names = ["broken.png", "page_001.png"]
    reported: list[tuple[list[pathlib.Path], list[tuple[pathlib.Path, str]]]] = []

    def present(loaded, failed, _parent) -> bool:
        reported.append((loaded, failed))
        return True

    main_window.project_files_dialog_factory = dialog_factory
    main_window.load_status_presenter = present
    respond_to_modal(accept=True)

    main_window.actionNewProject.trigger()

    assert len(reported) == 1
    loaded, failed = reported[0]
    assert loaded == [page_images.directory / "page_001.png"]
    assert [path.name for path, _ in failed] == ["broken.png"]
    assert main_window.project is not None
    assert len(main_window.pages()) == 1


def test_abandoning_the_load_status_dialog_creates_no_project(
    main_window: MainWindowManager,
    page_images: PageImages,
    dialog_factory: DialogFactory,
    respond_to_modal: Callable[..., None],
):
    (page_images.directory / "broken.png").write_bytes(b"not a png")
    dialog_factory.names = ["broken.png"]
    main_window.project_files_dialog_factory = dialog_factory
    main_window.load_status_presenter = lambda *_args: False
    respond_to_modal(accept=True)

    main_window.actionNewProject.trigger()

    assert main_window.project is None


def test_new_project_without_chosen_files_creates_no_project(
    main_window: MainWindowManager,
    dialog_factory: DialogFactory,
    respond_to_modal: Callable[..., None],
    monkeypatch,
):
    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dialog_factory.names = []
    main_window.project_files_dialog_factory = dialog_factory
    respond_to_modal(accept=True)

    main_window.actionNewProject.trigger()

    assert main_window.project is None
    assert len(warnings) == 1


def test_title_shows_unnamed_project_with_modified_marker(
    project_window: MainWindowManager,
):
    assert project_window.ui.windowTitle() == f"Unnamed[*] - {APP_TITLE}"
    assert project_window.ui.isWindowModified()


def test_project_actions_are_enabled_once_a_project_exists(
    project_window: MainWindowManager,
):
    assert project_window.actionSaveProject.isEnabled()
    assert project_window.actionSaveProjectAs.isEnabled()
    assert project_window.actionCloseProject.isEnabled()
    assert project_window.actionAddFiles.isEnabled()


def test_new_project_over_modified_project_asks_before_replacing_it(
    project_window: MainWindowManager,
    save_prompt: SavePromptScript,
    dialog_factory: DialogFactory,
):
    save_prompt.answers.append(SaveChoice.CANCEL)
    before = project_window.project

    project_window.actionNewProject.trigger()

    assert save_prompt.prompts == 1
    assert project_window.project is before
    assert len(dialog_factory.created) == 1
