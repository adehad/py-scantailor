"""Recent projects in the File menu and prompting only for real unsaved changes."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager, SaveChoice
from scantailor.core.image_io import list_image_pages
from scantailor.core.project import Project
from scantailor.core.recent_projects import DEFAULT_MAX_ITEMS
from scantailor.core.stage_sequence import FilterStage

if TYPE_CHECKING:
    from conftest import FileDialogStub, PageImages, SavePromptScript

RENDER_TIMEOUT_MS = 60_000
DESKEW_ANGLE = 1.5


def _write_project(page_images: PageImages, project_file: pathlib.Path) -> pathlib.Path:
    project = Project(output_directory=project_file.parent / "out")
    for path in page_images.paths:
        for image_id, metadata in list_image_pages(path):
            project.add_image(image_id, metadata)
    project.save(project_file)
    return project_file


def _open(
    window: MainWindowManager, file_dialogs: FileDialogStub, project_file: pathlib.Path
) -> None:
    file_dialogs.open_file = str(project_file)
    window.actionOpenProject.trigger()
    assert window.projectFile == str(project_file)


def _recent_entries(window: MainWindowManager) -> list[str]:
    return [action.toolTip() for action in window.menuOpenRecent.actions()]


def _set_deskew_angle(qtbot: QtBot, window: MainWindowManager, angle: float) -> None:
    window.actionSwitchFilter3.trigger()
    container = window.ui.findChild(QtWidgets.QWidget, "filterOptions")
    spin_box = container.findChild(QtWidgets.QDoubleSpinBox)
    with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
        spin_box.setValue(angle)


@pytest.fixture
def saved_project(page_images: PageImages, tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_project(page_images, tmp_path / "book.ScanTailor")


def test_recent_menu_sits_directly_below_open_project_and_starts_disabled(
    main_window: MainWindowManager,
):
    file_menu = main_window.ui.findChild(QtWidgets.QMenu, "menuFile")
    actions = file_menu.actions()

    following = actions[actions.index(main_window.actionOpenProject) + 1]

    assert following.menu() is main_window.menuOpenRecent
    assert not main_window.menuOpenRecent.isEnabled()


def test_recent_settings_are_isolated_from_the_user_configuration(
    main_window: MainWindowManager, isolated_settings: pathlib.Path
):
    assert pathlib.Path(main_window.settings.fileName()).is_relative_to(
        isolated_settings
    )


def test_opening_a_project_lists_it_in_the_recent_menu(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)

    assert _recent_entries(main_window) == [str(saved_project)]
    assert main_window.menuOpenRecent.isEnabled()


def test_saving_a_project_under_a_new_name_lists_it_in_the_recent_menu(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    file_dialogs.save_file = str(tmp_path / "fresh.ScanTailor")

    project_window.actionSaveProjectAs.trigger()

    assert _recent_entries(project_window) == [str(tmp_path / "fresh.ScanTailor")]


def test_most_recently_opened_project_comes_first_without_duplicates(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    first = _write_project(page_images, tmp_path / "a.ScanTailor")
    second = _write_project(page_images, tmp_path / "b.ScanTailor")

    _open(main_window, file_dialogs, first)
    _open(main_window, file_dialogs, second)
    _open(main_window, file_dialogs, first)

    assert _recent_entries(main_window) == [str(first), str(second)]


def test_recent_list_is_capped_at_the_most_recent_seven(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    files = [
        _write_project(page_images, tmp_path / f"p{n}.ScanTailor")
        for n in range(DEFAULT_MAX_ITEMS + 2)
    ]

    for file in files:
        _open(main_window, file_dialogs, file)

    assert DEFAULT_MAX_ITEMS == 7
    assert _recent_entries(main_window) == [str(f) for f in reversed(files[2:])]


def test_recent_list_survives_restarting_the_application(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)

    restarted = MainWindowManager()
    qtbot.addWidget(restarted.ui)

    assert _recent_entries(restarted) == [str(saved_project)]


def test_projects_whose_file_disappeared_are_dropped_when_the_menu_opens(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    kept = _write_project(page_images, tmp_path / "kept.ScanTailor")
    gone = _write_project(page_images, tmp_path / "gone.ScanTailor")
    _open(main_window, file_dialogs, gone)
    _open(main_window, file_dialogs, kept)
    gone.unlink()

    main_window.menuOpenRecent.aboutToShow.emit()

    assert _recent_entries(main_window) == [str(kept)]


def test_choosing_a_recent_project_opens_it(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)
    main_window.actionCloseProject.trigger()
    assert main_window.project is None

    main_window.menuOpenRecent.actions()[0].trigger()

    assert main_window.project is not None
    assert main_window.projectFile == str(saved_project)


def test_failing_to_open_a_project_does_not_add_it_to_the_recent_list(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_args: None)
    broken = tmp_path / "broken.ScanTailor"
    broken.write_text("not a project", encoding="utf-8")
    file_dialogs.open_file = str(broken)

    main_window.actionOpenProject.trigger()

    assert _recent_entries(main_window) == []


def test_opening_and_closing_a_project_without_edits_does_not_prompt(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)
    main_window.goNextPage()

    main_window.actionCloseProject.trigger()

    assert save_prompt.prompts == 0
    assert main_window.project is None


def test_browsing_through_the_stages_does_not_prompt_on_close(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)
    for stage in FilterStage:
        getattr(main_window, f"actionSwitchFilter{int(stage) + 1}").trigger()

    main_window.actionCloseProject.trigger()

    assert save_prompt.prompts == 0


def test_editing_the_deskew_angle_then_closing_prompts(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)
    _set_deskew_angle(qtbot, main_window, DESKEW_ANGLE)
    assert main_window.modified
    save_prompt.answers.append(SaveChoice.CANCEL)

    main_window.actionCloseProject.trigger()

    assert save_prompt.prompts == 1
    assert main_window.project is not None


def test_saving_the_edit_makes_closing_silent_again(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)
    _set_deskew_angle(qtbot, main_window, DESKEW_ANGLE)

    main_window.actionSaveProject.trigger()
    main_window.actionCloseProject.trigger()

    assert save_prompt.prompts == 0
    assert main_window.project is None


def test_choosing_save_in_the_prompt_writes_the_edit_to_the_file(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)
    _set_deskew_angle(qtbot, main_window, DESKEW_ANGLE)
    save_prompt.answers.append(SaveChoice.SAVE)

    main_window.actionCloseProject.trigger()

    assert "deskew" in Project.load(saved_project).stage_params


def test_a_modified_flag_without_any_content_change_does_not_prompt(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)
    main_window.modified = True

    main_window.actionCloseProject.trigger()

    assert save_prompt.prompts == 0
    assert main_window.project is None


def test_opening_another_project_after_an_edit_prompts_but_not_after_none(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    page_images: PageImages,
    saved_project: pathlib.Path,
    tmp_path: pathlib.Path,
):
    other = _write_project(page_images, tmp_path / "other.ScanTailor")
    _open(main_window, file_dialogs, saved_project)
    _open(main_window, file_dialogs, other)
    assert save_prompt.prompts == 0

    _set_deskew_angle(qtbot, main_window, DESKEW_ANGLE)
    save_prompt.answers.append(SaveChoice.DISCARD)
    _open(main_window, file_dialogs, saved_project)

    assert save_prompt.prompts == 1


def test_closing_the_window_applies_the_same_rule(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    saved_project: pathlib.Path,
):
    _open(main_window, file_dialogs, saved_project)

    assert main_window.ui.close()
    assert save_prompt.prompts == 0
