from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from scantailor.app.ui.MainWindow import APP_TITLE, MainWindowManager, SaveChoice

if TYPE_CHECKING:
    from conftest import FileDialogStub, SavePromptScript


def test_close_project_action_is_present_and_labelled(main_window: MainWindowManager):
    assert "close" in main_window.actionCloseProject.text().lower()


def test_close_project_action_is_reachable_from_the_file_menu(
    main_window: MainWindowManager,
):
    menus_with_action = [
        menu.title()
        for menu in main_window.ui.findChildren(QtWidgets.QMenu)
        if main_window.actionCloseProject in menu.actions()
    ]

    assert menus_with_action, "actionCloseProject is not in any menu"
    assert any("file" in title.lower() for title in menus_with_action)


def test_close_without_open_project_is_a_noop_without_prompt(
    main_window: MainWindowManager, save_prompt: SavePromptScript
):
    main_window.closeProject()

    assert save_prompt.prompts == 0
    assert QtWidgets.QApplication.activeModalWidget() is None
    assert main_window.ui.isVisible()


def test_close_project_with_unsaved_changes_prompts_and_cancel_keeps_project(
    project_window: MainWindowManager, save_prompt: SavePromptScript
):
    save_prompt.answers.append(SaveChoice.CANCEL)
    project = project_window.project

    project_window.actionCloseProject.trigger()

    assert save_prompt.prompts == 1
    assert project_window.project is project
    assert project_window.modified
    assert project_window.actionSaveProject.isEnabled()


def test_close_project_discard_drops_the_project_and_resets_window(
    project_window: MainWindowManager, save_prompt: SavePromptScript
):
    save_prompt.answers.append(SaveChoice.DISCARD)

    project_window.actionCloseProject.trigger()

    assert project_window.project is None
    assert project_window.current_page is None
    assert project_window.projectFile == ""
    assert project_window.ui.windowTitle() == APP_TITLE
    assert not project_window.ui.isWindowModified()
    assert project_window.ui.isVisible()


def test_close_project_disables_save_actions_afterwards(
    project_window: MainWindowManager, save_prompt: SavePromptScript
):
    save_prompt.answers.append(SaveChoice.DISCARD)

    project_window.actionCloseProject.trigger()

    assert not project_window.actionSaveProject.isEnabled()
    assert not project_window.actionSaveProjectAs.isEnabled()
    assert not project_window.actionCloseProject.isEnabled()


def test_close_project_save_choice_writes_the_file_then_closes(
    project_window: MainWindowManager,
    save_prompt: SavePromptScript,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    target = tmp_path / "closing.ScanTailor"
    file_dialogs.save_file = str(target)
    save_prompt.answers.append(SaveChoice.SAVE)

    project_window.actionCloseProject.trigger()

    assert target.exists()
    assert project_window.project is None


def test_close_project_save_choice_with_cancelled_file_dialog_keeps_project(
    project_window: MainWindowManager,
    save_prompt: SavePromptScript,
    file_dialogs: FileDialogStub,
):
    file_dialogs.save_file = ""
    save_prompt.answers.append(SaveChoice.SAVE)

    project_window.actionCloseProject.trigger()

    assert project_window.project is not None


def test_close_saved_unchanged_project_does_not_prompt(
    project_window: MainWindowManager,
    save_prompt: SavePromptScript,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    file_dialogs.save_file = str(tmp_path / "clean.ScanTailor")
    project_window.actionSaveProjectAs.trigger()

    project_window.actionCloseProject.trigger()

    assert save_prompt.prompts == 0
    assert project_window.project is None


def test_close_project_twice_is_idempotent(
    project_window: MainWindowManager, save_prompt: SavePromptScript
):
    save_prompt.answers.append(SaveChoice.DISCARD)

    project_window.closeProject()
    project_window.closeProject()

    assert save_prompt.prompts == 1
    assert project_window.project is None


def test_close_project_does_not_quit_the_application(
    project_window: MainWindowManager, save_prompt: SavePromptScript
):
    save_prompt.answers.append(SaveChoice.DISCARD)

    project_window.actionCloseProject.trigger()

    assert not project_window.closing


def test_closing_the_window_with_unsaved_changes_can_be_cancelled(
    project_window: MainWindowManager, save_prompt: SavePromptScript
):
    save_prompt.answers.append(SaveChoice.CANCEL)

    closed = project_window.ui.close()

    assert not closed
    assert project_window.ui.isVisible()
    assert project_window.project is not None


def test_closing_the_window_discarding_changes_closes_it(
    project_window: MainWindowManager, save_prompt: SavePromptScript
):
    save_prompt.answers.append(SaveChoice.DISCARD)

    closed = project_window.ui.close()

    assert closed
    assert not project_window.ui.isVisible()
