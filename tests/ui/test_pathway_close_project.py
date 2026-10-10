import pytest
from PySide6 import QtWidgets

from scantailor.app.ui.MainWindow import MainWindowManager


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


def test_close_project_trigger_logs_close_and_keeps_window_open(
    main_window: MainWindowManager, log_messages: list[str]
):
    main_window.actionCloseProject.trigger()

    assert log_messages.count("Close project") == 1
    assert main_window.ui.isVisible()


def test_close_project_without_open_project_shows_no_modal_dialog(
    main_window: MainWindowManager,
):
    main_window.actionCloseProject.trigger()

    assert QtWidgets.QApplication.activeModalWidget() is None


def test_close_project_twice_is_idempotent(
    main_window: MainWindowManager, log_messages: list[str]
):
    main_window.actionCloseProject.trigger()
    main_window.actionCloseProject.trigger()

    assert log_messages.count("Close project") == 2
    assert main_window.ui.isVisible()


def test_close_project_does_not_quit_the_application(
    main_window: MainWindowManager, log_messages: list[str]
):
    main_window.actionCloseProject.trigger()

    assert "Quit" not in log_messages


@pytest.mark.xfail(
    strict=True,
    reason="Project state is not implemented: closeProject only logs, so the "
    "window title is never reset and the project-dependent actions are never "
    "disabled (updateProjectActions/updateWindowTitle are commented out).",
)
def test_close_project_disables_save_actions_afterwards(
    main_window: MainWindowManager,
):
    main_window.actionCloseProject.trigger()

    assert not main_window.actionSaveProject.isEnabled()
    assert not main_window.actionCloseProject.isEnabled()
