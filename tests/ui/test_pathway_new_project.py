from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager


def test_new_project_action_is_enabled_with_ctrl_n_shortcut(
    main_window: MainWindowManager,
):
    action = main_window.actionNewProject

    assert action.isEnabled()
    assert action.shortcut().toString() == "Ctrl+N"


def test_triggering_new_project_action_is_handled_and_keeps_window_open(
    main_window: MainWindowManager, log_messages: list[str]
):
    main_window.actionNewProject.trigger()

    assert "New project" in log_messages
    assert main_window.ui.isVisible()


def test_new_project_shortcut_dispatches_to_the_action(
    qtbot: QtBot, main_window: MainWindowManager
):
    with qtbot.waitSignal(main_window.actionNewProject.triggered, timeout=2000):
        qtbot.keyClick(
            main_window.ui,
            QtCore.Qt.Key.Key_N,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )


def test_new_project_does_not_open_any_dialog(main_window: MainWindowManager):
    main_window.actionNewProject.trigger()

    assert QtWidgets.QApplication.activeModalWidget() is None
