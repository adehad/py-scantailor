import pathlib

from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager


def test_open_project_action_has_ctrl_o_shortcut(main_window: MainWindowManager):
    assert main_window.actionOpenProject.shortcut().toString() == "Ctrl+O"


def test_open_project_asks_for_scantailor_project_file_parented_to_window(
    main_window: MainWindowManager, file_dialogs
):
    main_window.actionOpenProject.trigger()

    assert len(file_dialogs.calls) == 1
    name, args, _kwargs = file_dialogs.calls[0]
    assert name == "getOpenFileName"
    assert args[0] is main_window.ui
    assert "*.ScanTailor" in args[3]


def test_open_project_with_chosen_file_logs_the_path(
    main_window: MainWindowManager,
    file_dialogs,
    tmp_path: pathlib.Path,
    log_messages: list[str],
):
    project_file = tmp_path / "book.ScanTailor"
    project_file.write_text("{}")
    file_dialogs.open_file = str(project_file)

    main_window.actionOpenProject.trigger()

    assert f"Opening project: {project_file}" in log_messages
    assert main_window.ui.isVisible()


def test_open_project_cancelled_dialog_does_nothing(
    main_window: MainWindowManager, file_dialogs, log_messages: list[str]
):
    file_dialogs.open_file = ""

    main_window.actionOpenProject.trigger()

    assert len(file_dialogs.calls) == 1
    assert not any(m.startswith("Opening project") for m in log_messages)
    assert main_window.ui.isVisible()


def test_open_project_shortcut_dispatches_to_the_action(
    qtbot: QtBot, main_window: MainWindowManager, file_dialogs
):
    with qtbot.waitSignal(main_window.actionOpenProject.triggered, timeout=2000):
        qtbot.keyClick(
            main_window.ui,
            QtCore.Qt.Key.Key_O,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

    assert len(file_dialogs.calls) == 1


def test_open_project_does_not_leave_a_modal_dialog_open(
    main_window: MainWindowManager, file_dialogs
):
    main_window.actionOpenProject.trigger()

    assert QtWidgets.QApplication.activeModalWidget() is None
