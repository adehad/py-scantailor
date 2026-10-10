import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager


def test_save_project_action_has_ctrl_s_shortcut(main_window: MainWindowManager):
    assert main_window.actionSaveProject.shortcut().toString() == "Ctrl+S"


def test_save_project_as_action_is_present_with_a_label(
    main_window: MainWindowManager,
):
    assert "save" in main_window.actionSaveProjectAs.text().lower()


def test_triggering_save_project_logs_and_keeps_window_open(
    main_window: MainWindowManager, log_messages: list[str]
):
    main_window.actionSaveProject.trigger()

    assert "Save project triggered" in log_messages
    assert "Save project as triggered" not in log_messages
    assert main_window.ui.isVisible()


def test_triggering_save_project_as_logs_and_keeps_window_open(
    main_window: MainWindowManager, log_messages: list[str]
):
    main_window.actionSaveProjectAs.trigger()

    assert "Save project as triggered" in log_messages
    assert "Save project triggered" not in log_messages
    assert main_window.ui.isVisible()


def test_ctrl_s_shortcut_dispatches_to_save_project(
    qtbot: QtBot, main_window: MainWindowManager, log_messages: list[str]
):
    with qtbot.waitSignal(main_window.actionSaveProject.triggered, timeout=2000):
        qtbot.keyClick(
            main_window.ui,
            QtCore.Qt.Key.Key_S,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

    assert "Save project triggered" in log_messages


def test_save_actions_do_not_open_any_modal_dialog(main_window: MainWindowManager):
    main_window.actionSaveProject.trigger()
    main_window.actionSaveProjectAs.trigger()

    assert QtWidgets.QApplication.activeModalWidget() is None


def test_save_actions_write_no_files_to_working_directory(
    main_window: MainWindowManager, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    main_window.actionSaveProject.trigger()
    main_window.actionSaveProjectAs.trigger()

    assert list(workdir.iterdir()) == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Save As has no QFileDialog.getSaveFileName call; "
        "see MainWindow.saveProjectAsTriggered"
    ),
)
def test_save_as_asks_for_a_destination_file(
    main_window: MainWindowManager, file_dialogs
):
    main_window.actionSaveProjectAs.trigger()

    assert [name for name, _, _ in file_dialogs.calls] == ["getSaveFileName"]


@pytest.mark.xfail(
    strict=True,
    reason="Save As is not wired to Project.save and nothing is written",
)
def test_save_as_writes_project_file_to_chosen_path(
    main_window: MainWindowManager, file_dialogs, tmp_path
):
    target = tmp_path / "out.ScanTailor"
    file_dialogs.save_file = str(target)

    main_window.actionSaveProjectAs.trigger()

    assert target.exists()


@pytest.mark.xfail(
    strict=True,
    reason="Cancelled Save As cannot be distinguished: no dialog exists to cancel",
)
def test_cancelled_save_as_dialog_writes_nothing_and_does_not_log_save(
    main_window: MainWindowManager, file_dialogs, log_messages: list[str], tmp_path
):
    file_dialogs.save_file = ""

    main_window.actionSaveProjectAs.trigger()

    assert file_dialogs.calls
    assert "Save project as triggered" not in log_messages
    assert list(tmp_path.glob("*.ScanTailor")) == []
