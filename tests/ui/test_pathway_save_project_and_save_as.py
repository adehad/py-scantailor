from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import APP_TITLE, MainWindowManager
from scantailor.core.models import OrthogonalRotation
from scantailor.core.project import Project
from scantailor.filters.deskew.params import Params as DeskewParams

if TYPE_CHECKING:
    from conftest import FileDialogStub


def test_save_project_action_has_ctrl_s_shortcut(main_window: MainWindowManager):
    assert main_window.actionSaveProject.shortcut().toString() == "Ctrl+S"


def test_save_project_as_action_is_present_with_a_label(
    main_window: MainWindowManager,
):
    assert "save" in main_window.actionSaveProjectAs.text().lower()


def test_save_actions_without_a_project_do_nothing(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    main_window.actionSaveProject.trigger()
    main_window.actionSaveProjectAs.trigger()
    main_window.saveProjectTriggered()
    main_window.saveProjectAsTriggered()

    assert file_dialogs.calls == []
    assert QtWidgets.QApplication.activeModalWidget() is None
    assert list(workdir.iterdir()) == []


def test_save_as_asks_for_a_destination_file_parented_to_the_window(
    project_window: MainWindowManager, file_dialogs: FileDialogStub
):
    project_window.actionSaveProjectAs.trigger()

    assert [name for name, _, _ in file_dialogs.calls] == ["getSaveFileName"]
    assert file_dialogs.calls[0][1][0] is project_window.ui
    assert "*.ScanTailor" in file_dialogs.calls[0][1][3]


def test_save_as_writes_project_file_to_chosen_path(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    target = tmp_path / "out.ScanTailor"
    file_dialogs.save_file = str(target)

    project_window.actionSaveProjectAs.trigger()

    assert target.exists()
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert len(saved["images"]) == 3


def test_save_as_appends_the_project_suffix_when_missing(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    file_dialogs.save_file = str(tmp_path / "book")

    project_window.actionSaveProjectAs.trigger()

    assert (tmp_path / "book.ScanTailor").exists()
    assert not (tmp_path / "book").exists()


def test_cancelled_save_as_dialog_writes_nothing_and_keeps_project_modified(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    file_dialogs.save_file = ""

    project_window.actionSaveProjectAs.trigger()

    assert file_dialogs.calls
    assert list(tmp_path.glob("*.ScanTailor")) == []
    assert project_window.modified
    assert project_window.projectFile == ""


def test_save_as_names_the_project_in_the_title_and_clears_modified_marker(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    file_dialogs.save_file = str(tmp_path / "my book.ScanTailor")

    project_window.actionSaveProjectAs.trigger()

    assert project_window.ui.windowTitle() == f"my book[*] - {APP_TITLE}"
    assert not project_window.ui.isWindowModified()


def test_save_of_unsaved_project_falls_back_to_save_as(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    file_dialogs.save_file = str(tmp_path / "first.ScanTailor")

    project_window.actionSaveProject.trigger()

    assert [name for name, _, _ in file_dialogs.calls] == ["getSaveFileName"]
    assert (tmp_path / "first.ScanTailor").exists()


def test_save_of_named_project_rewrites_the_file_without_asking(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    target = tmp_path / "again.ScanTailor"
    file_dialogs.save_file = str(target)
    project_window.actionSaveProjectAs.trigger()
    target.unlink()
    file_dialogs.calls.clear()

    project_window.modified = True
    project_window.actionSaveProject.trigger()

    assert file_dialogs.calls == []
    assert target.exists()
    assert not project_window.modified


def test_ctrl_s_shortcut_saves_a_named_project(
    qtbot: QtBot,
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    target = tmp_path / "keys.ScanTailor"
    file_dialogs.save_file = str(target)
    project_window.ui.activateWindow()
    qtbot.waitUntil(
        lambda: QtWidgets.QApplication.activeWindow() is project_window.ui,
        timeout=2000,
    )

    with qtbot.waitSignal(project_window.actionSaveProject.triggered, timeout=2000):
        qtbot.keyClick(
            project_window.ui,
            QtCore.Qt.Key.Key_S,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

    assert target.exists()


def test_save_to_unwritable_location_warns_and_keeps_project_modified(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    file_dialogs.save_file = str(tmp_path / "missing_dir" / "x.ScanTailor")

    project_window.actionSaveProjectAs.trigger()

    assert len(warnings) == 1
    assert project_window.modified
    assert project_window.projectFile == ""


def test_save_then_open_round_trips_pages_stage_params_and_current_page(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    pages = project_window.pages()
    project_window.stages.deskew_filter.settings.set_params(
        pages[1], DeskewParams().with_manual_angle(-2.5)
    )
    project_window.stages.fix_orientation_filter.settings.apply_rotation(
        pages[2].image_id, OrthogonalRotation(degrees=90)
    )
    project_window.goNextPage()
    project_window.goNextPage()
    target = tmp_path / "round.ScanTailor"
    file_dialogs.save_file = str(target)
    project_window.actionSaveProjectAs.trigger()
    project_window.actionCloseProject.trigger()
    assert project_window.project is None

    file_dialogs.open_file = str(target)
    project_window.actionOpenProject.trigger()

    assert project_window.project is not None
    assert project_window.pages() == pages
    assert project_window.current_page == pages[2]
    assert project_window.projectFile == str(target)
    assert not project_window.modified
    deskew = project_window.stages.deskew_filter.settings
    assert deskew.get_params(pages[1]) == DeskewParams().with_manual_angle(-2.5)
    assert not deskew.is_params_set(pages[0])
    orientation = project_window.stages.fix_orientation_filter.settings
    assert orientation.get_rotation_for(pages[2].image_id) == (
        OrthogonalRotation(degrees=90)
    )
    assert not orientation.is_rotation_set(pages[0].image_id)


def test_saved_file_loads_as_a_project_with_stage_params(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
):
    page = project_window.pages()[0]
    project_window.stages.deskew_filter.settings.set_params(
        page, DeskewParams().with_manual_angle(1.0)
    )
    target = tmp_path / "plain.ScanTailor"
    file_dialogs.save_file = str(target)

    project_window.actionSaveProjectAs.trigger()

    loaded = Project.load(target)
    assert [entry.page for entry in loaded.stage_params["deskew"]] == [page]
    assert loaded.selected_page == page
