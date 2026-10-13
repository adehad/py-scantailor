from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import APP_TITLE, MainWindowManager, SaveChoice
from scantailor.core.models import Dpi, ImageId
from scantailor.core.project import ImageMetadata, Project

if TYPE_CHECKING:
    from conftest import FileDialogStub, PageImages, SavePromptScript


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collect QMessageBox.warning texts instead of opening a modal box."""
    texts: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, text: texts.append(text),
    )
    return texts


def test_open_project_action_has_ctrl_o_shortcut(main_window: MainWindowManager):
    assert main_window.actionOpenProject.shortcut().toString() == "Ctrl+O"


def test_open_project_asks_for_scantailor_project_file_parented_to_window(
    main_window: MainWindowManager, file_dialogs: FileDialogStub
):
    main_window.actionOpenProject.trigger()

    assert len(file_dialogs.calls) == 1
    name, args, _kwargs = file_dialogs.calls[0]
    assert name == "getOpenFileName"
    assert args[0] is main_window.ui
    assert "*.ScanTailor" in args[3]


def test_open_project_with_chosen_file_loads_it_and_names_the_window(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
    page_images: PageImages,
):
    project = Project(output_directory=tmp_path)
    for path in page_images.paths:
        project.add_image(
            ImageId(file_path=path),
            ImageMetadata(width=10, height=10, dpi=Dpi.uniform(300)),
        )
    project_file = tmp_path / "book.ScanTailor"
    project.save(project_file)
    file_dialogs.open_file = str(project_file)

    main_window.actionOpenProject.trigger()

    assert len(main_window.pages()) == 3
    assert main_window.ui.windowTitle() == f"book[*] - {APP_TITLE}"
    assert not main_window.ui.isWindowModified()
    assert main_window.actionSaveProject.isEnabled()


def test_open_project_cancelled_dialog_does_nothing(
    main_window: MainWindowManager, file_dialogs: FileDialogStub
):
    file_dialogs.open_file = ""

    main_window.actionOpenProject.trigger()

    assert len(file_dialogs.calls) == 1
    assert main_window.project is None
    assert main_window.ui.isVisible()


@pytest.mark.parametrize("content", ["not a project", '{"images": 5}', ""])
def test_open_project_with_invalid_file_warns_and_keeps_current_state(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
    warnings: list[str],
    content: str,
):
    broken = tmp_path / "broken.ScanTailor"
    broken.write_text(content, encoding="utf-8")
    file_dialogs.open_file = str(broken)

    before = project_window.project

    project_window.actionOpenProject.trigger()

    assert len(warnings) == 1
    assert project_window.project is before
    assert project_window.modified


def test_open_project_with_missing_file_warns(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
    warnings: list[str],
):
    file_dialogs.open_file = str(tmp_path / "gone.ScanTailor")

    main_window.actionOpenProject.trigger()

    assert len(warnings) == 1
    assert main_window.project is None


def test_open_project_keeps_modified_project_when_save_prompt_is_cancelled(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    tmp_path: pathlib.Path,
):
    other = tmp_path / "other.ScanTailor"
    Project(output_directory=tmp_path).save(other)
    file_dialogs.open_file = str(other)
    save_prompt.answers.append(SaveChoice.CANCEL)
    before = project_window.project

    project_window.actionOpenProject.trigger()

    assert project_window.project is before


def test_open_project_discarding_changes_replaces_the_project(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    save_prompt: SavePromptScript,
    tmp_path: pathlib.Path,
):
    other = tmp_path / "other.ScanTailor"
    Project(output_directory=tmp_path).save(other)
    file_dialogs.open_file = str(other)
    save_prompt.answers.append(SaveChoice.DISCARD)

    project_window.actionOpenProject.trigger()

    assert project_window.projectFile == str(other)
    assert project_window.pages() == []


def test_open_project_shortcut_dispatches_to_the_action(
    qtbot: QtBot, main_window: MainWindowManager, file_dialogs: FileDialogStub
):
    with qtbot.waitSignal(main_window.actionOpenProject.triggered, timeout=2000):
        qtbot.keyClick(
            main_window.ui,
            QtCore.Qt.Key.Key_O,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

    assert len(file_dialogs.calls) == 1


def test_open_project_does_not_leave_a_modal_dialog_open(
    main_window: MainWindowManager, file_dialogs: FileDialogStub
):
    main_window.actionOpenProject.trigger()

    assert QtWidgets.QApplication.activeModalWidget() is None
