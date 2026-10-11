"""Relinking moved image folders and storing image paths relative to the project."""

from __future__ import annotations

import pathlib
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.relinking import RelinkingDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.image_io import list_image_pages
from scantailor.core.project import Project
from scantailor.core.stage_params import collect_stage_params
from scantailor.core.stage_sequence import StageSequence

if TYPE_CHECKING:
    from conftest import FileDialogStub, PageImages, SavePromptScript

DESKEW_ANGLE = 1.5


class RelinkingScript:
    """Relinking dialog factory that records the folders offered and relinks them."""

    def __init__(self, mapping: dict[pathlib.Path, pathlib.Path] | None = None):
        self.mapping = mapping or {}
        self.offered: list[list[pathlib.Path]] = []

    def __call__(
        self, directories: list[pathlib.Path], parent: QtWidgets.QWidget
    ) -> RelinkingDialog:
        self.offered.append(list(directories))
        dialog = RelinkingDialog(directories, parent)
        for old, new in self.mapping.items():
            dialog.relink_directory(old, new)
        return dialog


def _save_project(paths: list[pathlib.Path], project_file: pathlib.Path) -> Project:
    project = Project(output_directory=project_file.parent / "out")
    project.output_directory.mkdir(exist_ok=True)
    for path in paths:
        for image_id, metadata in list_image_pages(path):
            project.add_image(image_id, metadata)
    project.save(project_file)
    return project


def _open(
    window: MainWindowManager,
    file_dialogs: FileDialogStub,
    project_file: pathlib.Path,
) -> None:
    file_dialogs.open_file = str(project_file)
    window.actionOpenProject.trigger()


@pytest.fixture
def moved_project(
    page_images: PageImages, tmp_path: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """A saved project whose scans folder was renamed afterwards.

    Returns:
        The project file, the folder the images were saved in, and the folder
        they are in now.
    """
    project_file = tmp_path / "book.ScanTailor"
    _save_project(page_images.paths, project_file)
    old_dir = page_images.directory
    new_dir = tmp_path / "scans_moved"
    shutil.move(str(old_dir), str(new_dir))
    return project_file, old_dir, new_dir


def test_opening_a_project_with_moved_images_offers_the_missing_folder(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    moved_project: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
):
    project_file, old_dir, new_dir = moved_project
    script = RelinkingScript({old_dir: new_dir})
    main_window.relinking_dialog_factory = script
    respond_to_modal(accept=True)

    _open(main_window, file_dialogs, project_file)

    assert script.offered == [[old_dir]]
    assert main_window.project is not None


def test_relinking_remaps_every_page_below_the_chosen_folder(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    moved_project: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
):
    project_file, old_dir, new_dir = moved_project
    main_window.relinking_dialog_factory = RelinkingScript({old_dir: new_dir})
    respond_to_modal(accept=True)

    _open(main_window, file_dialogs, project_file)

    assert main_window.project is not None
    paths = [info.id.file_path for info in main_window.project.images]
    assert paths == [new_dir / f"page_{n:03d}.png" for n in (1, 2, 3)]
    assert all(path.exists() for path in paths)
    assert len(main_window.pages()) == 3
    assert main_window.current_page is not None
    assert main_window.current_page.image_id.file_path.parent == new_dir


def test_relinked_project_is_marked_modified_because_the_file_has_old_paths(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    moved_project: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
):
    project_file, old_dir, new_dir = moved_project
    main_window.relinking_dialog_factory = RelinkingScript({old_dir: new_dir})
    respond_to_modal(accept=True)

    _open(main_window, file_dialogs, project_file)

    assert main_window.modified


def test_cancelling_relinking_leaves_the_current_state_alone(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    moved_project: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
):
    project_file, old_dir, new_dir = moved_project
    main_window.relinking_dialog_factory = RelinkingScript({old_dir: new_dir})
    respond_to_modal(accept=False)

    _open(main_window, file_dialogs, project_file)

    assert main_window.project is None


def test_accepting_relinking_without_choosing_a_folder_opens_the_project_unchanged(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    moved_project: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
):
    project_file, old_dir, _new_dir = moved_project
    main_window.relinking_dialog_factory = RelinkingScript()
    respond_to_modal(accept=True)

    _open(main_window, file_dialogs, project_file)

    assert main_window.project is not None
    assert {info.id.file_path.parent for info in main_window.project.images} == {
        old_dir
    }
    assert not main_window.modified


def test_project_with_all_images_present_does_not_open_the_relinking_dialog(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    project_file = tmp_path / "book.ScanTailor"
    _save_project(page_images.paths, project_file)
    script = RelinkingScript()
    main_window.relinking_dialog_factory = script

    _open(main_window, file_dialogs, project_file)

    assert script.offered == []
    assert main_window.project is not None
    assert not main_window.modified


def test_project_and_images_moved_together_open_without_relinking(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    root = tmp_path / "work"
    root.mkdir()
    shutil.move(str(page_images.directory), str(root / "scans"))
    moved_paths = [root / "scans" / path.name for path in page_images.paths]
    _save_project(moved_paths, root / "book.ScanTailor")
    new_root = tmp_path / "elsewhere"
    shutil.move(str(root), str(new_root))
    script = RelinkingScript()
    main_window.relinking_dialog_factory = script

    _open(main_window, file_dialogs, new_root / "book.ScanTailor")

    assert script.offered == []
    assert main_window.project is not None
    assert [info.id.file_path for info in main_window.project.images] == [
        new_root / "scans" / path.name for path in page_images.paths
    ]
    assert not main_window.modified


def test_stage_parameters_follow_the_relinked_images(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    project_file = tmp_path / "book.ScanTailor"
    project = _save_project(page_images.paths, project_file)
    stages = StageSequence()
    page = project.get_pages()[1]
    params = stages.deskew_filter.get_params(page)
    stages.deskew_filter.settings.set_params(
        page, params.model_copy(update={"deskew_angle_deg": DESKEW_ANGLE})
    )
    project.stage_params = collect_stage_params(stages, project.get_pages())
    project.save(project_file)
    new_dir = tmp_path / "moved"
    shutil.move(str(page_images.directory), str(new_dir))
    main_window.relinking_dialog_factory = RelinkingScript(
        {page_images.directory: new_dir}
    )
    respond_to_modal(accept=True)

    _open(main_window, file_dialogs, project_file)

    assert main_window.project is not None
    relinked_page = main_window.pages()[1]
    assert relinked_page.image_id.file_path.parent == new_dir
    restored = main_window.stages.deskew_filter.get_params(relinked_page)
    assert restored.deskew_angle_deg == pytest.approx(DESKEW_ANGLE)


def test_relinking_action_rewrites_the_open_project_and_keeps_its_file(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    file_dialogs.save_file = str(tmp_path / "book.ScanTailor")
    project_window.actionSaveProjectAs.trigger()
    new_dir = tmp_path / "relocated"
    shutil.copytree(page_images.directory, new_dir)
    script = RelinkingScript({page_images.directory: new_dir})
    project_window.relinking_dialog_factory = script
    respond_to_modal(accept=True)

    project_window.actionRelinking.trigger()

    assert project_window.project is not None
    assert {info.id.file_path.parent for info in project_window.project.images} == {
        new_dir
    }
    assert project_window.projectFile == str(tmp_path / "book.ScanTailor")
    assert project_window.modified
    assert len(project_window.pages()) == 3


def test_relinking_action_without_any_choice_changes_nothing(
    project_window: MainWindowManager,
    respond_to_modal: Callable[..., None],
):
    project_window.modified = False
    project_window.relinking_dialog_factory = RelinkingScript()
    respond_to_modal(accept=True)

    project_window.actionRelinking.trigger()

    assert not project_window.modified


def test_relinking_dialog_shows_the_new_location_and_flags_missing_folders(
    qtbot: QtBot, tmp_path: pathlib.Path
):
    existing = tmp_path / "exists"
    existing.mkdir()
    gone = tmp_path / "gone"
    dialog = RelinkingDialog([gone])
    qtbot.addWidget(dialog)
    assert dialog.has_missing_directories()

    dialog.relink_directory(gone, existing)

    assert not dialog.has_missing_directories()
    row = dialog.findChild(QtWidgets.QListWidget).item(0)
    assert row.text() == str(existing)
    assert row.toolTip() == str(gone)


def test_relinking_dialog_undo_restores_the_previous_choice(
    qtbot: QtBot, tmp_path: pathlib.Path
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    gone = tmp_path / "gone"
    dialog = RelinkingDialog([gone])
    qtbot.addWidget(dialog)
    dialog.relink_directory(gone, first)
    dialog.relink_directory(gone, second)

    dialog.undo()
    assert dialog.relinker().substitute(gone) == first
    dialog.undo()
    assert dialog.relinker().substitute(gone) == gone
    dialog.undo()
    assert dialog.relinker().is_empty()


def test_relinking_dialog_relink_button_asks_for_a_folder(
    qtbot: QtBot,
    tmp_path: pathlib.Path,
    file_dialogs: FileDialogStub,
):
    target = tmp_path / "target"
    target.mkdir()
    gone = tmp_path / "gone"
    file_dialogs.directory = str(target)
    dialog = RelinkingDialog([gone])
    qtbot.addWidget(dialog)
    dialog.findChild(QtWidgets.QListWidget).selectAll()

    dialog.findChild(QtWidgets.QPushButton).click()

    assert dialog.relinker().substitute(gone) == target
    assert file_dialogs.calls[0][0] == "getExistingDirectory"


def test_open_ignores_unsaved_prompt_when_relinking_is_cancelled(
    project_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    save_prompt: SavePromptScript,
    moved_project: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
):
    project_file, old_dir, new_dir = moved_project
    project_window.relinking_dialog_factory = RelinkingScript({old_dir: new_dir})
    respond_to_modal(accept=False)
    before = project_window.project

    _open(project_window, file_dialogs, project_file)

    assert save_prompt.prompts == 0
    assert project_window.project is before


def test_missing_image_folder_offers_the_missing_output_folder_too(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    moved_project: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
):
    project_file, old_dir, new_dir = moved_project
    output = project_file.parent / "out"
    shutil.rmtree(output)
    script = RelinkingScript({old_dir: new_dir})
    main_window.relinking_dialog_factory = script
    respond_to_modal(accept=True)

    _open(main_window, file_dialogs, project_file)

    assert script.offered == [sorted([old_dir, output])]


def test_relinking_the_output_folder_updates_the_project_and_marks_it_modified(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    project_file = tmp_path / "book.ScanTailor"
    project = _save_project(page_images.paths, project_file)
    old_out = project.output_directory
    new_dir = tmp_path / "scans_moved"
    shutil.move(str(page_images.directory), str(new_dir))
    shutil.rmtree(old_out)
    new_out = tmp_path / "results"
    new_out.mkdir()
    main_window.relinking_dialog_factory = RelinkingScript(
        {page_images.directory: new_dir, old_out: new_out}
    )
    respond_to_modal(accept=True)

    _open(main_window, file_dialogs, project_file)

    assert main_window.project is not None
    assert main_window.project.output_directory == new_out
    assert main_window.modified


def test_missing_output_folder_alone_is_recreated_without_asking(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    project_file = tmp_path / "book.ScanTailor"
    project = _save_project(page_images.paths, project_file)
    shutil.rmtree(project.output_directory)
    script = RelinkingScript()
    main_window.relinking_dialog_factory = script

    _open(main_window, file_dialogs, project_file)

    assert script.offered == []
    assert project.output_directory.is_dir()


def test_relinking_action_always_lists_the_output_folder_with_the_image_folders(
    project_window: MainWindowManager,
    respond_to_modal: Callable[..., None],
    page_images: PageImages,
):
    script = RelinkingScript()
    project_window.relinking_dialog_factory = script
    respond_to_modal(accept=True)

    project_window.actionRelinking.trigger()

    assert project_window.project is not None
    assert script.offered == [
        sorted([page_images.directory, project_window.project.output_directory])
    ]


def test_relinking_action_moves_the_output_folder(
    project_window: MainWindowManager,
    respond_to_modal: Callable[..., None],
    tmp_path: pathlib.Path,
):
    assert project_window.project is not None
    old_out = project_window.project.output_directory
    new_out = tmp_path / "elsewhere"
    new_out.mkdir()
    project_window.relinking_dialog_factory = RelinkingScript({old_out: new_out})
    respond_to_modal(accept=True)

    project_window.actionRelinking.trigger()

    assert project_window.project is not None
    assert project_window.project.output_directory == new_out


def test_output_folder_inside_a_relinked_image_folder_moves_with_it(
    project_window: MainWindowManager,
    respond_to_modal: Callable[..., None],
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    assert project_window.project is not None
    project_window.project.output_directory = page_images.directory / "out"
    new_dir = tmp_path / "relocated"
    shutil.copytree(page_images.directory, new_dir)
    project_window.relinking_dialog_factory = RelinkingScript(
        {page_images.directory: new_dir}
    )
    respond_to_modal(accept=True)

    project_window.actionRelinking.trigger()

    assert project_window.project is not None
    assert project_window.project.output_directory == new_dir / "out"
