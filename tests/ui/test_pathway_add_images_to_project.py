"""Add-images pathway: scan folder -> choose pages -> project -> load status."""

import pathlib
import shutil
from typing import TYPE_CHECKING

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.load_files_status import LoadFilesStatusDialog
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.models import ImageId
from scantailor.core.project import Project

if TYPE_CHECKING:
    from conftest import PageImages

LEFT = QtCore.Qt.MouseButton.LeftButton


@pytest.fixture
def files_dialog(qtbot: QtBot, page_images: "PageImages") -> ProjectFilesDialog:
    dialog = ProjectFilesDialog(input_dir=page_images.directory)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def _string_list(model: QtCore.QAbstractItemModel) -> list[str]:
    return [model.index(row, 0).data() for row in range(model.rowCount())]


def _select_rows(view: QtWidgets.QListView, rows: list[int]) -> None:
    view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
    for row in rows:
        view.selectionModel().select(
            view.model().index(row, 0),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )


def test_dialog_lists_scanned_images_as_available_when_input_dir_given(
    files_dialog: ProjectFilesDialog,
):
    assert _string_list(files_dialog._off_project_model) == [
        "page_001.png",
        "page_002.png",
        "page_003.png",
    ]
    assert _string_list(files_dialog._in_project_model) == []


def test_dialog_ignores_non_image_files_in_input_dir(
    qtbot: QtBot, page_images: "PageImages"
):
    (page_images.directory / "notes.txt").write_text("not a scan")
    dialog = ProjectFilesDialog(input_dir=page_images.directory)
    qtbot.addWidget(dialog)

    available = _string_list(dialog._off_project_model)

    assert "notes.txt" not in available
    assert len(available) == 3


def test_browse_button_sets_input_dir_and_lists_its_images(
    qtbot: QtBot, page_images: "PageImages", file_dialogs
):
    dialog = ProjectFilesDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    file_dialogs.directory = str(page_images.directory)

    qtbot.mouseClick(dialog._input_dir_browse_btn, LEFT)

    assert [call[0] for call in file_dialogs.calls] == ["getExistingDirectory"]
    assert file_dialogs.calls[0][1][0] is dialog
    assert dialog.get_input_directory() == page_images.directory
    assert len(_string_list(dialog._off_project_model)) == 3


def test_browse_cancelled_keeps_input_dir_unset_and_lists_empty(
    qtbot: QtBot, file_dialogs
):
    dialog = ProjectFilesDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    file_dialogs.directory = ""

    qtbot.mouseClick(dialog._input_dir_browse_btn, LEFT)

    assert len(file_dialogs.calls) == 1
    assert dialog.get_input_directory() is None
    assert _string_list(dialog._off_project_model) == []


def test_browse_to_nonexistent_directory_lists_nothing(
    qtbot: QtBot, tmp_path: pathlib.Path, file_dialogs
):
    dialog = ProjectFilesDialog()
    qtbot.addWidget(dialog)
    file_dialogs.directory = str(tmp_path / "missing")

    qtbot.mouseClick(dialog._input_dir_browse_btn, LEFT)

    assert _string_list(dialog._off_project_model) == []


def test_add_button_moves_selected_files_into_project_and_emits_files_changed(
    qtbot: QtBot, files_dialog: ProjectFilesDialog
):
    _select_rows(files_dialog._off_project_list, [0, 2])

    with qtbot.waitSignal(files_dialog.files_changed, timeout=2000):
        qtbot.mouseClick(files_dialog._add_to_project_btn, LEFT)

    assert files_dialog.get_project_files() == ["page_001.png", "page_003.png"]
    assert _string_list(files_dialog._off_project_model) == ["page_002.png"]
    assert _string_list(files_dialog._in_project_model) == [
        "page_001.png",
        "page_003.png",
    ]


def test_remove_button_returns_file_to_available_list(
    qtbot: QtBot, files_dialog: ProjectFilesDialog
):
    _select_rows(files_dialog._off_project_list, [0, 1, 2])
    qtbot.mouseClick(files_dialog._add_to_project_btn, LEFT)
    _select_rows(files_dialog._in_project_list, [1])

    qtbot.mouseClick(files_dialog._remove_from_project_btn, LEFT)

    assert files_dialog.get_project_files() == ["page_001.png", "page_003.png"]
    assert _string_list(files_dialog._off_project_model) == ["page_002.png"]


def test_add_button_with_empty_selection_leaves_project_unchanged(
    qtbot: QtBot, files_dialog: ProjectFilesDialog
):
    qtbot.mouseClick(files_dialog._add_to_project_btn, LEFT)

    assert files_dialog.get_project_files() == []
    assert len(_string_list(files_dialog._off_project_model)) == 3


def test_files_already_in_project_are_not_offered_again(
    qtbot: QtBot, page_images: "PageImages"
):
    dialog = ProjectFilesDialog(
        input_dir=page_images.directory, project_files=["page_002.png"]
    )
    qtbot.addWidget(dialog)

    assert _string_list(dialog._off_project_model) == ["page_001.png", "page_003.png"]
    assert _string_list(dialog._in_project_model) == ["page_002.png"]


def test_dialog_ok_accepts_and_exposes_chosen_files_and_options(
    qtbot: QtBot, files_dialog: ProjectFilesDialog
):
    _select_rows(files_dialog._off_project_list, [0])
    qtbot.mouseClick(files_dialog._add_to_project_btn, LEFT)
    files_dialog._rtl_layout_cb.setChecked(True)

    files_dialog._button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).click()

    assert files_dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert files_dialog.get_project_files() == ["page_001.png"]
    assert files_dialog.is_rtl_layout()
    assert not files_dialog.is_force_fix_dpi()


def test_dialog_cancel_rejects(files_dialog: ProjectFilesDialog):
    files_dialog._button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Cancel
    ).click()

    assert files_dialog.result() == QtWidgets.QDialog.DialogCode.Rejected


def test_chosen_files_become_project_pages_in_order(
    qtbot: QtBot, files_dialog: ProjectFilesDialog, page_images: "PageImages"
):
    _select_rows(files_dialog._off_project_list, [0, 1, 2])
    qtbot.mouseClick(files_dialog._add_to_project_btn, LEFT)
    project = Project()

    for name in files_dialog.get_project_files():
        project.add_image(ImageId(file_path=page_images.directory / name))

    expected = ["page_001.png", "page_002.png", "page_003.png"]
    assert [info.id.file_path.name for info in project.images] == expected
    assert [page.image_id.file_path.name for page in project.get_pages()] == expected


def test_project_add_image_returns_the_stored_info(page_images: "PageImages"):
    project = Project()
    image_id = ImageId(file_path=page_images.paths[0])

    info = project.add_image(image_id)

    assert info.id == image_id
    assert project.images == [info]


def test_status_dialog_all_loaded_shows_counts_and_paths(
    qtbot: QtBot, page_images: "PageImages"
):
    dialog = LoadFilesStatusDialog(loaded_files=page_images.paths)
    qtbot.addWidget(dialog)

    assert not dialog.has_failures()
    assert dialog.get_loaded_count() == 3
    assert dialog._tab_widget.tabText(0) == "Loaded successfully: 3"
    assert dialog._tab_widget.tabText(1) == "Failed to load: 0"
    assert dialog._tab_widget.currentIndex() == 0
    assert str(page_images.paths[1]) in dialog._loaded_text.toPlainText()


def test_status_dialog_with_failures_focuses_failed_tab_and_shows_errors(
    qtbot: QtBot, page_images: "PageImages", tmp_path: pathlib.Path
):
    broken = tmp_path / "broken.png"
    dialog = LoadFilesStatusDialog(
        loaded_files=page_images.paths[:2],
        failed_files=[(broken, "unsupported format")],
    )
    qtbot.addWidget(dialog)

    assert dialog.has_failures()
    assert dialog.get_failed_count() == 1
    assert dialog._tab_widget.currentIndex() == 1
    assert dialog._failed_text.toPlainText() == f"{broken}: unsupported format"


def test_status_dialog_reports_corrupt_file_on_disk_as_failure(
    qtbot: QtBot, page_images: "PageImages", tmp_path: pathlib.Path
):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a png")
    good = tmp_path / "ok.png"
    shutil.copy(page_images.paths[0], good)

    loaded: list[pathlib.Path] = []
    failed: list[tuple[pathlib.Path, str]] = []
    for path in (good, corrupt):
        if QtGui.QImage(str(path)).isNull():
            failed.append((path, "cannot decode image"))
        else:
            loaded.append(path)
    dialog = LoadFilesStatusDialog(loaded, failed)
    qtbot.addWidget(dialog)

    assert dialog.get_loaded_count() == 1
    assert dialog.get_failed_count() == 1
    assert "corrupt.png" in dialog._failed_text.toPlainText()


def test_show_status_returns_true_when_user_accepts(
    page_images: "PageImages", respond_to_modal
):
    respond_to_modal(accept=True)

    assert LoadFilesStatusDialog.show_status(page_images.paths, []) is True


def test_show_status_returns_false_when_user_cancels(
    page_images: "PageImages", respond_to_modal
):
    respond_to_modal(accept=False)

    assert LoadFilesStatusDialog.show_status(page_images.paths, []) is False


@pytest.mark.xfail(
    strict=True,
    reason="MainWindow has no add-files action; ProjectFilesDialog is never opened",
)
def test_main_window_add_files_action_opens_project_files_dialog(
    main_window: MainWindowManager, respond_to_modal
):
    action = getattr(main_window, "actionAddFiles", None)
    assert action is not None

    respond_to_modal(accept=False)
    action.trigger()

    assert main_window.ui.findChild(ProjectFilesDialog) is not None


@pytest.mark.xfail(
    strict=True,
    reason="No code connects ProjectFilesDialog to Project.add_image; "
    "MainWindowManager holds no Project",
)
def test_accepting_project_files_dialog_adds_pages_to_main_window_project(
    main_window: MainWindowManager,
):
    assert hasattr(main_window, "project")
    assert main_window.project.images
