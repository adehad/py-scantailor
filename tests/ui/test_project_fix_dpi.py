"""The Fix DPI dialog: after importing images and from the Fix DPI action."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from PIL import Image
from PySide6 import QtCore, QtWidgets

from scantailor.app.ui.dialogs.fix_dpi import FixDpiDialog
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.image_io import list_image_pages
from scantailor.core.models import Dpi, ImageId
from scantailor.core.project import ImageInfo, ImageMetadata

if TYPE_CHECKING:
    from conftest import PageImages


def _write_scans(
    directory: pathlib.Path, names: list[str], dpi: int | None
) -> list[pathlib.Path]:
    """Write small grey pages; ``dpi=None`` leaves the resolution undefined."""
    directory.mkdir(exist_ok=True)
    paths = []
    for name in names:
        path = directory / name
        image = Image.new("L", (600, 800), 255)
        if dpi is None:
            image.save(path)
        else:
            image.save(path, dpi=(dpi, dpi))
        paths.append(path)
    return paths


def _files_dialog_factory(
    input_dir: pathlib.Path, names: list[str], output_dir: pathlib.Path
) -> Callable[[QtWidgets.QWidget], ProjectFilesDialog]:
    def build(parent: QtWidgets.QWidget) -> ProjectFilesDialog:
        return ProjectFilesDialog(
            input_dir=input_dir,
            output_dir=output_dir,
            project_files=names,
            parent=parent,
        )

    return build


class FixDpiDriver:
    """Operates the Fix DPI dialog from the event loop while it blocks in exec()."""

    def __init__(self) -> None:
        self.dialogs: list[FixDpiDialog] = []
        self.listed: list[list[str]] = []
        self.ok_enabled_before_apply: list[bool] = []
        self.ok_enabled_after_apply: list[bool] = []

    def schedule(
        self, dpi: int | None, *, tab: int = 0, accept: bool = True, timeout_ms=5000
    ) -> None:
        """Answer the next Fix DPI dialog: apply ``dpi`` to every row, then close."""
        deadline = QtCore.QDeadlineTimer(timeout_ms)

        def poll() -> None:
            modal = QtWidgets.QApplication.activeModalWidget()
            if not isinstance(modal, FixDpiDialog):
                if not deadline.hasExpired():
                    QtCore.QTimer.singleShot(10, poll)
                return
            self.dialogs.append(modal)
            self._operate(modal, dpi, tab, accept)

        QtCore.QTimer.singleShot(10, poll)

    def _operate(
        self, dialog: FixDpiDialog, dpi: int | None, tab: int, accept: bool
    ) -> None:
        tabs = dialog.findChild(QtWidgets.QTabWidget, "tabWidget")
        view_name = "undefinedDpiView" if tab == 0 else "allPagesView"
        view = dialog.findChild(QtWidgets.QTreeView, view_name)
        buttons = dialog.findChild(QtWidgets.QDialogButtonBox, "buttonBox")
        ok_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        tabs.setCurrentIndex(tab)
        model = view.model()
        self.listed.append(
            [model.index(row, 0).data() for row in range(model.rowCount())]
        )
        self.ok_enabled_before_apply.append(ok_button.isEnabled())
        if dpi is not None:
            view.selectAll()
            dialog.findChild(QtWidgets.QLineEdit, "xDpi").setText(str(dpi))
            dialog.findChild(QtWidgets.QLineEdit, "yDpi").setText(str(dpi))
            dialog.findChild(QtWidgets.QPushButton, "applyBtn").click()
        self.ok_enabled_after_apply.append(ok_button.isEnabled())
        if accept:
            dialog.accept()
        else:
            dialog.reject()


@pytest.fixture
def fix_dpi() -> FixDpiDriver:
    return FixDpiDriver()


def _new_project(
    window: MainWindowManager,
    factory: Callable[[QtWidgets.QWidget], ProjectFilesDialog],
    respond_to_modal: Callable[..., None],
) -> None:
    window.project_files_dialog_factory = factory
    respond_to_modal(accept=True)
    window.actionNewProject.trigger()


def test_importing_images_with_undefined_dpi_asks_and_stores_the_chosen_dpi(
    main_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
    respond_to_modal: Callable[..., None],
    tmp_path: pathlib.Path,
):
    names = ["a.png", "b.png"]
    _write_scans(tmp_path / "raw", names, dpi=None)
    fix_dpi.schedule(300)

    _new_project(
        main_window,
        _files_dialog_factory(tmp_path / "raw", names, tmp_path / "out"),
        respond_to_modal,
    )

    assert len(fix_dpi.dialogs) == 1
    assert fix_dpi.listed == [names]
    assert main_window.project is not None
    assert [info.metadata.dpi for info in main_window.project.images] == [
        Dpi.uniform(300)
    ] * 2


def test_ok_is_only_available_once_every_image_has_a_usable_dpi(
    main_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
    respond_to_modal: Callable[..., None],
    tmp_path: pathlib.Path,
):
    _write_scans(tmp_path / "raw", ["a.png"], dpi=None)
    fix_dpi.schedule(300)

    _new_project(
        main_window,
        _files_dialog_factory(tmp_path / "raw", ["a.png"], tmp_path / "out"),
        respond_to_modal,
    )

    assert fix_dpi.ok_enabled_before_apply == [False]
    assert fix_dpi.ok_enabled_after_apply == [True]


def test_images_with_out_of_range_dpi_are_offered_for_fixing(
    main_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
    respond_to_modal: Callable[..., None],
    tmp_path: pathlib.Path,
):
    _write_scans(tmp_path / "raw", ["low.png"], dpi=72)
    fix_dpi.schedule(400)

    _new_project(
        main_window,
        _files_dialog_factory(tmp_path / "raw", ["low.png"], tmp_path / "out"),
        respond_to_modal,
    )

    assert fix_dpi.listed == [["low.png"]]
    assert main_window.project is not None
    assert main_window.project.images[0].metadata.dpi == Dpi.uniform(400)


def test_images_with_good_dpi_do_not_open_the_dialog(
    main_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
    respond_to_modal: Callable[..., None],
    tmp_path: pathlib.Path,
):
    _write_scans(tmp_path / "raw", ["a.png"], dpi=300)
    created: list[ImageInfo] = []
    main_window.fix_dpi_dialog_factory = lambda images, parent: (
        created.extend(images) or FixDpiDialog(images, parent)
    )

    _new_project(
        main_window,
        _files_dialog_factory(tmp_path / "raw", ["a.png"], tmp_path / "out"),
        respond_to_modal,
    )

    assert created == []
    assert main_window.project is not None


def test_cancelling_the_fix_dpi_dialog_abandons_the_new_project(
    main_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
    respond_to_modal: Callable[..., None],
    tmp_path: pathlib.Path,
):
    _write_scans(tmp_path / "raw", ["a.png"], dpi=None)
    fix_dpi.schedule(None, accept=False)

    _new_project(
        main_window,
        _files_dialog_factory(tmp_path / "raw", ["a.png"], tmp_path / "out"),
        respond_to_modal,
    )

    assert len(fix_dpi.dialogs) == 1
    assert main_window.project is None


def test_adding_files_with_undefined_dpi_asks_only_about_the_new_files(
    project_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
    respond_to_modal: Callable[..., None],
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    extra = _write_scans(tmp_path / "extra", ["new.png"], dpi=None)
    project_window.project_files_dialog_factory = _files_dialog_factory(
        extra[0].parent, ["new.png"], tmp_path / "out"
    )
    respond_to_modal(accept=True)
    fix_dpi.schedule(300)

    project_window.actionAddFiles.trigger()

    assert fix_dpi.listed == [["new.png"]]
    assert project_window.project is not None
    added = project_window.project.images[-1]
    assert added.id.file_path == extra[0]
    assert added.metadata.dpi == Dpi.uniform(300)
    assert len(project_window.project.images) == len(page_images.paths) + 1


def test_fix_dpi_action_applies_the_dpi_to_the_chosen_images_and_marks_modified(
    project_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
):
    project_window.modified = False
    fix_dpi.schedule(600, tab=1)

    project_window.actionFixDpi.trigger()

    assert project_window.project is not None
    assert {info.metadata.dpi for info in project_window.project.images} == {
        Dpi.uniform(600)
    }
    assert project_window.modified


def test_fix_dpi_action_without_changes_leaves_the_project_unmodified(
    project_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
):
    project_window.modified = False
    fix_dpi.schedule(None, tab=1)

    project_window.actionFixDpi.trigger()

    assert len(fix_dpi.dialogs) == 1
    assert not project_window.modified


def test_fix_dpi_action_needs_an_open_project(main_window: MainWindowManager):
    assert not main_window.actionFixDpi.isEnabled()


def _dialog_for(infos: list[ImageInfo], qtbot) -> FixDpiDialog:
    dialog = FixDpiDialog(infos)
    qtbot.addWidget(dialog)
    return dialog


def _image(name: str, dpi: Dpi, page: int = 0) -> ImageInfo:
    return ImageInfo(
        id=ImageId(file_path=pathlib.Path(name), page=page),
        metadata=ImageMetadata(width=2480, height=3508, dpi=dpi),
    )


def test_dialog_lists_images_needing_a_fix_apart_from_all_images(qtbot):
    dialog = _dialog_for(
        [_image("ok.png", Dpi.uniform(300)), _image("bad.png", Dpi())], qtbot
    )

    needing = dialog.findChild(QtWidgets.QTreeView, "undefinedDpiView").model()
    everything = dialog.findChild(QtWidgets.QTreeView, "allPagesView").model()
    assert [needing.index(r, 0).data() for r in range(needing.rowCount())] == [
        "bad.png"
    ]
    assert everything.rowCount() == 2


def test_dialog_reports_only_changed_images_keyed_by_image_id(qtbot):
    bad = _image("bad.png", Dpi())
    good = _image("ok.png", Dpi.uniform(300))
    dialog = _dialog_for([good, bad], qtbot)
    dialog.findChild(QtWidgets.QTreeView, "undefinedDpiView").selectAll()
    dialog.findChild(QtWidgets.QLineEdit, "xDpi").setText("400")
    dialog.findChild(QtWidgets.QLineEdit, "yDpi").setText("600")

    dialog.findChild(QtWidgets.QPushButton, "applyBtn").click()

    assert dialog.get_dpi_changes() == {bad.id: Dpi(horizontal=400, vertical=600)}
    assert dialog.images_needing_fix() == []


def test_dialog_refuses_a_dpi_that_is_implausible_for_the_image(qtbot):
    dialog = _dialog_for([_image("bad.png", Dpi())], qtbot)
    dialog.findChild(QtWidgets.QTreeView, "undefinedDpiView").selectAll()

    dialog.findChild(QtWidgets.QLineEdit, "xDpi").setText("50")
    dialog.findChild(QtWidgets.QLineEdit, "yDpi").setText("50")

    assert not dialog.findChild(QtWidgets.QPushButton, "applyBtn").isEnabled()


def test_dialog_tells_pages_of_one_multi_page_file_apart(qtbot):
    first = _image("scan.tif", Dpi(), page=1)
    second = _image("scan.tif", Dpi.uniform(300), page=2)
    dialog = _dialog_for([first, second], qtbot)
    dialog.findChild(QtWidgets.QTreeView, "undefinedDpiView").selectAll()
    dialog.findChild(QtWidgets.QLineEdit, "xDpi").setText("300")
    dialog.findChild(QtWidgets.QLineEdit, "yDpi").setText("300")

    dialog.findChild(QtWidgets.QPushButton, "applyBtn").click()

    assert list(dialog.get_dpi_changes()) == [first.id]


def test_dialog_presets_fill_both_dpi_fields(qtbot):
    dialog = _dialog_for([_image("bad.png", Dpi())], qtbot)

    dialog.findChild(QtWidgets.QComboBox, "dpiCombo").setCurrentText("600")

    assert dialog.findChild(QtWidgets.QLineEdit, "xDpi").text() == "600"
    assert dialog.findChild(QtWidgets.QLineEdit, "yDpi").text() == "600"


def test_image_without_dpi_is_read_as_undefined(tmp_path: pathlib.Path):
    [path] = _write_scans(tmp_path / "raw", ["a.png"], dpi=None)

    [(_, metadata)] = list_image_pages(path)

    assert not metadata.has_plausible_dpi()


def _save_undefined_dpi_project(tmp_path: pathlib.Path) -> pathlib.Path:
    from scantailor.core.project import Project

    paths = _write_scans(tmp_path / "raw", ["a.png"], dpi=None)
    project = Project(output_directory=tmp_path / "out")
    for image_id, metadata in list_image_pages(paths[0]):
        project.add_image(image_id, metadata)
    project_file = tmp_path / "book.ScanTailor"
    project.save(project_file)
    return project_file


def test_opening_a_project_with_undefined_dpi_asks_and_stores_the_chosen_dpi(
    main_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
    file_dialogs,
    tmp_path: pathlib.Path,
):
    fix_dpi.schedule(300)
    file_dialogs.open_file = str(_save_undefined_dpi_project(tmp_path))

    main_window.actionOpenProject.trigger()

    assert len(fix_dpi.dialogs) == 1
    assert main_window.project is not None
    assert main_window.project.images[0].metadata.dpi == Dpi.uniform(300)
    assert main_window.modified


def test_cancelling_fix_dpi_while_opening_a_project_leaves_it_unopened(
    main_window: MainWindowManager,
    fix_dpi: FixDpiDriver,
    file_dialogs,
    tmp_path: pathlib.Path,
):
    fix_dpi.schedule(None, accept=False)
    file_dialogs.open_file = str(_save_undefined_dpi_project(tmp_path))

    main_window.actionOpenProject.trigger()

    assert len(fix_dpi.dialogs) == 1
    assert main_window.project is None
