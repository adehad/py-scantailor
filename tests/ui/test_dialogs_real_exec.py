"""Run each dialog through its real ``exec()``, without the conftest dialog factory.

A dialog whose ``.ui`` root is a QDialog used to open as an empty window because the
loaded form stayed a hidden top-level window. Every test here drives the dialog from
a timer while ``exec()`` blocks, so what is checked is what a user would see.
"""

import pathlib
from collections.abc import Callable

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.dialogs.load_files_status import LoadFilesStatusDialog
from scantailor.app.ui.dialogs.out_of_memory import OutOfMemoryDialog
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.dialogs.remove_pages import RemovePagesDialog
from scantailor.app.ui.dialogs.split_mode import SplitModeDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.page_split.layout_type import LayoutType

# 300 dpi, so no Fix DPI dialog interrupts project creation.
_DOTS_PER_METER = 11811
_POLL_MS = 20
_TIMEOUT_MS = 10000


def _find(
    dialog: QtWidgets.QDialog, widget_type: type[QtWidgets.QWidget], name: str
) -> QtWidgets.QWidget:
    widget = dialog.findChild(widget_type, name)
    assert widget is not None, f"{name} not found in {type(dialog).__name__}"
    return widget


def _assert_shown(dialog: QtWidgets.QDialog, *widgets: QtWidgets.QWidget) -> None:
    assert dialog.isVisible()
    for widget in widgets:
        name = widget.objectName()
        assert widget.isVisible(), f"{name} is not visible"
        assert widget.width() > 0, f"{name} has zero width"
        assert widget.height() > 0, f"{name} has zero height"


def _exec_while(
    dialog: QtWidgets.QDialog, drive: Callable[[QtWidgets.QDialog], None]
) -> QtWidgets.QDialog.DialogCode:
    """Run ``dialog.exec()`` and call ``drive`` once the dialog is on screen.

    Fails instead of hanging when the dialog never becomes visible or ``drive``
    does not end it.
    """
    errors: list[BaseException] = []
    clock = QtCore.QElapsedTimer()
    clock.start()
    timer = QtCore.QTimer()
    timer.setInterval(_POLL_MS)

    def poll() -> None:
        if not dialog.isVisible():
            if clock.hasExpired(_TIMEOUT_MS):
                timer.stop()
                errors.append(AssertionError("dialog never became visible"))
                dialog.reject()
            return
        timer.stop()
        try:
            drive(dialog)
        except BaseException as error:
            errors.append(error)
        if dialog.isVisible():
            dialog.reject()

    timer.timeout.connect(poll)
    timer.start()
    result = dialog.exec()
    timer.stop()
    if errors:
        raise errors[0]
    return QtWidgets.QDialog.DialogCode(result)


def _write_png(path: pathlib.Path) -> None:
    image = QtGui.QImage(60, 80, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor("white"))
    image.setDotsPerMeterX(_DOTS_PER_METER)
    image.setDotsPerMeterY(_DOTS_PER_METER)
    assert image.save(str(path)), f"could not write {path}"


def _pick_all_files(dialog: QtWidgets.QDialog) -> None:
    """Move every file listed as available into the project, as a user would."""
    off_list = _find(dialog, QtWidgets.QListView, "offProjectList")
    assert isinstance(off_list, QtWidgets.QListView)
    off_list.selectAll()
    _find(dialog, QtWidgets.QToolButton, "addToProjectBtn").click()


def test_project_files_dialog_shows_its_form_and_accepts(
    qtbot: QtBot, tmp_path: pathlib.Path
):
    for name in ("p2.png", "p10.png"):
        _write_png(tmp_path / name)
    dialog = ProjectFilesDialog(input_dir=tmp_path)
    qtbot.addWidget(dialog)
    seen: dict[str, list[str]] = {}

    def drive(dlg: QtWidgets.QDialog) -> None:
        names = (
            ("inpDirLine", QtWidgets.QLineEdit),
            ("outDirLine", QtWidgets.QLineEdit),
            ("offProjectList", QtWidgets.QListView),
            ("inProjectList", QtWidgets.QListView),
            ("addToProjectBtn", QtWidgets.QToolButton),
            ("buttonBox", QtWidgets.QDialogButtonBox),
        )
        _assert_shown(dlg, *(_find(dlg, kind, name) for name, kind in names))
        _pick_all_files(dlg)
        assert isinstance(dlg, ProjectFilesDialog)
        seen["files"] = dlg.get_project_files()
        button_box = _find(dlg, QtWidgets.QDialogButtonBox, "buttonBox")
        assert isinstance(button_box, QtWidgets.QDialogButtonBox)
        button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    assert _exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Accepted
    assert seen["files"] == ["p2.png", "p10.png"]


def test_project_files_dialog_cancel_rejects(qtbot: QtBot, tmp_path: pathlib.Path):
    dialog = ProjectFilesDialog(input_dir=tmp_path)
    qtbot.addWidget(dialog)

    def drive(dlg: QtWidgets.QDialog) -> None:
        button_box = _find(dlg, QtWidgets.QDialogButtonBox, "buttonBox")
        assert isinstance(button_box, QtWidgets.QDialogButtonBox)
        button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).click()

    assert _exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Rejected


def test_load_files_status_dialog_lists_failures_and_accepts(qtbot: QtBot):
    dialog = LoadFilesStatusDialog(
        loaded_files=[pathlib.Path("good.png")],
        failed_files=[(pathlib.Path("bad.png"), "damaged header")],
    )
    qtbot.addWidget(dialog)
    seen: dict[str, str] = {}

    def drive(dlg: QtWidgets.QDialog) -> None:
        tabs = _find(dlg, QtWidgets.QTabWidget, "tabWidget")
        failed = _find(dlg, QtWidgets.QPlainTextEdit, "failedFiles")
        button_box = _find(dlg, QtWidgets.QDialogButtonBox, "buttonBox")
        # The failed tab is the one shown first when anything failed.
        _assert_shown(dlg, tabs, failed, button_box)
        assert isinstance(failed, QtWidgets.QPlainTextEdit)
        seen["failed"] = failed.toPlainText()
        assert isinstance(button_box, QtWidgets.QDialogButtonBox)
        button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    assert _exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Accepted
    assert "damaged header" in seen["failed"]


def test_remove_pages_dialog_asks_with_the_page_count_and_accepts(qtbot: QtBot):
    dialog = RemovePagesDialog(page_count=3)
    qtbot.addWidget(dialog)
    seen: dict[str, str] = {}

    def drive(dlg: QtWidgets.QDialog) -> None:
        text = _find(dlg, QtWidgets.QLabel, "text")
        icon = _find(dlg, QtWidgets.QLabel, "icon")
        button_box = _find(dlg, QtWidgets.QDialogButtonBox, "buttonBox")
        _assert_shown(dlg, text, icon, button_box)
        assert isinstance(text, QtWidgets.QLabel)
        seen["text"] = text.text()
        assert isinstance(button_box, QtWidgets.QDialogButtonBox)
        button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    assert _exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Accepted
    assert "3 pages" in seen["text"]


def test_out_of_memory_dialog_offers_saving_and_declines(qtbot: QtBot):
    dialog = OutOfMemoryDialog()
    qtbot.addWidget(dialog)
    requested: list[str] = []
    dialog.save_project_requested.connect(lambda: requested.append("save"))

    def drive(dlg: QtWidgets.QDialog) -> None:
        save = _find(dlg, QtWidgets.QPushButton, "saveProjectBtn")
        decline = _find(dlg, QtWidgets.QPushButton, "dontSaveBtn")
        tabs = _find(dlg, QtWidgets.QTabWidget, "tabWidget")
        _assert_shown(dlg, save, decline, tabs)
        assert isinstance(save, QtWidgets.QPushButton)
        assert isinstance(decline, QtWidgets.QPushButton)
        save.click()
        decline.click()

    assert _exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Rejected
    assert requested == ["save"]


def test_out_of_memory_dialog_success_page_closes(qtbot: QtBot):
    dialog = OutOfMemoryDialog()
    qtbot.addWidget(dialog)

    def drive(dlg: QtWidgets.QDialog) -> None:
        assert isinstance(dlg, OutOfMemoryDialog)
        dlg.show_save_success()
        button_box = _find(dlg, QtWidgets.QDialogButtonBox, "buttonBox")
        _assert_shown(dlg, button_box)
        assert isinstance(button_box, QtWidgets.QDialogButtonBox)
        button_box.rejected.emit()

    assert _exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Rejected


def test_about_dialog_shows_version_and_license_and_closes(qtbot: QtBot):
    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    seen: dict[str, str] = {}

    def drive(dlg: QtWidgets.QDialog) -> None:
        version = _find(dlg, QtWidgets.QLabel, "version")
        button_box = _find(dlg, QtWidgets.QDialogButtonBox, "buttonBox")
        _assert_shown(dlg, version, button_box)
        tabs = _find(dlg, QtWidgets.QTabWidget, "tabWidget")
        assert isinstance(tabs, QtWidgets.QTabWidget)
        tabs.setCurrentIndex(tabs.count() - 1)
        QtWidgets.QApplication.processEvents()
        license_viewer = _find(dlg, QtWidgets.QTextBrowser, "licenseViewer")
        _assert_shown(dlg, license_viewer)
        assert isinstance(version, QtWidgets.QLabel)
        seen["version"] = version.text()
        assert isinstance(button_box, QtWidgets.QDialogButtonBox)
        button_box.buttons()[0].click()

    assert _exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Accepted
    assert seen["version"].startswith("version ")


def test_split_mode_dialog_reports_chosen_settings_on_ok(qtbot: QtBot):
    page = PageId(
        image_id=ImageId(file_path=pathlib.Path("a.png"), page=0),
        sub_page=SubPage.SINGLE_PAGE,
    )
    dialog = SplitModeDialog(
        current_page=page,
        all_pages=[page],
        layout_type=LayoutType.AUTO_LAYOUT_TYPE,
    )
    qtbot.addWidget(dialog)
    accepted: list[tuple[set[PageId], LayoutType, bool]] = []
    dialog.settings_accepted.connect(
        lambda pages, layout, cut: accepted.append((pages, layout, cut))
    )

    def drive(dlg: QtWidgets.QDialog) -> None:
        mode_auto = _find(dlg, QtWidgets.QRadioButton, "modeAuto")
        this_page = _find(dlg, QtWidgets.QRadioButton, "thisPageRB")
        button_box = _find(dlg, QtWidgets.QDialogButtonBox, "buttonBox")
        _assert_shown(dlg, mode_auto, this_page, button_box)
        assert isinstance(button_box, QtWidgets.QDialogButtonBox)
        button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    assert _exec_while(dialog, drive) == QtWidgets.QDialog.DialogCode.Accepted
    assert accepted == [({page}, LayoutType.AUTO_LAYOUT_TYPE, False)]


@pytest.mark.parametrize(
    "dialog_factory",
    [
        lambda: AboutDialog(),
        lambda: LoadFilesStatusDialog(),
        lambda: RemovePagesDialog(page_count=1),
        lambda: OutOfMemoryDialog(),
        lambda: ProjectFilesDialog(),
        lambda: SplitModeDialog(),
    ],
    ids=["about", "load_status", "remove_pages", "out_of_memory", "files", "split"],
)
def test_embedded_form_is_a_child_widget_not_a_second_window(
    qtbot: QtBot, dialog_factory: Callable[[], QtWidgets.QDialog]
):
    dialog = dialog_factory()
    qtbot.addWidget(dialog)

    form = next(
        child for child in dialog.children() if isinstance(child, QtWidgets.QDialog)
    )
    assert isinstance(form, QtWidgets.QDialog)
    assert not form.isWindow()
    assert dialog.windowTitle() == form.windowTitle()


def test_new_project_through_the_real_project_files_dialog(
    main_window: MainWindowManager, tmp_path: pathlib.Path
):
    scans = tmp_path / "scans"
    scans.mkdir()
    for name in ("p10.png", "p2.png", "p1.png"):
        _write_png(scans / name)
    out = tmp_path / "out"

    def drive(dlg: QtWidgets.QDialog) -> None:
        input_line = _find(dlg, QtWidgets.QLineEdit, "inpDirLine")
        output_line = _find(dlg, QtWidgets.QLineEdit, "outDirLine")
        _assert_shown(dlg, input_line, output_line)
        assert isinstance(input_line, QtWidgets.QLineEdit)
        assert isinstance(output_line, QtWidgets.QLineEdit)
        input_line.setText(str(scans))
        output_line.setText(str(out))
        _pick_all_files(dlg)
        button_box = _find(dlg, QtWidgets.QDialogButtonBox, "buttonBox")
        assert isinstance(button_box, QtWidgets.QDialogButtonBox)
        button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    def find_files_dialog() -> ProjectFilesDialog | None:
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(widget, ProjectFilesDialog) and widget.isVisible():
                return widget
        return None

    clock = QtCore.QElapsedTimer()
    clock.start()
    failures: list[BaseException] = []

    def poll() -> None:
        dialog = find_files_dialog()
        if dialog is None:
            if not clock.hasExpired(_TIMEOUT_MS):
                QtCore.QTimer.singleShot(_POLL_MS, poll)
            return
        try:
            drive(dialog)
        except BaseException as error:
            failures.append(error)
            dialog.reject()

    QtCore.QTimer.singleShot(_POLL_MS, poll)
    main_window.actionNewProject.trigger()

    if failures:
        raise failures[0]
    project = main_window.project
    assert project is not None
    assert [info.id.file_path.name for info in project.images] == [
        "p1.png",
        "p2.png",
        "p10.png",
    ]
    assert project.output_directory == out
    assert out.is_dir()


def test_project_files_dialog_keeps_files_of_several_folders_apart(
    qtbot: QtBot, tmp_path: pathlib.Path
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for folder in (first, second):
        folder.mkdir()
        _write_png(folder / "p1.png")
    dialog = ProjectFilesDialog(input_dir=first)
    qtbot.addWidget(dialog)
    _pick_all_files(dialog)

    dialog._input_dir_line.setText(str(second))
    _pick_all_files(dialog)

    assert dialog.get_project_paths() == [first / "p1.png", second / "p1.png"]
    in_project = _find(dialog, QtWidgets.QListView, "inProjectList")
    assert isinstance(in_project, QtWidgets.QListView)
    model = in_project.model()
    assert [model.index(row, 0).data() for row in range(model.rowCount())] == [
        f"p1.png ({first})",
        f"p1.png ({second})",
    ]


def test_project_files_dialog_resolves_bare_names_against_the_input_folder(
    qtbot: QtBot, tmp_path: pathlib.Path
):
    dialog = ProjectFilesDialog(input_dir=tmp_path, project_files=["p10.png", "p2.png"])
    qtbot.addWidget(dialog)

    assert dialog.get_project_paths() == [tmp_path / "p2.png", tmp_path / "p10.png"]
