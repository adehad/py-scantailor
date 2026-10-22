"""The error panel shown in place of a page whose source image cannot be loaded."""

from __future__ import annotations

import pathlib
import shutil
from collections.abc import Callable

from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.relinking import RelinkingDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.widgets.error_widget import ErrorWidget, load_failure_message

TIMEOUT_MS = 10_000


class RelinkingScript:
    """Relinking dialog factory that records each call and relinks one folder."""

    def __init__(self, old: pathlib.Path, new: pathlib.Path) -> None:
        self.old, self.new = old, new
        self.calls = 0

    def __call__(
        self, directories: list[pathlib.Path], parent: QtWidgets.QWidget
    ) -> RelinkingDialog:
        self.calls += 1
        dialog = RelinkingDialog(directories, parent)
        dialog.relink_directory(self.old, self.new)
        return dialog


def _move_scans_away(window: MainWindowManager) -> tuple[pathlib.Path, pathlib.Path]:
    """Move the project's scans folder and forget the decoded sources.

    Returns:
        The folder the scans were in and the folder they are in now.
    """
    assert window.project is not None
    old = window.project.images[0].id.file_path.parent
    new = old.with_name("scans_moved")
    shutil.move(str(old), str(new))
    window.renderer.sources.clear()
    return old, new


def _wait_for_error_panel(qtbot: QtBot, window: MainWindowManager) -> None:
    qtbot.waitUntil(
        lambda: window.imageFrameLayout.currentWidget() is window.load_error_widget,
        timeout=TIMEOUT_MS,
    )


def test_missing_source_shows_the_error_panel_with_a_relinking_link(
    qtbot: QtBot, project_window: MainWindowManager
):
    _move_scans_away(project_window)

    project_window.goNextPage()

    _wait_for_error_panel(qtbot, project_window)
    label = project_window.load_error_widget.text_label
    assert label.textFormat() == QtCore.Qt.TextFormat.RichText
    assert "doesn't exist" in label.text()
    assert '<a href="#relink">Relinking Tool</a>' in label.text()
    page = project_window.current_page
    assert page is not None
    assert QtCore.QDir.toNativeSeparators(str(page.image_id.file_path)) in label.text()
    assert project_window._panels is not None
    assert project_window._panels.options.isHidden()


def test_unreadable_source_shows_the_error_panel_without_a_link(
    qtbot: QtBot, project_window: MainWindowManager
):
    assert project_window.project is not None
    second = project_window.project.images[1].id.file_path
    second.write_bytes(b"not an image")
    project_window.renderer.sources.clear()

    project_window.goNextPage()

    _wait_for_error_panel(qtbot, project_window)
    label = project_window.load_error_widget.text_label
    assert label.textFormat() == QtCore.Qt.TextFormat.PlainText
    expected = QtCore.QDir.toNativeSeparators(str(second))
    assert label.text() == f"The following file could not be loaded:\n{expected}"


def test_going_to_a_loadable_page_replaces_the_error_panel_with_the_view(
    qtbot: QtBot, project_window: MainWindowManager
):
    assert project_window.project is not None
    project_window.project.images[1].id.file_path.write_bytes(b"not an image")
    project_window.renderer.sources.clear()
    project_window.goNextPage()
    _wait_for_error_panel(qtbot, project_window)

    project_window.goPrevPage()

    panels = project_window._panels
    assert panels is not None
    assert project_window.imageFrameLayout.currentWidget() is panels.view
    assert not panels.options.isHidden()


def test_clicking_the_link_opens_relinking_and_the_relinked_page_loads(
    qtbot: QtBot,
    project_window: MainWindowManager,
    respond_to_modal: Callable[..., None],
):
    old, new = _move_scans_away(project_window)
    project_window.goNextPage()
    _wait_for_error_panel(qtbot, project_window)
    script = RelinkingScript(old, new)
    project_window.relinking_dialog_factory = script
    respond_to_modal(accept=True)

    project_window.load_error_widget.text_label.linkActivated.emit("#relink")

    assert script.calls == 1
    page = project_window.current_page
    assert page is not None
    assert page.image_id.file_path.parent == new
    panels = project_window._panels
    assert panels is not None
    qtbot.waitUntil(lambda: panels._source is not None, timeout=TIMEOUT_MS)
    assert project_window.imageFrameLayout.currentWidget() is panels.view
    project_window.renderer.wait_for_done(TIMEOUT_MS)


def test_load_failure_message_escapes_the_path_of_a_missing_file(
    tmp_path: pathlib.Path,
):
    missing = tmp_path / "a&b<c>.png"

    text, text_format = load_failure_message(missing)

    assert text_format == QtCore.Qt.TextFormat.RichText
    assert "a&amp;b&lt;c&gt;.png" in text


def test_error_widget_reports_the_clicked_link(qtbot: QtBot):
    widget = ErrorWidget()
    qtbot.addWidget(widget)
    widget.set_message('<a href="#relink">here</a>', QtCore.Qt.TextFormat.RichText)

    with qtbot.waitSignal(widget.link_activated) as blocker:
        widget.text_label.linkActivated.emit("#relink")

    assert blocker.args == ["#relink"]
