"""The window title names the application and its version from the package metadata."""

import importlib.metadata

from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.app.ui.MainWindow import MainWindowManager

EXPECTED_TITLE = f"pyScanTailor - {importlib.metadata.version('py-scantailor')}"


def test_without_a_project_the_title_is_the_app_name_and_version(
    main_window: MainWindowManager,
):
    assert main_window.project is None
    assert main_window.ui.windowTitle() == EXPECTED_TITLE


def test_a_project_name_and_the_modified_marker_come_before_the_app_name(
    project_window: MainWindowManager,
):
    assert project_window.ui.windowTitle() == f"Unnamed[*] - {EXPECTED_TITLE}"

    project_window.modified = True

    assert project_window.ui.isWindowModified()


def test_closing_the_project_goes_back_to_the_plain_title(
    project_window: MainWindowManager,
):
    project_window.modified = False

    project_window.actionCloseProject.trigger()

    assert project_window.project is None
    assert project_window.ui.windowTitle() == EXPECTED_TITLE


def test_the_about_dialog_is_titled_with_the_app_name(qtbot: QtBot):
    dialog = AboutDialog(parent=None)
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "pyScanTailor"
