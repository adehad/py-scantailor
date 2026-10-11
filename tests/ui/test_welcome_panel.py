"""The welcome panel shown in the image area while no project is open."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.panels.new_open_project import NewOpenProjectPanel
from scantailor.core.image_io import list_image_pages
from scantailor.core.project import Project
from scantailor.core.recent_projects import RecentProjects

if TYPE_CHECKING:
    from conftest import PageImages


def _panel(window: MainWindowManager) -> NewOpenProjectPanel | None:
    current = window.imageFrameLayout.currentWidget()
    return current.findChild(NewOpenProjectPanel) if current else None


def _click(qtbot: QtBot, label: QtWidgets.QLabel) -> None:
    qtbot.mouseClick(label, QtCore.Qt.MouseButton.LeftButton)


def _recent_labels(panel: NewOpenProjectPanel) -> list[QtWidgets.QLabel]:
    group = panel.findChild(QtWidgets.QGroupBox, "recentProjectsGroup")
    return group.findChildren(QtWidgets.QLabel)


def _save_project(page_images: PageImages, path: pathlib.Path) -> pathlib.Path:
    project = Project(output_directory=path.parent / "out")
    (path.parent / "out").mkdir(exist_ok=True)
    for image_path in page_images.paths:
        for image_id, metadata in list_image_pages(image_path):
            project.add_image(image_id, metadata)
    project.save(path)
    return path


def _remember(window: MainWindowManager, *paths: pathlib.Path) -> None:
    RecentProjects([str(path) for path in paths]).write(window.settings)
    window._update_welcome_panel()


def test_welcome_panel_fills_the_image_area_when_no_project_is_open(
    main_window: MainWindowManager,
):
    assert _panel(main_window) is not None
    assert _panel(main_window).isVisible()


def test_recent_group_is_hidden_without_recent_projects(
    main_window: MainWindowManager,
):
    group = _panel(main_window).findChild(QtWidgets.QGroupBox, "recentProjectsGroup")

    assert not group.isVisible()


def test_recent_projects_are_listed_newest_first_by_base_name(
    main_window: MainWindowManager, tmp_path: pathlib.Path
):
    first, second = tmp_path / "alpha.ScanTailor", tmp_path / "beta.v2.ScanTailor"
    first.touch()
    second.touch()

    _remember(main_window, second, first)

    labels = _recent_labels(_panel(main_window))
    assert [label.text() for label in labels] == ["beta.v2", "alpha"]
    assert [label.toolTip() for label in labels] == [str(second), str(first)]


def test_missing_recent_projects_are_dropped_and_forgotten(
    main_window: MainWindowManager, tmp_path: pathlib.Path
):
    present = tmp_path / "here.ScanTailor"
    present.touch()

    _remember(main_window, tmp_path / "gone.ScanTailor", present)

    assert [label.text() for label in _recent_labels(_panel(main_window))] == ["here"]
    assert RecentProjects.read(main_window.settings).enumerate() == [str(present)]


def test_clicking_a_recent_project_opens_it(
    qtbot: QtBot,
    main_window: MainWindowManager,
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    project_file = _save_project(page_images, tmp_path / "book.ScanTailor")
    _remember(main_window, project_file)

    _click(qtbot, _recent_labels(_panel(main_window))[0])

    qtbot.waitUntil(lambda: main_window.projectFile == str(project_file))
    assert main_window.project is not None


def test_welcome_panel_gives_way_to_the_project_and_returns_on_close(
    main_window: MainWindowManager,
    page_images: PageImages,
    tmp_path: pathlib.Path,
):
    project_file = _save_project(page_images, tmp_path / "book.ScanTailor")

    assert main_window.open_project_file(project_file)
    assert _panel(main_window) is None

    main_window.closeProject()
    assert _panel(main_window) is not None


def test_a_recent_project_that_cannot_be_opened_is_dropped_from_the_panel(
    qtbot: QtBot,
    main_window: MainWindowManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    broken = tmp_path / "broken.ScanTailor"
    broken.write_text("not a project")
    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2]),
    )
    _remember(main_window, broken)

    _click(qtbot, _recent_labels(_panel(main_window))[0])

    qtbot.waitUntil(lambda: bool(warnings))
    assert main_window.project is None
    assert _panel(main_window) is not None


@pytest.mark.parametrize(
    ("label_name", "handler"),
    [("newProjectLabel", "newProject"), ("openProjectLabel", "openProject")],
)
def test_new_and_open_links_trigger_the_matching_action(
    qtbot: QtBot,
    main_window: MainWindowManager,
    monkeypatch: pytest.MonkeyPatch,
    label_name: str,
    handler: str,
):
    calls: list[str] = []
    monkeypatch.setattr(main_window, handler, lambda: calls.append(handler))

    _click(qtbot, _panel(main_window).findChild(QtWidgets.QLabel, label_name))

    qtbot.waitUntil(lambda: calls == [handler])


def test_a_right_click_on_a_link_does_nothing(qtbot: QtBot):
    panel = NewOpenProjectPanel()
    qtbot.addWidget(panel)
    seen: list[str] = []
    panel.new_project_requested.connect(lambda: seen.append("new"))

    qtbot.mouseClick(
        panel.findChild(QtWidgets.QLabel, "newProjectLabel"),
        QtCore.Qt.MouseButton.RightButton,
    )

    assert seen == []
