"""The window opens a project file named on the command line."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from PySide6 import QtWidgets

from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.image_io import list_image_pages
from scantailor.core.project import Project
from scantailor.main import open_project_from_arguments, project_file_from_arguments

if TYPE_CHECKING:
    from conftest import PageImages


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collect the text of each warning box instead of showing it."""
    shown: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: shown.append(args[2]),
    )
    return shown


def _save_project(page_images: PageImages, path: pathlib.Path) -> pathlib.Path:
    project = Project(output_directory=path.parent / "out")
    (path.parent / "out").mkdir(exist_ok=True)
    for image_path in page_images.paths:
        for image_id, metadata in list_image_pages(image_path):
            project.add_image(image_id, metadata)
    project.save(path)
    return path


def test_no_argument_after_the_program_name_names_no_project() -> None:
    assert project_file_from_arguments(["ScanTailor"]) is None


def test_an_empty_argument_list_names_no_project() -> None:
    assert project_file_from_arguments([]) is None


def test_the_first_argument_is_the_project_made_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.chdir(tmp_path)

    project_file = project_file_from_arguments(["ScanTailor", "book.ScanTailor"])

    assert project_file == tmp_path / "book.ScanTailor"


def test_arguments_after_the_project_are_ignored(tmp_path: pathlib.Path) -> None:
    first = tmp_path / "first.ScanTailor"

    project_file = project_file_from_arguments(
        ["ScanTailor", str(first), str(tmp_path / "second.ScanTailor")]
    )

    assert project_file == first


def _arguments_left_by_qt(arguments: list[str]) -> list[str]:
    """Run the arguments through a QApplication in a child interpreter.

    The test process already has its QApplication, and Qt allows only one.
    """
    script = (
        "import sys\n"
        "from PySide6 import QtWidgets\n"
        "app = QtWidgets.QApplication(sys.argv[1:])\n"
        "print('\\n'.join(app.arguments()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    return result.stdout.splitlines()


def test_qt_options_before_the_project_are_not_taken_for_it() -> None:
    arguments = _arguments_left_by_qt(
        ["ScanTailor", "-style", "fusion", "book.ScanTailor"]
    )

    assert project_file_from_arguments(arguments) == (
        pathlib.Path("book.ScanTailor").absolute()
    )


def test_a_project_named_on_the_command_line_opens_in_the_window(
    main_window: MainWindowManager,
    page_images: PageImages,
    tmp_path: pathlib.Path,
    warnings: list[str],
) -> None:
    project_file = _save_project(page_images, tmp_path / "book.ScanTailor")

    assert open_project_from_arguments(main_window, ["ScanTailor", str(project_file)])

    assert main_window.project is not None
    assert main_window.projectFile == str(project_file)
    assert warnings == []


def test_without_a_project_argument_the_window_stays_empty(
    main_window: MainWindowManager, warnings: list[str]
) -> None:
    assert not open_project_from_arguments(main_window, ["ScanTailor"])

    assert main_window.project is None
    assert warnings == []


def _leave_missing(_path: pathlib.Path) -> None:
    pass


def _write_garbage(path: pathlib.Path) -> None:
    path.write_text("not a project")


@pytest.mark.parametrize(
    "make_bad_file",
    [
        pytest.param(_leave_missing, id="missing"),
        pytest.param(_write_garbage, id="not-a-project"),
        pytest.param(pathlib.Path.mkdir, id="a-folder"),
    ],
)
def test_a_bad_project_argument_warns_and_leaves_no_project_open(
    main_window: MainWindowManager,
    tmp_path: pathlib.Path,
    warnings: list[str],
    make_bad_file: Callable[[pathlib.Path], object],
) -> None:
    bad_file = tmp_path / "book.ScanTailor"
    make_bad_file(bad_file)

    assert not open_project_from_arguments(main_window, ["ScanTailor", str(bad_file)])

    assert main_window.project is None
    assert len(warnings) == 1
    assert warnings[0].startswith("Unable to open the project file.")
