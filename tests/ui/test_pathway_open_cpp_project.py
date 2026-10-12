"""Opening a C++ ScanTailor XML project through the normal Open Project action."""

from __future__ import annotations

import pathlib
import shutil
from typing import TYPE_CHECKING

import pytest
from PySide6 import QtWidgets

from scantailor.core.models import SubPage
from scantailor.filters.deskew.params import AutoManualMode
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.page_split.layout_type import LayoutType

if TYPE_CHECKING:
    from collections.abc import Callable

    from conftest import FileDialogStub

    from scantailor.app.ui.MainWindow import MainWindowManager

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures" / "xml_projects"


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


def _copy_fixture(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    target = tmp_path / name
    shutil.copytree(FIXTURES / name, target)
    return target / f"{name}.ScanTailor"


def test_open_project_reads_a_c_plus_plus_project_with_its_pages_and_settings(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    wait_for_thumbnails: Callable[[MainWindowManager], None],
    tmp_path: pathlib.Path,
):
    project_file = _copy_fixture(tmp_path, "split_and_mixed")
    file_dialogs.open_file = str(project_file)

    main_window.actionOpenProject.trigger()
    wait_for_thumbnails(main_window)

    left, right, cover = main_window.pages()
    assert [page.sub_page for page in (left, right, cover)] == [
        SubPage.LEFT_PAGE,
        SubPage.RIGHT_PAGE,
        SubPage.SINGLE_PAGE,
    ]
    assert main_window.current_page == right
    assert main_window.ui.windowTitle().startswith("split_and_mixed")

    stages = main_window.stages
    deskew = stages.deskew_filter.settings.get_params(left)
    assert deskew.mode is AutoManualMode.MANUAL
    assert deskew.deskew_angle_deg == pytest.approx(-1.5)
    assert stages.output_filter.settings.get_params(left).color_mode is ColorMode.MIXED
    assert (
        stages.output_filter.settings.get_params(right).dewarping.mode
        is DewarpingMode.MANUAL
    )
    assert (
        stages.page_split_filter.settings.get_params(left).layout_type
        is LayoutType.TWO_PAGES
    )
    assert stages.page_layout_filter.settings.get_params(left).hard_margins_mm.left == (
        12.5
    )


def test_opened_c_plus_plus_project_saves_as_json_and_reopens_with_its_settings(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    wait_for_thumbnails: Callable[[MainWindowManager], None],
    tmp_path: pathlib.Path,
):
    file_dialogs.open_file = str(_copy_fixture(tmp_path, "split_and_mixed"))
    main_window.actionOpenProject.trigger()
    wait_for_thumbnails(main_window)
    saved = tmp_path / "converted.ScanTailor"
    file_dialogs.save_file = str(saved)

    main_window.actionSaveProjectAs.trigger()

    assert saved.read_text(encoding="utf-8").lstrip().startswith("{")
    file_dialogs.open_file = str(saved)
    main_window.modified = False
    main_window.actionOpenProject.trigger()
    wait_for_thumbnails(main_window)
    left = main_window.pages()[0]
    assert main_window.stages.deskew_filter.settings.get_params(left).is_manual()


def test_open_project_refuses_a_newer_c_plus_plus_project_with_a_warning(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: pathlib.Path,
    warnings: list[str],
):
    file_dialogs.open_file = str(_copy_fixture(tmp_path, "newer_version"))

    main_window.actionOpenProject.trigger()

    assert len(warnings) == 1
    assert "newer" in warnings[0]
    assert main_window.project is None
