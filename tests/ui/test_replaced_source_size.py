"""The project stores the new size of a source file replaced by a different image."""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.models import ImageId

TIMEOUT_MS = 10_000


def _replace_second_scan(
    window: MainWindowManager, size: tuple[int, int], dpi: int
) -> pathlib.Path:
    """Overwrite the second page's file with a blank page of another size."""
    assert window.project is not None
    path = window.project.images[1].id.file_path
    width, height = size
    Image.fromarray(np.full((height, width), 255, dtype=np.uint8)).save(
        path, dpi=(dpi, dpi)
    )
    window.renderer.sources.clear()
    return path


def _wait_for_source(qtbot: QtBot, window: MainWindowManager, image: ImageId) -> None:
    def shown() -> bool:
        panels = window._panels
        return (
            panels is not None
            and panels._source is not None
            and panels._source.image_id == image
        )

    qtbot.waitUntil(shown, timeout=TIMEOUT_MS)


def test_a_replaced_file_stores_its_new_size_and_marks_the_project_modified(
    qtbot: QtBot, project_window: MainWindowManager
) -> None:
    assert project_window.project is not None
    _replace_second_scan(project_window, (900, 700), dpi=300)
    project_window.modified = False

    project_window.goNextPage()

    info = project_window.project.images[1]
    qtbot.waitUntil(lambda: info.metadata.width == 900, timeout=TIMEOUT_MS)
    assert info.metadata.height == 700
    assert project_window.modified


def test_a_replaced_file_keeps_the_resolution_the_project_holds(
    qtbot: QtBot, project_window: MainWindowManager
) -> None:
    assert project_window.project is not None
    stored_dpi = project_window.project.images[1].metadata.dpi
    _replace_second_scan(project_window, (900, 700), dpi=150)

    project_window.goNextPage()

    info = project_window.project.images[1]
    qtbot.waitUntil(lambda: info.metadata.width == 900, timeout=TIMEOUT_MS)
    assert info.metadata.dpi == stored_dpi


def test_an_unchanged_file_leaves_the_project_unmodified(
    qtbot: QtBot, project_window: MainWindowManager
) -> None:
    assert project_window.project is not None
    before = project_window.project.images[1].metadata
    project_window.modified = False

    project_window.goNextPage()

    _wait_for_source(qtbot, project_window, project_window.project.images[1].id)
    assert project_window.project.images[1].metadata == before
    assert not project_window.modified
