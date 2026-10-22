"""The project stores the new size of a source file replaced by a different image."""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.batch import PageDetected, PageDone, RenderInfo
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.transformation import ImageTransformation, Rect

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


def test_a_thumbnail_render_of_a_replaced_file_stores_its_new_size(
    qtbot: QtBot, project_window: MainWindowManager
) -> None:
    """The page is never opened, so only the thumbnail worker decodes it."""
    assert project_window.project is not None
    _replace_second_scan(project_window, (900, 700), dpi=300)
    project_window.modified = False

    project_window._invalidate_thumbnails()
    project_window.thumbnails.view.viewport().repaint()

    info = project_window.project.images[1]
    qtbot.waitUntil(lambda: info.metadata.width == 900, timeout=TIMEOUT_MS)
    assert info.metadata.height == 700
    assert project_window.modified
    current = project_window.current_page
    assert current is not None and current.image_id != info.id


def test_a_batch_detection_of_a_replaced_file_stores_its_new_size(
    project_window: MainWindowManager,
) -> None:
    assert project_window.project is not None
    info = project_window.project.images[1]
    page = PageId(image_id=info.id)

    project_window._on_batch_detected(PageDetected(page, source_size=(900, 700)))

    assert (info.metadata.width, info.metadata.height) == (900, 700)
    assert project_window.modified


def test_a_batch_render_of_a_replaced_file_stores_its_new_size(
    project_window: MainWindowManager,
) -> None:
    assert project_window.project is not None
    info = project_window.project.images[1]
    page = PageId(image_id=info.id)
    render_info = RenderInfo(
        ImageTransformation(Rect(0, 0, 900, 700), Dpi.uniform(300)),
        Dpi.uniform(300),
        was_dewarped=False,
        source_size=(900, 700),
    )

    project_window._on_batch_progress(
        PageDone(page, info.id.file_path, render_info, files=())
    )

    assert (info.metadata.width, info.metadata.height) == (900, 700)
    assert project_window.modified
