"""What a thumbnail kept on disk is named by, so a later session can trust it."""

import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QImage
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager, thumbnail_cache_directory
from scantailor.app.ui.page_renderer import thumbnail_file_digest
from scantailor.app.ui.thumbnails.cache import ThumbnailKey, page_file_prefix
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.project import Project
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.color_mode import ColorMode


def _write_scan(path: Path, height: int = 120) -> None:
    write_image(path, np.full((height, 90), 200, dtype=np.uint8), Dpi.uniform(300))


@pytest.fixture
def page(tmp_path: Path) -> PageId:
    path = tmp_path / "page.png"
    _write_scan(path)
    return PageId(image_id=ImageId(file_path=path))


def test_the_same_page_and_parameters_give_the_same_digest_in_new_stages(
    page: PageId,
) -> None:
    """New stages stand for a later session that loaded the same project."""
    first = thumbnail_file_digest([page], StageSequence(), FilterStage.DESKEW)
    second = thumbnail_file_digest([page], StageSequence(), FilterStage.DESKEW)

    assert first is not None
    assert first == second


def test_a_replaced_source_file_changes_the_digest(page: PageId) -> None:
    stages = StageSequence()
    before = thumbnail_file_digest([page], stages, FilterStage.DESKEW)
    _write_scan(page.image_id.file_path, height=150)
    stat = page.image_id.file_path.stat()
    os.utime(page.image_id.file_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    assert thumbnail_file_digest([page], stages, FilterStage.DESKEW) != before


def test_an_edit_at_the_shown_stage_changes_the_digest(page: PageId) -> None:
    stages = StageSequence()
    before = thumbnail_file_digest([page], stages, FilterStage.DESKEW)
    stages.deskew_filter.set_manual_angle(page, 2.0)

    assert thumbnail_file_digest([page], stages, FilterStage.DESKEW) != before


def test_an_output_edit_changes_the_output_digest_only(page: PageId) -> None:
    stages = StageSequence()
    deskew_before = thumbnail_file_digest([page], stages, FilterStage.DESKEW)
    output_before = thumbnail_file_digest([page], stages, FilterStage.OUTPUT)
    params = stages.output_filter.get_params(page)
    stages.output_filter.settings.set_params(
        page, params.model_copy(update={"color_mode": ColorMode.MIXED})
    )

    assert thumbnail_file_digest([page], stages, FilterStage.OUTPUT) != output_before
    assert thumbnail_file_digest([page], stages, FilterStage.DESKEW) == deskew_before


def test_a_project_resolution_changes_the_digest(page: PageId) -> None:
    stages = StageSequence()
    before = thumbnail_file_digest([page], stages, FilterStage.DESKEW)
    overrides = {page.image_id: Dpi.uniform(150)}

    assert thumbnail_file_digest([page], stages, FilterStage.DESKEW, overrides) != (
        before
    )


def test_a_removed_half_changes_the_digest(page: PageId) -> None:
    stages = StageSequence()
    kept = thumbnail_file_digest([page], stages, FilterStage.PAGE_SPLIT)
    removed = thumbnail_file_digest(
        [page], stages, FilterStage.PAGE_SPLIT, removed_halves=(True, False)
    )

    assert kept != removed


def test_a_missing_source_file_gives_no_digest(tmp_path: Path) -> None:
    missing = PageId(image_id=ImageId(file_path=tmp_path / "gone.png"))

    assert thumbnail_file_digest([missing], StageSequence(), FilterStage.DESKEW) is (
        None
    )


def test_thumbnails_are_kept_under_the_output_folder(tmp_path: Path) -> None:
    project = Project(output_directory=tmp_path / "out")

    assert thumbnail_cache_directory(project) == tmp_path / "out" / "cache" / "thumbs"


def test_no_thumbnail_folder_without_a_project_or_with_a_relative_output_folder() -> (
    None
):
    assert thumbnail_cache_directory(None) is None
    assert thumbnail_cache_directory(Project(output_directory=Path("out"))) is None


def test_an_open_project_keeps_its_thumbnails_under_its_output_folder(
    qtbot: QtBot, project_window: MainWindowManager
) -> None:
    assert project_window.project is not None
    directory = project_window.project.output_directory / "cache" / "thumbs"

    project_window.thumbnails.view.viewport().repaint()

    assert project_window.thumbnails.model.cache.disk_directory == directory
    qtbot.waitUntil(lambda: any(directory.glob("*.png")), timeout=10_000)


def test_removing_a_page_deletes_its_kept_thumbnails_and_keeps_the_others(
    qtbot: QtBot, project_window: MainWindowManager
) -> None:
    assert project_window.project is not None
    cache = project_window.thumbnails.model.cache
    pages = project_window.pages()
    image = QImage(4, 4, QImage.Format.Format_RGB32)
    views = (("0", (160, 160)), ("3", (160, 160)), ("3", (320, 320)))
    for page in pages:
        for stage, size in views:
            cache.store_to_disk(ThumbnailKey(page, stage, 0, size), "d", image)
    directory = cache.disk_directory
    assert directory is not None
    project_window.remove_pages_prompt = lambda _count, _parent: False

    project_window.remove_pages([pages[1]])

    remaining = {path.name for path in directory.glob("*.png")}
    assert not any(name.startswith(page_file_prefix(pages[1])) for name in remaining)
    for kept in (pages[0], pages[2]):
        for stage, size in views:
            stem = ThumbnailKey(kept, stage, 0, size).file_stem()
            assert f"{stem}.png" in remaining
