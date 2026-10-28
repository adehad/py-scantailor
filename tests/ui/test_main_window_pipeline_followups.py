"""Auto-save after a batch, aggregate size changes and defaults for new pages."""

import pathlib
import shutil

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.batch import BatchResult
from scantailor.core.default_params import DefaultParams, PageLayoutDefaults
from scantailor.core.models import Margins, PageId, SubPage
from scantailor.core.settings import ApplicationSettings
from scantailor.filters.page_layout.params import ContentRect, ContentSize
from tests.ui.conftest import settle


@pytest.fixture
def auto_save_enabled():
    settings = ApplicationSettings.get_instance()
    previous = settings.auto_save_project
    settings.auto_save_project = True
    yield settings
    settings.auto_save_project = previous


@pytest.fixture
def settled_window(qtbot: QtBot, project_window: MainWindowManager):
    settle(project_window)
    return project_window


def _pretend_batch(monkeypatch: pytest.MonkeyPatch, window: MainWindowManager):
    """Make the batch runner report a running batch until the flag is cleared."""
    state = {"running": True}
    monkeypatch.setattr(
        type(window.batch_runner), "is_running", property(lambda _: state["running"])
    )
    return state


class TestAutoSaveDuringBatch:
    def test_an_auto_save_that_falls_due_in_a_batch_runs_when_the_batch_ends(
        self,
        settled_window: MainWindowManager,
        auto_save_enabled,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        window = settled_window
        target = tmp_path / "auto.ScanTailor"
        window.projectFile = str(target)
        batch = _pretend_batch(monkeypatch, window)

        window._auto_save_project()
        assert not target.exists()
        batch["running"] = False
        window._on_batch_finished(BatchResult())

        assert target.is_file()
        window.renderer.wait_for_done()

    def test_a_failed_batch_also_releases_the_deferred_save(
        self,
        settled_window: MainWindowManager,
        auto_save_enabled,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        window = settled_window
        target = tmp_path / "auto.ScanTailor"
        window.projectFile = str(target)
        batch = _pretend_batch(monkeypatch, window)
        monkeypatch.setattr(QMessageBox, "warning", lambda *_a, **_k: None)

        window._auto_save_project()
        batch["running"] = False
        window._on_batch_failed("boom")

        assert target.is_file()
        window.renderer.wait_for_done()

    def test_a_batch_that_ends_without_a_due_auto_save_writes_nothing(
        self,
        settled_window: MainWindowManager,
        auto_save_enabled,
        tmp_path: pathlib.Path,
    ):
        window = settled_window
        target = tmp_path / "auto.ScanTailor"
        window.projectFile = str(target)

        window._on_batch_finished(BatchResult())

        assert not target.exists()
        window.renderer.wait_for_done()

    def test_the_deferred_save_is_not_repeated_after_it_ran(
        self,
        settled_window: MainWindowManager,
        auto_save_enabled,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        window = settled_window
        target = tmp_path / "auto.ScanTailor"
        window.projectFile = str(target)
        batch = _pretend_batch(monkeypatch, window)
        window._auto_save_project()
        batch["running"] = False
        window._on_batch_finished(BatchResult())
        target.unlink()

        window._on_batch_finished(BatchResult())

        assert not target.exists()
        window.renderer.wait_for_done()

    def test_with_auto_save_off_nothing_is_deferred_or_written(
        self,
        settled_window: MainWindowManager,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        window = settled_window
        settings = ApplicationSettings.get_instance()
        previous = settings.auto_save_project
        settings.auto_save_project = False
        try:
            target = tmp_path / "auto.ScanTailor"
            window.projectFile = str(target)
            batch = _pretend_batch(monkeypatch, window)
            window._auto_save_project()
            batch["running"] = False

            window._on_batch_finished(BatchResult())
        finally:
            settings.auto_save_project = previous

        assert not target.exists()
        window.renderer.wait_for_done()


class TestThumbnailDigestPerStage:
    @staticmethod
    def _halves(window: MainWindowManager) -> tuple[PageId, PageId]:
        image = window.pages()[0].image_id
        return (
            PageId(image_id=image, sub_page=SubPage.LEFT_PAGE),
            PageId(image_id=image, sub_page=SubPage.RIGHT_PAGE),
        )

    @pytest.mark.parametrize("stage", ["0", "1"])
    def test_editing_either_half_makes_the_scan_thumbnail_stale_where_scans_are_listed(
        self, settled_window: MainWindowManager, stage: str
    ):
        left, right = self._halves(settled_window)
        before = settled_window._thumbnail_digest(left, stage)

        settled_window._invalidate_thumbnails(right)

        assert settled_window._thumbnail_digest(left, stage) != before

    @pytest.mark.parametrize("stage", ["2", "5"])
    def test_editing_one_half_keeps_the_other_halfs_thumbnail_where_pages_are_listed(
        self, settled_window: MainWindowManager, stage: str
    ):
        left, right = self._halves(settled_window)
        before = settled_window._thumbnail_digest(left, stage)

        settled_window._invalidate_thumbnails(right)

        assert settled_window._thumbnail_digest(left, stage) == before


class TestAggregateSizeChange:
    @staticmethod
    def _record_content(window: MainWindowManager, page: PageId, width: float) -> None:
        rect = ContentRect(x=0, y=0, width=100, height=100)
        window.stages.page_layout_filter.settings.update_content_size_and_get_aggregate(
            page, rect, rect, ContentSize(width_mm=width, height_mm=50.0)
        )

    def test_a_render_that_enlarges_the_aggregate_refreshes_other_pages(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        current, other = window.pages()[0], window.pages()[1]
        window._remember_aggregate_size()
        before = window._thumbnail_digest(other, "5")

        self._record_content(window, current, 400.0)
        window._on_render_done(other, np.zeros((4, 4), dtype=np.uint8))

        assert window._thumbnail_digest(other, "5") != before

    def test_a_render_that_leaves_the_aggregate_alone_keeps_other_thumbnails(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        other = window.pages()[1]
        window._remember_aggregate_size()
        before = window._thumbnail_digest(other, "5")

        window._on_render_done(other, np.zeros((4, 4), dtype=np.uint8))

        assert window._thumbnail_digest(other, "5") == before

    def test_a_change_is_reported_once(self, settled_window: MainWindowManager):
        window = settled_window
        window._remember_aggregate_size()
        self._record_content(window, window.pages()[0], 400.0)

        assert window._remember_aggregate_size()
        assert not window._remember_aggregate_size()


class TestNewPagesUseTheDefaults:
    DEFAULT_MARGINS = Margins(left=22.0, right=22.0, top=11.0, bottom=11.0)

    @pytest.fixture
    def new_image(self, page_images, dialog_factory) -> pathlib.Path:
        path = page_images.directory / "added.png"
        shutil.copy(page_images.paths[0], path)
        dialog_factory.names = [path.name]
        return path

    def test_an_inserted_page_starts_from_the_default_margins(
        self,
        settled_window: MainWindowManager,
        new_image: pathlib.Path,
        respond_to_modal,
        monkeypatch: pytest.MonkeyPatch,
    ):
        window = settled_window
        defaults = DefaultParams(
            page_layout=PageLayoutDefaults(hard_margins_mm=self.DEFAULT_MARGINS)
        )
        monkeypatch.setattr(MainWindow, "active_default_params", lambda: defaults)
        existing = window.pages()[0]
        existing_before = window.stages.page_layout_filter.get_params(existing)
        respond_to_modal(accept=True)

        window.insert_images_at_end()

        added = [
            page for page in window.pages() if page.image_id.file_path == new_image
        ]
        assert added
        for page in added:
            params = window.stages.page_layout_filter.get_params(page)
            assert params.hard_margins_mm == self.DEFAULT_MARGINS
        assert window.stages.page_layout_filter.get_params(existing) == existing_before
        window.renderer.wait_for_done()
