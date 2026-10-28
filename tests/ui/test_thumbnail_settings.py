"""Thumbnail cell size, render resolution and column mode follow the settings."""

import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from scantailor.app.ui.thumbnails import ThumbnailListWidget
from scantailor.core.models import ImageId, PageId
from scantailor.core.project import ImageMetadata, Project
from scantailor.core.settings import ApplicationSettings
from tests.ui.conftest import flush_events
from tests.ui.hosting import show_in_host

SOURCE_SIZE = QSize(800, 1200)


# Two worker threads making their first PySide call at the same moment deadlock
# in shiboken's per-module feature import.
_RENDER_LOCK = threading.Lock()


def render(_page: PageId, _stage: str) -> QImage:
    with _RENDER_LOCK:
        image = QImage(SOURCE_SIZE, QImage.Format.Format_RGB32)
        image.fill(QColor("steelblue"))
    return image


@pytest.fixture
def project() -> Project:
    project = Project()
    for number in range(1, 5):
        project.add_image(
            ImageId(file_path=Path(f"page_{number}.png")),
            ImageMetadata(width=800, height=1200),
        )
    return project


@pytest.fixture
def widget(qtbot: QtBot, host: QWidget, project: Project) -> ThumbnailListWidget:
    thumbnails = ThumbnailListWidget(
        render,
        project,
        stage="s",
        thumbnail_size=QSize(100, 100),
        render_size=QSize(100, 100),
        parent=host,
    )
    show_in_host(qtbot, thumbnails, 700, 700)
    return thumbnails


def rendered_size(qtbot: QtBot, widget: ThumbnailListWidget, row: int = 0) -> QSize:
    index = widget.model.index(row)
    qtbot.waitUntil(
        lambda: widget.model.data(index, Qt.ItemDataRole.DecorationRole) is not None,
        timeout=5000,
    )
    return widget.model.data(index, Qt.ItemDataRole.DecorationRole).size()


def cell(widget: ThumbnailListWidget, row: int) -> tuple[int, int]:
    rect = widget.item_rect(widget.pages()[row])
    return rect.x(), rect.y()


class TestCellSize:
    def test_a_larger_cell_size_makes_the_cells_larger(
        self, widget: ThumbnailListWidget
    ) -> None:
        before = widget.item_rect(widget.pages()[0])

        widget.set_cell_size(QSize(200, 160))

        after = widget.item_rect(widget.pages()[0])
        assert after.width() - before.width() == 100
        assert after.height() - before.height() == 60

    def test_the_view_is_wide_enough_for_one_cell_and_a_scroll_bar(
        self, widget: ThumbnailListWidget
    ) -> None:
        widget.set_cell_size(QSize(300, 160))

        assert widget.view.minimumWidth() > widget.cell_size.width() + 24

    def test_cell_size_does_not_change_what_is_rendered(
        self, qtbot: QtBot, widget: ThumbnailListWidget
    ) -> None:
        before = rendered_size(qtbot, widget)

        widget.set_cell_size(QSize(300, 300))

        assert rendered_size(qtbot, widget) == before


class TestRenderSize:
    def test_thumbnails_are_kept_within_the_render_size(
        self, qtbot: QtBot, widget: ThumbnailListWidget
    ) -> None:
        size = rendered_size(qtbot, widget)

        assert size.height() == 100
        assert size.width() == pytest.approx(100 * 800 / 1200, abs=1)

    def test_a_larger_render_size_renders_larger_thumbnails(
        self, qtbot: QtBot, widget: ThumbnailListWidget
    ) -> None:
        rendered_size(qtbot, widget)

        widget.set_render_size(QSize(300, 300))

        assert rendered_size(qtbot, widget).height() == 300

    def test_the_render_size_is_exposed_to_the_renderer(
        self, widget: ThumbnailListWidget
    ) -> None:
        widget.set_render_size(QSize(240, 240))

        assert widget.render_size == QSize(240, 240)


class TestColumns:
    def test_default_fills_the_width_with_several_columns(
        self, widget: ThumbnailListWidget
    ) -> None:
        assert cell(widget, 0)[1] == cell(widget, 1)[1]
        assert cell(widget, 0)[0] != cell(widget, 1)[0]

    def test_single_column_stacks_the_thumbnails(
        self, widget: ThumbnailListWidget
    ) -> None:
        widget.set_single_column(True)

        assert cell(widget, 0)[0] == cell(widget, 1)[0]
        assert cell(widget, 0)[1] < cell(widget, 1)[1]

    def test_multi_column_comes_back(self, widget: ThumbnailListWidget) -> None:
        widget.set_single_column(True)

        widget.set_single_column(False)

        assert cell(widget, 0)[1] == cell(widget, 1)[1]

    def test_a_narrow_view_wraps_to_further_rows(
        self, qtbot: QtBot, widget: ThumbnailListWidget
    ) -> None:
        widget.set_cell_size(QSize(250, 160))
        widget.window().resize(widget.view.minimumWidth() + 40, 700)
        flush_events()

        assert cell(widget, 0)[0] == cell(widget, 1)[0]


class TestApplySettings:
    def test_all_three_options_are_taken_from_the_settings(
        self, qtbot: QtBot, widget: ThumbnailListWidget
    ) -> None:
        settings = ApplicationSettings(
            thumbnail_quality_width=240,
            thumbnail_quality_height=240,
            max_thumbnail_width=300.0,
            max_thumbnail_height=192.0,
            single_column_thumbnail_display=True,
        )

        widget.apply_settings(settings)

        assert widget.render_size == QSize(240, 240)
        assert widget.cell_size == QSize(300, 192)
        assert cell(widget, 0)[0] == cell(widget, 1)[0]
        assert rendered_size(qtbot, widget).height() == 240

    def test_changed_settings_replace_earlier_ones(
        self, widget: ThumbnailListWidget
    ) -> None:
        widget.apply_settings(ApplicationSettings(single_column_thumbnail_display=True))

        widget.apply_settings(
            ApplicationSettings(
                single_column_thumbnail_display=False,
                max_thumbnail_width=120.0,
                max_thumbnail_height=77.0,
            )
        )

        assert widget.cell_size == QSize(120, 77)
        assert cell(widget, 0)[1] == cell(widget, 1)[1]
