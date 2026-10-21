"""The thumbnail of the page on screen reuses the render the view already shows."""

import gc
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow as main_window_module
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.page_renderer import PageRenderer, shrink_to_thumbnail
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_layout.params import ContentRect, ContentSize
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

RENDER_TIMEOUT_MS = 60_000


@pytest.fixture
def page(tmp_path: Path) -> PageId:
    pixels = np.full((400, 300, 3), 220, dtype=np.uint8)
    pixels[100:300, 80:220] = (40, 60, 90)
    path = tmp_path / "page.tif"
    write_image(path, pixels, Dpi.uniform(300))
    return PageId(image_id=ImageId(file_path=path))


def _stages(page: PageId) -> StageSequence:
    stages = StageSequence()
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 0.0)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=80, y=100, width=140, height=200)
    )
    return stages


def _render(qtbot: QtBot, renderer: PageRenderer, *args, **kwargs) -> np.ndarray:
    with qtbot.waitSignal(renderer.render_done, timeout=RENDER_TIMEOUT_MS) as blocker:
        renderer.request(*args, **kwargs)
    renderer.wait_for_done()
    return blocker.args[1]


def test_a_tagged_render_is_handed_out_for_its_page_stage_and_tag(
    qtbot: QtBot, page: PageId
) -> None:
    renderer = PageRenderer()

    image = _render(
        qtbot, renderer, page, _stages(page), FilterStage.DESKEW, tag="params-1"
    )

    assert renderer.rendered(page, FilterStage.DESKEW, "params-1") is image
    assert renderer.rendered(page, FilterStage.DESKEW, "params-2") is None
    assert renderer.rendered(page, FilterStage.SELECT_CONTENT, "params-1") is None


def test_an_untagged_render_keeps_no_image(qtbot: QtBot, page: PageId) -> None:
    renderer = PageRenderer()

    _render(qtbot, renderer, page, _stages(page), FilterStage.DESKEW)

    assert renderer.rendered(page, FilterStage.DESKEW, None) is None


def test_a_new_request_or_a_cancel_drops_the_kept_image(
    qtbot: QtBot, page: PageId
) -> None:
    renderer = PageRenderer()
    stages = _stages(page)
    _render(qtbot, renderer, page, stages, FilterStage.DESKEW, tag="a")

    _render(qtbot, renderer, page, stages, FilterStage.SELECT_CONTENT)

    assert renderer.rendered(page, FilterStage.DESKEW, "a") is None
    _render(qtbot, renderer, page, stages, FilterStage.DESKEW, tag="a")
    renderer.cancel()
    assert renderer.rendered(page, FilterStage.DESKEW, "a") is None


def test_a_renderer_that_has_rendered_is_freed_without_the_cycle_collector(
    qtbot: QtBot, page: PageId
) -> None:
    renderer = PageRenderer()
    _render(qtbot, renderer, page, _stages(page), FilterStage.DESKEW, tag="a")
    destroyed: list[bool] = []
    renderer.destroyed.connect(lambda: destroyed.append(True))
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del renderer
        assert destroyed
    finally:
        if was_enabled:
            gc.enable()


def test_a_shrunk_page_fits_the_box_and_keeps_its_shape() -> None:
    image = np.zeros((400, 200, 3), dtype=np.uint8)

    thumbnail = shrink_to_thumbnail(image, QtCore.QSize(100, 100))

    assert (thumbnail.width(), thumbnail.height()) == (50, 100)


@pytest.mark.slow
def test_the_output_thumbnail_of_the_page_on_screen_reuses_its_render(
    qtbot: QtBot,
    project_window: MainWindowManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    getattr(
        project_window, f"actionSwitchFilter{int(FilterStage.OUTPUT) + 1}"
    ).trigger()
    layout = project_window.stages.page_layout_filter.settings
    for each in project_window.pages():
        layout.update_content_size_and_get_aggregate(
            each,
            ContentRect(width=600, height=800),
            ContentRect(width=500, height=700),
            ContentSize(width_mm=100, height_mm=150),
        )
    # A render, or the thumbnails of the other pages it sets off, can move the
    # aggregate page size, which makes every thumbnail render again. Render
    # until one leaves the thumbnails alone: that is what a revisit looks like.
    for _ in range(4):
        epoch = project_window._thumbnail_epoch
        with qtbot.waitSignal(
            project_window.renderer.render_done, timeout=RENDER_TIMEOUT_MS
        ):
            project_window._request_render()
        project_window.renderer.wait_for_done()
        project_window.thumbnails.model.wait_for_idle(RENDER_TIMEOUT_MS)
        if project_window._thumbnail_epoch == epoch:
            break
    current = project_window._current_page
    assert current is not None
    pipeline_runs: list[PageId] = []

    def counted(page_id: PageId, *args, **kwargs):
        pipeline_runs.append(page_id)
        return real(page_id, *args, **kwargs)

    real = main_window_module.render_thumbnail
    monkeypatch.setattr(main_window_module, "render_thumbnail", counted)
    stage = str(int(FilterStage.OUTPUT))

    reused = project_window._render_page_thumbnail(current, stage)
    project_window._invalidate_thumbnails(current)
    rendered_again = project_window._render_page_thumbnail(current, stage)

    assert not reused.isNull()
    assert pipeline_runs == [current]
    assert (reused.width(), reused.height()) == (
        rendered_again.width(),
        rendered_again.height(),
    )
