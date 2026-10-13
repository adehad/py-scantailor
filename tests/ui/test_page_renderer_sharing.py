"""The renderer shares decoded sources and lets the page on screen go first."""

import threading
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from scantailor.app.ui import page_renderer, source_cache, stage_panels
from scantailor.app.ui.page_renderer import PageRenderer, render_thumbnail
from scantailor.app.ui.source_cache import SourceCache
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

RENDER_TIMEOUT_MS = 30_000


def _page(path: Path) -> PageId:
    pixels = np.full((400, 300, 3), 220, dtype=np.uint8)
    pixels[100:300, 80:220] = (40, 60, 90)
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


@pytest.fixture
def decodes(monkeypatch: pytest.MonkeyPatch) -> list[ImageId]:
    seen: list[ImageId] = []
    real = source_cache.load_page

    def counting(image_id, dpi_overrides=None):
        seen.append(image_id)
        return real(image_id, dpi_overrides)

    monkeypatch.setattr(source_cache, "load_page", counting)
    monkeypatch.setattr(page_renderer, "load_page", counting)
    monkeypatch.setattr(stage_panels, "load_page", counting)
    return seen


def _render(qtbot: QtBot, renderer: PageRenderer, *args, **kwargs) -> object:
    with qtbot.waitSignal(renderer.render_done, timeout=RENDER_TIMEOUT_MS) as blocker:
        renderer.request(*args, **kwargs)
    renderer.wait_for_done()
    return blocker.args[1]


def test_renders_of_one_page_decode_it_once(
    qtbot: QtBot, tmp_path: Path, decodes: list[ImageId]
) -> None:
    page = _page(tmp_path / "a.tif")
    stages = _stages(page)
    renderer = PageRenderer()

    _render(qtbot, renderer, page, stages, FilterStage.DESKEW)
    _render(qtbot, renderer, page, stages, FilterStage.SELECT_CONTENT)

    assert decodes == [page.image_id]


def test_stage_panels_reuse_the_source_the_renderer_decoded(
    qtbot: QtBot, tmp_path: Path, decodes: list[ImageId]
) -> None:
    page = _page(tmp_path / "a.tif")
    stages = _stages(page)
    renderer = PageRenderer()
    _render(qtbot, renderer, page, stages, FilterStage.DESKEW)
    panels = stage_panels.StagePanels(FilterStage.DESKEW, stages)
    panels.set_source_cache(renderer.sources)

    panels.set_page(page)

    assert decodes == [page.image_id]
    assert panels._source is not None
    assert panels._source.image_id == page.image_id


def test_stage_panels_decode_through_the_cache_the_renderer_then_reuses(
    qtbot: QtBot, tmp_path: Path, decodes: list[ImageId]
) -> None:
    page = _page(tmp_path / "a.tif")
    stages = _stages(page)
    renderer = PageRenderer()
    panels = stage_panels.StagePanels(FilterStage.DESKEW, stages)
    panels.set_source_cache(renderer.sources)

    panels.set_page(page)
    qtbot.waitUntil(lambda: panels._source is not None, timeout=RENDER_TIMEOUT_MS)
    _render(qtbot, renderer, page, stages, FilterStage.DESKEW)

    assert decodes == [page.image_id]


def test_a_render_without_pixels_delivers_an_empty_image(
    qtbot: QtBot, tmp_path: Path
) -> None:
    page = _page(tmp_path / "a.tif")
    renderer = PageRenderer()

    image = _render(
        qtbot, renderer, page, _stages(page), FilterStage.DESKEW, pixels=False
    )

    assert isinstance(image, np.ndarray)
    assert image.size == 0


def test_a_render_with_pixels_delivers_the_page_as_rgb(
    qtbot: QtBot, tmp_path: Path
) -> None:
    page = _page(tmp_path / "a.tif")
    renderer = PageRenderer()

    image = _render(qtbot, renderer, page, _stages(page), FilterStage.DESKEW)

    assert isinstance(image, np.ndarray)
    assert image.shape == (400, 300, 3)
    assert tuple(image[200, 150]) == (90, 60, 40)


def test_thumbnail_uses_a_cached_source_without_caching_its_own(
    tmp_path: Path, decodes: list[ImageId]
) -> None:
    on_screen = _page(tmp_path / "a.tif")
    passing = _page(tmp_path / "b.tif")
    sources = SourceCache()
    sources.load(on_screen.image_id)
    size = QtCore.QSize(64, 64)

    render_thumbnail(
        on_screen, _stages(on_screen), FilterStage.DESKEW, size, None, sources
    )
    render_thumbnail(passing, _stages(passing), FilterStage.DESKEW, size, None, sources)

    assert decodes == [on_screen.image_id, passing.image_id]
    assert sources.peek(passing.image_id) is None
    assert sources.peek(on_screen.image_id) is not None


class TestYieldingToTheRenderTheUserWaitsFor:
    def test_the_renderer_is_idle_before_any_request(self) -> None:
        assert PageRenderer().wait_until_idle(0.0)

    def test_workers_wait_while_a_render_runs_and_resume_when_it_returns(
        self, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _page(tmp_path / "a.tif")
        stages = _stages(page)
        renderer = PageRenderer()
        entered = threading.Event()
        release = threading.Event()
        real = page_renderer.pipeline.process_page

        def slow(image, page_id, sequence, options):
            entered.set()
            assert release.wait(timeout=30)
            return real(image, page_id, sequence, options)

        monkeypatch.setattr(page_renderer.pipeline, "process_page", slow)

        renderer.request(page, stages, FilterStage.DESKEW)
        assert entered.wait(timeout=10)
        waiting = renderer.wait_until_idle(0.05)
        release.set()
        assert renderer.wait_for_done(RENDER_TIMEOUT_MS)

        # No event loop ran: workers must not depend on the GUI thread to resume.
        assert not waiting
        assert renderer.wait_until_idle(0.0)

    def test_cancelling_releases_waiting_workers(
        self, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _page(tmp_path / "a.tif")
        renderer = PageRenderer()
        entered = threading.Event()
        release = threading.Event()
        real = page_renderer.pipeline.process_page

        def slow(image, page_id, sequence, options):
            entered.set()
            assert release.wait(timeout=30)
            return real(image, page_id, sequence, options)

        monkeypatch.setattr(page_renderer.pipeline, "process_page", slow)
        renderer.request(page, _stages(page), FilterStage.DESKEW)
        assert entered.wait(timeout=10)

        renderer.cancel()

        assert renderer.wait_until_idle(0.0)
        release.set()
        renderer.wait_for_done()

    def test_a_failed_render_also_releases_waiting_workers(
        self, qtbot: QtBot, tmp_path: Path
    ) -> None:
        page = PageId(image_id=ImageId(file_path=tmp_path / "missing.tif"))
        renderer = PageRenderer()

        with qtbot.waitSignal(renderer.render_failed, timeout=RENDER_TIMEOUT_MS):
            renderer.request(page, StageSequence(), FilterStage.DESKEW)
        renderer.wait_for_done()

        assert renderer.wait_until_idle(0.0)
