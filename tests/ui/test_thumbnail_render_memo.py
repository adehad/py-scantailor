"""Thumbnails before Output are reused or made from a shrunk source when they can be."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6 import QtCore, QtGui

from scantailor.app.ui import page_renderer
from scantailor.app.ui.page_renderer import render_thumbnail
from scantailor.app.ui.thumbnails.render_memo import (
    ReducedSource,
    ThumbnailMemo,
    reduced_copy,
)
from scantailor.core import pipeline
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, Margins, PageId
from scantailor.core.page_source import load_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

BOX = QtCore.QSize(64, 64)
# Taken before any test replaces it with a counting one.
_PROCESS_PAGE = pipeline.process_page


def _write_scan(path: Path, shade: int = 30) -> None:
    pixels = np.full((1200, 900, 3), (222, 230, 236), dtype=np.uint8)
    for line in range(18):
        cv2.putText(
            pixels,
            "the mill stood where the river bent",
            (100, 200 + line * 50),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1.0,
            (shade, shade, shade),
            2,
        )
    write_image(path, pixels, Dpi.uniform(300))


@pytest.fixture
def page(tmp_path: Path) -> PageId:
    path = tmp_path / "page.tif"
    _write_scan(path)
    return PageId(image_id=ImageId(file_path=path))


def _stored(page: PageId) -> StageSequence:
    """Parameters as a batch leaves them: nothing is left to detect or record."""
    stages = StageSequence()
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 1.0)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=80, y=150, width=740, height=900)
    )
    loaded = load_page(page.image_id)
    options = pipeline.PipelineOptions(
        end_stage=FilterStage.PAGE_LAYOUT, dpi=300.0, render_image=False
    )
    _PROCESS_PAGE(loaded.pixels, page, stages, options)
    return stages


@pytest.fixture
def full_renders(monkeypatch: pytest.MonkeyPatch) -> list[FilterStage]:
    """Record the end stage of every pipeline run on a full-size source."""
    seen: list[FilterStage] = []
    real = pipeline.process_page

    def counting(
        image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
        options: pipeline.PipelineOptions | None = None,
    ) -> pipeline.PipelineResult:
        assert options is not None
        seen.append(options.end_stage)
        return real(image, page_id, sequence, options)

    monkeypatch.setattr(pipeline, "process_page", counting)
    return seen


def _pixels(image: QtGui.QImage) -> np.ndarray:
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    data = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    return data.reshape(height, image.bytesPerLine())[:, : width * 3].copy()


class TestUnchangedParameters:
    def test_a_thumbnail_is_handed_out_again_without_decoding_or_rendering(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        first = render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)

        again = render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)

        assert full_renders == [FilterStage.DESKEW]
        assert again is first

    def test_a_different_box_is_a_different_thumbnail(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        small = render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)

        larger = render_thumbnail(
            page, stages, FilterStage.DESKEW, QtCore.QSize(96, 96), memo=memo
        )

        assert max(larger.width(), larger.height()) == 96
        assert max(small.width(), small.height()) == 64

    def test_a_rewritten_source_file_is_rendered_again(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)
        _write_scan(page.image_id.file_path, shade=120)

        render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)

        assert full_renders == [FilterStage.DESKEW, FilterStage.DESKEW]

    def test_output_thumbnails_are_rendered_every_time(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()

        render_thumbnail(page, stages, FilterStage.OUTPUT, BOX, memo=memo)
        render_thumbnail(page, stages, FilterStage.OUTPUT, BOX, memo=memo)

        assert full_renders == [FilterStage.OUTPUT, FilterStage.OUTPUT]


class TestShrunkSource:
    def test_a_settled_page_renders_its_layout_from_the_shrunk_source(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        render_thumbnail(page, stages, FilterStage.SELECT_CONTENT, BOX, memo=memo)
        expected = render_thumbnail(
            page, _stored(page), FilterStage.PAGE_LAYOUT, BOX, memo=ThumbnailMemo()
        )
        full_renders.clear()

        thumbnail = render_thumbnail(
            page, stages, FilterStage.PAGE_LAYOUT, BOX, memo=memo
        )

        assert full_renders == []
        actual, wanted = _pixels(thumbnail), _pixels(expected)
        assert actual.shape == wanted.shape
        assert np.abs(actual.astype(int) - wanted.astype(int)).mean() < 3.0

    def test_an_edit_after_the_detecting_stages_keeps_using_the_shrunk_source(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        render_thumbnail(page, stages, FilterStage.PAGE_LAYOUT, BOX, memo=memo)
        stages.page_layout_filter.set_margins(
            page, Margins(top=30.0, bottom=30.0, left=30.0, right=30.0)
        )

        render_thumbnail(page, stages, FilterStage.PAGE_LAYOUT, BOX, memo=memo)

        assert full_renders == [FilterStage.PAGE_LAYOUT]

    def test_an_edit_of_a_detecting_stage_needs_the_full_size_source(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        render_thumbnail(page, stages, FilterStage.SELECT_CONTENT, BOX, memo=memo)
        stages.deskew_filter.set_manual_angle(page, -2.0)

        render_thumbnail(page, stages, FilterStage.SELECT_CONTENT, BOX, memo=memo)

        assert full_renders == [FilterStage.SELECT_CONTENT] * 2

    def test_an_earlier_stage_of_a_settled_page_uses_the_shrunk_source(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        render_thumbnail(page, stages, FilterStage.SELECT_CONTENT, BOX, memo=memo)

        render_thumbnail(page, stages, FilterStage.PAGE_SPLIT, BOX, memo=memo)

        assert full_renders == [FilterStage.SELECT_CONTENT]

    def test_a_later_stage_than_any_rendered_needs_the_full_size_source(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        render_thumbnail(page, stages, FilterStage.PAGE_SPLIT, BOX, memo=memo)

        render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)

        assert full_renders == [FilterStage.PAGE_SPLIT, FilterStage.DESKEW]

    def test_a_box_too_large_for_the_shrunk_source_needs_the_full_size_source(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)

        render_thumbnail(
            page, stages, FilterStage.DESKEW, QtCore.QSize(400, 400), memo=memo
        )

        assert full_renders == [FilterStage.DESKEW] * 2


class TestDetection:
    def test_a_render_that_detected_is_not_reused(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = StageSequence(), ThumbnailMemo()
        stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)

        render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)
        render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)
        render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)

        assert stages.deskew_filter.settings.is_params_set(page)
        assert full_renders == [FilterStage.DESKEW] * 2

    def test_a_page_never_rendered_in_full_needs_the_full_size_source(
        self, page: PageId, full_renders: list[FilterStage]
    ) -> None:
        stages, memo = _stored(page), ThumbnailMemo()
        render_thumbnail(page, stages, FilterStage.FIX_ORIENTATION, BOX, memo=memo)

        render_thumbnail(page, stages, FilterStage.DESKEW, BOX, memo=memo)

        assert full_renders == [FilterStage.FIX_ORIENTATION, FilterStage.DESKEW]


@pytest.fixture
def shared_memo() -> Iterator[ThumbnailMemo]:
    memo = page_renderer._SHARED_MEMO
    memo.clear()
    yield memo
    memo.clear()


def test_the_shared_memo_serves_callers_that_pass_none(
    page: PageId, full_renders: list[FilterStage], shared_memo: ThumbnailMemo
) -> None:
    stages = _stored(page)

    render_thumbnail(page, stages, FilterStage.DESKEW, BOX)
    render_thumbnail(page, stages, FilterStage.DESKEW, BOX)

    assert full_renders == [FilterStage.DESKEW]


class TestThumbnailMemo:
    def test_sources_beyond_the_budget_evict_the_least_recently_used(self) -> None:
        memo = ThumbnailMemo(source_budget=250)
        sources = {
            name: ReducedSource(np.zeros((10, 10), np.uint8), (100, 100), 300.0)
            for name in "abc"
        }
        memo.keep_reduced_source("a", sources["a"])
        memo.keep_reduced_source("b", sources["b"])
        memo.reduced_source("a")

        memo.keep_reduced_source("c", sources["c"])

        assert memo.reduced_source("a") is sources["a"]
        assert memo.reduced_source("b") is None
        assert memo.reduced_source("c") is sources["c"]

    def test_a_kept_source_is_read_only(self) -> None:
        memo = ThumbnailMemo()
        source = ReducedSource(np.zeros((10, 10), np.uint8), (100, 100), 300.0)

        memo.keep_reduced_source("a", source)

        assert not source.pixels.flags.writeable

    def test_a_thumbnail_larger_than_the_budget_is_not_kept(self) -> None:
        memo = ThumbnailMemo(image_budget=100)
        image = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB888)

        memo.keep_image("a", image)

        assert memo.image("a") is None

    def test_settled_digests_beyond_the_capacity_are_forgotten_oldest_first(
        self,
    ) -> None:
        memo = ThumbnailMemo(settled_capacity=2)
        for digest in ("a", "b", "c"):
            memo.settle(digest)

        assert not memo.is_settled("a")
        assert memo.is_settled("b")
        assert memo.is_settled("c")

    def test_clear_forgets_everything(self) -> None:
        memo = ThumbnailMemo()
        memo.settle("a")
        memo.keep_image("a", QtGui.QImage(4, 4, QtGui.QImage.Format.Format_RGB888))

        memo.clear()

        assert not memo.is_settled("a")
        assert memo.image("a") is None


class TestReducedCopy:
    def test_the_longer_side_is_shrunk_to_the_length_asked_for(self) -> None:
        reduced = reduced_copy(np.zeros((1000, 700, 3), np.uint8), 600.0, 200)

        assert reduced is not None
        assert reduced.pixels.shape == (200, 140, 3)
        assert reduced.full_size == (700, 1000)
        assert reduced.dpi == 600.0

    def test_a_copy_not_even_half_as_large_is_not_made(self) -> None:
        assert reduced_copy(np.zeros((300, 200), np.uint8), 300.0, 150) is None
