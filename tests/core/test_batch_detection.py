"""Tests for defaults reaching batch workers and for pooled geometry detection."""

from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PIL import Image

from scantailor.core.batch import (
    BatchResult,
    PageDetected,
    PageDone,
    PageFailed,
    apply_stage_defaults,
    collect_effective_stage_params,
    collect_stage_defaults,
    pool_size,
    run_batch,
    spawn_pool,
)
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, fix_aggregate_page_size
from scantailor.core.stage_params import collect_stage_params
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

SOURCE_DPI = 300


def _page(path: Path) -> PageId:
    return PageId(image_id=ImageId(file_path=path), sub_page=SubPage.SINGLE_PAGE)


def _write_text_like_page(path: Path, block: tuple[int, int, int, int]) -> None:
    """A 300 dpi page of horizontal dark stripes inside ``block`` (x, y, w, h)."""
    x, y, width, height = block
    pixels = np.full((600, 450), 235, dtype=np.uint8)
    for top in range(y, y + height, 12):
        pixels[top : top + 5, x : x + width] = 25
    write_image(path, pixels, Dpi.uniform(SOURCE_DPI))


@pytest.fixture
def detected_sources(tmp_path: Path) -> list[PageId]:
    blocks = [(60, 80, 300, 380), (90, 120, 240, 300), (50, 60, 340, 440)]
    pages = []
    for index, block in enumerate(blocks):
        path = tmp_path / "in" / f"page{index}.tif"
        path.parent.mkdir(exist_ok=True)
        _write_text_like_page(path, block)
        pages.append(_page(path))
    return pages


class TestStageDefaultsData:
    def test_defaults_round_trip_into_fresh_stages(self):
        stages = StageSequence()
        stages.output_filter.settings.set_default_dpi(Dpi.uniform(300))
        stages.output_filter.settings.set_default_color_mode(ColorMode.COLOR_GRAYSCALE)
        stages.output_filter.settings.set_default_despeckle_level(DespeckleLevel.OFF)
        stages.page_split_filter.settings.set_default_layout_type(
            LayoutType.SINGLE_PAGE_UNCUT
        )
        stages.select_content_filter.settings.set_page_detection_box(210.0, 297.0)
        stages.select_content_filter.settings.set_page_detection_tolerance(0.25)

        fresh = StageSequence()
        apply_stage_defaults(fresh, collect_stage_defaults(stages))

        assert collect_stage_defaults(fresh) == collect_stage_defaults(stages)
        assert fresh.output_filter.settings.get_default_dpi() == Dpi.uniform(300)

    def test_effective_params_include_pages_that_only_have_defaults(self, tmp_path):
        stages = StageSequence()
        stages.output_filter.settings.set_default_dpi(Dpi.uniform(300))
        page = _page(tmp_path / "unset.tif")

        data = collect_effective_stage_params(stages, [page])

        (entry,) = data["output"]
        assert entry["params"]["output_dpi"] == {"horizontal": 300, "vertical": 300}
        assert not stages.output_filter.settings.is_params_set(page)

    def test_effective_params_keep_explicit_page_values(self, tmp_path):
        stages = StageSequence()
        stages.page_split_filter.settings.set_default_layout_type(
            LayoutType.SINGLE_PAGE_UNCUT
        )
        page = _page(tmp_path / "explicit.tif")
        stages.deskew_filter.set_manual_angle(page, 1.5)

        data = collect_effective_stage_params(stages, [page])

        assert data["deskew"][0]["params"]["deskew_angle_deg"] == 1.5
        assert (
            data["page_split"][0]["params"]["layout_type"]
            == LayoutType.SINGLE_PAGE_UNCUT.value
        )


class TestEmptyAndCancelledBatches:
    def test_empty_project_reports_nothing_and_writes_nothing(self, tmp_path: Path):
        events = []

        result = run_batch(
            StageSequence(), [], tmp_path / "out", on_progress=events.append
        )

        assert result == BatchResult()
        assert events == []

    def test_cancel_before_detection_reports_every_page_as_cancelled(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        cancel = Event()
        cancel.set()
        events = []

        result = run_batch(
            StageSequence(),
            detected_sources,
            tmp_path / "out",
            cancel=cancel,
            on_progress=events.append,
        )

        assert result.cancelled == detected_sources
        assert result.written == {} and events == []


@pytest.mark.slow
class TestDefaultsReachWorkers:
    def test_default_output_dpi_alone_decides_the_written_resolution(
        self, tmp_path: Path
    ):
        source = tmp_path / "in" / "page.tif"
        source.parent.mkdir()
        _write_text_like_page(source, (60, 80, 300, 380))
        page = _page(source)
        stages = StageSequence()
        stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
        stages.deskew_filter.set_manual_angle(page, 0.0)
        stages.select_content_filter.set_manual_content(
            page, ContentBox(x=60, y=80, width=300, height=380)
        )
        stages.output_filter.settings.set_default_dpi(Dpi.uniform(300))

        result = run_batch(stages, [page], tmp_path / "out", max_workers=1)

        assert result.failed == {}
        with Image.open(result.written[page]) as written:
            assert written.info["dpi"] == pytest.approx((300, 300), abs=1)

    def test_default_layout_type_applies_to_pages_detected_in_workers(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        stages = StageSequence()
        stages.page_split_filter.settings.set_default_layout_type(
            LayoutType.SINGLE_PAGE_UNCUT
        )

        run_batch(stages, detected_sources, tmp_path / "out", max_workers=2)

        for page in detected_sources:
            stored = stages.page_split_filter.get_params(page)
            assert stored.layout_type is LayoutType.SINGLE_PAGE_UNCUT


@pytest.mark.slow
class TestPooledDetection:
    def test_detected_params_match_the_serial_pass(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        serial = StageSequence()
        images = []
        for page in detected_sources:
            with Image.open(page.image_id.file_path) as source:
                images.append((np.asarray(source), page))
        fix_aggregate_page_size(images, serial, PipelineOptions(dpi=float(SOURCE_DPI)))
        pooled = StageSequence()

        run_batch(pooled, detected_sources, tmp_path / "out", max_workers=2)

        assert collect_stage_params(pooled, detected_sources) == collect_stage_params(
            serial, detected_sources
        )

    def test_outputs_match_a_run_from_serially_detected_params(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        serial = StageSequence()
        for page in detected_sources:
            with Image.open(page.image_id.file_path) as source:
                fix_aggregate_page_size(
                    [(np.asarray(source), page)],
                    serial,
                    PipelineOptions(dpi=float(SOURCE_DPI)),
                )
        pooled = StageSequence()

        pooled_result = run_batch(
            pooled, detected_sources, tmp_path / "pooled", max_workers=2
        )
        serial_result = run_batch(
            serial, detected_sources, tmp_path / "serial", max_workers=2
        )

        assert pooled_result.failed == {} and serial_result.failed == {}
        for page in detected_sources:
            assert (
                pooled_result.written[page].read_bytes()
                == serial_result.written[page].read_bytes()
            )

    def test_progress_reports_detection_before_any_page_is_done(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        events = []

        run_batch(
            StageSequence(),
            detected_sources,
            tmp_path / "out",
            max_workers=2,
            on_progress=events.append,
        )

        kinds = [type(event) for event in events]
        assert kinds.count(PageDetected) == len(detected_sources)
        assert kinds.count(PageDone) == len(detected_sources)
        assert max(i for i, k in enumerate(kinds) if k is PageDetected) < min(
            i for i, k in enumerate(kinds) if k is PageDone
        )

    def test_unreadable_page_fails_during_detection_and_others_continue(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        missing = _page(tmp_path / "in" / "missing.tif")
        events = []

        result = run_batch(
            StageSequence(),
            [detected_sources[0], missing, *detected_sources[1:]],
            tmp_path / "out",
            max_workers=2,
            on_progress=events.append,
        )

        assert set(result.failed) == {missing}
        assert set(result.written) == set(detected_sources)
        assert [e.page for e in events if isinstance(e, PageFailed)] == [missing]
        assert missing not in [e.page for e in events if isinstance(e, PageDetected)]


@pytest.mark.slow
class TestPooledDetectionOfSplitSpreads:
    def test_both_halves_of_a_spread_match_the_serial_pass(self, tmp_path: Path):
        source = tmp_path / "in" / "spread.tif"
        source.parent.mkdir()
        pixels = np.full((600, 900), 235, dtype=np.uint8)
        for left_edge in (60, 510):
            for top in range(80, 460, 12):
                pixels[top : top + 5, left_edge : left_edge + 300] = 25
        write_image(source, pixels, Dpi.uniform(SOURCE_DPI))
        image_id = ImageId(file_path=source)
        halves = [
            PageId(image_id=image_id, sub_page=SubPage.LEFT_PAGE),
            PageId(image_id=image_id, sub_page=SubPage.RIGHT_PAGE),
        ]
        serial = StageSequence()
        for half in halves:
            fix_aggregate_page_size(
                [(pixels, half)], serial, PipelineOptions(dpi=float(SOURCE_DPI))
            )
        pooled = StageSequence()

        result = run_batch(pooled, halves, tmp_path / "out", max_workers=2)

        assert result.failed == {}
        assert collect_stage_params(pooled, halves) == collect_stage_params(
            serial, halves
        )


class TestPoolSize:
    def test_default_pool_size_is_accepted_on_machines_with_many_cores(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("os.cpu_count", lambda: 128)

        with spawn_pool(pool_size(None)) as pool:
            assert pool._max_workers >= 1


class TestBatchToAnEarlierStage:
    def test_batch_to_deskew_measures_deskew_but_not_the_content_box(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        stages = StageSequence()

        result = run_batch(
            stages,
            detected_sources,
            tmp_path / "out",
            max_workers=1,
            end_stage=FilterStage.DESKEW,
        )

        assert result.detected == detected_sources
        for page in detected_sources:
            assert stages.deskew_filter.settings.is_params_set(page)
            assert not stages.select_content_filter.settings.is_params_set(page)

    def test_batch_to_margins_writes_no_files_and_reports_no_render(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        stages = StageSequence()
        events = []

        result = run_batch(
            stages,
            detected_sources,
            tmp_path / "out",
            max_workers=1,
            on_progress=events.append,
            end_stage=FilterStage.PAGE_LAYOUT,
        )

        assert not (tmp_path / "out").exists()
        assert result.written == {}
        assert sorted(result.detected, key=str) == sorted(detected_sources, key=str)
        assert all(isinstance(event, PageDetected) for event in events)
        for page in detected_sources:
            assert stages.page_layout_filter.settings.is_params_set(page)

    def test_batch_to_output_records_every_detected_page(
        self, tmp_path: Path, detected_sources: list[PageId]
    ):
        result = run_batch(
            StageSequence(), detected_sources, tmp_path / "out", max_workers=1
        )

        assert sorted(result.detected, key=str) == sorted(detected_sources, key=str)
        assert set(result.written) == set(detected_sources)
