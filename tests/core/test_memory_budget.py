"""Memory a page takes through the pipeline, and how many pages a batch runs at once."""

import tracemalloc
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core import batch
from scantailor.core.batch import (
    PAGE_BYTES_PER_PIXEL,
    WORKER_BASE_BYTES,
    PageLister,
    largest_page_bytes,
    memory_worker_limit,
)
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel

_SLACK = 64 << 10
_GIB = 1 << 30


def _book_page(height: int, width: int) -> NDArray[np.uint8]:
    """Lines of text over tinted paper, a photograph and scattered specks."""
    rng = np.random.default_rng(3)
    page = np.full((height, width, 3), (232, 238, 241), dtype=np.uint8)
    for top in range(height // 12, height * 11 // 20, height // 40):
        cv2.putText(
            page,
            "The quick brown fox jumps.",
            (width // 10, top),
            cv2.FONT_HERSHEY_SIMPLEX,
            width / 1500,
            (40, 35, 30),
            max(1, width // 600),
        )
    top, left = height * 3 // 5, width // 6
    picture = (height // 4, width * 2 // 3, 3)
    page[top : top + picture[0], left : left + picture[1]] = rng.integers(
        30, 230, picture, dtype=np.uint8
    )
    specks = height * width // 4000
    page[rng.integers(0, height, specks), rng.integers(0, width, specks)] = 0
    return page


def _page_id(name: str = "budget.tif") -> PageId:
    return PageId(image_id=ImageId(file_path=Path(name)), sub_page=SubPage.SINGLE_PAGE)


def _stages(page: PageId, mode: ColorMode, despeckle: DespeckleLevel) -> StageSequence:
    stages = StageSequence()
    output = stages.output_filter
    output.set_params(
        page,
        output.get_params(page).model_copy(
            update={
                "color_mode": mode,
                "despeckle_level": despeckle,
                "output_dpi": Dpi.uniform(300),
            }
        ),
    )
    return stages


@pytest.mark.slow
@pytest.mark.parametrize(
    ("mode", "despeckle"),
    [
        (ColorMode.BLACK_AND_WHITE, DespeckleLevel.CAUTIOUS),
        (ColorMode.BLACK_AND_WHITE, DespeckleLevel.AGGRESSIVE),
        (ColorMode.MIXED, DespeckleLevel.CAUTIOUS),
        (ColorMode.MIXED, DespeckleLevel.AGGRESSIVE),
        # Colour output skips despeckling, so one level covers it.
        (ColorMode.COLOR_GRAYSCALE, DespeckleLevel.CAUTIOUS),
    ],
)
def test_a_page_through_every_stage_stays_within_its_modes_bytes_per_pixel(
    one_opencv_thread: None, mode: ColorMode, despeckle: DespeckleLevel
) -> None:
    height, width = 2400, 1800
    image = _book_page(height, width)
    page = _page_id()
    stages = _stages(page, mode, despeckle)

    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        process_page(image, page, stages, PipelineOptions(dpi=300.0))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak - before <= PAGE_BYTES_PER_PIXEL[mode] * height * width + _SLACK


@pytest.mark.slow
def test_a_page_upscaled_to_its_output_resolution_stays_within_its_price(
    one_opencv_thread: None, tmp_path: Path
) -> None:
    height, width = 1200, 900
    image = _book_page(height, width)
    source = tmp_path / "budget.tif"
    write_image(source, image, Dpi.uniform(150))
    page = _page_id(str(source))
    stages = _stages(page, ColorMode.MIXED, DespeckleLevel.CAUTIOUS)
    price = largest_page_bytes(stages, [page])

    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        process_page(image, page, stages, PipelineOptions(dpi=150.0))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert price == PAGE_BYTES_PER_PIXEL[ColorMode.MIXED] * height * width * 4
    assert peak - before <= price + _SLACK


class TestMemoryWorkerLimit:
    @pytest.fixture
    def machine(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int | None]:
        """Stand in for the machine's RAM and this process's resident memory."""
        memory: dict[str, int | None] = {"ram": 16 * _GIB, "rss": _GIB // 2}
        monkeypatch.setattr(batch, "total_ram_bytes", lambda: memory["ram"])
        monkeypatch.setattr(batch, "process_rss_bytes", lambda: memory["rss"])
        return memory

    def test_pages_that_fit_in_three_quarters_of_ram_run_at_once(
        self, machine: dict[str, int | None]
    ) -> None:
        page_bytes = _GIB

        limit = memory_worker_limit(32, page_bytes)

        expected = int((0.75 * 16 * _GIB - _GIB // 2) // (WORKER_BASE_BYTES + _GIB))
        assert limit == expected == 10

    def test_the_cpu_count_caps_the_limit_when_memory_is_plenty(
        self, machine: dict[str, int | None]
    ) -> None:
        machine["ram"] = 1024 * _GIB

        assert memory_worker_limit(8, _GIB) == 8

    def test_one_page_runs_even_when_none_fits(
        self, machine: dict[str, int | None]
    ) -> None:
        machine["ram"] = _GIB

        assert memory_worker_limit(8, 4 * _GIB) == 1

    def test_without_a_reading_of_the_ram_every_worker_runs(
        self, machine: dict[str, int | None]
    ) -> None:
        machine["ram"] = None

        assert memory_worker_limit(6, 100 * _GIB) == 6

    def test_an_unknown_resident_size_counts_as_nothing(
        self, machine: dict[str, int | None]
    ) -> None:
        machine["rss"] = None

        limit = memory_worker_limit(32, _GIB)

        assert limit == int(0.75 * 16 * _GIB // (WORKER_BASE_BYTES + _GIB))


class TestLargestPageBytes:
    def test_the_largest_page_is_priced_by_its_pixels_and_mode(
        self, tmp_path: Path
    ) -> None:
        small, large = tmp_path / "small.tif", tmp_path / "large.tif"
        write_image(small, np.zeros((100, 80), np.uint8), Dpi.uniform(300))
        write_image(large, np.zeros((300, 200), np.uint8), Dpi.uniform(300))
        pages = [_page_id(str(small)), _page_id(str(large))]
        stages = StageSequence()
        for page in pages:
            stages.output_filter.set_params(
                page,
                stages.output_filter.get_params(page).model_copy(
                    update={
                        "color_mode": ColorMode.MIXED,
                        "output_dpi": Dpi.uniform(300),
                    }
                ),
            )

        priced = largest_page_bytes(stages, pages)

        assert priced == int(PAGE_BYTES_PER_PIXEL[ColorMode.MIXED] * 300 * 200)

    @pytest.mark.parametrize(
        ("file_dpi", "override", "default_dpi", "upscale"),
        [
            (150, None, 300.0, 16),
            (600, None, 300.0, 1),
            (600, Dpi.uniform(200), 300.0, 9),
            (0, None, 100.0, 36),
            (Dpi(horizontal=300, vertical=150), None, 300.0, 8),
        ],
    )
    def test_a_page_is_priced_at_its_output_resolution_when_that_is_finer(
        self,
        tmp_path: Path,
        file_dpi: int | Dpi,
        override: Dpi | None,
        default_dpi: float,
        upscale: int,
    ) -> None:
        source = tmp_path / "page.tif"
        dpi = file_dpi if isinstance(file_dpi, Dpi) else Dpi.uniform(file_dpi)
        write_image(source, np.zeros((30, 20), np.uint8), dpi)
        page = _page_id(str(source))
        stages = _stages(page, ColorMode.BLACK_AND_WHITE, DespeckleLevel.CAUTIOUS)
        overrides = {} if override is None else {page.image_id: override}
        stages.output_filter.set_params(
            page,
            stages.output_filter.get_params(page).model_copy(
                update={"output_dpi": Dpi.uniform(600)}
            ),
        )

        priced = largest_page_bytes(stages, [page], default_dpi, overrides)

        bytes_per_pixel = PAGE_BYTES_PER_PIXEL[ColorMode.BLACK_AND_WHITE]
        assert priced == bytes_per_pixel * 30 * 20 * upscale

    def test_a_missing_file_is_priced_at_nothing(self, tmp_path: Path) -> None:
        assert largest_page_bytes(StageSequence(), [_page_id(str(tmp_path / "x"))]) == 0

    def test_mixed_mode_costs_more_than_colour_and_black_and_white(self) -> None:
        assert (
            PAGE_BYTES_PER_PIXEL[ColorMode.MIXED]
            > PAGE_BYTES_PER_PIXEL[ColorMode.BLACK_AND_WHITE]
            > PAGE_BYTES_PER_PIXEL[ColorMode.COLOR_GRAYSCALE]
        )


def test_a_batch_runs_no_more_pages_at_once_than_fit_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[int] = []

    def fake_render(*args: object) -> None:
        running_limit = args[7]
        asked.append(running_limit())

    monkeypatch.setattr(batch, "memory_worker_limit", lambda workers, page: 2)
    monkeypatch.setattr(batch, "_detect_geometry", lambda *args: list(args[1]))
    monkeypatch.setattr(batch, "_render_ready_pages", fake_render)
    monkeypatch.setattr(batch, "spawn_pool", _NoPool)

    batch.run_batch(
        StageSequence(),
        [_page_id()],
        Path("unused"),
        max_workers=8,
        worker_limit=lambda: 6,
    )

    assert asked == [2]


def test_a_batch_reprices_its_pages_after_page_split_lists_new_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single = _page_id()
    halves = [
        PageId(image_id=single.image_id, sub_page=SubPage.LEFT_PAGE),
        PageId(image_id=single.image_id, sub_page=SubPage.RIGHT_PAGE),
    ]
    asked: list[int] = []

    def fake_render(*args: object) -> None:
        running_limit = args[7]
        asked.append(running_limit())

    monkeypatch.setattr(
        batch, "largest_page_bytes", lambda stages, pages, *args: len(pages)
    )
    monkeypatch.setattr(batch, "memory_worker_limit", lambda workers, page: page)
    monkeypatch.setattr(batch, "_split_images", lambda *args: halves)
    monkeypatch.setattr(batch, "_detect_geometry", lambda *args: list(args[1]))
    monkeypatch.setattr(batch, "_render_ready_pages", fake_render)
    monkeypatch.setattr(batch, "spawn_pool", _NoPool)

    batch.run_batch(
        StageSequence(),
        [single],
        Path("unused"),
        max_workers=8,
        page_lister=cast(PageLister, object()),
    )

    assert asked == [2]


class _NoPool:
    def __init__(self, *args: object) -> None:
        pass

    def __enter__(self) -> _NoPool:
        return self

    def __exit__(self, *args: object) -> None:
        return None
