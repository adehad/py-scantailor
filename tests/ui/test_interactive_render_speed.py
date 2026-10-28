"""The page on screen costs no more full-page passes than the batch spends on it.

A pass is an OpenCV call over an array of at least half the page's pixels; the
interactive and batch paths are counted on the same dense 600 dpi page with
nothing remembered from an earlier run.
"""

import collections
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from scantailor.app.ui import source_cache
from scantailor.app.ui.page_renderer import render_page_with_model
from scantailor.app.ui.source_cache import SourceCache
from scantailor.core import pipeline
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.page_source import load_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.select_content import content_finder

DPI = 600
HEIGHT, WIDTH = 1750, 1240
BENCHMARK_HEIGHT, BENCHMARK_WIDTH = 3500, 2480
_PASS_KERNELS = (
    "warpAffine",
    "remap",
    "resize",
    "cvtColor",
    "transform",
    "GaussianBlur",
    "sepFilter2D",
    "filter2D",
    "boxFilter",
    "blur",
    "integral",
    "morphologyEx",
    "erode",
    "dilate",
    "connectedComponentsWithStats",
    "threshold",
    "calcHist",
    "copyMakeBorder",
    "LUT",
)


def _write_dense_page(path: Path, height: int, width: int) -> PageId:
    """Lines of text-like words over most of a 600 dpi page."""
    rng = np.random.default_rng(1)
    pixels = np.full((height, width), 238, dtype=np.uint8)
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "elit"]
    y = 300
    while y < height - 300:
        x = 250
        while x < width - 350:
            word = words[int(rng.integers(len(words)))]
            cv2.putText(
                pixels, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3, cv2.LINE_AA
            )
            x += 30 * len(word) + 20
        y += 55
    write_image(path, pixels, Dpi.uniform(DPI))
    return PageId(image_id=ImageId(file_path=path))


def _batch_detection(page: PageId) -> None:
    """What a batch worker does to find a page's geometry."""
    loaded = load_page(page.image_id)
    options = pipeline.PipelineOptions(dpi=loaded.resolution())
    pipeline.fix_aggregate_page_size([(loaded.pixels, page)], StageSequence(), options)


def _batch_render(page: PageId) -> None:
    """What a batch worker does to produce a page's output, before writing it."""
    loaded = load_page(page.image_id)
    options = pipeline.PipelineOptions(dpi=loaded.resolution())
    pipeline.process_page(loaded.pixels, page, StageSequence(), options)


@pytest.fixture(scope="module")
def dense_page(tmp_path_factory: pytest.TempPathFactory) -> PageId:
    return _write_dense_page(
        tmp_path_factory.mktemp("dense") / "page.tif", HEIGHT, WIDTH
    )


class _FullPagePasses:
    """Counts OpenCV calls given an array of at least half the page's pixels."""

    def __init__(self, page_pixels: int) -> None:
        self._threshold = page_pixels // 2
        self.count = 0

    def wrap(self, kernel: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            if any(
                isinstance(arg, np.ndarray)
                and arg.ndim >= 2
                and arg.shape[0] * arg.shape[1] >= self._threshold
                for arg in args
            ):
                self.count += 1
            return kernel(*args, **kwargs)

        return counted

    def of(self, work: Callable[[], object], *, fresh: bool = True) -> int:
        if fresh:
            content_finder._CONTENT_MEMO.clear()  # the batch runs in other processes
        self.count = 0
        work()
        return self.count


@pytest.fixture
def passes(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FullPagePasses]:
    counter = _FullPagePasses(HEIGHT * WIDTH)
    for name in _PASS_KERNELS:
        monkeypatch.setattr(cv2, name, counter.wrap(getattr(cv2, name)))
    yield counter


def _interactive(page: PageId, stage: FilterStage) -> None:
    render_page_with_model(
        page,
        StageSequence(),
        stage,
        sources=SourceCache(),
        pixels=stage == FilterStage.OUTPUT,
    )


def test_select_content_makes_no_more_full_page_passes_than_batch_detection(
    dense_page: PageId, passes: _FullPagePasses
) -> None:
    batch = passes.of(lambda: _batch_detection(dense_page))

    interactive = passes.of(
        lambda: _interactive(dense_page, FilterStage.SELECT_CONTENT)
    )

    assert batch > 0
    assert interactive <= batch


def test_output_makes_no_more_full_page_passes_than_the_batch_render(
    dense_page: PageId, passes: _FullPagePasses
) -> None:
    batch = passes.of(lambda: _batch_render(dense_page))

    interactive = passes.of(lambda: _interactive(dense_page, FilterStage.OUTPUT))

    assert batch > 0
    assert interactive <= batch


def test_showing_a_stage_again_reuses_the_decoded_page_and_the_detected_content(
    dense_page: PageId, passes: _FullPagePasses, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = SourceCache()
    stages = StageSequence()
    calls: collections.Counter[str] = collections.Counter()

    def counting(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return function(*args, **kwargs)

        return counted

    monkeypatch.setattr(
        source_cache, "_decode", counting("decode", source_cache._decode)
    )
    monkeypatch.setattr(
        content_finder,
        "find_content_rect",
        counting("detect", content_finder.find_content_rect),
    )

    def show() -> None:
        render_page_with_model(
            dense_page,
            stages,
            FilterStage.SELECT_CONTENT,
            sources=sources,
            pixels=False,
        )

    first = passes.of(show)
    first_calls = calls.copy()
    calls.clear()
    again = passes.of(show, fresh=False)

    assert first_calls == {"decode": 1, "detect": 1}
    assert not calls
    assert again < first


@pytest.fixture(scope="module")
def benchmark_page(tmp_path_factory: pytest.TempPathFactory) -> PageId:
    page = _write_dense_page(
        tmp_path_factory.mktemp("dense") / "page.tif",
        BENCHMARK_HEIGHT,
        BENCHMARK_WIDTH,
    )
    _batch_render(page)  # compile what the first page would compile
    return page


def _best_of(runs: int, work: Callable[[], object]) -> float:
    best = float("inf")
    for _ in range(runs):
        content_finder._CONTENT_MEMO.clear()
        start = time.perf_counter()
        work()
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "stage", [FilterStage.SELECT_CONTENT, FilterStage.OUTPUT], ids=lambda s: s.name
)
def test_interactive_and_batch_time_on_a_dense_page(
    benchmark_page: PageId,
    stage: FilterStage,
    record_property: Callable[[str, object], None],
) -> None:
    batch = _batch_render if stage == FilterStage.OUTPUT else _batch_detection
    record_property(
        "benchmark_seconds_batch", _best_of(2, lambda: batch(benchmark_page))
    )
    record_property(
        "benchmark_seconds_interactive",
        _best_of(2, lambda: _interactive(benchmark_page, stage)),
    )
