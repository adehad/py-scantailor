"""The page on screen appears about as fast as the batch handles it.

Both sides are timed in the same process on the same dense 600 dpi page, with
nothing remembered from an earlier run, so the ratio holds on any machine.
"""

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from scantailor.app.ui.page_renderer import render_page_with_model
from scantailor.app.ui.source_cache import SourceCache
from scantailor.core import pipeline
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.page_source import load_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.select_content import content_finder

pytestmark = pytest.mark.slow

DPI = 600
HEIGHT, WIDTH = 3500, 2480
# The interactive view may cost this many times what the batch spends on a page.
ALLOWED_RATIO = 2.0


def _write_dense_page(path: Path) -> PageId:
    """Lines of text-like words over most of a 600 dpi page."""
    rng = np.random.default_rng(1)
    pixels = np.full((HEIGHT, WIDTH), 238, dtype=np.uint8)
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "elit"]
    y = 300
    while y < HEIGHT - 300:
        x = 250
        while x < WIDTH - 350:
            word = words[int(rng.integers(len(words)))]
            cv2.putText(
                pixels, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3, cv2.LINE_AA
            )
            x += 30 * len(word) + 20
        y += 55
    write_image(path, pixels, Dpi.uniform(DPI))
    return PageId(image_id=ImageId(file_path=path))


def _best_of(runs: int, work: Callable[[], object]) -> float:
    best = float("inf")
    for _ in range(runs):
        content_finder._CONTENT_MEMO.clear()  # the batch runs in other processes
        start = time.perf_counter()
        work()
        best = min(best, time.perf_counter() - start)
    return best


def _batch_detection(page: PageId) -> None:
    """What a batch worker does to find a page's geometry."""
    loaded = load_page(page.image_id)
    options = pipeline.PipelineOptions(dpi=loaded.horizontal_dpi())
    pipeline.fix_aggregate_page_size([(loaded.pixels, page)], StageSequence(), options)


def _batch_render(page: PageId) -> None:
    """What a batch worker does to produce a page's output, before writing it."""
    loaded = load_page(page.image_id)
    options = pipeline.PipelineOptions(dpi=loaded.horizontal_dpi())
    pipeline.process_page(loaded.pixels, page, StageSequence(), options)


@pytest.fixture(scope="module")
def dense_page(tmp_path_factory: pytest.TempPathFactory) -> PageId:
    page = _write_dense_page(tmp_path_factory.mktemp("dense") / "page.tif")
    _batch_render(page)  # compile what the first page would compile
    return page


def _interactive(page: PageId, stage: FilterStage) -> None:
    render_page_with_model(
        page,
        StageSequence(),
        stage,
        sources=SourceCache(),
        pixels=stage == FilterStage.OUTPUT,
    )


def test_select_content_appears_within_twice_the_batch_detection_time(
    dense_page: PageId,
) -> None:
    batch = _best_of(2, lambda: _batch_detection(dense_page))

    interactive = _best_of(
        2, lambda: _interactive(dense_page, FilterStage.SELECT_CONTENT)
    )

    assert interactive < ALLOWED_RATIO * batch, (interactive, batch)


def test_output_appears_within_twice_the_batch_render_time(dense_page: PageId) -> None:
    batch = _best_of(2, lambda: _batch_render(dense_page))

    interactive = _best_of(2, lambda: _interactive(dense_page, FilterStage.OUTPUT))

    assert interactive < ALLOWED_RATIO * batch, (interactive, batch)


def test_showing_a_stage_again_is_faster_than_the_first_time(
    dense_page: PageId,
) -> None:
    sources = SourceCache()
    stages = StageSequence()
    content_finder._CONTENT_MEMO.clear()

    def show() -> float:
        start = time.perf_counter()
        render_page_with_model(
            dense_page,
            stages,
            FilterStage.SELECT_CONTENT,
            sources=sources,
            pixels=False,
        )
        return time.perf_counter() - start

    first = show()
    again = show()

    assert again < first
