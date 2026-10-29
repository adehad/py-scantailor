"""A page on screen detects the same geometry as the batch geometry pass."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import cv2
import pytest
from PIL import Image

from scantailor.app.ui.page_renderer import render_page_with_model
from scantailor.app.ui.source_cache import SourceCache
from scantailor.core import batch
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.stage_params import collect_stage_params
from scantailor.core.stage_sequence import FilterStage, StageSequence

DPI = 600


def _sample_file(sample_book_tool: ModuleType, name: str, path: Path) -> list[PageId]:
    """Write a sample page scaled up to 600 dpi and name the pages it holds."""
    canvas = sample_book_tool.render_sample_page(name).image
    factor = DPI // sample_book_tool.SAMPLE_DPI
    height, width = canvas.shape[:2]
    pixels = cv2.resize(
        canvas, (width * factor, height * factor), interpolation=cv2.INTER_CUBIC
    )
    Image.fromarray(pixels).save(path, dpi=(DPI, DPI), quality=90)
    image = ImageId(file_path=path)
    if "spread" not in name:
        return [PageId(image_id=image, sub_page=SubPage.SINGLE_PAGE)]
    return [
        PageId(image_id=image, sub_page=SubPage.LEFT_PAGE),
        PageId(image_id=image, sub_page=SubPage.RIGHT_PAGE),
    ]


def _batch_geometry(pages: list[PageId], output: Path) -> StageSequence:
    stages = StageSequence()
    with ThreadPoolExecutor(1) as pool:
        result = batch.run_batch(
            stages,
            pages,
            output,
            end_stage=FilterStage.PAGE_LAYOUT,
            executor=pool,  # type: ignore[arg-type]
        )
    assert result.failed == {}
    return stages


def _gui_geometry(pages: list[PageId]) -> StageSequence:
    stages = StageSequence()
    sources = SourceCache()
    for page in pages:
        render_page_with_model(
            page, stages, FilterStage.PAGE_LAYOUT, sources=sources, pixels=False
        )
    return stages


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "suffix"),
    [
        ("02_spread_light_gutter", ".jpg"),
        ("09_specks", ".png"),
    ],
)
def test_a_600_dpi_page_detects_the_same_geometry_on_screen_as_in_a_batch(
    sample_book_tool: ModuleType, tmp_path: Path, name: str, suffix: str
) -> None:
    pages = _sample_file(sample_book_tool, name, tmp_path / f"{name}{suffix}")

    in_batch = _batch_geometry(pages, tmp_path / "out")
    on_screen = _gui_geometry(pages)

    assert collect_stage_params(on_screen, pages) == collect_stage_params(
        in_batch, pages
    )
