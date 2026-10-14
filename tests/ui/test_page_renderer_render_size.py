"""Stages whose view shows the source do not pay for resampling the page."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.app.ui.page_renderer import render_page_with_model
from scantailor.app.ui.source_cache import SourceCache
from scantailor.core import pipeline
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

HEIGHT, WIDTH = 500, 360
_RESAMPLE = pipeline.render_transformed


@pytest.fixture
def page(tmp_path: Path) -> PageId:
    pixels = np.full((HEIGHT, WIDTH, 3), 225, dtype=np.uint8)
    pixels[120:380, 80:280] = 25
    path = tmp_path / "page.png"
    write_image(path, pixels, Dpi.uniform(300))
    return PageId(image_id=ImageId(file_path=path))


def _stored(page: PageId) -> StageSequence:
    """Parameters that are all stored, as after a first visit or a batch."""
    stages = StageSequence()
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 1.5)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=60, y=90, width=240, height=320)
    )
    return stages


def _resampled_sizes(
    monkeypatch: pytest.MonkeyPatch, page: PageId, pixels: bool
) -> list[tuple[int, ...]]:
    sizes: list[tuple[int, ...]] = []

    def recorded(image: np.ndarray, xform: object) -> np.ndarray:
        result = _RESAMPLE(image, xform)  # type: ignore[arg-type]
        sizes.append(result.shape[:2])
        return result

    monkeypatch.setattr(pipeline, "render_transformed", recorded)
    render_page_with_model(
        page,
        _stored(page),
        FilterStage.PAGE_LAYOUT,
        sources=SourceCache(),
        pixels=pixels,
    )
    return sizes


def test_a_render_that_is_not_drawn_skips_resampling_the_final_page(
    monkeypatch: pytest.MonkeyPatch, page: PageId
) -> None:
    drawn = _resampled_sizes(monkeypatch, page, pixels=True)

    not_drawn = _resampled_sizes(monkeypatch, page, pixels=False)

    assert not_drawn == drawn[:-1]


def test_a_render_that_is_not_drawn_returns_an_empty_image(page: PageId) -> None:
    image, _model = render_page_with_model(
        page, _stored(page), FilterStage.PAGE_LAYOUT, pixels=False
    )

    assert image.size == 0
