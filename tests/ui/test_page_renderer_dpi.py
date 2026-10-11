"""The renderer and the source decoder of the stage panels honour the project DPI."""

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore

from scantailor.app.ui import stage_panels
from scantailor.app.ui.page_renderer import render_page, render_thumbnail
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

FILE_DPI = 72
FIXED_DPI = 300


def _source(path: Path, dpi: int) -> PageId:
    pixels = np.full((400, 300), 220, dtype=np.uint8)
    pixels[100:300, 80:220] = 40
    write_image(path, pixels, Dpi.uniform(dpi))
    return PageId(image_id=ImageId(file_path=path))


def _stages(page: PageId) -> StageSequence:
    stages = StageSequence()
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 0.0)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=80, y=100, width=140, height=200)
    )
    stages.output_filter.set_output_dpi(page, Dpi.uniform(FIXED_DPI))
    return stages


def test_render_with_fixed_dpi_matches_a_file_saved_at_that_dpi(tmp_path: Path):
    low = _source(tmp_path / "low.tif", FILE_DPI)
    high = _source(tmp_path / "high.tif", FIXED_DPI)
    overrides = {low.image_id: Dpi.uniform(FIXED_DPI)}

    fixed = render_page(low, _stages(low), FilterStage.OUTPUT, overrides)
    reference = render_page(high, _stages(high), FilterStage.OUTPUT)

    assert fixed.shape == reference.shape


def test_render_without_override_uses_the_file_dpi(tmp_path: Path):
    low = _source(tmp_path / "low.tif", FILE_DPI)
    overrides = {low.image_id: Dpi.uniform(FIXED_DPI)}

    unfixed = render_page(low, _stages(low), FilterStage.OUTPUT)
    fixed = render_page(low, _stages(low), FilterStage.OUTPUT, overrides)

    assert unfixed.shape != fixed.shape


def test_thumbnail_is_rendered_at_the_project_dpi(tmp_path: Path):
    low = _source(tmp_path / "low.tif", FILE_DPI)
    unbounded = QtCore.QSize(10_000, 10_000)
    overrides = {low.image_id: Dpi.uniform(FIXED_DPI)}

    unfixed = render_thumbnail(low, _stages(low), FilterStage.OUTPUT, unbounded)
    fixed = render_thumbnail(
        low, _stages(low), FilterStage.OUTPUT, unbounded, overrides
    )

    assert unfixed.size() != fixed.size()


@pytest.mark.parametrize(
    ("override", "expected"),
    [(Dpi.uniform(FIXED_DPI), float(FIXED_DPI)), (None, float(FILE_DPI))],
    ids=["fixed", "unfixed"],
)
def test_stage_panel_source_is_decoded_with_the_effective_dpi(
    tmp_path: Path, override: Dpi | None, expected: float
):
    page = _source(tmp_path / "low.tif", FILE_DPI)
    overrides = {page.image_id: override} if override else {}
    signals = stage_panels._SourceSignals()
    loaded: list[float] = []
    signals.loaded.connect(lambda _gen, _id, _pixels, dpi: loaded.append(dpi))

    stage_panels._SourceTask(1, page.image_id, overrides, signals).run()

    assert loaded == [expected]
