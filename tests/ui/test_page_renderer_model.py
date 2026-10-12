"""The renderer hands the dewarping model a render built to the GUI thread."""

from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from scantailor.app.ui.page_renderer import PageRenderer, render_page_with_model
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox
from tests.dewarping.synthetic import (
    PAGE_DPI,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    curved_text_page,
)

RENDER_TIMEOUT_MS = 20_000


def _curved_page(tmp_path: Path, mode: DewarpingMode) -> tuple[PageId, StageSequence]:
    path = tmp_path / "curved.tif"
    write_image(path, curved_text_page(), Dpi.uniform(PAGE_DPI))
    page = PageId(image_id=ImageId(file_path=path))
    stages = StageSequence()
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 0.0)
    stages.select_content_filter.set_manual_content(
        page,
        ContentBox(x=100, y=100, width=PAGE_WIDTH - 200, height=PAGE_HEIGHT - 200),
    )
    params = stages.output_filter.get_params(page).model_copy(
        update={
            "color_mode": ColorMode.COLOR_GRAYSCALE,
            "output_dpi": Dpi.uniform(PAGE_DPI),
            "dewarping": DewarpingOptions(mode=mode, post_deskew=False),
        }
    )
    stages.output_filter.set_params(page, params)
    return page, stages


def test_render_with_model_returns_the_edges_automatic_dewarping_found(tmp_path: Path):
    page, stages = _curved_page(tmp_path, DewarpingMode.AUTO)

    image, model = render_page_with_model(page, stages, FilterStage.OUTPUT)

    assert image.size > 0
    assert model is not None and model.is_defined()


def test_render_with_model_returns_none_when_dewarping_is_off(tmp_path: Path):
    page, stages = _curved_page(tmp_path, DewarpingMode.OFF)

    _image, model = render_page_with_model(page, stages, FilterStage.OUTPUT)

    assert model is None


def test_renderer_emits_the_model_before_the_image(qtbot: QtBot, tmp_path: Path):
    page, stages = _curved_page(tmp_path, DewarpingMode.AUTO)
    renderer = PageRenderer()
    order: list[str] = []
    renderer.distortion_model_found.connect(lambda *_: order.append("model"))
    renderer.render_done.connect(lambda *_: order.append("image"))

    with qtbot.waitSignal(renderer.render_done, timeout=RENDER_TIMEOUT_MS):
        renderer.request(page, stages, FilterStage.OUTPUT)

    assert order == ["model", "image"]
    renderer.wait_for_done()


def test_stages_before_output_report_no_model(qtbot: QtBot, tmp_path: Path):
    page, stages = _curved_page(tmp_path, DewarpingMode.AUTO)
    renderer = PageRenderer()
    found: list[object] = []
    renderer.distortion_model_found.connect(lambda *args: found.append(args))

    with qtbot.waitSignal(renderer.render_done, timeout=RENDER_TIMEOUT_MS):
        renderer.request(page, stages, FilterStage.PAGE_LAYOUT)

    assert found == []
    assert (
        np.asarray(stages.output_filter.get_params(page).distortion_model.top).size == 0
    )
    renderer.wait_for_done()
