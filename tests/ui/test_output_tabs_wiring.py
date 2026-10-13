"""The Output view's dewarping, despeckling and picture zone tabs get their pictures."""

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import ImageViewTab
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.stage_panels import (
    _model_in_page_pixels,
    _model_in_view_pixels,
)
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.output.params import ColorMode, DistortionModelSpec
from scantailor.filters.page_layout.params import ContentRect, ContentSize

RENDER_TIMEOUT_MS = 60_000


def test_a_model_mapped_into_the_view_and_back_is_unchanged() -> None:
    xform = ImageTransformation.from_size(400, 300, Dpi.uniform(300))
    scale = np.array([1.5, 2.0])
    model = DistortionModelSpec(
        top=((10.0, 20.0), (200.0, 15.0), (390.0, 22.0)),
        bottom=((12.0, 280.0), (200.0, 285.0), (388.0, 279.0)),
    )

    there = _model_in_view_pixels(model, xform, scale)
    back = _model_in_page_pixels(there, xform, scale)

    assert there.top[0] == pytest.approx((15.0, 40.0))
    assert np.allclose(back.top, model.top)
    assert np.allclose(back.bottom, model.bottom)


def test_an_undefined_model_stays_empty_through_the_mapping() -> None:
    xform = ImageTransformation.from_size(400, 300, Dpi.uniform(300))

    mapped = _model_in_view_pixels(DistortionModelSpec(), xform, np.ones(2))

    assert mapped == DistortionModelSpec()


def _page_with_text() -> np.ndarray:
    page = np.full((400, 300, 3), 235, dtype=np.uint8)
    for top in range(60, 340, 20):
        page[top : top + 6, 40:260] = 30
    return page


def test_the_pipeline_returns_the_tab_pictures_only_when_asked() -> None:
    page_id = PageId(
        image_id=ImageId(file_path=Path("/book.tiff")), sub_page=SubPage.SINGLE_PAGE
    )
    for asked in (False, True):
        sequence = StageSequence()
        params = sequence.output_filter.get_params(page_id)
        sequence.output_filter.set_params(
            page_id, params.model_copy(update={"color_mode": ColorMode.BLACK_AND_WHITE})
        )

        result = process_page(
            _page_with_text(),
            page_id,
            sequence,
            PipelineOptions(dpi=300.0, keep_output_extras=asked),
        )

        if not asked:
            assert result.output_extras is None
            continue
        extras = result.output_extras
        assert extras is not None
        assert extras.page.shape[:2] == result.image.shape[:2]
        assert extras.despeckle_source is not None
        assert set(np.unique(extras.despeckle_source)) <= {0, 255}
        assert extras.despeckle_dpi == result.dpi.horizontal


@pytest.mark.slow
def test_rendering_the_output_fills_the_view_tabs(
    qtbot: QtBot, project_window: MainWindowManager
) -> None:
    getattr(
        project_window, f"actionSwitchFilter{int(FilterStage.OUTPUT) + 1}"
    ).trigger()
    layout = project_window.stages.page_layout_filter.settings
    for page in project_window.pages():
        layout.update_content_size_and_get_aggregate(
            page,
            ContentRect(width=600, height=800),
            ContentRect(width=500, height=700),
            ContentSize(width_mm=100, height_mm=150),
        )

    with qtbot.waitSignal(
        project_window.renderer.output_extras_found, timeout=RENDER_TIMEOUT_MS
    ):
        project_window._request_render()
    view = project_window._panels.view
    qtbot.waitUntil(lambda: view.image is not None, timeout=RENDER_TIMEOUT_MS)
    qtbot.waitUntil(
        lambda: ImageViewTab.DEWARPING in view._tab_images, timeout=RENDER_TIMEOUT_MS
    )

    assert view.tabs() == list(ImageViewTab)
    assert ImageViewTab.PICTURE_ZONES in view._tab_images
    options = project_window._panels.options
    assert isinstance(options, QtWidgets.QWidget)
    view.set_current_tab(ImageViewTab.DEWARPING)
    assert options._dewarping_tab_active
    view.set_current_tab(ImageViewTab.OUTPUT)
    assert not options._dewarping_tab_active
