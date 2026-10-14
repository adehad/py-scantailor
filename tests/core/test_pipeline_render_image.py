"""A run that ends before Output can skip resampling the page it ends with."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core import pipeline
from scantailor.core.models import ImageId, PageId
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

PAGE = PageId(image_id=ImageId(file_path=Path("render_image.png")))
HEIGHT, WIDTH = 400, 300
_RESAMPLE = pipeline.render_transformed


def _stages() -> StageSequence:
    """Stages whose parameters are all stored, so nothing has to be detected."""
    stages = StageSequence()
    stages.page_split_filter.set_layout_type(PAGE, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(PAGE, 2.0)
    stages.select_content_filter.set_manual_content(
        PAGE, ContentBox(x=40, y=50, width=200, height=280)
    )
    return stages


def _source() -> np.ndarray:
    source = np.full((HEIGHT, WIDTH, 3), 230, dtype=np.uint8)
    source[100:300, 60:240] = 30
    return source


def _run(render_image: bool) -> pipeline.PipelineResult:
    options = PipelineOptions(
        end_stage=FilterStage.PAGE_LAYOUT, dpi=300, render_image=render_image
    )
    return process_page(_source(), PAGE, _stages(), options)


def test_without_render_image_the_source_comes_back_unresampled() -> None:
    result = _run(render_image=False)

    assert result.image.shape == (HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(result.image, _source())


def test_without_render_image_the_transformation_and_metadata_are_unchanged() -> None:
    rendered, skipped = _run(render_image=True), _run(render_image=False)

    assert rendered.transformation is not None
    assert skipped.transformation is not None
    np.testing.assert_allclose(
        skipped.transformation.transform, rendered.transformation.transform
    )
    assert skipped.metadata == rendered.metadata


def test_render_image_gives_the_page_at_the_size_of_the_transformation() -> None:
    result = _run(render_image=True)

    assert result.transformation is not None
    rect = result.transformation.resulting_rect
    assert result.image.shape[:2] == (round(rect.height), round(rect.width))


def _resamples(monkeypatch: pytest.MonkeyPatch, render_image: bool) -> int:
    """Count the full page resamples one run makes."""
    calls: list[tuple[int, ...]] = []

    def counted(image: np.ndarray, xform: object) -> np.ndarray:
        calls.append(image.shape)
        return _RESAMPLE(image, xform)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "render_transformed", counted)
    _run(render_image=render_image)
    return len(calls)


def test_without_render_image_the_cropped_page_is_not_resampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = _resamples(monkeypatch, render_image=True)

    skipped = _resamples(monkeypatch, render_image=False)

    assert skipped == rendered - 1


def test_with_every_parameter_stored_a_run_without_render_image_never_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _resamples(monkeypatch, render_image=False) == 0


def test_the_view_size_matches_the_view_without_making_it() -> None:
    state = pipeline._PageState(_source(), 300)
    state.apply_deskew(
        _stages().deskew_filter, _stages().deskew_filter.get_params(PAGE)
    )

    size = state.view_size()

    height, width = state.view().shape[:2]
    assert size == (width, height)
