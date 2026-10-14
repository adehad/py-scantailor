"""Rendering a stage before Output from a shrunk copy of the source."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core import pipeline
from scantailor.core.models import ImageId, OrthogonalRotation, PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.fix_orientation.image_trim import ImageTrim
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

HEIGHT, WIDTH = 1800, 1300
DPI = 300.0
BOX = (160, 160)
# Generous for the two area averagings of the shrunk path against the one of the
# full-size path; a wrong geometry moves whole lines of text and fails by far.
MEAN_TOLERANCE = 3.0
P99_TOLERANCE = 35
STAGES_BEFORE_OUTPUT = [stage for stage in FilterStage if stage < FilterStage.OUTPUT]


def _scan() -> NDArray[np.uint8]:
    """A paper-coloured page with lines of text and a dark picture."""
    rng = np.random.default_rng(36)
    page = np.full((HEIGHT, WIDTH, 3), (222, 230, 236), dtype=np.uint8)
    for line in range(28):
        words = "".join(chr(int(c)) for c in rng.integers(97, 123, size=24))
        cv2.putText(
            page,
            words,
            (150, 260 + line * 48),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1.6,
            (30, 30, 30),
            2,
        )
    page[1500:1700, 700:1150] = (60, 90, 40)
    return page


def _page() -> PageId:
    return PageId(image_id=ImageId(file_path=Path("reduced.tif")))


def _stored_stages(page: PageId) -> StageSequence:
    """Parameters that need no detection, with a rotation and a trim."""
    stages = StageSequence()
    orientation = stages.fix_orientation_filter
    orientation.set_rotation(page.image_id, OrthogonalRotation(degrees=90))
    orientation.set_trim(
        page.image_id, ImageTrim(enabled=True, left=40, top=30, right=20, bottom=50)
    )
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 1.5)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=200, y=150, width=1300, height=900)
    )
    return stages


def _shrunk(image: NDArray[np.uint8], scale: float) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    size = (round(width * scale), round(height * scale))
    return np.asarray(
        cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8
    )


def _fitted(image: NDArray[np.uint8], full_size: tuple[int, int]) -> NDArray:
    """Shrink to the size a full-size view of ``full_size`` gets in ``BOX``."""
    width, height = full_size
    scale = min(BOX[0] / width, BOX[1] / height, 1.0)
    size = (round(width * scale), round(height * scale))
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA))


def _options(stage: FilterStage) -> pipeline.PipelineOptions:
    return pipeline.PipelineOptions(end_stage=stage, dpi=DPI)


@pytest.mark.parametrize("stage", STAGES_BEFORE_OUTPUT)
def test_a_view_from_the_shrunk_source_matches_the_full_size_view_shrunk(
    stage: FilterStage,
) -> None:
    page, scan = _page(), _scan()
    stages = _stored_stages(page)
    full = pipeline.process_page(scan, page, stages, _options(stage)).image

    rendered = pipeline.render_from_reduced_source(
        _shrunk(scan, 0.25),
        (WIDTH, HEIGHT),
        page,
        stages,
        _options(stage),
        (BOX[0] * 2, BOX[1] * 2),
    )

    assert rendered is not None
    view, full_size = rendered
    assert full_size == (full.shape[1], full.shape[0])
    expected = _fitted(full, full_size)
    actual = _fitted(view, full_size)
    assert actual.shape == expected.shape
    difference = np.abs(actual.astype(int) - expected.astype(int))
    assert difference.mean() < MEAN_TOLERANCE
    assert np.percentile(difference, 99) < P99_TOLERANCE


def test_the_view_keeps_the_shrunk_sources_scale() -> None:
    page, scan = _page(), _scan()
    stages = _stored_stages(page)

    rendered = pipeline.render_from_reduced_source(
        _shrunk(scan, 0.25),
        (WIDTH, HEIGHT),
        page,
        stages,
        _options(FilterStage.DESKEW),
        BOX,
    )

    assert rendered is not None
    view, (full_width, full_height) = rendered
    assert view.shape[1] == pytest.approx(full_width * 0.25, abs=1)
    assert view.shape[0] == pytest.approx(full_height * 0.25, abs=1)


def test_a_source_too_coarse_for_the_box_renders_nothing() -> None:
    page, scan = _page(), _scan()

    rendered = pipeline.render_from_reduced_source(
        _shrunk(scan, 0.05),
        (WIDTH, HEIGHT),
        page,
        _stored_stages(page),
        _options(FilterStage.DESKEW),
        (1000, 1000),
    )

    assert rendered is None


def test_the_output_stage_cannot_be_rendered_from_a_shrunk_source() -> None:
    page, scan = _page(), _scan()

    with pytest.raises(ValueError, match="Output"):
        pipeline.render_from_reduced_source(
            _shrunk(scan, 0.25),
            (WIDTH, HEIGHT),
            page,
            _stored_stages(page),
            _options(FilterStage.OUTPUT),
            BOX,
        )


def test_no_stage_detects_when_rendering_from_a_shrunk_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page, scan = _page(), _scan()
    stages = StageSequence()

    def no_detection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("detection ran")

    monkeypatch.setattr(stages.page_split_filter, "process", no_detection)
    monkeypatch.setattr(stages.deskew_filter, "detect_skew", no_detection)
    monkeypatch.setattr(stages.select_content_filter, "process", no_detection)

    rendered = pipeline.render_from_reduced_source(
        _shrunk(scan, 0.25),
        (WIDTH, HEIGHT),
        page,
        stages,
        _options(FilterStage.SELECT_CONTENT),
        BOX,
    )

    assert rendered is not None


def test_the_shrunk_source_is_left_as_it_was() -> None:
    page, scan = _page(), _scan()
    reduced = _shrunk(scan, 0.25)
    reduced.flags.writeable = False

    pipeline.render_from_reduced_source(
        reduced,
        (WIDTH, HEIGHT),
        page,
        _stored_stages(page),
        _options(FilterStage.PAGE_LAYOUT),
        BOX,
    )

    assert np.array_equal(reduced, _shrunk(scan, 0.25))
