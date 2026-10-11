"""Transformations rebuilt from stored parameters match what the pipeline renders."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, Margins, PageId, SubPage
from scantailor.core.page_geometry import (
    build_transformation,
    content_rect_of,
    transformation_dpi,
)
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

WIDTH = 600
HEIGHT = 800
DPI = 300.0


def _page(sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    return PageId(image_id=ImageId(file_path=Path("scan.png")), sub_page=sub_page)


def _scan() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    for top in range(100, 700, 40):
        image[top : top + 14, 80:520] = 30
    return image


def _sequence_with_manual_geometry(page: PageId, angle: float) -> StageSequence:
    sequence = StageSequence()
    sequence.deskew_filter.set_manual_angle(page, angle)
    sequence.select_content_filter.set_manual_content(
        page, ContentBox(x=60, y=80, width=400, height=600)
    )
    return sequence


def test_dpi_is_rounded_and_never_below_two():
    assert transformation_dpi(299.6).horizontal == 300
    assert transformation_dpi(0.2).horizontal == 2


def test_the_first_stage_has_the_identity_transformation():
    xform = build_transformation(
        StageSequence(), _page(), (WIDTH, HEIGHT), DPI, FilterStage.FIX_ORIENTATION
    )

    assert xform.transform_point(10, 20) == pytest.approx((10, 20))
    assert xform.resulting_rect.width == WIDTH
    assert xform.resulting_rect.height == HEIGHT


def test_page_split_crops_to_the_page_of_the_stored_layout():
    page = _page(SubPage.RIGHT_PAGE)
    sequence = StageSequence()
    sequence.page_split_filter.set_manual_split(page, 250.0, WIDTH, HEIGHT)

    xform = build_transformation(
        sequence, page, (WIDTH, HEIGHT), DPI, FilterStage.PAGE_SPLIT
    )

    assert xform.resulting_rect.width == pytest.approx(WIDTH - 250)
    assert xform.resulting_rect.height == pytest.approx(HEIGHT)
    assert xform.transform_point(250, 0) == pytest.approx((0, 0))


def test_deskew_rotates_by_the_stored_angle():
    page = _page()
    sequence = _sequence_with_manual_geometry(page, 3.0)

    xform = build_transformation(
        sequence, page, (WIDTH, HEIGHT), DPI, FilterStage.DESKEW
    )

    assert xform.post_rotation == 3.0
    assert xform.resulting_rect.width > WIDTH


@pytest.mark.parametrize("angle", [0.0, 2.5, -4.0])
@pytest.mark.parametrize(
    "stage",
    [
        FilterStage.PAGE_SPLIT,
        FilterStage.DESKEW,
        FilterStage.SELECT_CONTENT,
        FilterStage.PAGE_LAYOUT,
    ],
)
def test_the_rebuilt_transformation_has_the_size_of_the_pipelines_render(
    stage: FilterStage, angle: float
):
    page = _page()
    sequence = _sequence_with_manual_geometry(page, angle)
    sequence.page_layout_filter.set_margins(
        page, Margins(left=8, right=6, top=4, bottom=12)
    )
    rendered = process_page(
        _scan(), page, sequence, PipelineOptions(end_stage=stage, dpi=DPI)
    ).image

    through = FilterStage.DESKEW if stage == FilterStage.SELECT_CONTENT else stage
    xform = build_transformation(sequence, page, (WIDTH, HEIGHT), DPI, through)

    assert xform.resulting_rect.width == pytest.approx(rendered.shape[1], abs=1)
    assert xform.resulting_rect.height == pytest.approx(rendered.shape[0], abs=1)


def test_the_final_transformation_places_the_content_box_inside_the_page():
    page = _page()
    sequence = _sequence_with_manual_geometry(page, 0.0)
    sequence.page_layout_filter.set_margins(
        page, Margins(left=10, right=10, top=5, bottom=5)
    )
    process_page(
        _scan(),
        page,
        sequence,
        PipelineOptions(end_stage=FilterStage.PAGE_LAYOUT, dpi=DPI),
    )

    xform = build_transformation(
        sequence, page, (WIDTH, HEIGHT), DPI, FilterStage.PAGE_LAYOUT
    )

    left_margin_px = 10 / 25.4 * DPI
    top_margin_px = 5 / 25.4 * DPI
    x, y = xform.transform_point(60, 80)
    assert x == pytest.approx(left_margin_px, abs=1)
    assert y == pytest.approx(top_margin_px, abs=1)


def test_building_a_transformation_does_not_change_any_parameters():
    page = _page()
    sequence = StageSequence()

    build_transformation(sequence, page, (WIDTH, HEIGHT), DPI, FilterStage.PAGE_LAYOUT)

    assert not sequence.page_layout_filter.settings.is_params_set(page)
    assert not sequence.page_split_filter.is_params_set(page)
    assert (
        sequence.page_split_filter.get_params(page).layout_type
        == LayoutType.AUTO_LAYOUT_TYPE
    )


def test_content_rect_of_returns_the_stored_box():
    page = _page()
    sequence = _sequence_with_manual_geometry(page, 0.0)

    rect = content_rect_of(sequence, page)

    assert (rect.x, rect.y, rect.width, rect.height) == (60, 80, 400, 600)
