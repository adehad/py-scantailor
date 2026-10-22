"""The Output stage stores auto picture zones in source coordinates."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.batch import run_batch
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.zones import ZoneCategory
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions
from tests.core.test_pipeline_output import _page_id, _sequence_for, _set_output
from tests.filters.output.test_mixed_output import (
    PAGE_HEIGHT,
    PAGE_WIDTH,
    PHOTO,
    _synthetic_page,
)

SOURCE_DPI = 300.0
RECTANGULAR = PictureShapeOptions(shape=PictureShape.RECTANGULAR)


def _mixed_rectangular_page(output_dpi: int = 300, page_id: PageId | None = None):
    page_id = page_id or _page_id()
    sequence = _sequence_for(page_id, (0, 0, PAGE_WIDTH, PAGE_HEIGHT))
    _set_output(
        sequence,
        page_id,
        color_mode=ColorMode.MIXED,
        output_dpi=Dpi.uniform(output_dpi),
        picture_shape=RECTANGULAR,
    )
    return page_id, sequence


def _render(page_id, sequence):
    page, _ = _synthetic_page()
    return process_page(page, page_id, sequence, PipelineOptions(dpi=SOURCE_DPI))


def _photo_corners() -> np.ndarray:
    rows, cols = PHOTO
    return np.array(
        [
            (cols.start, rows.start),
            (cols.stop - 1, rows.start),
            (cols.stop - 1, rows.stop - 1),
            (cols.start, rows.stop - 1),
        ],
        dtype=float,
    )


@pytest.mark.parametrize("output_dpi", [300, 600])
def test_the_photo_becomes_an_auto_zone_in_source_coordinates(output_dpi):
    page_id, sequence = _mixed_rectangular_page(output_dpi)

    result = _render(page_id, sequence)

    params = sequence.output_filter.get_params(page_id)
    (zone,) = params.picture_zones
    assert zone.properties.category is ZoneCategory.AUTO
    np.testing.assert_allclose(np.array(zone.points), _photo_corners(), atol=8)
    assert params.auto_picture_zones_shape == RECTANGULAR
    assert result.auto_picture_zones is not None
    assert result.auto_picture_zones.zones == params.picture_zones


def test_a_second_render_keeps_the_stored_auto_zones():
    page_id, sequence = _mixed_rectangular_page()
    _render(page_id, sequence)
    stored = sequence.output_filter.get_params(page_id)

    result = _render(page_id, sequence)

    assert result.auto_picture_zones is None
    assert sequence.output_filter.get_params(page_id) == stored


def test_new_options_detect_again_and_keep_an_edited_zone():
    page_id, sequence = _mixed_rectangular_page()
    _render(page_id, sequence)
    output = sequence.output_filter
    (auto,) = output.get_params(page_id).picture_zones
    edited = auto.model_copy(
        update={
            "points": ((10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)),
            "properties": auto.properties.model_copy(
                update={"category": ZoneCategory.MANUAL}
            ),
        }
    )
    sensitive = RECTANGULAR.model_copy(update={"higher_search_sensitivity": True})
    output.set_params(
        page_id,
        output.get_params(page_id)
        .with_picture_zones((edited,))
        .with_picture_shape(sensitive),
    )

    _render(page_id, sequence)

    params = output.get_params(page_id)
    manual, detected = params.picture_zones
    assert manual == edited
    assert detected.properties.category is ZoneCategory.AUTO
    assert params.auto_picture_zones_shape == sensitive


def test_switching_to_free_shape_removes_the_auto_zones():
    page_id, sequence = _mixed_rectangular_page()
    _render(page_id, sequence)
    output = sequence.output_filter
    output.set_params(
        page_id,
        output.get_params(page_id).with_picture_shape(PictureShapeOptions()),
    )

    _render(page_id, sequence)

    params = output.get_params(page_id)
    assert params.picture_zones == ()
    assert params.auto_picture_zones_shape is None


@pytest.mark.slow
def test_a_batch_worker_hands_its_auto_zones_back_to_the_project(tmp_path: Path):
    source = tmp_path / "in" / "mixed.tif"
    source.parent.mkdir()
    page, _ = _synthetic_page()
    write_image(source, page, Dpi.uniform(round(SOURCE_DPI)))
    page_id, sequence = _mixed_rectangular_page(
        page_id=PageId(image_id=ImageId(file_path=source), sub_page=SubPage.SINGLE_PAGE)
    )

    result = run_batch(sequence, [page_id], tmp_path / "out", max_workers=1)

    assert page_id in result.written
    params = sequence.output_filter.get_params(page_id)
    (zone,) = params.picture_zones
    assert zone.properties.category is ZoneCategory.AUTO
    assert params.auto_picture_zones_shape == RECTANGULAR
