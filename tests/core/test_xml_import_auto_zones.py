"""Auto picture zones of an upstream project are kept, not detected again."""

from pathlib import Path

from scantailor.core.zones import ZoneCategory
from scantailor.filters.output.picture_shape import PictureShape
from tests.core.test_xml_import_output_1_2_1 import _import, _output_params


def test_found_auto_zones_of_a_rectangular_page_are_recorded_as_current(
    tmp_path: Path,
):
    project = _import(
        tmp_path, "split_and_mixed", {'autoZonesFound="0"': 'autoZonesFound="1"'}
    )

    params = _output_params(project)[project.get_pages()[0]]

    assert params.picture_shape.shape is PictureShape.RECTANGULAR
    assert params.auto_picture_zones_shape == params.picture_shape
    assert params.picture_zones[1].properties.category is ZoneCategory.AUTO


def test_auto_zones_not_yet_found_are_detected_on_the_first_render(tmp_path: Path):
    project = _import(tmp_path, "split_and_mixed", {})

    params = _output_params(project)[project.get_pages()[0]]

    assert params.auto_picture_zones_shape is None


def test_free_shape_pages_record_no_auto_zones_even_when_upstream_says_found(
    tmp_path: Path,
):
    project = _import(
        tmp_path, "split_and_mixed", {'autoZonesFound="0"': 'autoZonesFound="1"'}
    )

    free_pages = [
        params
        for params in _output_params(project).values()
        if params.picture_shape.shape is PictureShape.FREE
    ]

    assert free_pages
    assert all(params.auto_picture_zones_shape is None for params in free_pages)
