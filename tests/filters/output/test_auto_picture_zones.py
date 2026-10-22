"""Rectangular picture detection as editable auto picture zones."""

import numpy as np
import pytest

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.zones import (
    PictureLayer,
    PictureZoneProperties,
    Zone,
    ZoneCategory,
    ZoneSpline,
)
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import (
    AutoPictureZones,
    Params,
    PictureZoneSpec,
)
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions
from scantailor.filters.output.settings import Settings
from tests.filters.output.test_mixed_output import (
    PHOTO,
    _interior,
    _mixed_params,
    _synthetic_page,
)

RECTANGULAR = PictureShapeOptions(shape=PictureShape.RECTANGULAR)
AUTO = PictureZoneProperties(layer=PictureLayer.PAINTER2, category=ZoneCategory.AUTO)
MANUAL = PictureZoneProperties(layer=PictureLayer.PAINTER2)
SQUARE = ((10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0))
ELSEWHERE = ((300.0, 20.0), (400.0, 20.0), (400.0, 120.0), (300.0, 120.0))


def _zone(points, properties: PictureZoneProperties) -> Zone:
    return Zone(spline=ZoneSpline(points=np.array(points)), properties=properties)


def _bounds(zone: Zone) -> tuple[float, float, float, float]:
    left, top = zone.spline.points.min(axis=0)
    right, bottom = zone.spline.points.max(axis=0)
    return left, top, right, bottom


class TestDetection:
    """A mixed render with rectangular pictures detects auto zones when due."""

    def test_rectangular_detection_returns_one_auto_zone_around_the_photo(self):
        page, _ = _synthetic_page()

        result = generate_output(page, _mixed_params(picture_shape=RECTANGULAR))

        assert result.auto_picture_zones is not None
        (zone,) = result.auto_picture_zones
        assert zone.properties == AUTO
        rows, cols = PHOTO
        left, top, right, bottom = _bounds(zone)
        assert left == pytest.approx(cols.start, abs=6)
        assert right == pytest.approx(cols.stop - 1, abs=6)
        assert top == pytest.approx(rows.start, abs=6)
        assert bottom == pytest.approx(rows.stop - 1, abs=6)

    def test_the_detected_zone_is_drawn_in_the_same_render(self):
        page, _ = _synthetic_page()

        result = generate_output(page, _mixed_params(picture_shape=RECTANGULAR))

        assert result.picture_mask is not None
        assert result.picture_mask[_interior(PHOTO)].all()
        assert not result.picture_mask[:250].any()

    def test_up_to_date_auto_zones_are_drawn_as_given_without_detecting(self):
        page, _ = _synthetic_page()
        params = _mixed_params(
            picture_shape=RECTANGULAR, auto_picture_zones_shape=RECTANGULAR
        )

        result = generate_output(page, params, picture_zones=[_zone(ELSEWHERE, AUTO)])

        assert result.auto_picture_zones is None
        assert result.picture_mask is not None
        assert result.picture_mask[30:110, 310:390].all()
        assert not result.picture_mask[_interior(PHOTO)].any()

    def test_changed_sensitivity_detects_again_and_drops_the_old_auto_zones(self):
        page, _ = _synthetic_page()
        params = _mixed_params(
            picture_shape=RECTANGULAR.model_copy(update={"sensitivity": 50}),
            auto_picture_zones_shape=RECTANGULAR,
        )

        result = generate_output(page, params, picture_zones=[_zone(ELSEWHERE, AUTO)])

        assert result.auto_picture_zones is not None
        assert len(result.auto_picture_zones) == 1
        assert result.picture_mask is not None
        assert not result.picture_mask[30:110, 310:390].any()

    def test_manual_zones_are_drawn_beside_freshly_detected_ones(self):
        page, _ = _synthetic_page()

        result = generate_output(
            page,
            _mixed_params(picture_shape=RECTANGULAR),
            picture_zones=[_zone(ELSEWHERE, MANUAL)],
        )

        assert result.picture_mask is not None
        assert result.picture_mask[30:110, 310:390].all()
        assert result.picture_mask[_interior(PHOTO)].all()

    @pytest.mark.parametrize("shape", [PictureShape.FREE, PictureShape.OFF])
    def test_other_shapes_clear_the_auto_zones_and_do_not_draw_them(self, shape):
        page, _ = _synthetic_page()
        params = _mixed_params(
            picture_shape=PictureShapeOptions(shape=shape),
            auto_picture_zones_shape=RECTANGULAR,
        )

        result = generate_output(page, params, picture_zones=[_zone(ELSEWHERE, AUTO)])

        assert result.auto_picture_zones == ()
        assert result.picture_mask is not None
        assert not result.picture_mask[30:110, 310:390].any()

    def test_a_free_shape_page_without_auto_zones_reports_no_change(self):
        page, _ = _synthetic_page()

        result = generate_output(page, _mixed_params())

        assert result.auto_picture_zones is None

    def test_black_and_white_output_leaves_auto_zones_alone(self):
        page, _ = _synthetic_page()
        params = Params(picture_shape=RECTANGULAR, output_dpi=Dpi.uniform(300))

        result = generate_output(page, params)

        assert result.auto_picture_zones is None


class TestStoringAutoZones:
    """Params and Settings replace auto zones and keep manual ones."""

    def _page(self) -> PageId:
        return PageId(
            image_id=ImageId(file_path="page.tiff"), sub_page=SubPage.SINGLE_PAGE
        )

    def test_found_zones_replace_auto_zones_and_keep_manual_ones(self):
        params = Params(
            picture_shape=RECTANGULAR,
            picture_zones=(
                PictureZoneSpec(points=SQUARE, properties=MANUAL),
                PictureZoneSpec(points=ELSEWHERE, properties=AUTO),
            ),
        )
        found = PictureZoneSpec(points=SQUARE, properties=AUTO)

        updated = params.with_auto_picture_zones(
            AutoPictureZones(zones=(found,), picture_shape=RECTANGULAR)
        )

        assert updated.picture_zones == (params.picture_zones[0], found)
        assert updated.auto_picture_zones_shape == RECTANGULAR

    def test_found_nothing_for_a_free_shape_clears_the_record(self):
        params = Params(
            auto_picture_zones_shape=RECTANGULAR,
            picture_zones=(PictureZoneSpec(points=SQUARE, properties=AUTO),),
        )

        updated = params.with_auto_picture_zones(
            AutoPictureZones(picture_shape=params.picture_shape)
        )

        assert updated.picture_zones == ()
        assert updated.auto_picture_zones_shape is None

    def test_settings_store_zones_found_with_the_current_options(self):
        settings = Settings()
        page = self._page()
        settings.set_params(page, Params(picture_shape=RECTANGULAR))
        zone = PictureZoneSpec(points=SQUARE, properties=AUTO)

        changed = settings.replace_auto_picture_zones(
            page, AutoPictureZones(zones=(zone,), picture_shape=RECTANGULAR)
        )

        assert changed
        assert settings.get_params(page).picture_zones == (zone,)

    def test_settings_ignore_zones_found_before_the_options_changed(self):
        settings = Settings()
        page = self._page()
        newer = RECTANGULAR.model_copy(update={"sensitivity": 40})
        settings.set_params(page, Params(picture_shape=newer))

        changed = settings.replace_auto_picture_zones(
            page,
            AutoPictureZones(
                zones=(PictureZoneSpec(points=SQUARE, properties=AUTO),),
                picture_shape=RECTANGULAR,
            ),
        )

        assert not changed
        assert settings.get_params(page).picture_zones == ()
        assert settings.get_params(page).auto_picture_zones_shape is None

    def test_project_json_without_the_record_loads_as_never_detected(self):
        stored = Params(picture_shape=RECTANGULAR).model_dump(mode="json")
        del stored["auto_picture_zones_shape"]

        params = Params.model_validate(stored)

        assert params.auto_picture_zones_shape is None

    def test_the_record_survives_a_json_round_trip(self):
        params = Params(picture_shape=RECTANGULAR, auto_picture_zones_shape=RECTANGULAR)

        loaded = Params.model_validate_json(params.model_dump_json())

        assert loaded == params
