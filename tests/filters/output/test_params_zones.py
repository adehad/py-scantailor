"""Zones, distortion model and depth perception in the output Params."""

import pytest
from pydantic import ValidationError

from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
    Zone,
)
from scantailor.filters.output.params import (
    DistortionModelSpec,
    FillZoneSpec,
    Params,
    PictureZoneSpec,
)

SQUARE = ((10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0))


class TestZoneSpecs:
    """Specs describe zones in plain, serialisable values."""

    def test_picture_zone_spec_makes_a_picture_zone(self) -> None:
        spec = PictureZoneSpec(
            points=SQUARE,
            properties=PictureZoneProperties(layer=PictureLayer.ERASER1),
        )

        zone = spec.to_zone()

        assert isinstance(zone, Zone)
        assert zone.is_picture_zone()
        assert zone.is_valid()
        assert zone.properties.layer == PictureLayer.ERASER1
        assert zone.spline.points.tolist() == [list(point) for point in SQUARE]

    def test_fill_zone_spec_makes_a_fill_zone(self) -> None:
        spec = FillZoneSpec(
            points=SQUARE,
            is_smooth=True,
            properties=FillZoneProperties(color=(1, 2, 3)),
        )

        zone = spec.to_zone()

        assert zone.is_fill_zone()
        assert zone.properties.color == (1, 2, 3)
        assert zone.spline.is_smooth

    def test_spec_without_points_is_not_a_valid_zone(self) -> None:
        assert not FillZoneSpec().to_zone().is_valid()


class TestParams:
    """Params carry the zones and the manual model."""

    def test_defaults_have_no_zones_and_the_default_depth(self) -> None:
        params = Params()

        assert params.picture_zones == ()
        assert params.fill_zones == ()
        assert params.depth_perception == 2.0
        assert not params.distortion_model.is_defined()

    def test_json_round_trip_keeps_zones_and_model(self) -> None:
        params = (
            Params()
            .with_picture_zones(
                (
                    PictureZoneSpec(
                        points=SQUARE,
                        properties=PictureZoneProperties(layer=PictureLayer.PAINTER2),
                    ),
                )
            )
            .with_fill_zones(
                (
                    FillZoneSpec(
                        points=SQUARE, properties=FillZoneProperties(color=(9, 8, 7))
                    ),
                )
            )
            .with_distortion_model(
                DistortionModelSpec(
                    top=((0.0, 0.0), (100.0, 5.0)), bottom=((0.0, 90.0), (100.0, 95.0))
                )
            )
            .with_depth_perception(2.5)
        )

        restored = Params.model_validate(params.model_dump(mode="json"))

        assert restored == params
        assert restored.picture_zones[0].properties.layer == PictureLayer.PAINTER2
        assert restored.fill_zones[0].properties.color == (9, 8, 7)
        assert restored.distortion_model.is_defined()

    def test_older_json_without_the_new_fields_still_loads(self) -> None:
        data = Params().model_dump(mode="json")
        for key in (
            "picture_zones",
            "fill_zones",
            "distortion_model",
            "depth_perception",
        ):
            del data[key]

        assert Params.model_validate(data) == Params()

    @pytest.mark.parametrize("value", [0.9, 3.1])
    def test_depth_perception_outside_the_range_is_rejected(self, value: float) -> None:
        with pytest.raises(ValidationError):
            Params(depth_perception=value)

    def test_zone_and_model_setters_return_copies(self) -> None:
        params = Params()

        changed = params.with_depth_perception(1.5)

        assert changed is not params
        assert params.depth_perception == 2.0
        assert changed.depth_perception == 1.5


class TestDistortionModelSpec:
    """A model is defined once both edges have two points."""

    def test_needs_both_edges(self) -> None:
        edge = ((0.0, 0.0), (10.0, 0.0))

        assert DistortionModelSpec(top=edge, bottom=edge).is_defined()
        assert not DistortionModelSpec(top=edge).is_defined()
        assert not DistortionModelSpec(top=edge[:1], bottom=edge).is_defined()
