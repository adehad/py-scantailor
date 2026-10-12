"""Merging one option group of a page's Params into another page's Params."""

import pytest

from scantailor.core.models import Dpi
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import ColorCommonOptions
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.params import (
    ApplyGroup,
    DistortionModelSpec,
    FillZoneSpec,
    Params,
    PictureZoneSpec,
)
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions

_MODEL = DistortionModelSpec(top=((0, 0), (10, 1)), bottom=((0, 9), (10, 10)))
_PICTURE = PictureZoneSpec(points=((0, 0), (5, 0), (5, 5)))
_FILL = FillZoneSpec(points=((1, 1), (4, 1), (4, 4)))

_GROUP_FIELDS = {
    ApplyGroup.DPI: {"output_dpi"},
    ApplyGroup.COLOR_MODE: {
        "color_mode",
        "color_common",
        "picture_shape",
        "black_on_white",
    },
    ApplyGroup.SPLITTING: {"splitting"},
    ApplyGroup.BINARIZATION: {"binarization"},
    ApplyGroup.DESPECKLE: {"despeckle_level"},
    ApplyGroup.DEWARPING: {"dewarping"},
    ApplyGroup.DEPTH_PERCEPTION: {"depth_perception"},
}


def _source() -> Params:
    """Page whose option groups all differ from the defaults."""
    base = Params()
    return (
        base.with_output_dpi(Dpi.uniform(300))
        .with_color_mode(ColorMode.MIXED)
        .with_black_on_white(False)
        .with_color_common(ColorCommonOptions(fill_offcut=False))
        .with_picture_shape(PictureShapeOptions(shape=PictureShape.OFF))
        .with_despeckle_level(DespeckleLevel.AGGRESSIVE)
        .with_dewarping(base.dewarping.with_mode(DewarpingMode.MARGINAL))
        .with_depth_perception(2.7)
        .with_binarization(base.binarization.model_copy(update={"window_size": 321}))
        .with_splitting(base.splitting.model_copy(update={"split_output": True}))
    )


def _target() -> Params:
    """Page with its own zones and manual dewarp model."""
    return (
        Params()
        .with_picture_zones((_PICTURE,))
        .with_fill_zones((_FILL,))
        .with_distortion_model(_MODEL)
        .with_dewarping(Params().dewarping.with_mode(DewarpingMode.MANUAL))
    )


def test_colour_mode_group_keeps_the_target_zones_and_distortion_model():
    merged = _target().with_group_from(_source(), ApplyGroup.COLOR_MODE)

    assert merged.color_mode == ColorMode.MIXED
    assert merged.black_on_white is False
    assert merged.picture_zones == (_PICTURE,)
    assert merged.fill_zones == (_FILL,)
    assert merged.distortion_model == _MODEL


def test_despeckle_group_changes_only_the_despeckle_level():
    target = _target()

    merged = target.with_group_from(_source(), ApplyGroup.DESPECKLE)

    assert merged.despeckle_level == DespeckleLevel.AGGRESSIVE
    assert merged == target.with_despeckle_level(DespeckleLevel.AGGRESSIVE)


def test_dewarping_group_copies_the_mode_but_not_the_manual_model():
    source = _source().with_distortion_model(
        DistortionModelSpec(top=((0, 5), (1, 5)), bottom=((0, 6), (1, 6)))
    )

    merged = _target().with_group_from(source, ApplyGroup.DEWARPING)

    assert merged.dewarping.mode == DewarpingMode.MARGINAL
    assert merged.distortion_model == _MODEL


@pytest.mark.parametrize("group", list(ApplyGroup))
def test_every_group_leaves_other_groups_and_per_page_geometry_alone(
    group: ApplyGroup,
):
    source, target = _source(), _target()

    merged = target.with_group_from(source, group)

    for field in _GROUP_FIELDS[group]:
        assert getattr(source, field) != getattr(target, field), field
    for field in Params.model_fields:
        expected = source if field in _GROUP_FIELDS[group] else target
        assert getattr(merged, field) == getattr(expected, field), field
    assert merged.picture_zones == target.picture_zones
    assert merged.fill_zones == target.fill_zones
    assert merged.distortion_model == target.distortion_model


def test_merging_does_not_modify_either_page():
    source, target = _source(), _target()
    source_before, target_before = source.model_copy(), target.model_copy()

    target.with_group_from(source, ApplyGroup.BINARIZATION)

    assert source == source_before
    assert target == target_before
