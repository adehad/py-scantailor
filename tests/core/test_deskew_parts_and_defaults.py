"""Apply To parts and the "Automatic oblique correction" default of Deskew."""

from pathlib import Path

import pytest

from scantailor.core.default_params import (
    DefaultParams,
    DeskewDefaults,
    apply_default_params,
)
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.propagation import apply_deskew
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.deskew.params import AutoManualMode, Params

AUTO = AutoManualMode.AUTO
MANUAL = AutoManualMode.MANUAL


def _pages(count: int) -> list[PageId]:
    return [
        PageId(
            image_id=ImageId(file_path=Path(f"/scans/p{number}.png")),
            sub_page=SubPage.SINGLE_PAGE,
        )
        for number in range(count)
    ]


def _source_params() -> Params:
    return Params(
        deskew_angle_deg=2.0,
        mode=MANUAL,
        oblique_angle_deg=-3.0,
        oblique_mode=MANUAL,
    )


class TestApplyParts:
    """The Apply To checkboxes choose which of rotation and oblique are copied."""

    def test_both_parts_are_copied_by_default(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.deskew_filter.set_params(source, _source_params())

        apply_deskew(stages, source, [target])

        params = stages.deskew_filter.get_params(target)
        assert (params.deskew_angle_deg, params.oblique_angle_deg) == (2.0, -3.0)
        assert params.is_manual() and params.is_oblique_manual()

    def test_only_the_rotation_is_copied_when_oblique_is_unchecked(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.deskew_filter.set_params(source, _source_params())
        stages.deskew_filter.set_manual_oblique(target, 1.0)

        apply_deskew(stages, source, [target], apply_oblique=False)

        params = stages.deskew_filter.get_params(target)
        assert params.deskew_angle_deg == 2.0
        assert params.oblique_angle_deg == 1.0

    def test_only_the_oblique_is_copied_when_deskew_is_unchecked(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.deskew_filter.set_params(source, _source_params())
        stages.deskew_filter.set_manual_angle(target, 0.5)

        apply_deskew(stages, source, [target], apply_deskew=False)

        params = stages.deskew_filter.get_params(target)
        assert params.deskew_angle_deg == 0.5
        assert params.oblique_angle_deg == -3.0
        assert params.is_oblique_manual()

    def test_neither_part_leaves_the_target_as_it_was(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.deskew_filter.set_params(source, _source_params())
        stages.deskew_filter.set_manual_angle(target, 0.5)
        before = stages.deskew_filter.get_params(target)

        apply_deskew(stages, source, [target], apply_deskew=False, apply_oblique=False)

        assert stages.deskew_filter.get_params(target) == before

    def test_an_automatic_oblique_makes_the_targets_detect_their_own(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.deskew_filter.set_params(
            source,
            Params(
                deskew_angle_deg=2.0,
                mode=MANUAL,
                oblique_angle_deg=1.0,
                oblique_mode=AUTO,
            ),
        )
        stages.deskew_filter.set_manual_oblique(target, 4.0)

        apply_deskew(stages, source, [target])

        params = stages.deskew_filter.get_params(target)
        assert params.is_oblique_auto()
        assert params.oblique_angle_deg == 0.0
        assert params.redetect_oblique
        assert params.deskew_angle_deg == 2.0

    def test_a_target_without_params_keeps_the_default_for_the_part_not_copied(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.deskew_filter.set_params(source, _source_params())

        apply_deskew(stages, source, [target], apply_oblique=False)

        params = stages.deskew_filter.get_params(target)
        assert params.oblique_mode is MANUAL
        assert params.oblique_angle_deg == 0.0

    def test_every_target_gets_the_parts(self):
        stages = StageSequence()
        source, *targets = _pages(4)
        stages.deskew_filter.set_params(source, _source_params())

        apply_deskew(stages, source, targets)

        for target in targets:
            assert stages.deskew_filter.get_params(target).oblique_angle_deg == -3.0


class TestDefaults:
    """The profile's "Automatic oblique correction"."""

    def test_it_is_off_by_default(self):
        assert DeskewDefaults().auto_oblique is False
        assert DefaultParams().deskew.auto_oblique is False

    def test_off_gives_a_manual_unsheared_page(self):
        params = DeskewDefaults(auto_oblique=False).to_params()

        assert params.oblique_mode is MANUAL
        assert params.oblique_angle_deg == 0.0
        assert not params.redetect_oblique

    def test_on_gives_an_automatic_oblique_that_is_detected(self):
        params = DeskewDefaults(auto_oblique=True).to_params()

        assert params.oblique_mode is AUTO
        assert params.redetect_oblique

    def test_it_round_trips_through_json(self):
        original = DefaultParams(deskew=DeskewDefaults(auto_oblique=True))

        restored = DefaultParams.model_validate_json(original.model_dump_json())

        assert restored.deskew.auto_oblique is True

    def test_a_profile_saved_without_it_loads_as_off(self):
        restored = DeskewDefaults.model_validate(
            {"deskew_angle_deg": 0.0, "mode": "auto"}
        )

        assert restored.auto_oblique is False

    @pytest.mark.parametrize("enabled", [True, False])
    def test_a_new_project_takes_the_default(self, enabled: bool):
        stages = StageSequence()
        page = _pages(1)[0]

        apply_default_params(
            stages,
            [page.image_id],
            DefaultParams(deskew=DeskewDefaults(auto_oblique=enabled)),
        )

        assert stages.deskew_filter.auto_oblique_default is enabled
