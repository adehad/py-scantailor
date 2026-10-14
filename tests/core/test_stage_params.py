"""Tests for exporting and restoring per-page stage parameters."""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from scantailor.core.models import Dpi, ImageId, Margins, OrthogonalRotation, PageId
from scantailor.core.project import Project, StageParamsEntry
from scantailor.core.stage_params import (
    STAGE_KEYS,
    apply_stage_defaults,
    apply_stage_params,
    collect_stage_defaults,
    collect_stage_params,
)
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.deskew.params import AutoManualMode
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.fix_orientation.image_trim import ImageTrim
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.output.picture_shape import PictureShape
from scantailor.filters.page_layout.alignment import (
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_layout.params import ContentSize
from scantailor.filters.page_layout.params import Params as PageLayoutParams
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.params import Params as PageSplitParams
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)
from scantailor.filters.select_content.params import Params as SelectContentParams


def _page(name: str) -> PageId:
    return PageId(image_id=ImageId(file_path=Path("/scans") / name))


@pytest.fixture
def pages() -> list[PageId]:
    return [_page("a.png"), _page("b.png"), _page("c.png")]


@pytest.fixture
def configured(pages: list[PageId]) -> StageSequence:
    """Stages with non-default parameters on the second page only."""
    stages = StageSequence()
    page = pages[1]
    stages.fix_orientation_filter.settings.apply_rotation(
        page.image_id, OrthogonalRotation(degrees=270)
    )
    stages.page_split_filter.settings.set_params(
        page, PageSplitParams(layout_type=LayoutType.TWO_PAGES)
    )
    stages.deskew_filter.settings.set_params(
        page, DeskewParams().with_manual_angle(3.5)
    )
    stages.select_content_filter.settings.set_params(
        page,
        SelectContentParams(content_detection_mode=ContentDetectionMode.MANUAL),
    )
    stages.page_layout_filter.settings.set_params(
        page,
        PageLayoutParams(hard_margins_mm=Margins.uniform(12.5), auto_margins=True),
    )
    stages.output_filter.settings.set_params(
        page,
        OutputParams(output_dpi=Dpi.uniform(300), color_mode=ColorMode.MIXED),
    )
    return stages


def test_collect_from_untouched_stages_is_empty(pages: list[PageId]):
    assert collect_stage_params(StageSequence(), pages) == {}


def test_collect_includes_every_stage_that_has_explicit_params(
    configured: StageSequence, pages: list[PageId]
):
    collected = collect_stage_params(configured, pages)

    assert set(collected) == set(STAGE_KEYS.values())
    assert [entry.page for entry in collected["deskew"]] == [pages[1]]
    assert [entry.page.image_id for entry in collected["fix_orientation"]] == [
        pages[1].image_id
    ]


def test_collect_ignores_pages_outside_the_given_list(
    configured: StageSequence, pages: list[PageId]
):
    assert collect_stage_params(configured, [pages[0], pages[2]]) == {}


def test_collect_does_not_modify_the_stage_settings(
    configured: StageSequence, pages: list[PageId]
):
    collect_stage_params(configured, pages)

    assert not configured.deskew_filter.settings.is_params_set(pages[0])
    assert configured.deskew_filter.settings.is_params_set(pages[1])


def test_json_round_trip_restores_every_stage(
    configured: StageSequence, pages: list[PageId], tmp_path: Path
):
    project = Project(stage_params=collect_stage_params(configured, pages))
    project_file = tmp_path / "p.ScanTailor"
    project.save(project_file)
    restored = StageSequence()

    apply_stage_params(restored, Project.load(project_file).stage_params)

    page = pages[1]
    assert restored.fix_orientation_filter.settings.get_rotation_for(
        page.image_id
    ) == OrthogonalRotation(degrees=270)
    assert restored.page_split_filter.settings.get_params(page) == (
        configured.page_split_filter.settings.get_params(page)
    )
    assert restored.deskew_filter.settings.get_params(page) == (
        DeskewParams().with_manual_angle(3.5)
    )
    assert restored.select_content_filter.settings.get_params(page) == (
        configured.select_content_filter.settings.get_params(page)
    )
    assert restored.page_layout_filter.settings.get_params(page) == (
        configured.page_layout_filter.settings.get_params(page)
    )
    assert restored.output_filter.settings.get_params(page) == (
        configured.output_filter.settings.get_params(page)
    )


def test_restore_leaves_pages_without_entries_on_defaults(
    configured: StageSequence, pages: list[PageId]
):
    restored = StageSequence()

    apply_stage_params(restored, collect_stage_params(configured, pages))

    assert not restored.deskew_filter.settings.is_params_set(pages[0])
    assert not restored.output_filter.settings.is_params_set(pages[2])
    assert not restored.fix_orientation_filter.settings.is_rotation_set(
        pages[0].image_id
    )


def test_apply_ignores_unknown_stage_keys(pages: list[PageId]):
    stages = StageSequence()

    apply_stage_params(stages, {"not_a_stage": []})

    assert not stages.deskew_filter.settings.is_params_set(pages[0])


def test_apply_rejects_entries_that_do_not_match_the_stage_params(
    configured: StageSequence, pages: list[PageId]
):
    collected = collect_stage_params(configured, pages)
    collected["deskew"][0].params["deskew_angle_deg"] = 1000.0

    with pytest.raises(ValueError, match="deskew_angle_deg"):
        apply_stage_params(StageSequence(), collected)


class TestEnumValuesInProjectFiles:
    """Mode and level choices are stored as their plain values, as before."""

    STORED: ClassVar[dict[FilterStage, dict[str, Any]]] = {
        FilterStage.DESKEW: {"mode": "manual", "oblique_mode": "auto"},
        FilterStage.SELECT_CONTENT: {
            "content_detection_mode": "disabled",
            "page_detection_mode": "auto",
        },
        FilterStage.PAGE_LAYOUT: {
            "alignment": {"vertical": "bottom", "horizontal": "right"}
        },
        FilterStage.OUTPUT: {
            "color_mode": "mixed",
            "despeckle_level": "aggressive",
            "dewarping": {"mode": "marginal"},
            "picture_shape": {"shape": "rectangular"},
        },
    }

    def _write_existing_file(self, path: Path, page: PageId) -> None:
        stage_params = {
            STAGE_KEYS[stage]: [StageParamsEntry(page=page, params=params)]
            for stage, params in self.STORED.items()
        }
        Project(stage_params=stage_params).save(path)

    def test_a_file_with_plain_values_loads_every_choice(
        self, pages: list[PageId], tmp_path: Path
    ):
        project_file = tmp_path / "p.ScanTailor"
        self._write_existing_file(project_file, pages[0])
        stages = StageSequence()

        apply_stage_params(stages, Project.load(project_file).stage_params)

        page = pages[0]
        deskew = stages.deskew_filter.settings.get_params(page)
        assert deskew.mode is AutoManualMode.MANUAL
        assert deskew.oblique_mode is AutoManualMode.AUTO
        content = stages.select_content_filter.settings.get_params(page)
        assert content.content_detection_mode is ContentDetectionMode.DISABLED
        assert content.page_detection_mode is PageDetectionMode.AUTO
        alignment = stages.page_layout_filter.settings.get_params(page).alignment
        assert alignment.vertical is VerticalAlignment.BOTTOM
        assert alignment.horizontal is HorizontalAlignment.RIGHT
        output = stages.output_filter.settings.get_params(page)
        assert output.color_mode is ColorMode.MIXED
        assert output.despeckle_level is DespeckleLevel.AGGRESSIVE
        assert output.dewarping.mode is DewarpingMode.MARGINAL
        assert output.picture_shape.shape is PictureShape.RECTANGULAR

    def test_saving_the_loaded_choices_writes_the_same_plain_values(
        self, pages: list[PageId], tmp_path: Path
    ):
        first = tmp_path / "first.ScanTailor"
        self._write_existing_file(first, pages[0])
        stages = StageSequence()
        apply_stage_params(stages, Project.load(first).stage_params)
        second = tmp_path / "second.ScanTailor"

        Project(stage_params=collect_stage_params(stages, pages)).save(second)

        saved = json.loads(second.read_text(encoding="utf-8"))["stage_params"]
        for stage, stored in self.STORED.items():
            params = saved[STAGE_KEYS[stage]][0]["params"]
            for key, value in stored.items():
                if isinstance(value, dict):
                    assert {k: params[key][k] for k in value} == value
                else:
                    assert params[key] == value
        assert "ColorMode" not in second.read_text(encoding="utf-8")


def test_stage_keys_cover_all_stages_with_unique_names():
    assert set(STAGE_KEYS) == set(FilterStage)
    assert len(set(STAGE_KEYS.values())) == len(FilterStage)


class TestStageDefaults:
    @pytest.fixture
    def customised(self) -> StageSequence:
        stages = StageSequence()
        stages.page_split_filter.settings.set_default_layout_type(LayoutType.TWO_PAGES)
        output = stages.output_filter.settings
        output.set_default_dpi(Dpi.uniform(600))
        output.set_default_color_mode(ColorMode.MIXED)
        output.set_default_despeckle_level(DespeckleLevel.AGGRESSIVE)
        select_content = stages.select_content_filter.settings
        select_content.set_page_detection_box(210.0, 297.0)
        select_content.set_page_detection_tolerance(0.25)
        return stages

    def test_defaults_survive_a_project_file_round_trip(
        self, customised: StageSequence, tmp_path: Path
    ):
        project = Project()
        project.stage_defaults = collect_stage_defaults(customised)
        project_file = tmp_path / "p.json"

        project.save(project_file)
        restored = StageSequence()
        apply_stage_defaults(restored, Project.load(project_file).stage_defaults)

        assert collect_stage_defaults(restored) == collect_stage_defaults(customised)
        assert (
            restored.page_split_filter.settings.get_default_layout_type()
            is LayoutType.TWO_PAGES
        )
        output = restored.output_filter.settings
        assert output.get_default_dpi() == Dpi.uniform(600)
        assert output.get_default_color_mode() is ColorMode.MIXED
        assert output.get_default_despeckle_level() is DespeckleLevel.AGGRESSIVE
        select_content = restored.select_content_filter.settings
        assert select_content.get_page_detection_box() == (210.0, 297.0)
        assert select_content.get_page_detection_tolerance() == pytest.approx(0.25)

    def test_empty_defaults_leave_the_stage_defaults_alone(self):
        stages = StageSequence()
        before = collect_stage_defaults(stages)

        apply_stage_defaults(stages, {})

        assert collect_stage_defaults(stages) == before

    def test_partial_defaults_change_only_the_keys_present(self):
        stages = StageSequence()
        before = collect_stage_defaults(stages)

        apply_stage_defaults(stages, {"output_color_mode": ColorMode.MIXED.value})

        after = collect_stage_defaults(stages)
        assert after.pop("output_color_mode") == ColorMode.MIXED.value
        before.pop("output_color_mode")
        assert after == before

    def test_invalid_stored_value_is_rejected(self):
        with pytest.raises(ValueError):
            apply_stage_defaults(StageSequence(), {"output_color_mode": "no-such-mode"})


class TestUpstreamSettingsPersistence:
    """Trims, the top-edge deskew flag and the frozen aggregate size survive a save."""

    def test_trim_is_exported_beside_the_rotation_and_restored(
        self, pages: list[PageId]
    ):
        trim = ImageTrim(enabled=True, left=4, top=5, right=6, bottom=7)
        stages = StageSequence()
        orientation = stages.fix_orientation_filter.settings
        orientation.set_trim(pages[0].image_id, trim)
        orientation.apply_rotation(pages[0].image_id, OrthogonalRotation(degrees=90))

        exported = collect_stage_params(stages, pages)
        restored = StageSequence()
        apply_stage_params(restored, exported)

        (entry,) = exported[STAGE_KEYS[FilterStage.FIX_ORIENTATION]]
        assert entry.params["trim"]["left"] == 4
        restored_settings = restored.fix_orientation_filter.settings
        assert restored_settings.get_trim(pages[0].image_id) == trim
        assert restored_settings.get_rotation_for(pages[0].image_id).degrees == 90

    def test_a_trim_only_image_restores_without_a_rotation(self, pages: list[PageId]):
        stages = StageSequence()
        stages.fix_orientation_filter.settings.set_trim(
            pages[0].image_id, ImageTrim(enabled=True, left=3)
        )

        restored = StageSequence()
        apply_stage_params(restored, collect_stage_params(stages, pages))

        settings = restored.fix_orientation_filter.settings
        assert settings.get_trim(pages[0].image_id).left == 3
        assert not settings.is_rotation_set(pages[0].image_id)

    def test_top_edge_deskew_flag_round_trips(self):
        stages = StageSequence()
        stages.deskew_filter.content_based = False

        restored = StageSequence()
        apply_stage_defaults(restored, collect_stage_defaults(stages))

        assert restored.deskew_filter.content_based is False

    def test_a_file_without_the_flag_keeps_content_based_deskew(self):
        stages = StageSequence()

        apply_stage_defaults(stages, {})

        assert stages.deskew_filter.content_based is True

    def test_frozen_aggregate_size_round_trips(self):
        size = ContentSize(width_mm=120.0, height_mm=180.0)
        stages = StageSequence()
        stages.page_layout_filter.settings.restore_frozen_aggregate_hard_size_mm(size)

        defaults = collect_stage_defaults(stages)
        restored = StageSequence()
        apply_stage_defaults(restored, defaults)

        settings = restored.page_layout_filter.settings
        assert settings.get_frozen_aggregate_hard_size_mm() == size

    def test_an_unfrozen_aggregate_stays_unfrozen(self):
        defaults = collect_stage_defaults(StageSequence())
        restored = StageSequence()

        apply_stage_defaults(restored, defaults)

        assert defaults["page_layout_frozen_aggregate_size"] is None
        assert not restored.page_layout_filter.settings.is_aggregate_hard_size_frozen()
