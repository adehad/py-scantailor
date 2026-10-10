"""Tests for exporting and restoring per-page stage parameters."""

from pathlib import Path

import pytest

from scantailor.core.models import Dpi, ImageId, Margins, OrthogonalRotation, PageId
from scantailor.core.project import Project
from scantailor.core.stage_params import (
    STAGE_KEYS,
    apply_stage_params,
    collect_stage_params,
)
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.page_layout.params import Params as PageLayoutParams
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.params import Params as PageSplitParams
from scantailor.filters.select_content.detection import ContentDetectionMode
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


def test_stage_keys_cover_all_stages_with_unique_names():
    assert set(STAGE_KEYS) == set(FilterStage)
    assert len(set(STAGE_KEYS.values())) == len(FilterStage)
