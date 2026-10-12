"""Tests for seeding a new project's stages from a default-params profile."""

from pathlib import Path

from scantailor.core.default_params import (
    DefaultParams,
    DeskewDefaults,
    OutputDefaults,
    apply_default_params,
)
from scantailor.core.models import ImageId, OrthogonalRotation, PageId, SubPage
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.deskew.params import AutoManualMode
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.detection import ContentDetectionMode

IMAGE = ImageId(file_path=Path("/scans/a.png"))


def _page(sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    return PageId(image_id=IMAGE, sub_page=sub_page)


class TestApplyDefaultParams:
    def test_factory_profile_leaves_every_stage_unseeded(self) -> None:
        stages = StageSequence()

        apply_default_params(stages, [IMAGE], DefaultParams())

        assert not stages.deskew_filter.settings.is_params_set(_page())
        assert not stages.select_content_filter.settings.is_params_set(_page())
        assert not stages.page_layout_filter.settings.is_params_set(_page())
        assert not stages.output_filter.settings.is_params_set(_page())

    def test_source_profile_seeds_the_pages_a_split_creates_later(self) -> None:
        stages = StageSequence()

        apply_default_params(stages, [IMAGE], DefaultParams.source())

        for sub_page in SubPage:
            page = _page(sub_page)
            assert (
                stages.deskew_filter.settings.get_params(page).mode
                is AutoManualMode.MANUAL
            )
            assert (
                stages.select_content_filter.settings.get_params(
                    page
                ).content_detection_mode
                is ContentDetectionMode.DISABLED
            )
            assert stages.page_layout_filter.settings.get_params(page).alignment.is_null
            assert (
                stages.output_filter.settings.get_params(page).color_mode
                is ColorMode.COLOR_GRAYSCALE
            )
        assert (
            stages.page_split_filter.settings.get_default_layout_type()
            is LayoutType.SINGLE_PAGE_UNCUT
        )

    def test_automatic_deskew_is_not_seeded_so_it_still_detects(self) -> None:
        stages = StageSequence()
        defaults = DefaultParams(deskew=DeskewDefaults(deskew_angle_deg=2.0))

        apply_default_params(stages, [IMAGE], defaults)

        assert not stages.deskew_filter.settings.is_params_set(_page())

    def test_custom_output_and_rotation_reach_every_image(self) -> None:
        other = ImageId(file_path=Path("/scans/b.png"))
        stages = StageSequence()
        defaults = DefaultParams(output=OutputDefaults(color_mode=ColorMode.MIXED))
        defaults.fix_orientation.image_rotation = OrthogonalRotation(degrees=90)

        apply_default_params(stages, [IMAGE, other], defaults)

        for image in (IMAGE, other):
            assert (
                stages.output_filter.settings.get_params(
                    PageId(image_id=image)
                ).color_mode
                is ColorMode.MIXED
            )
            assert (
                stages.fix_orientation_filter.settings.get_rotation_for(image).degrees
                == 90
            )
