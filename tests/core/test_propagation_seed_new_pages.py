"""Images added to an open project start from the default parameters."""

from pathlib import Path

from scantailor.core.default_params import (
    DefaultParams,
    OutputDefaults,
    PageLayoutDefaults,
)
from scantailor.core.models import ImageId, Margins, PageId, SubPage
from scantailor.core.propagation import seed_new_pages
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.output.color_mode import ColorMode

NEW = ImageId(file_path=Path("new.tiff"))
OLD = ImageId(file_path=Path("old.tiff"))
MARGINS = Margins(left=30.0, right=30.0, top=12.0, bottom=12.0)


def _pages(image: ImageId) -> list[PageId]:
    return [PageId(image_id=image, sub_page=sub_page) for sub_page in SubPage]


def _margin_defaults() -> DefaultParams:
    return DefaultParams(page_layout=PageLayoutDefaults(hard_margins_mm=MARGINS))


def test_a_new_page_takes_the_default_margins():
    stages = StageSequence()

    seed_new_pages(stages, [NEW], _margin_defaults())

    for page in _pages(NEW):
        assert stages.page_layout_filter.get_params(page).hard_margins_mm == MARGINS


def test_pages_split_from_the_new_image_later_are_seeded_too():
    stages = StageSequence()

    seed_new_pages(stages, [NEW], _margin_defaults())

    left = PageId(image_id=NEW, sub_page=SubPage.LEFT_PAGE)
    assert stages.page_layout_filter.settings.is_params_set(left)


def test_a_page_that_has_parameters_keeps_them():
    stages = StageSequence()
    own = Margins(left=1.0, right=2.0, top=3.0, bottom=4.0)
    page = PageId(image_id=NEW, sub_page=SubPage.SINGLE_PAGE)
    stages.page_layout_filter.settings.set_margins(page, own)

    seed_new_pages(stages, [NEW], _margin_defaults())

    assert stages.page_layout_filter.get_params(page).hard_margins_mm == own


def test_other_images_are_left_alone():
    stages = StageSequence()
    old_page = PageId(image_id=OLD, sub_page=SubPage.SINGLE_PAGE)

    seed_new_pages(stages, [NEW], _margin_defaults())

    assert not stages.page_layout_filter.settings.is_params_set(old_page)


def test_factory_defaults_leave_every_stage_without_stored_parameters():
    stages = StageSequence()

    seed_new_pages(stages, [NEW], DefaultParams())

    for page in _pages(NEW):
        assert not stages.page_layout_filter.settings.is_params_set(page)
        assert not stages.output_filter.settings.is_params_set(page)
        assert not stages.select_content_filter.settings.is_params_set(page)


def test_output_defaults_are_seeded_without_touching_the_stage_wide_values():
    stages = StageSequence()
    defaults = DefaultParams(output=OutputDefaults(color_mode=ColorMode.MIXED))
    layout_before = stages.select_content_filter.settings.get_page_detection_box()

    seed_new_pages(stages, [NEW], defaults)

    page = PageId(image_id=NEW, sub_page=SubPage.SINGLE_PAGE)
    assert stages.output_filter.get_params(page).color_mode is ColorMode.MIXED
    assert (
        stages.select_content_filter.settings.get_page_detection_box() == layout_before
    )
