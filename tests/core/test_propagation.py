"""Page selection, propagators and "Apply to..." copying."""

from pathlib import Path

import pytest

from scantailor.core.models import (
    Dpi,
    ImageId,
    Margins,
    OrthogonalRotation,
    PageId,
    SubPage,
)
from scantailor.core.project import ImageMetadata, Project
from scantailor.core.propagation import (
    ApplyScope,
    advise_logical_page_count,
    apply_alignment,
    apply_deskew,
    apply_margins,
    apply_orientation,
    apply_output,
    apply_select_content,
    propagate_content_boxes,
    propagate_page_orientations,
    select_pages,
)
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.params import ApplyGroup
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.page_layout.alignment import (
    Alignment,
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_layout.params import ContentRect, ContentSize
from scantailor.filters.page_layout.params import Params as LayoutParams
from scantailor.filters.page_layout.settings import Settings as LayoutSettings
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.params import Params as SplitParams
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.filters.select_content.params import Params as SelectContentParams

DPI = Dpi.uniform(300)


def _page(name: str, sub: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(f"/scans/{name}.png")), sub_page=sub)


def _pages(count: int) -> list[PageId]:
    return [_page(f"p{index}") for index in range(count)]


def _project(*sizes: tuple[int, int]) -> Project:
    project = Project()
    for index, (width, height) in enumerate(sizes):
        project.add_image(
            ImageId(file_path=Path(f"/scans/p{index}.png")),
            ImageMetadata(width=width, height=height, dpi=DPI),
        )
    return project


class TestSelectPages:
    """The seven scopes of the apply dialog."""

    def test_this_page_is_only_the_current_page(self):
        pages = _pages(4)

        assert select_pages(ApplyScope.THIS_PAGE, pages[2], pages) == [pages[2]]

    def test_all_pages_keeps_display_order(self):
        pages = _pages(4)

        assert select_pages(ApplyScope.ALL_PAGES, pages[2], pages) == pages

    def test_followers_start_at_the_current_page(self):
        pages = _pages(5)

        result = select_pages(ApplyScope.THIS_PAGE_AND_FOLLOWERS, pages[2], pages)

        assert result == pages[2:]

    def test_this_page_every_other_skips_alternate_followers(self):
        pages = _pages(6)

        result = select_pages(ApplyScope.THIS_PAGE_EVERY_OTHER, pages[1], pages)

        assert result == [pages[1], pages[3], pages[5]]

    @pytest.mark.parametrize(("current", "expected"), [(1, [1, 3, 5]), (2, [0, 2, 4])])
    def test_every_other_page_shares_the_parity_of_the_current_page(
        self, current: int, expected: list[int]
    ):
        pages = _pages(6)

        result = select_pages(ApplyScope.EVERY_OTHER, pages[current], pages)

        assert result == [pages[i] for i in expected]

    def test_selected_pages_are_returned_as_selected(self):
        pages = _pages(6)

        result = select_pages(
            ApplyScope.SELECTED_PAGES, pages[0], pages, [pages[1], pages[4]]
        )

        assert result == [pages[1], pages[4]]

    def test_every_other_selected_includes_the_current_page(self):
        pages = _pages(8)
        selected = [pages[1], pages[2], pages[4], pages[5], pages[7]]

        result = select_pages(
            ApplyScope.EVERY_OTHER_SELECTED, pages[2], pages, selected
        )

        assert result == [pages[2], pages[5]]

    def test_every_other_selected_starts_at_the_first_when_current_is_not_selected(
        self,
    ):
        pages = _pages(6)
        selected = [pages[1], pages[3], pages[5]]

        result = select_pages(
            ApplyScope.EVERY_OTHER_SELECTED, pages[0], pages, selected
        )

        assert result == [pages[1], pages[5]]

    def test_unknown_current_page_falls_back_to_itself(self):
        pages = _pages(3)
        stranger = _page("other")

        assert select_pages(ApplyScope.THIS_PAGE_AND_FOLLOWERS, stranger, pages) == [
            stranger
        ]


class TestAdviseLogicalPageCount:
    """A physically landscape image is a two-page spread."""

    def test_landscape_image_is_a_spread(self):
        metadata = ImageMetadata(width=2000, height=1000, dpi=DPI)

        assert advise_logical_page_count(metadata, OrthogonalRotation(degrees=0)) == 2

    def test_portrait_image_is_one_page(self):
        metadata = ImageMetadata(width=1000, height=2000, dpi=DPI)

        assert advise_logical_page_count(metadata, OrthogonalRotation(degrees=0)) == 1

    def test_rotating_a_landscape_image_by_a_quarter_turn_gives_one_page(self):
        metadata = ImageMetadata(width=2000, height=1000, dpi=DPI)

        assert advise_logical_page_count(metadata, OrthogonalRotation(degrees=90)) == 1

    def test_resolution_decides_when_the_pixel_sizes_are_equal(self):
        metadata = ImageMetadata(
            width=1000, height=1000, dpi=Dpi(horizontal=150, vertical=300)
        )

        assert advise_logical_page_count(metadata, OrthogonalRotation(degrees=0)) == 2


class TestPropagatePageOrientations:
    """Orientation decides how many logical pages an image has."""

    def test_rotating_a_spread_to_portrait_leaves_one_page(self):
        project = _project((2000, 1000))
        project.images[0].num_sub_pages = 2
        stages = StageSequence()
        image_id = project.images[0].id
        stages.fix_orientation_filter.set_rotation(
            image_id, OrthogonalRotation(degrees=90)
        )

        changed = propagate_page_orientations(project, stages)

        assert changed == [image_id]
        assert project.images[0].num_sub_pages == 1

    def test_unrotated_spread_gets_two_pages(self):
        project = _project((2000, 1000))

        propagate_page_orientations(project, StageSequence())

        assert len(project.get_pages()) == 2

    def test_unchanged_page_count_reports_nothing(self):
        project = _project((1000, 2000))

        assert propagate_page_orientations(project, StageSequence()) == []

    def test_manually_chosen_layout_type_is_kept(self):
        project = _project((2000, 1000))
        stages = StageSequence()
        page = project.get_pages()[0]
        stages.page_split_filter.settings.set_params(
            page, SplitParams(layout_type=LayoutType.SINGLE_PAGE_UNCUT)
        )

        changed = propagate_page_orientations(project, stages)

        assert changed == []
        assert project.images[0].num_sub_pages == 1

    def test_single_removed_half_keeps_a_single_page(self):
        project = _project((2000, 1000))
        project.images[0].left_half_removed = True

        propagate_page_orientations(project, StageSequence())

        assert project.images[0].num_sub_pages == 1

    def test_automatic_split_is_dropped_when_the_page_count_changes(self):
        project = _project((2000, 1000))
        stages = StageSequence()
        page = project.get_pages()[0]
        stages.page_split_filter.settings.set_params(page, SplitParams())

        propagate_page_orientations(project, stages)

        assert not stages.page_split_filter.settings.is_params_set(page)


def _select_content_params() -> SelectContentParams:
    return SelectContentParams(
        content_box=ContentBox(x=100, y=100, width=600, height=800),
        page_box=PageBox(x=0, y=0, width=800, height=1000),
    )


class TestPropagateContentBoxes:
    """Content sizes reach the page layout without detection or pixels."""

    def test_measured_page_records_its_content_size_for_the_layout(self):
        project = _project((800, 1000))
        stages = StageSequence()
        page = project.get_pages()[0]
        stages.select_content_filter.set_params(page, _select_content_params())

        propagate_content_boxes(project, stages)

        params = stages.page_layout_filter.get_params(page)
        assert params.content_rect.width == pytest.approx(600)
        assert params.content_size_mm.width_mm == pytest.approx(600 / 300 * 25.4)
        assert params.content_size_mm.height_mm == pytest.approx(800 / 300 * 25.4)

    def test_unmeasured_page_loses_its_recorded_geometry_but_keeps_its_margins(self):
        project = _project((800, 1000))
        stages = StageSequence()
        page = project.get_pages()[0]
        settings = stages.page_layout_filter.settings
        settings.set_params(
            page,
            LayoutParams(
                hard_margins_mm=Margins.uniform(7.0),
                page_rect=ContentRect(width=800, height=1000),
                content_size_mm=ContentSize(width_mm=50, height_mm=60),
            ),
        )

        propagate_content_boxes(project, stages)

        params = settings.get_params(page)
        assert params.content_size_mm.is_empty()
        assert params.page_rect.is_empty()
        assert params.hard_margins_mm == Margins.uniform(7.0)

    def test_every_page_is_defined_once_all_are_measured(self):
        project = _project((800, 1000), (800, 1000))
        stages = StageSequence()
        for page in project.get_pages():
            stages.select_content_filter.set_params(page, _select_content_params())

        propagate_content_boxes(project, stages)

        assert stages.page_layout_filter.settings.check_everything_defined(
            project.get_pages()
        )

    def test_a_page_that_was_never_measured_leaves_the_layout_undefined(self):
        project = _project((800, 1000), (800, 1000))
        stages = StageSequence()
        measured = project.get_pages()[0]
        stages.select_content_filter.set_params(measured, _select_content_params())

        propagate_content_boxes(project, stages)

        assert not stages.page_layout_filter.settings.check_everything_defined(
            project.get_pages()
        )

    def test_detection_is_not_run(self):
        project = _project((800, 1000))
        stages = StageSequence()

        propagate_content_boxes(project, stages)

        page = project.get_pages()[0]
        assert not stages.deskew_filter.settings.is_params_set(page)
        assert not stages.page_split_filter.settings.is_params_set(page)


class TestCheckEverythingDefined:
    """The output readiness rule of the page layout settings."""

    def _measured(self) -> LayoutParams:
        return LayoutParams(content_size_mm=ContentSize(width_mm=10, height_mm=10))

    def test_no_pages_is_defined(self):
        assert LayoutSettings().check_everything_defined([])

    def test_page_without_params_is_undefined(self):
        assert not LayoutSettings().check_everything_defined(_pages(1))

    def test_margins_alone_do_not_define_a_page(self):
        settings = LayoutSettings()
        page = _pages(1)[0]
        settings.set_margins(page, Margins.uniform(3.0))

        assert not settings.check_everything_defined([page])

    def test_measured_pages_are_defined(self):
        settings = LayoutSettings()
        pages = _pages(2)
        for page in pages:
            settings.set_params(page, self._measured())

        assert settings.check_everything_defined(pages)

    def test_ignored_page_does_not_count(self):
        settings = LayoutSettings()
        measured, other = _pages(2)
        settings.set_params(measured, self._measured())

        assert not settings.check_everything_defined([measured, other])
        assert settings.check_everything_defined([measured, other], ignore=other)


class TestApplyFunctions:
    """Copying one page's settings to others."""

    def test_orientation_reaches_the_images_of_the_targets(self):
        stages = StageSequence()
        source, *targets = _pages(3)
        stages.fix_orientation_filter.set_rotation(
            source.image_id, OrthogonalRotation(degrees=270)
        )

        apply_orientation(stages, source, targets)

        for target in targets:
            assert stages.fix_orientation_filter.get_rotation(
                target.image_id
            ) == OrthogonalRotation(degrees=270)

    def test_manual_deskew_angle_is_copied(self):
        stages = StageSequence()
        source, *targets = _pages(3)
        stages.deskew_filter.set_params(source, DeskewParams().with_manual_angle(2.5))

        apply_deskew(stages, source, targets)

        for target in targets:
            params = stages.deskew_filter.get_params(target)
            assert params.deskew_angle_deg == pytest.approx(2.5)
            assert params.is_manual()

    def test_automatic_deskew_makes_targets_detect_again(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.deskew_filter.set_params(source, DeskewParams().with_auto_angle(1.0))
        stages.deskew_filter.set_params(target, DeskewParams().with_manual_angle(4.0))

        apply_deskew(stages, source, [target])

        assert not stages.deskew_filter.settings.is_params_set(target)

    def test_select_content_copies_boxes_and_modes(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.select_content_filter.set_params(
            source,
            _select_content_params().with_manual_content(
                ContentBox(x=1, y=2, width=3, height=4)
            ),
        )

        apply_select_content(stages, source, [target])

        params = stages.select_content_filter.get_params(target)
        assert params.is_content_manual()
        assert params.content_box == ContentBox(x=1, y=2, width=3, height=4)

    def test_select_content_keeps_the_old_manual_box_when_the_option_is_off(self):
        stages = StageSequence()
        source, target = _pages(2)
        old_box = ContentBox(x=9, y=9, width=9, height=9)
        stages.select_content_filter.set_params(
            source,
            _select_content_params().with_manual_content(
                ContentBox(x=1, y=2, width=3, height=4)
            ),
        )
        stages.select_content_filter.set_params(
            target, SelectContentParams(content_box=old_box)
        )

        apply_select_content(stages, source, [target], apply_content_box=False)

        params = stages.select_content_filter.get_params(target)
        assert params.content_box == old_box
        assert params.is_content_manual()

    def test_select_content_skips_the_source_page(self):
        stages = StageSequence()
        source, target = _pages(2)
        stages.select_content_filter.set_params(source, _select_content_params())

        apply_select_content(stages, source, [source, target])

        assert (
            stages.select_content_filter.get_params(source) == _select_content_params()
        )

    def test_margins_are_copied_without_touching_alignment(self):
        stages = StageSequence()
        source, target = _pages(2)
        settings = stages.page_layout_filter.settings
        settings.set_margins(source, Margins(left=1, top=2, right=3, bottom=4))
        settings.set_alignment(
            target,
            Alignment(
                vertical=VerticalAlignment.TOP, horizontal=HorizontalAlignment.LEFT
            ),
        )

        apply_margins(stages, source, [target])

        params = settings.get_params(target)
        assert params.hard_margins_mm == Margins(left=1, top=2, right=3, bottom=4)
        assert params.alignment.vertical is VerticalAlignment.TOP

    def test_margins_that_follow_the_content_stay_automatic(self):
        stages = StageSequence()
        source, target = _pages(2)
        settings = stages.page_layout_filter.settings
        settings.set_params(source, LayoutParams().with_auto_margins())

        apply_margins(stages, source, [target])

        assert settings.get_params(target).auto_margins

    def test_automatic_margins_leave_the_targets_own_margins_and_need_measuring_again(
        self,
    ):
        stages = StageSequence()
        source, target = _pages(2)
        settings = stages.page_layout_filter.settings
        settings.set_params(source, LayoutParams().with_auto_margins())
        settings.set_params(
            target,
            LayoutParams(
                hard_margins_mm=Margins.uniform(7.0),
                page_rect=ContentRect(x=0, y=0, width=100, height=100),
                content_rect=ContentRect(x=10, y=10, width=80, height=80),
                content_size_mm=ContentSize(width_mm=20, height_mm=20),
            ),
        )

        apply_margins(stages, source, [source, target])

        params = settings.get_params(target)
        assert params.auto_margins
        assert params.hard_margins_mm == Margins.uniform(7.0)
        assert params.content_size_mm.is_empty()
        assert not settings.check_everything_defined([target])

    def test_alignment_is_copied_without_touching_margins(self):
        stages = StageSequence()
        source, target = _pages(2)
        settings = stages.page_layout_filter.settings
        alignment = Alignment(
            vertical=VerticalAlignment.BOTTOM, horizontal=HorizontalAlignment.RIGHT
        )
        settings.set_alignment(source, alignment)
        settings.set_margins(target, Margins.uniform(8.0))

        apply_alignment(stages, source, [target])

        params = settings.get_params(target)
        assert params.alignment == alignment
        assert params.hard_margins_mm == Margins.uniform(8.0)

    def test_output_parameters_are_copied(self):
        stages = StageSequence()
        source, target = _pages(2)
        params = OutputParams(color_mode=ColorMode.COLOR_GRAYSCALE)
        stages.output_filter.set_params(source, params)

        apply_output(stages, source, [target], ApplyGroup.COLOR_MODE)

        assert stages.output_filter.get_params(target).color_mode == (
            ColorMode.COLOR_GRAYSCALE
        )
