"""Apply Margins: per-side copying and the result for the current page."""

from pathlib import Path

from scantailor.core.models import ImageId, Margins, PageId, SubPage
from scantailor.core.propagation import ApplyScope, apply_margins, select_pages
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.page_layout.params import ContentRect, ContentSize, Params


def _pages(count: int) -> list[PageId]:
    return [
        PageId(
            image_id=ImageId(file_path=Path(f"/{index}.tiff")),
            sub_page=SubPage.SINGLE_PAGE,
        )
        for index in range(count)
    ]


def _margins(stages: StageSequence, page: PageId) -> Margins:
    return stages.page_layout_filter.get_margins(page)


class TestPerSideMargins:
    def test_only_the_chosen_sides_are_copied(self) -> None:
        stages = StageSequence()
        source, target = _pages(2)
        stages.page_layout_filter.set_margins(
            source, Margins(top=11, bottom=12, left=13, right=14)
        )
        stages.page_layout_filter.set_margins(
            target, Margins(top=1, bottom=2, left=3, right=4)
        )

        apply_margins(stages, source, [target], {"top", "right"})

        assert _margins(stages, target) == Margins(top=11, bottom=2, left=3, right=14)

    def test_every_side_is_copied_unless_told_otherwise(self) -> None:
        stages = StageSequence()
        source, target = _pages(2)
        stages.page_layout_filter.set_margins(source, Margins.uniform(9.0))

        apply_margins(stages, source, [target])

        assert _margins(stages, target) == Margins.uniform(9.0)

    def test_each_target_keeps_its_own_unchosen_sides(self) -> None:
        stages = StageSequence()
        source, first, second = _pages(3)
        filter_ = stages.page_layout_filter
        filter_.set_margins(source, Margins.uniform(50.0))
        filter_.set_margins(first, Margins(top=1, bottom=1, left=1, right=1))
        filter_.set_margins(second, Margins(top=2, bottom=2, left=2, right=2))

        apply_margins(stages, source, [first, second], {"left"})

        assert _margins(stages, first) == Margins(top=1, bottom=1, left=50, right=1)
        assert _margins(stages, second) == Margins(top=2, bottom=2, left=50, right=2)

    def test_automatic_margins_ignore_the_chosen_sides(self) -> None:
        stages = StageSequence()
        source, target = _pages(2)
        settings = stages.page_layout_filter.settings
        settings.set_params(source, Params().with_auto_margins())
        settings.set_margins(target, Margins.uniform(7.0))

        apply_margins(stages, source, [target], {"top"})

        params = settings.get_params(target)
        assert params.auto_margins
        assert params.hard_margins_mm == Margins.uniform(7.0)


class TestCurrentPageAmongTheTargets:
    """Upstream a00c0282 stopped skipping the current page in Apply Margins."""

    def test_current_page_keeps_its_margins_when_it_is_a_target(self) -> None:
        stages = StageSequence()
        source, other = _pages(2)
        stages.page_layout_filter.set_margins(
            source, Margins(top=1, bottom=2, left=3, right=4)
        )

        apply_margins(stages, source, [source, other], {"top", "left"})

        assert _margins(stages, source) == Margins(top=1, bottom=2, left=3, right=4)
        assert _margins(stages, other) == Margins(top=1, bottom=5, left=3, right=10)

    def test_selection_that_includes_the_current_page_reaches_the_others(self) -> None:
        stages = StageSequence()
        pages = _pages(4)
        stages.page_layout_filter.set_margins(pages[1], Margins.uniform(8.0))
        selection = select_pages(ApplyScope.SELECTED_PAGES, pages[1], pages, pages[1:3])

        apply_margins(stages, pages[1], selection)

        assert pages[1] in selection
        assert _margins(stages, pages[1]) == Margins.uniform(8.0)
        assert _margins(stages, pages[2]) == Margins.uniform(8.0)
        assert _margins(stages, pages[0]) == Margins(
            top=5.0, bottom=5.0, left=10.0, right=10.0
        )

    def test_automatic_margins_of_the_current_page_stay_measured(self) -> None:
        stages = StageSequence()
        source, other = _pages(2)
        settings = stages.page_layout_filter.settings
        measured = Params(
            content_rect=ContentRect(x=0, y=0, width=10, height=10),
            page_rect=ContentRect(x=0, y=0, width=20, height=20),
            content_size_mm=ContentSize(width_mm=5, height_mm=5),
        ).with_auto_margins()
        settings.set_params(source, measured)

        apply_margins(stages, source, [source, other])

        assert settings.get_params(source) == measured
        assert settings.get_params(other).auto_margins
        assert settings.get_params(other).content_size_mm.is_empty()
