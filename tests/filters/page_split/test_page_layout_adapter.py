"""Adapting a manual page layout to a new outline, like C++ PageLayoutAdapter."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId
from scantailor.core.propagation import apply_page_split
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.page_split.dependencies import Dependencies
from scantailor.filters.page_split.filter import Filter
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType
from scantailor.filters.page_split.page_layout_adapter import (
    adapt_page_layout,
    correct_page_layout_type,
)
from scantailor.filters.page_split.params import AutoManualMode, Params


def _page(name: str = "scan.tiff") -> PageId:
    return PageId(image_id=ImageId(file_path=Path(name)))


def _blank(width: int, height: int) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _stored(
    filter_: Filter,
    page: PageId,
    layout: PageLayout,
    *,
    requested: LayoutType,
    mode: AutoManualMode,
) -> None:
    """Store a layout as if it had been detected or placed on its own image."""
    _, _, width, height = layout.outline
    filter_.set_params(
        page,
        Params(
            layout_type=requested,
            page_layout=layout,
            split_line_mode=mode,
            dependencies=Dependencies(
                image_size=(int(width), int(height)), layout_type=requested
            ),
        ),
    )


class TestAdaptPageLayout:
    def test_same_outline_returns_the_layout_unchanged(self):
        layout = PageLayout.two_pages(1000, 500, split_x=420)

        assert adapt_page_layout(layout, (0, 0, 1000, 500)) is layout

    def test_split_line_is_extended_to_the_new_outline_height(self):
        layout = PageLayout.two_pages(1000, 500, split_x=420)

        adapted = adapt_page_layout(layout, (0, 0, 500, 1000))

        assert adapted.layout_type is PageLayoutType.TWO_PAGES
        assert adapted.outline == (0, 0, 500, 1000)
        assert adapted.cutter1 is not None
        (x1, y1), (x2, y2) = adapted.cutter1
        assert (x1, y1, x2, y2) == pytest.approx((420, 0, 420, 1000))

    def test_split_line_beyond_the_new_width_is_clamped_and_cuts_nothing(self):
        layout = PageLayout.two_pages(1000, 500, split_x=950)

        adapted = adapt_page_layout(layout, (0, 0, 500, 1000))

        assert adapted.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT

    def test_slanted_cutter_is_cut_off_at_the_new_top_and_bottom_edges(self):
        layout = PageLayout(
            outline=(0, 0, 1000, 500),
            layout_type=PageLayoutType.TWO_PAGES,
            cutter1=((400, 0), (500, 500)),
            cutter2=((400, 0), (500, 500)),
        )

        adapted = adapt_page_layout(layout, (0, 100, 1000, 300))

        assert adapted.cutter1 is not None
        (x1, y1), (x2, y2) = adapted.cutter1
        assert (x1, y1, x2, y2) == pytest.approx((420, 100, 480, 400))

    def test_cut_layout_keeps_both_cutters(self):
        layout = PageLayout.single_page_cut(1000, 500, left_x=100, right_x=900)

        adapted = adapt_page_layout(layout, (0, 0, 800, 400))

        assert adapted.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert adapted.cutter1 is not None
        assert adapted.cutter2 is not None
        assert adapted.cutter1[0][0] == pytest.approx(100)
        assert adapted.cutter2[0][0] == pytest.approx(800)

    def test_cutters_that_would_cross_inside_the_outline_meet_at_the_nearer_edge(self):
        layout = PageLayout(
            outline=(0, 0, 1000, 500),
            layout_type=PageLayoutType.SINGLE_PAGE_CUT,
            cutter1=((100, 0), (300, 500)),
            cutter2=((300, 0), (100, 500)),
        )

        adapted = adapt_page_layout(layout, (0, 0, 1000, 600))

        assert adapted.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert adapted.cutter1 is not None
        assert adapted.cutter2 is not None
        assert adapted.cutter1[0] == pytest.approx((200, 0))
        assert adapted.cutter2[0] == pytest.approx((200, 0))

    def test_uncut_layout_stays_uncut(self):
        layout = PageLayout.single_page_uncut(1000, 500)

        adapted = adapt_page_layout(layout, (0, 0, 800, 400))

        assert adapted == PageLayout.single_page_uncut(800, 400)

    def test_cutter_parallel_to_the_top_edge_is_left_alone(self):
        layout = PageLayout(
            outline=(0, 0, 1000, 500),
            layout_type=PageLayoutType.TWO_PAGES,
            cutter1=((0, 250), (1000, 250)),
            cutter2=((0, 250), (1000, 250)),
        )

        adapted = adapt_page_layout(layout, (0, 0, 800, 400))

        assert adapted.cutter1 == ((0, 250), (1000, 250))


class TestCorrectPageLayoutType:
    def test_two_pages_split_on_the_left_edge_becomes_single_uncut(self):
        layout = PageLayout.two_pages(1000, 500, split_x=0)

        assert (
            correct_page_layout_type(layout).layout_type
            is PageLayoutType.SINGLE_PAGE_UNCUT
        )

    def test_cut_with_both_cutters_on_the_edges_becomes_single_uncut(self):
        layout = PageLayout.single_page_cut(1000, 500, left_x=0, right_x=1000)

        assert (
            correct_page_layout_type(layout).layout_type
            is PageLayoutType.SINGLE_PAGE_UNCUT
        )

    def test_cut_with_one_cutter_on_an_edge_stays_cut(self):
        layout = PageLayout.single_page_cut(1000, 500, left_x=0, right_x=900)

        assert correct_page_layout_type(layout) is layout

    def test_two_pages_inside_the_outline_stays_two_pages(self):
        layout = PageLayout.two_pages(1000, 500, split_x=500)

        assert correct_page_layout_type(layout) is layout

    def test_cutters_are_kept_when_the_type_is_corrected(self):
        layout = PageLayout.two_pages(1000, 500, split_x=1000)

        corrected = correct_page_layout_type(layout)

        assert corrected.cutter1 == layout.cutter1


class TestFilterKeepsManualLayouts:
    def test_manual_split_survives_a_rotation_that_changes_the_image_size(self):
        filter_ = Filter()
        page = _page()
        filter_.detect_split(_blank(1000, 500), page, rotation_degrees=0)
        filter_.set_manual_split(page, split_x=420, width=1000, height=500)

        params = filter_.process(_blank(500, 1000), page, rotation_degrees=90)

        assert params.is_manual()
        assert params.page_layout is not None
        assert params.page_layout.outline == (0, 0, 500, 1000)
        assert params.page_layout.get_split_line_x() == pytest.approx(420)
        assert params.dependencies.image_size == (500, 1000)
        assert params.dependencies.rotation_degrees == 90

    def test_manual_split_survives_a_new_image_size_without_rotation(self):
        filter_ = Filter()
        page = _page()
        filter_.detect_split(_blank(1000, 500), page)
        filter_.set_manual_split(page, split_x=300, width=1000, height=500)

        params = filter_.process(_blank(800, 400), page)

        assert params.page_layout is not None
        assert params.page_layout.outline == (0, 0, 800, 400)
        assert params.page_layout.get_split_line_x() == pytest.approx(300)

    def test_auto_layout_of_an_automatic_request_is_detected_again(self):
        filter_ = Filter()
        page = _page()
        filter_.detect_split(_blank(1000, 500), page)

        params = filter_.process(_blank(800, 400), page)

        assert params.page_layout is not None
        assert params.page_layout.outline == (0, 0, 800, 400)
        assert params.is_auto()

    def test_manual_split_line_comes_back_after_switching_to_single_page_and_back(
        self,
    ):
        filter_ = Filter()
        page = _page()
        image = _blank(1000, 500)
        filter_.detect_split(image, page)
        filter_.set_manual_split(page, split_x=420, width=1000, height=500)

        filter_.retype_layout(page, LayoutType.SINGLE_PAGE_UNCUT)
        single = filter_.process(image, page)
        filter_.retype_layout(page, LayoutType.TWO_PAGES)
        back = filter_.process(image, page)

        assert single.page_layout is not None
        assert single.page_layout.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT
        assert single.is_manual()
        assert back.page_layout is not None
        assert back.page_layout.layout_type is PageLayoutType.TWO_PAGES
        assert back.page_layout.get_split_line_x() == pytest.approx(420)
        assert back.is_manual()

    def test_manual_split_changed_to_page_plus_offcut_is_dropped_for_detection(
        self,
    ):
        filter_ = Filter()
        page = _page()
        image = _blank(1000, 500)
        filter_.detect_split(image, page)
        filter_.set_manual_split(page, split_x=420, width=1000, height=500)

        filter_.set_layout_type(page, LayoutType.PAGE_PLUS_OFFCUT)
        params = filter_.process(image, page)

        assert params.layout_type is LayoutType.PAGE_PLUS_OFFCUT
        assert params.is_auto()
        assert params.page_layout is not None
        assert params.page_layout.get_split_line_x() is None

    def test_auto_detected_two_pages_kept_when_two_pages_is_requested(self):
        filter_ = Filter()
        page = _page()
        detected = PageLayout.two_pages(600, 400, split_x=280)
        _stored(
            filter_,
            page,
            detected,
            requested=LayoutType.AUTO_LAYOUT_TYPE,
            mode=AutoManualMode.AUTO,
        )

        filter_.set_layout_type(page, LayoutType.TWO_PAGES)
        params = filter_.process(_blank(600, 400), page)

        assert params.layout_type is LayoutType.TWO_PAGES
        assert params.page_layout == detected
        assert params.is_auto()
        assert params.dependencies.layout_type is LayoutType.TWO_PAGES

    def test_auto_detected_two_pages_is_not_detected_again_for_a_resized_image(self):
        filter_ = Filter()
        page = _page()
        _stored(
            filter_,
            page,
            PageLayout.two_pages(600, 400, split_x=280),
            requested=LayoutType.AUTO_LAYOUT_TYPE,
            mode=AutoManualMode.AUTO,
        )
        filter_.set_layout_type(page, LayoutType.TWO_PAGES)

        params = filter_.process(_blank(1200, 800), page)

        assert params.page_layout is not None
        assert params.page_layout.outline == (0, 0, 1200, 800)
        assert params.page_layout.get_split_line_x() == pytest.approx(280)

    def test_auto_detected_single_page_asked_to_be_two_pages_is_detected(self):
        filter_ = Filter()
        page = _page()
        _stored(
            filter_,
            page,
            PageLayout.single_page_uncut(600, 400),
            requested=LayoutType.AUTO_LAYOUT_TYPE,
            mode=AutoManualMode.AUTO,
        )

        filter_.set_layout_type(page, LayoutType.TWO_PAGES)
        params = filter_.process(_blank(600, 400), page)

        assert params.page_layout is not None
        assert params.page_layout.layout_type is PageLayoutType.TWO_PAGES

    def test_cutters_on_the_edges_are_corrected_to_single_uncut_when_processed(self):
        filter_ = Filter()
        page = _page()
        layout = PageLayout.two_pages(600, 400, split_x=0)
        _stored(
            filter_,
            page,
            layout,
            requested=LayoutType.TWO_PAGES,
            mode=AutoManualMode.MANUAL,
        )

        params = filter_.process(_blank(600, 400), page)

        assert params.layout_type is LayoutType.SINGLE_PAGE_UNCUT
        assert params.page_layout is not None
        assert params.page_layout.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT
        assert params.is_manual()

    def test_both_halves_of_an_image_share_one_record(self):
        from scantailor.core.models import SubPage

        filter_ = Filter()
        left = PageId(
            image_id=ImageId(file_path=Path("a.tiff")), sub_page=SubPage.LEFT_PAGE
        )
        right = PageId(image_id=left.image_id, sub_page=SubPage.RIGHT_PAGE)

        filter_.set_layout_type(left, LayoutType.TWO_PAGES)

        assert filter_.get_layout_type(right) is LayoutType.TWO_PAGES


class TestApplyLayoutType:
    def _sequence_with_targets(self) -> tuple[StageSequence, PageId, PageId]:
        stages = StageSequence()
        split = stages.page_split_filter
        source, target = _page("source.tiff"), _page("target.tiff")
        _stored(
            split,
            source,
            PageLayout.two_pages(1000, 500, split_x=450),
            requested=LayoutType.TWO_PAGES,
            mode=AutoManualMode.MANUAL,
        )
        _stored(
            split,
            target,
            PageLayout.single_page_uncut(600, 400),
            requested=LayoutType.SINGLE_PAGE_UNCUT,
            mode=AutoManualMode.AUTO,
        )
        return stages, source, target

    def test_manual_split_is_adapted_to_a_differently_sized_target(self):
        stages, source, target = self._sequence_with_targets()

        apply_page_split(stages, source, [target], LayoutType.TWO_PAGES, apply_cut=True)

        params = stages.page_split_filter.get_params(target)
        assert params.layout_type is LayoutType.TWO_PAGES
        assert params.page_layout is not None
        assert params.page_layout.outline == (0, 0, 600, 400)
        assert params.page_layout.cutter1 is not None
        (x1, y1), (x2, y2) = params.page_layout.cutter1
        assert (x1, y1, x2, y2) == pytest.approx((450, 0, 450, 400))
        assert params.is_manual()

    def test_split_beyond_the_target_width_cuts_nothing(self):
        stages, source, target = self._sequence_with_targets()
        split = stages.page_split_filter
        _stored(
            split,
            source,
            PageLayout.two_pages(1000, 500, split_x=900),
            requested=LayoutType.TWO_PAGES,
            mode=AutoManualMode.MANUAL,
        )

        apply_page_split(stages, source, [target], LayoutType.TWO_PAGES, apply_cut=True)

        params = split.get_params(target)
        assert params.layout_type is LayoutType.SINGLE_PAGE_UNCUT
        assert params.page_layout is not None
        assert params.page_layout.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT

    def test_adapted_layout_is_kept_when_the_target_is_processed(self):
        stages, source, target = self._sequence_with_targets()
        apply_page_split(stages, source, [target], LayoutType.TWO_PAGES, apply_cut=True)

        params = stages.page_split_filter.process(_blank(600, 400), target)

        assert params.page_layout is not None
        assert params.page_layout.get_split_line_x() == pytest.approx(450)

    def test_without_apply_cut_the_target_only_gets_the_type(self):
        stages, source, target = self._sequence_with_targets()

        apply_page_split(
            stages, source, [target], LayoutType.TWO_PAGES, apply_cut=False
        )

        params = stages.page_split_filter.get_params(target)
        assert params.layout_type is LayoutType.TWO_PAGES
        assert params.page_layout is None

    def test_target_without_a_layout_receives_the_source_layout_as_it_is(self):
        stages, source, _ = self._sequence_with_targets()
        fresh = _page("fresh.tiff")

        apply_page_split(stages, source, [fresh], LayoutType.TWO_PAGES, apply_cut=True)

        params = stages.page_split_filter.get_params(fresh)
        assert params.page_layout == PageLayout.two_pages(1000, 500, split_x=450)
        assert params.is_manual()

    def test_automatic_layout_type_makes_targets_detect_their_own(self):
        stages, source, target = self._sequence_with_targets()

        apply_page_split(
            stages, source, [target], LayoutType.AUTO_LAYOUT_TYPE, apply_cut=True
        )
        params = stages.page_split_filter.process(_blank(600, 400), target)

        assert params.layout_type is LayoutType.AUTO_LAYOUT_TYPE
        assert params.dependencies.layout_type is LayoutType.AUTO_LAYOUT_TYPE
        assert params.is_auto()

    def test_source_layout_of_another_type_than_requested_is_not_copied(self):
        stages, source, _ = self._sequence_with_targets()
        split = stages.page_split_filter
        _stored(
            split,
            source,
            PageLayout.single_page_cut(1000, 500, left_x=100, right_x=900),
            requested=LayoutType.PAGE_PLUS_OFFCUT,
            mode=AutoManualMode.MANUAL,
        )
        fresh = _page("fresh.tiff")

        apply_page_split(stages, source, [fresh], LayoutType.TWO_PAGES, apply_cut=True)

        assert split.get_params(fresh).page_layout is None
