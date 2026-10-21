"""Detection through the page split Filter, as the batch pipeline and users call it."""

from pathlib import Path
from typing import Any

import pytest

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.page_split import filter as split_filter_module
from scantailor.filters.page_split.filter import Filter
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType
from scantailor.filters.page_split.settings import Settings
from scantailor.filters.page_split.split_finder import SplitResult

from . import scan_pages as pages


def _page_id() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("/scan.tiff")), sub_page=SubPage.SINGLE_PAGE
    )


class TestSinglePageWhoseInsideEdgeWasCutOff:
    """A scanner dropped the inside edge, so the text reaches the left border."""

    def test_page_plus_offcut_by_default_gives_a_cut_layout_not_an_uncut_one(
        self,
    ) -> None:
        settings = Settings()
        settings.set_default_layout_type(LayoutType.PAGE_PLUS_OFFCUT)
        page_filter = Filter(settings)
        page_id = _page_id()

        result = page_filter.detect_split(
            pages.single_page_missing_inside_edge(), page_id
        )

        assert result.layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        stored = page_filter.get_page_layout(page_id)
        assert stored is not None
        assert stored.layout_type is PageLayoutType.SINGLE_PAGE_CUT

    def test_the_cut_page_spans_the_image_when_no_edge_can_be_seen(self) -> None:
        page_filter = Filter()
        page_id = _page_id()
        page_filter.set_layout_type(page_id, LayoutType.PAGE_PLUS_OFFCUT)

        layout = page_filter.detect_split(
            pages.single_page_missing_inside_edge(), page_id
        ).layout

        outline = layout.get_single_page_outline()
        assert outline is not None
        assert outline[:, 0].min() < 1.0
        assert outline[:, 0].max() > 999.0

    def test_a_dark_edge_strip_is_cut_at_when_the_layout_is_automatic(self) -> None:
        page_filter = Filter()
        page_id = _page_id()

        layout = page_filter.detect_split(
            pages.single_page_with_dark_gutter_strip(), page_id
        ).layout

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter1 is not None
        assert 40 < 0.5 * (layout.cutter1[0][0] + layout.cutter1[1][0]) < 90


def test_a_two_page_spread_is_split_in_the_gutter_through_the_pipeline_entry() -> None:
    page_filter = Filter()
    page_id = _page_id()

    params = page_filter.process(pages.two_page_spread(), page_id, dpi=300.0)

    assert params.page_layout is not None
    assert params.page_layout.layout_type is PageLayoutType.TWO_PAGES
    assert params.page_layout.get_split_line_x() == 750.0
    assert params.is_auto()


class TestLayoutTypeChosenWhileProcessing:
    """Processing runs on worker threads; Apply To can land while it runs."""

    def test_a_detection_finishing_after_apply_to_keeps_the_applied_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page_filter = Filter()
        page_id = _page_id()
        detect = split_filter_module.detect_split

        def apply_to_during_detection(*args: Any) -> SplitResult:
            page_filter.apply_layout_type(
                page_id, [page_id], LayoutType.TWO_PAGES, apply_cut=False
            )
            return detect(*args)

        monkeypatch.setattr(
            split_filter_module, "detect_split", apply_to_during_detection
        )

        page_filter.process(pages.single_page_with_dark_gutter_strip(), page_id)

        assert page_filter.get_layout_type(page_id) is LayoutType.TWO_PAGES
        assert page_filter.get_page_layout(page_id) is None

    def test_an_adaptation_finishing_after_apply_to_keeps_the_applied_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page_filter = Filter()
        page_id = _page_id()
        page_filter.set_manual_split(page_id, 300.0, 600.0, 800.0)
        adapt = split_filter_module.adapt_page_layout

        def apply_to_during_adaptation(*args: Any) -> PageLayout:
            page_filter.apply_layout_type(
                page_id, [page_id], LayoutType.SINGLE_PAGE_UNCUT, apply_cut=False
            )
            return adapt(*args)

        monkeypatch.setattr(
            split_filter_module, "adapt_page_layout", apply_to_during_adaptation
        )

        page_filter.process(pages.two_page_spread(), page_id)

        assert page_filter.get_layout_type(page_id) is LayoutType.SINGLE_PAGE_UNCUT
        assert page_filter.get_page_layout(page_id) is None
