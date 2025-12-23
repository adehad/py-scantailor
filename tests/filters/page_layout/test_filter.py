"""Tests for Page Layout filter."""

from __future__ import annotations

from pathlib import Path

from scantailor.core import ImageId, Margins, PageId, SubPage
from scantailor.filters.page_layout import Alignment, Filter, Params, Settings
from scantailor.filters.page_layout.params import ContentSize


def make_page_id(name: str, sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    """Helper to create PageId instances."""
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}.tiff")),
        sub_page=sub_page,
    )


class TestFilter:
    """Tests for Filter class."""

    def test_filter_name(self):
        """Filter should have correct name."""
        filter = Filter()
        assert filter.name == "Page Layout"

    def test_creates_default_settings(self):
        """Filter should create settings if not provided."""
        filter = Filter()
        assert filter.settings is not None

    def test_uses_provided_settings(self):
        """Filter should use provided settings."""
        settings = Settings()
        filter = Filter(settings)
        assert filter.settings is settings

    def test_get_params_default(self):
        """get_params should return default params."""
        filter = Filter()
        page_id = make_page_id("test")

        params = filter.get_params(page_id)

        assert params.hard_margins_mm.top == 0.0
        assert params.alignment.aligned_with_others() is True

    def test_get_margins(self):
        """get_margins should return the margins."""
        filter = Filter()
        page_id = make_page_id("test")
        margins = Margins.uniform(10)
        filter.set_margins(page_id, margins)

        result = filter.get_margins(page_id)

        assert result == margins

    def test_get_alignment(self):
        """get_alignment should return the alignment."""
        filter = Filter()
        page_id = make_page_id("test")
        alignment = Alignment.top_left()
        filter.set_alignment(page_id, alignment)

        result = filter.get_alignment(page_id)

        assert result == alignment

    def test_set_uniform_margins(self):
        """set_uniform_margins should set equal margins on all sides."""
        filter = Filter()
        page_id = make_page_id("test")

        filter.set_uniform_margins(page_id, 15.0)

        margins = filter.get_margins(page_id)
        assert margins.top == 15.0
        assert margins.bottom == 15.0
        assert margins.left == 15.0
        assert margins.right == 15.0

    def test_apply_margins_to_pages(self):
        """apply_margins_to_pages should set margins for all pages."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        margins = Margins.uniform(20)

        filter.apply_margins_to_pages([page1, page2], margins)

        assert filter.get_margins(page1) == margins
        assert filter.get_margins(page2) == margins

    def test_apply_alignment_to_pages(self):
        """apply_alignment_to_pages should set alignment for all pages."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        alignment = Alignment.top_left()

        filter.apply_alignment_to_pages([page1, page2], alignment)

        assert filter.get_alignment(page1) == alignment
        assert filter.get_alignment(page2) == alignment


class TestFilterAggregateSize:
    """Tests for aggregate size calculation."""

    def test_aggregate_size_single_page(self):
        """Aggregate size should match single page size."""
        filter = Filter()
        page_id = make_page_id("test")
        params = Params(
            hard_margins_mm=Margins.uniform(10),
            content_size_mm=ContentSize(width_mm=100, height_mm=150),
        )
        filter.set_params(page_id, params)

        size = filter.get_aggregate_hard_size()

        # 100 + 10 + 10 = 120, 150 + 10 + 10 = 170
        assert size.width_mm == 120.0
        assert size.height_mm == 170.0

    def test_aggregate_size_multiple_pages(self):
        """Aggregate size should be maximum across aligned pages."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        # Page 1: 100 + 10 + 10 = 120 wide, 150 + 10 + 10 = 170 tall
        filter.set_params(
            page1,
            Params(
                hard_margins_mm=Margins.uniform(10),
                content_size_mm=ContentSize(width_mm=100, height_mm=150),
            ),
        )

        # Page 2: 150 + 5 + 5 = 160 wide, 100 + 5 + 5 = 110 tall
        filter.set_params(
            page2,
            Params(
                hard_margins_mm=Margins.uniform(5),
                content_size_mm=ContentSize(width_mm=150, height_mm=100),
            ),
        )

        size = filter.get_aggregate_hard_size()

        assert size.width_mm == 160.0  # max(120, 160)
        assert size.height_mm == 170.0  # max(170, 110)

    def test_aggregate_size_excludes_independent(self):
        """Aggregate size should exclude independent (is_null) pages."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        # Page 1: aligned, 120x170
        filter.set_params(
            page1,
            Params(
                hard_margins_mm=Margins.uniform(10),
                content_size_mm=ContentSize(width_mm=100, height_mm=150),
            ),
        )

        # Page 2: independent (is_null), larger but should be excluded
        filter.set_params(
            page2,
            Params(
                hard_margins_mm=Margins.uniform(50),
                content_size_mm=ContentSize(width_mm=200, height_mm=200),
                alignment=Alignment.independent(),
            ),
        )

        size = filter.get_aggregate_hard_size()

        # Should only consider page1
        assert size.width_mm == 120.0
        assert size.height_mm == 170.0


class TestFilterSoftMargins:
    """Tests for soft margin calculation."""

    def test_soft_margins_independent_page(self):
        """Independent pages should have zero soft margins."""
        filter = Filter()
        page_id = make_page_id("test")
        filter.set_params(
            page_id,
            Params(
                content_size_mm=ContentSize(width_mm=100, height_mm=100),
                alignment=Alignment.independent(),
            ),
        )

        soft = filter.calculate_soft_margins(page_id)

        assert soft.top == 0.0
        assert soft.bottom == 0.0
        assert soft.left == 0.0
        assert soft.right == 0.0

    def test_soft_margins_centered(self):
        """Centered alignment should distribute soft margins equally."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        # Page 1: larger (120x170)
        filter.set_params(
            page1,
            Params(
                hard_margins_mm=Margins.uniform(10),
                content_size_mm=ContentSize(width_mm=100, height_mm=150),
                alignment=Alignment.centered(),
            ),
        )

        # Page 2: smaller (60x60)
        filter.set_params(
            page2,
            Params(
                hard_margins_mm=Margins.uniform(5),
                content_size_mm=ContentSize(width_mm=50, height_mm=50),
                alignment=Alignment.centered(),
            ),
        )

        soft = filter.calculate_soft_margins(page2)

        # Page 2 hard size: 60x60, aggregate: 120x170
        # Width diff: 60, height diff: 110
        # Centered: split equally
        assert soft.left == 30.0
        assert soft.right == 30.0
        assert soft.top == 55.0
        assert soft.bottom == 55.0

    def test_soft_margins_top_left(self):
        """Top-left alignment should put soft margins at bottom-right."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        # Page 1: larger (120x170)
        filter.set_params(
            page1,
            Params(
                hard_margins_mm=Margins.uniform(10),
                content_size_mm=ContentSize(width_mm=100, height_mm=150),
            ),
        )

        # Page 2: smaller with top-left alignment
        filter.set_params(
            page2,
            Params(
                hard_margins_mm=Margins.uniform(5),
                content_size_mm=ContentSize(width_mm=50, height_mm=50),
                alignment=Alignment.top_left(),
            ),
        )

        soft = filter.calculate_soft_margins(page2)

        # Width diff: 60, height diff: 110
        # Top-left: margin goes to bottom and right
        assert soft.left == 0.0
        assert soft.right == 60.0
        assert soft.top == 0.0
        assert soft.bottom == 110.0

    def test_total_margins(self):
        """get_total_margins should combine hard and soft margins."""
        filter = Filter()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        # Page 1: larger
        filter.set_params(
            page1,
            Params(
                hard_margins_mm=Margins.uniform(10),
                content_size_mm=ContentSize(width_mm=100, height_mm=150),
            ),
        )

        # Page 2: smaller with hard margins
        filter.set_params(
            page2,
            Params(
                hard_margins_mm=Margins(top=5, bottom=5, left=5, right=5),
                content_size_mm=ContentSize(width_mm=50, height_mm=50),
                alignment=Alignment.centered(),
            ),
        )

        total = filter.get_total_margins(page2)

        # Hard: 5 all around
        # Soft (centered): left=30, right=30, top=55, bottom=55
        assert total.left == 35.0
        assert total.right == 35.0
        assert total.top == 60.0
        assert total.bottom == 60.0
