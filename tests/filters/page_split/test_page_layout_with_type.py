"""Tests for PageLayout.with_type, the equivalent of C++ PageLayout::setType."""

import pytest

from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType


def test_manual_split_line_survives_switch_to_single_and_back():
    two = PageLayout.two_pages(1000, 500, split_x=420)

    single = two.with_type(PageLayoutType.SINGLE_PAGE_UNCUT)
    back = single.with_type(PageLayoutType.TWO_PAGES)

    assert single.layout_type == PageLayoutType.SINGLE_PAGE_UNCUT
    assert back.layout_type == PageLayoutType.TWO_PAGES
    assert back.get_split_line_x() == pytest.approx(420)


def test_two_pages_shares_first_cutter_with_second():
    cut = PageLayout.single_page_cut(1000, 500, left_x=100, right_x=900)

    two = cut.with_type(PageLayoutType.TWO_PAGES)

    assert two.cutter1 == cut.cutter1
    assert two.cutter2 == cut.cutter1


def test_cutter_the_layout_never_had_defaults_to_the_outline_edges():
    uncut = PageLayout.single_page_uncut(1000, 500, x=10, y=20)

    cut = uncut.with_type(PageLayoutType.SINGLE_PAGE_CUT)

    assert cut.cutter1 == ((10, 20), (10, 520))
    assert cut.cutter2 == ((1010, 20), (1010, 520))


def test_outline_is_preserved():
    two = PageLayout.two_pages(1000, 500, split_x=500, x=10, y=20)

    assert two.with_type(PageLayoutType.SINGLE_PAGE_UNCUT).outline == two.outline
