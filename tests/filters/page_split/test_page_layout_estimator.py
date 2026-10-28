"""Layout decisions of the page layout estimator."""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.page_split import page_layout_estimator as estimator
from scantailor.filters.page_split.content_span_finder import Span
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayoutType
from scantailor.filters.page_split.page_layout_estimator import estimate_page_layout
from tests.dewarping.synthetic import straight_text_page

from . import scan_pages as pages


def _mid_x(line) -> float:
    return 0.5 * (line[0][0] + line[1][0])


class TestTwoPagesFromSpans:
    def test_splits_in_the_middle_of_the_gap_between_balanced_spans(self) -> None:
        spans = [Span(100, 280), Span(320, 500)]

        cutters = estimator._process_spans_two_pages(spans, 600)

        assert _mid_x(cutters.lines[0]) == 300.0

    def test_splits_at_the_centre_when_there_is_no_content(self) -> None:
        cutters = estimator._process_spans_two_pages([], 600)

        assert _mid_x(cutters.lines[0]) == 300.0

    def test_prefers_the_widest_of_the_nearly_equally_balanced_gaps(self) -> None:
        spans = [Span(50, 250), Span(262, 300), Span(340, 540)]

        cutters = estimator._process_spans_two_pages(spans, 600)

        # The gap 300-340 is wider than 250-262 and almost as well centred.
        assert _mid_x(cutters.lines[0]) == 320.0

    def test_a_single_span_in_the_middle_is_cut_beside_it(self) -> None:
        cutters = estimator._process_spans_two_pages([Span(250, 350)], 600)

        assert _mid_x(cutters.lines[0]) in (235.0, 365.0)

    def test_a_single_span_far_from_the_centre_leaves_the_split_in_the_middle(
        self,
    ) -> None:
        cutters = estimator._process_spans_two_pages([Span(20, 200)], 600)

        assert _mid_x(cutters.lines[0]) == 300.0


class TestLayoutKinds:
    @pytest.mark.parametrize("layout_type", list(LayoutType))
    @pytest.mark.parametrize(
        "factory",
        [
            pages.two_page_spread,
            pages.single_page,
            pages.single_page_missing_inside_edge,
        ],
    )
    def test_the_layout_agrees_with_the_requested_type(
        self, layout_type, factory
    ) -> None:
        layout = estimate_page_layout(factory(), layout_type, 300.0)

        if layout_type is not LayoutType.AUTO_LAYOUT_TYPE:
            assert layout.to_layout_type() is layout_type

    def test_single_page_uncut_is_not_analysed(self) -> None:
        layout = estimate_page_layout(
            pages.two_page_spread(), LayoutType.SINGLE_PAGE_UNCUT, 300.0
        )

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT
        assert layout.outline == (0.0, 0.0, 1500.0, 1000.0)

    def test_a_wide_image_is_two_pages_and_a_tall_one_is_one_page(self) -> None:
        wide = estimate_page_layout(
            pages.two_page_spread(), LayoutType.AUTO_LAYOUT_TYPE, 300.0
        )
        tall = estimate_page_layout(
            pages.single_page(), LayoutType.AUTO_LAYOUT_TYPE, 300.0
        )

        assert wide.layout_type is PageLayoutType.TWO_PAGES
        assert tall.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT

    def test_a_clean_spread_splits_in_the_middle_of_the_gutter(self) -> None:
        layout = estimate_page_layout(
            pages.two_page_spread(), LayoutType.TWO_PAGES, 300.0
        )

        assert layout.get_split_line_x() == pytest.approx(750.0, abs=1.0)

    def test_a_folding_line_decides_the_split_over_the_whitespace(self) -> None:
        layout = estimate_page_layout(
            pages.spread_with_folding_line(), LayoutType.AUTO_LAYOUT_TYPE, 300.0
        )

        assert layout.layout_type is PageLayoutType.TWO_PAGES
        assert layout.get_split_line_x() == pytest.approx(773.0, abs=8.0)

    def test_black_shadows_at_the_borders_do_not_move_the_split(self) -> None:
        layout = estimate_page_layout(
            pages.spread_with_dark_borders(), LayoutType.AUTO_LAYOUT_TYPE, 300.0
        )

        assert layout.get_split_line_x() == pytest.approx(750.0, abs=1.0)

    def test_a_skewed_spread_gets_a_tilted_split_line(self) -> None:
        layout = estimate_page_layout(
            pages.skewed_spread(2.0), LayoutType.TWO_PAGES, 300.0
        )

        assert layout.cutter1 is not None
        (x_top, _), (x_bottom, _) = layout.cutter1
        assert x_bottom - x_top == pytest.approx(
            1000 * np.tan(np.radians(2.0)), rel=0.3
        )
        assert layout.get_split_line_x() == pytest.approx(750.0, abs=6.0)

    def test_edge_lines_make_a_single_page_cut_at_them(self) -> None:
        layout = estimate_page_layout(
            pages.single_page_with_edge_lines(), LayoutType.AUTO_LAYOUT_TYPE, 300.0
        )

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter1 is not None
        assert layout.cutter2 is not None
        assert _mid_x(layout.cutter1) == pytest.approx(81.0, abs=15.0)
        assert _mid_x(layout.cutter2) == pytest.approx(919.0, abs=15.0)

    def test_one_edge_line_is_paired_with_the_far_image_edge(self) -> None:
        layout = estimate_page_layout(
            pages.single_page_with_right_edge_line(), LayoutType.AUTO_LAYOUT_TYPE, 300.0
        )

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter1 is not None
        assert layout.cutter2 is not None
        assert _mid_x(layout.cutter1) == pytest.approx(0.0)
        assert _mid_x(layout.cutter2) == pytest.approx(921.0, abs=15.0)

    def test_cutters_span_the_whole_image_height(self) -> None:
        layout = estimate_page_layout(
            pages.skewed_spread(2.0), LayoutType.TWO_PAGES, 300.0
        )

        assert layout.cutter1 is not None
        assert layout.cutter1[0][1] == 0.0
        assert layout.cutter1[1][1] == 1000.0


class TestSinglePageWithoutFoldingLine:
    @pytest.mark.parametrize(
        "factory",
        [pages.single_page, pages.single_page_missing_inside_edge],
    )
    def test_an_automatic_request_leaves_a_page_without_an_offcut_uncut(
        self, factory
    ) -> None:
        layout = estimate_page_layout(factory(), LayoutType.AUTO_LAYOUT_TYPE, 300.0)

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT

    def test_an_automatic_request_cuts_the_neighbours_text_off_the_left_edge(
        self,
    ) -> None:
        layout = estimate_page_layout(
            pages.single_page_with_neighbours_letters_on_left_edge(),
            LayoutType.AUTO_LAYOUT_TYPE,
            300.0,
        )

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter1 is not None
        assert layout.cutter2 is not None
        # The stray text ends at x = 70 and the page's text starts at 200.
        assert 70.0 < _mid_x(layout.cutter1) < 200.0
        assert _mid_x(layout.cutter2) == pytest.approx(1000.0, abs=0.5)

    def test_an_automatic_request_cuts_the_neighbours_text_off_the_right_edge(
        self,
    ) -> None:
        layout = estimate_page_layout(
            pages.single_page_with_neighbours_letters_on_right_edge(),
            LayoutType.AUTO_LAYOUT_TYPE,
            300.0,
        )

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter1 is not None
        assert layout.cutter2 is not None
        assert _mid_x(layout.cutter1) == pytest.approx(0.0, abs=0.5)
        assert 800.0 < _mid_x(layout.cutter2) < 930.0

    def test_a_pages_own_text_running_off_the_edge_stays_uncut(self) -> None:
        """The last word of each line touches the right edge and is a span of its
        own, across an ordinary word gap. Upstream 1.2.1 would cut there if its
        whitespace cutter ran for an Auto single page; it never does (#220).
        """
        page = straight_text_page()
        assert page[:, -4:].min() < 100

        layout = estimate_page_layout(page, LayoutType.AUTO_LAYOUT_TYPE, 200.0)

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT

    def test_a_page_plus_offcut_request_cuts_at_the_image_edges_when_nothing_is_found(
        self,
    ) -> None:
        page = pages.single_page_missing_inside_edge()

        layout = estimate_page_layout(page, LayoutType.PAGE_PLUS_OFFCUT, 300.0)

        assert layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
        assert layout.cutter1 is not None
        assert layout.cutter2 is not None
        assert _mid_x(layout.cutter1) == pytest.approx(0.0, abs=0.5)
        assert _mid_x(layout.cutter2) == pytest.approx(1000.0, abs=0.5)


class TestThreshold:
    def test_an_explicit_threshold_replaces_otsus(self) -> None:
        page = pages.blank(1500, 1000)
        pages.add_text(page, 110, 500, 100, 900, seed=1)
        pages.add_text(page, 800, 1390, 100, 900, seed=2)

        by_default = estimate_page_layout(page, LayoutType.TWO_PAGES, 300.0)
        # Nothing is darker than 10, so there is no content and the split falls
        # back to the middle of the image.
        no_content = estimate_page_layout(
            page, LayoutType.TWO_PAGES, 300.0, bw_threshold=10
        )

        assert by_default.get_split_line_x() == pytest.approx(650.0, abs=1.0)
        assert no_content.get_split_line_x() == pytest.approx(750.0)


class TestOffcutStripRule:
    """Which edge span the whitespace cutter takes for a neighbour's offcut."""

    WIDTH = 500

    def test_a_narrow_strip_across_a_gutter_is_cut_in_the_gutter(self) -> None:
        spans = [Span(0, 35), Span(100, 435)]

        assert estimator._left_cut_position(spans, self.WIDTH) == 67.5
        mirrored = [Span(65, 400), Span(465, 500)]
        assert estimator._right_cut_position(mirrored, self.WIDTH) == 432.5

    def test_the_last_word_of_text_running_off_the_edge_is_not_an_offcut(
        self,
    ) -> None:
        words = [Span(40, 110), Span(124, 180), Span(194, 260), Span(274, 350)]

        assert (
            estimator._right_cut_position([*words, Span(364, 500)], self.WIDTH) is None
        )
        assert estimator._left_cut_position([Span(0, 26), *words], self.WIDTH) is None

    def test_a_gutter_must_be_wider_than_the_gaps_inside_the_content(self) -> None:
        wide_gaps = [Span(40, 110), Span(140, 180), Span(210, 260), Span(290, 350)]
        same_as_inside = [*wide_gaps, Span(380, 500)]
        twice_as_wide = [*wide_gaps, Span(410, 500)]

        assert estimator._right_cut_position(same_as_inside, self.WIDTH) is None
        assert estimator._right_cut_position(twice_as_wide, self.WIDTH) == 380.0

    def test_a_column_wider_than_a_quarter_of_the_page_is_content(self) -> None:
        two_columns = [Span(40, 230), Span(290, 500)]

        assert estimator._right_cut_position(two_columns, self.WIDTH) is None

    def test_a_gutter_narrower_than_the_minimum_is_a_word_gap(self) -> None:
        spans = [Span(0, 30), Span(45, 435)]

        assert estimator._left_cut_position(spans, self.WIDTH) is None


class TestTwoPagesGapChoice:
    def test_a_wider_gap_to_the_right_that_is_nearly_as_balanced_wins(self) -> None:
        # Gap 290-296 balances best (0.95); gap 300-330 balances 0.90, within 90 %
        # of the best, and is five times as wide.
        spans = [Span(0, 290), Span(296, 300), Span(330, 600)]

        cutters = estimator._process_spans_two_pages(spans, 600)

        assert _mid_x(cutters.lines[0]) == 315.0

    def test_a_narrower_neighbour_gap_does_not_replace_the_best_one(self) -> None:
        # Gap 280-300 balances 0.93 and gap 310-320 0.90, close enough to be
        # weighed but only half as wide.
        spans = [Span(0, 280), Span(300, 310), Span(320, 600)]

        cutters = estimator._process_spans_two_pages(spans, 600)

        assert _mid_x(cutters.lines[0]) == 290.0

    def test_a_wider_gap_to_the_left_that_is_nearly_as_balanced_wins(self) -> None:
        spans = [Span(0, 270), Span(300, 304), Span(310, 600)]

        cutters = estimator._process_spans_two_pages(spans, 600)

        assert _mid_x(cutters.lines[0]) == 285.0

    def test_content_too_lopsided_to_split_is_cut_beside_as_one_block(self) -> None:
        # 290 columns of content left of the only gap and 10 right of it: below
        # the 0.25 balance, so the spans count as one block 10-330 that reaches
        # the centre, and the cut goes 15 columns clear of its emptier side.
        spans = [Span(10, 300), Span(320, 330)]

        cutters = estimator._process_spans_two_pages(spans, 600)

        assert cutters.layout_type is PageLayoutType.TWO_PAGES
        assert _mid_x(cutters.lines[0]) == 345.0

    def test_a_single_block_over_the_centre_is_cut_on_its_wider_margin_side(
        self,
    ) -> None:
        cutters = estimator._process_spans_two_pages([Span(280, 500)], 600)

        assert _mid_x(cutters.lines[0]) == 265.0


class TestRemoveInsignificantEdgeSpans:
    def test_small_spans_at_both_edges_are_dropped_smallest_first(self) -> None:
        # 391 columns of content allow 26 to be dropped: 5 at the left, then 6 at
        # the right, and the 180 column span after that is far over what is left.
        spans = [Span(0, 5), Span(100, 300), Span(320, 500), Span(590, 596)]

        estimator._remove_insignificant_edge_spans(spans)

        assert spans == [Span(100, 300), Span(320, 500)]

    def test_the_allowance_is_shared_between_the_two_edges(self) -> None:
        # 340 columns allow 22: the right 20 column span goes, which leaves 2,
        # too few for the left one of the same width.
        spans = [Span(0, 20), Span(100, 400), Span(580, 600)]

        estimator._remove_insignificant_edge_spans(spans)

        assert spans == [Span(0, 20), Span(100, 400)]

    def test_the_last_span_is_never_dropped(self) -> None:
        spans = [Span(0, 3)]

        estimator._remove_insignificant_edge_spans(spans)

        assert spans == [Span(0, 3)]

    def test_no_spans_stay_no_spans(self) -> None:
        spans: list[Span] = []

        estimator._remove_insignificant_edge_spans(spans)

        assert spans == []


_SINGLE_PAGE_CASES = [
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(60, 440)],
        (False, False),
        None,
        id="no_offcut_stays_uncut",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(0, 35), Span(100, 435), Span(465, 500)],
        (True, True),
        None,
        id="marks_at_both_edges_stay_uncut",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [],
        (True, False),
        (0.0, 500.0),
        id="left_no_content_cuts_at_the_edge",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(30, 440)],
        (True, False),
        (15.0, 500.0),
        id="left_content_clear_of_the_edge_cut_halfway_to_it",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(0, 35), Span(100, 435)],
        (True, False),
        (67.5, 500.0),
        id="left_strip_cut_in_its_gutter",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(0, 30), Span(45, 435)],
        (True, False),
        None,
        id="left_strip_behind_a_word_gap_is_page_text",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(0, 300), Span(320, 480)],
        (True, False),
        None,
        id="left_span_over_half_the_page_is_content",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(0, 40)],
        (True, False),
        (60.0, 500.0),
        id="left_lone_strip_cut_20_columns_past_it",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [],
        (False, True),
        (0.0, 500.0),
        id="right_no_content_cuts_at_the_edge",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(60, 470)],
        (False, True),
        (0.0, 485.0),
        id="right_content_clear_of_the_edge_cut_halfway_to_it",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(65, 400), Span(465, 500)],
        (False, True),
        (0.0, 432.5),
        id="right_strip_cut_in_its_gutter",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(460, 500)],
        (False, True),
        (0.0, 440.0),
        id="right_lone_strip_cut_20_columns_before_it",
    ),
    pytest.param(
        LayoutType.AUTO_LAYOUT_TYPE,
        [Span(10, 100), Span(200, 500)],
        (False, True),
        None,
        id="right_span_over_half_the_page_is_content",
    ),
    pytest.param(
        LayoutType.PAGE_PLUS_OFFCUT,
        [Span(0, 35), Span(100, 435)],
        (True, False),
        (0.0, 500.0),
        id="page_plus_offcut_cuts_at_the_image_edges",
    ),
    pytest.param(
        LayoutType.PAGE_PLUS_OFFCUT,
        [Span(60, 440)],
        (False, False),
        (0.0, 500.0),
        id="page_plus_offcut_without_marks_cuts_at_the_image_edges",
    ),
]


class TestSinglePageFromSpans:
    """Whitespace decisions for one page 500 columns wide.

    Only an automatic request looks for an offcut, and only when marks reach one
    edge alone.
    """

    @pytest.mark.parametrize(
        ("layout_type", "spans", "offcuts", "expected"), _SINGLE_PAGE_CASES
    )
    def test_cutters_for_the_spans_and_the_marks_at_the_edges(
        self,
        layout_type: LayoutType,
        spans: list[Span],
        offcuts: tuple[bool, bool],
        expected: tuple[float, float] | None,
    ) -> None:
        cutters = estimator._process_spans_single_page(
            layout_type, spans, 500, 700, *offcuts
        )

        if expected is None:
            assert cutters.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT
            assert cutters.lines == ()
        else:
            assert cutters.layout_type is PageLayoutType.SINGLE_PAGE_CUT
            assert tuple(_mid_x(line) for line in cutters.lines) == expected


class TestOffcutMarks:
    """Marks in the 3 columns that start 2 pixels inside an edge mean an offcut.

    The 2 rows at the top and the foot are not looked at.
    """

    @pytest.mark.parametrize(
        ("columns", "rows", "left", "right"),
        [
            pytest.param(slice(3, 4), slice(40, 50), True, False, id="left_band"),
            pytest.param(slice(2, 3), slice(40, 50), True, False, id="left_first"),
            pytest.param(slice(4, 5), slice(40, 50), True, False, id="left_last"),
            pytest.param(slice(0, 2), slice(40, 50), False, False, id="left_margin"),
            pytest.param(slice(5, 6), slice(40, 50), False, False, id="past_left"),
            pytest.param(slice(3, 4), slice(0, 2), False, False, id="top_rows"),
            pytest.param(slice(3, 4), slice(98, 100), False, False, id="foot_rows"),
            pytest.param(slice(97, 98), slice(40, 50), False, True, id="right_band"),
            pytest.param(slice(95, 96), slice(40, 50), False, True, id="right_first"),
            pytest.param(slice(98, 100), slice(40, 50), False, False, id="r_margin"),
            pytest.param(slice(94, 95), slice(40, 50), False, False, id="before_right"),
            pytest.param(slice(0, 100), slice(40, 50), True, True, id="across"),
        ],
    )
    def test_only_marks_in_the_edge_bands_count(
        self, columns: slice, rows: slice, left: bool, right: bool
    ) -> None:
        page = np.zeros((100, 100), dtype=np.bool_)
        page[rows, columns] = True

        assert estimator._has_left_offcut(page) is left
        assert estimator._has_right_offcut(page) is right


class TestGarbageAndShadowRemoval:
    """A 1000 x 800 page at 300 dpi; sizes in the names are at the 150 dpi output."""

    @staticmethod
    def _blank() -> NDArray[np.bool_]:
        """An empty 1000 x 800 binary page.

        Returns:
            NDArray[np.bool_]: (800, 1000) all-False page.
        """
        return np.zeros((800, 1000), dtype=np.bool_)

    def test_the_result_is_half_the_size(self) -> None:
        assert estimator._remove_garbage_and_halve(self._blank()).shape == (400, 500)

    def test_a_block_with_bars_of_4_pixels_is_kept_whole(self) -> None:
        page = self._blank()
        page[100:120, 200:220] = True

        cleaned = estimator._remove_garbage_and_halve(page)

        expected = np.zeros((400, 500), dtype=np.bool_)
        expected[50:60, 100:110] = True
        np.testing.assert_array_equal(cleaned, expected)

    @pytest.mark.parametrize(
        ("height", "width"),
        [(6, 6), (2, 6), (6, 2)],
        ids=["3x3_speck", "3_wide_dash", "3_tall_dash"],
    )
    def test_a_mark_without_a_4_pixel_bar_is_dropped(
        self, height: int, width: int
    ) -> None:
        page = self._blank()
        page[100 : 100 + height, 200 : 200 + width] = True

        assert not estimator._remove_garbage_and_halve(page).any()

    def test_a_one_pixel_rule_is_kept(self) -> None:
        page = self._blank()
        page[100:140, 200:202] = True

        assert estimator._remove_garbage_and_halve(page)[50:70, 100].all()

    @pytest.mark.parametrize(
        ("height", "width", "removed"),
        [
            pytest.param(28, 400, True, id="200x14_wide_shadow"),
            pytest.param(28, 396, False, id="198_wide_is_no_shadow"),
            pytest.param(26, 400, False, id="13_thick_is_no_shadow"),
            pytest.param(600, 28, True, id="14x300_tall_shadow"),
            pytest.param(596, 28, False, id="298_tall_is_no_shadow"),
            pytest.param(600, 26, False, id="13_wide_is_no_shadow"),
        ],
    )
    def test_a_bar_is_a_shadow_from_200_by_14_or_14_by_300(
        self, height: int, width: int, removed: bool
    ) -> None:
        page = self._blank()
        page[100 : 100 + height, 200 : 200 + width] = True

        cleaned = estimator._remove_garbage_and_halve(page)

        assert bool(cleaned.any()) is not removed

    def test_a_narrow_band_on_the_image_edge_is_a_shadow(self) -> None:
        """Outside the image counts as black, so a band 10 wide on the border is
        part of a shadow 14 wide; the same band inside the page is not."""
        on_edge = self._blank()
        on_edge[100:700, 0:20] = True
        inside = self._blank()
        inside[100:700, 400:420] = True

        assert not estimator._remove_garbage_and_halve(on_edge).any()
        assert estimator._remove_garbage_and_halve(inside).any()

    @pytest.mark.parametrize(
        ("gap", "kept"), [(0, False), (8, True)], ids=["touching", "4_px_clear"]
    )
    def test_a_mark_touching_a_shadow_goes_with_it(self, gap: int, kept: bool) -> None:
        page = self._blank()
        page[100:700, 200:228] = True
        mark = 228 + gap
        page[300:320, mark : mark + 20] = True

        cleaned = estimator._remove_garbage_and_halve(page)

        assert not cleaned[:, 100:114].any()
        assert bool(cleaned[150:160, mark // 2 : mark // 2 + 10].all()) is kept


def _vertical(x: float, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """A vertical line at column ``x`` spanning the image height.

    Args:
        x (float): Column of the line.
        height (int): Height of the image.

    Returns:
        tuple[tuple[float, float], tuple[float, float]]: Its top and foot.
    """
    return ((x, 0.0), (x, float(height)))


_FOLDING_CASES = [
    pytest.param("tall", "auto", [], None, id="tall_auto_no_line"),
    pytest.param("tall", "auto", [260], ("uncut",), id="tall_auto_near_centre"),
    pytest.param("tall", "auto", [40], ("cut", 40, 500), id="tall_auto_left"),
    pytest.param("tall", "auto", [460], ("cut", 0, 460), id="tall_auto_right"),
    pytest.param("tall", "auto", [470, 30], ("cut", 30, 470), id="tall_auto_both"),
    pytest.param("tall", "auto", [240, 270], ("uncut",), id="tall_auto_two_central"),
    pytest.param("tall", "auto", [30, 60], ("cut", 60, 500), id="tall_auto_one_side"),
    pytest.param(
        "tall", "auto", [30, 200], ("cut", 30, 200), id="tall_auto_one_side_central"
    ),
    pytest.param("tall", "offcut", [], None, id="tall_offcut_no_line"),
    pytest.param(
        "tall", "offcut", [260], ("cut", 0, 260), id="tall_offcut_central_cuts"
    ),
    pytest.param("wide", "offcut", [], None, id="wide_offcut_no_line"),
    pytest.param("wide", "offcut", [40, 760], ("cut", 40, 760), id="wide_offcut_two"),
    pytest.param("tall", "two", [], None, id="tall_two_no_line"),
    pytest.param("tall", "two", [240], ("two", 240), id="tall_two_one_line"),
    pytest.param("tall", "two", [60], None, id="tall_two_line_at_edge"),
    pytest.param("tall", "two", [200, 290], ("two", 290), id="tall_two_nearest"),
    pytest.param("wide", "auto", [], None, id="wide_auto_no_line"),
    pytest.param("wide", "auto", [420], ("two", 420), id="wide_auto_one_line"),
    pytest.param("wide", "auto", [100], None, id="wide_auto_line_at_edge"),
    pytest.param(
        "wide", "auto", [100, 450, 380, 700], ("two", 380), id="wide_auto_nearest"
    ),
    pytest.param("wide", "two", [420], ("two", 420), id="wide_two_one_line"),
]
_REQUESTS = {
    "auto": LayoutType.AUTO_LAYOUT_TYPE,
    "offcut": LayoutType.PAGE_PLUS_OFFCUT,
    "two": LayoutType.TWO_PAGES,
}
_KINDS = {
    "uncut": PageLayoutType.SINGLE_PAGE_UNCUT,
    "cut": PageLayoutType.SINGLE_PAGE_CUT,
    "two": PageLayoutType.TWO_PAGES,
}


class TestFoldingLineRules:
    """Cutters chosen from the folding lines found, for each requested layout.

    A wide image is 800 x 500 and a tall one 500 x 800. One page: lines within
    0.65 of the half width of the centre leave it uncut, and lines all on one side
    and at least 0.3 of the half width from the centre reduce to the nearest.
    Lines on one side with one nearer the centre than that are not reduced, so
    when any line is beyond the uncut band the page is cut between the outermost
    lines; a single line is cut against the opposite image edge. Two pages: only
    lines within 0.6 of the half width of the centre count.
    """

    @pytest.mark.parametrize(
        ("shape", "request_name", "xs", "expected"), _FOLDING_CASES
    )
    def test_cutters_for_the_lines_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        shape: str,
        request_name: str,
        xs: list[float],
        expected: tuple[str | float, ...] | None,
    ) -> None:
        height, width = (800, 500) if shape == "tall" else (500, 800)
        lines = [_vertical(x, height) for x in xs]
        monkeypatch.setattr(estimator, "find_lines", lambda *_args: list(lines))
        gray = np.zeros((height, width), dtype=np.uint8)

        cutters = estimator._try_cut_at_folding_line(
            _REQUESTS[request_name], gray, 300.0
        )

        if expected is None:
            assert cutters is None
            return
        assert cutters is not None
        kind, *positions = expected
        assert cutters.layout_type is _KINDS[str(kind)]
        assert [_mid_x(line) for line in cutters.lines] == positions

    def test_a_tilted_line_is_placed_by_its_middle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Its foot is beyond 0.6 of the half width from the centre; its middle
        # is not, so it counts.
        tilted = ((560.0, 0.0), (680.0, 500.0))
        monkeypatch.setattr(estimator, "find_lines", lambda *_args: [tilted])

        cutters = estimator._try_cut_at_folding_line(
            LayoutType.TWO_PAGES, np.zeros((500, 800), dtype=np.uint8), 300.0
        )

        assert cutters is not None
        assert cutters.lines == (tilted,)
