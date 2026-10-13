"""Layout decisions of the page layout estimator."""

import numpy as np
import pytest

from scantailor.filters.page_split import page_layout_estimator as estimator
from scantailor.filters.page_split.content_span_finder import Span
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayoutType
from scantailor.filters.page_split.page_layout_estimator import estimate_page_layout
from tests.dewarping.synthetic import straight_text_page

from . import cpp_reference as cpp
from . import scan_pages as pages

WIDTH = 600
HEIGHT = 800


def _random_spans(seed: int, count: int | None = None) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    count = count if count is not None else int(rng.integers(0, 9))
    spans, x = [], int(rng.integers(0, 60))
    for _ in range(count):
        begin = x + int(rng.integers(0, 90))
        end = begin + int(rng.integers(2, 120))
        if end > WIDTH:
            break
        spans.append((begin, end))
        x = end + 8
    return spans


def _mid_x(line) -> float:
    return 0.5 * (line[0][0] + line[1][0])


def port_keeps_the_strip(
    spans: list[Span], width: int, left_offcut: bool, right_offcut: bool
) -> bool:
    """Whether upstream would cut the page's own text where the port does not.

    Upstream takes any edge span narrower than half the page for an offcut when
    another span follows; the port asks ``_is_offcut_strip`` first.
    """
    if len(spans) < 2:
        return False
    first, last = spans[0], spans[-1]
    if left_offcut and not right_offcut and first.begin == 0:
        return first.width <= width // 2 and not estimator._is_offcut_strip(
            first, Span.between(first, spans[1]), spans[1:], width
        )
    if right_offcut and not left_offcut and last.end == width:
        return last.width <= width // 2 and not estimator._is_offcut_strip(
            last, Span.between(spans[-2], last), spans[:-1], width
        )
    return False


class TestWhitespaceDecisionsMatchUpstream:
    @pytest.mark.parametrize("seed", range(60))
    def test_two_pages(self, seed: int) -> None:
        spans = _random_spans(seed)

        got = estimator._process_spans_two_pages([Span(a, b) for a, b in spans], WIDTH)

        kind, line, _ = cpp.process_content_spans_two_pages(
            [cpp.Span(a, b) for a, b in spans], WIDTH
        )
        assert got.layout_type is PageLayoutType.TWO_PAGES
        assert kind == cpp.TWO
        assert _mid_x(got.lines[0]) == pytest.approx(_mid_x(line))

    @pytest.mark.parametrize("seed", range(40))
    def test_edge_spans_removed_before_a_two_page_split(self, seed: int) -> None:
        spans = _random_spans(seed + 100)
        mine = [Span(a, b) for a, b in spans]
        theirs = [cpp.Span(a, b) for a, b in spans]

        estimator._remove_insignificant_edge_spans(mine)
        cpp.remove_insignificant_edge_spans(theirs)

        assert [(s.begin, s.end) for s in mine] == [(s.begin, s.end) for s in theirs]

    @pytest.mark.parametrize("seed", range(40))
    @pytest.mark.parametrize("left", [False, True])
    @pytest.mark.parametrize("right", [False, True])
    @pytest.mark.parametrize(
        ("requested", "upstream"),
        [
            (LayoutType.AUTO_LAYOUT_TYPE, cpp.AUTO),
            (LayoutType.PAGE_PLUS_OFFCUT, cpp.PLUS_OFFCUT),
        ],
    )
    def test_single_page(
        self,
        seed: int,
        left: bool,
        right: bool,
        requested: LayoutType,
        upstream: str,
    ) -> None:
        spans = _random_spans(seed + 200)
        # Reach the edges in some cases so the "content touches the border" paths run.
        if spans and seed % 3 == 0:
            spans[0] = (0, spans[0][1])
        if spans and seed % 4 == 0:
            spans[-1] = (spans[-1][0], WIDTH)

        mine = [Span(a, b) for a, b in spans]
        got = estimator._process_spans_single_page(
            requested, mine, WIDTH, HEIGHT, left, right
        )

        kind, line1, line2 = cpp.process_content_spans_single_page(
            upstream, [cpp.Span(a, b) for a, b in spans], WIDTH, HEIGHT, left, right
        )
        if requested is LayoutType.AUTO_LAYOUT_TYPE and port_keeps_the_strip(
            mine, WIDTH, left, right
        ):
            assert kind == cpp.CUT
            assert got.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT
            return
        assert got.layout_type.value == kind
        if line1 is not None:
            assert _mid_x(got.lines[0]) == pytest.approx(_mid_x(line1))
            assert _mid_x(got.lines[1]) == pytest.approx(_mid_x(line2))


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
