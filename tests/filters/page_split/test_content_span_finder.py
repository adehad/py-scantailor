"""Content spans of a column histogram."""

import pytest

from scantailor.filters.page_split.content_span_finder import Span, find_content_spans


def _spans(
    histogram: list[int], min_content: int, min_whitespace: int
) -> list[tuple[int, int]]:
    return [
        (span.begin, span.end)
        for span in find_content_spans(histogram, min_content, min_whitespace)
    ]


class TestFindContentSpans:
    def test_separates_runs_by_enough_whitespace(self) -> None:
        histogram = [0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 0]

        assert _spans(histogram, 1, 3) == [(2, 5), (10, 12)]

    def test_merges_runs_closer_than_the_whitespace_width(self) -> None:
        histogram = [1, 1, 0, 0, 1, 1]

        assert _spans(histogram, 1, 3) == [(0, 6)]

    def test_drops_a_narrow_run_that_nothing_joins(self) -> None:
        histogram = [1, 0, 0, 0, 0, 5, 5, 5]

        assert _spans(histogram, 2, 3) == [(5, 8)]

    def test_keeps_a_narrow_run_that_joins_a_wide_one(self) -> None:
        histogram = [1, 0, 5, 5, 5]

        assert _spans(histogram, 3, 3) == [(0, 5)]

    def test_a_run_reaching_the_end_is_closed(self) -> None:
        assert _spans([0, 0, 4, 4], 1, 2) == [(2, 4)]

    def test_narrow_runs_closer_than_the_whitespace_join_into_one_wide_span(
        self,
    ) -> None:
        assert _spans([1, 0, 1, 0, 1], 3, 2) == [(0, 5)]

    def test_narrow_runs_apart_by_the_whitespace_width_are_all_dropped(self) -> None:
        assert _spans([1, 0, 0, 1, 0, 0, 1], 2, 2) == []

    def test_with_a_whitespace_width_of_one_every_empty_column_separates(
        self,
    ) -> None:
        assert _spans([2, 0, 2, 2, 0, 2], 1, 1) == [(0, 1), (2, 4), (5, 6)]

    def test_a_gap_of_exactly_the_whitespace_width_separates(self) -> None:
        assert _spans([3, 3, 0, 0, 0, 3, 3], 1, 3) == [(0, 2), (5, 7)]

    def test_a_gap_one_column_short_of_the_whitespace_width_merges(self) -> None:
        assert _spans([3, 3, 0, 0, 3, 3], 1, 3) == [(0, 6)]

    @pytest.mark.parametrize(
        ("min_content", "expected"), [(3, [(0, 3)]), (4, [])], ids=["kept", "dropped"]
    )
    def test_a_lone_span_is_kept_from_the_minimum_width_up(
        self, min_content: int, expected: list[tuple[int, int]]
    ) -> None:
        assert _spans([1, 1, 1, 0, 0, 0, 0], min_content, 2) == expected

    def test_an_empty_histogram_has_no_spans(self) -> None:
        assert find_content_spans([], 1, 1) == []
        assert find_content_spans([0, 0, 0], 1, 1) == []


class TestSpan:
    def test_width_and_center(self) -> None:
        span = Span(10, 50)

        assert span.width == 40
        assert span.center == 30.0

    def test_between_two_spans_is_the_gap(self) -> None:
        gap = Span.between(Span(0, 10), Span(25, 30))

        assert (gap.begin, gap.end) == (10, 25)

    def test_between_a_position_and_a_span_uses_the_position_as_it_is(self) -> None:
        assert Span.between(0, Span(25, 30)).width == 25
        assert Span.between(Span(0, 10), 100).width == 90
