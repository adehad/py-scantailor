"""Content spans of a column histogram."""

import numpy as np
import pytest

from scantailor.filters.page_split.content_span_finder import Span, find_content_spans

from . import cpp_reference as cpp


def _spans(
    histogram: list[int], min_content: int, min_whitespace: int
) -> list[tuple[int, int]]:
    return [
        (span.begin, span.end)
        for span in find_content_spans(histogram, min_content, min_whitespace)
    ]


class TestFindContentSpans:
    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize(
        ("min_content", "min_whitespace"), [(1, 1), (2, 8), (5, 3), (3, 20)]
    )
    def test_matches_upstream_on_random_histograms(
        self, seed: int, min_content: int, min_whitespace: int
    ) -> None:
        rng = np.random.default_rng(seed)
        histogram = np.where(
            rng.random(300) < 0.55, rng.integers(1, 9, 300), 0
        ).tolist()

        want = [
            (s.begin, s.end)
            for s in cpp.content_span_finder(histogram, min_content, min_whitespace)
        ]
        assert _spans(histogram, min_content, min_whitespace) == want

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
