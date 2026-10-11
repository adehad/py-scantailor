"""Text line tracing on rendered pages, including polarity and speed."""

import time

import numpy as np
import pytest

from scantailor.dewarping.text_lines import trace_text_lines
from tests.dewarping.synthetic import (
    BACKGROUND,
    FIRST_BASELINE,
    LINE_SPACING,
    PAGE_DPI,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    TEXT_LEFT,
    curved_text_page,
    straight_text_page,
)

EXPECTED_LINES = (PAGE_HEIGHT - 2 * FIRST_BASELINE) // LINE_SPACING + 1


class TestTracedLines:
    """Lines are found one per text line and follow the page's curve."""

    def test_one_line_per_text_line_on_a_straight_page(self) -> None:
        result = trace_text_lines(straight_text_page(), float(PAGE_DPI))

        assert len(result.lines) == EXPECTED_LINES
        heights = [np.mean([p[1] for p in line.points]) for line in result.lines]
        assert np.all(np.diff(sorted(heights)) == pytest.approx(LINE_SPACING, abs=3))

    def test_traced_lines_are_level_on_a_straight_page(self) -> None:
        result = trace_text_lines(straight_text_page(), float(PAGE_DPI))

        for line in result.lines:
            ys = np.array([p[1] for p in line.points])
            assert np.ptp(ys) < 8.0

    def test_traced_lines_follow_the_bow_of_a_curved_page(self) -> None:
        result = trace_text_lines(curved_text_page(bow=60.0), float(PAGE_DPI))

        assert len(result.lines) == EXPECTED_LINES
        for line in result.lines:
            xs = np.array([p[0] for p in line.points])
            ys = np.array([p[1] for p in line.points])
            middle = ys[np.argmin(np.abs(xs - PAGE_WIDTH / 2))]
            ends = 0.5 * (ys[0] + ys[-1])
            assert middle - ends > 35.0

    def test_vertical_bounds_enclose_the_text(self) -> None:
        bounds = trace_text_lines(straight_text_page(), float(PAGE_DPI)).vertical_bounds

        assert bounds is not None
        assert bounds.left_top[0] == pytest.approx(TEXT_LEFT, abs=6.0)
        assert bounds.left_top[0] == pytest.approx(bounds.left_bottom[0], abs=1.0)
        assert bounds.right_top[0] > PAGE_WIDTH - 40

    def test_lines_are_returned_in_source_pixels_for_other_resolutions(self) -> None:
        """A page at 400 dpi is reduced for tracing and mapped back."""
        page = straight_text_page()
        doubled = np.kron(page, np.ones((2, 2), dtype=np.uint8))

        result = trace_text_lines(doubled, 2.0 * PAGE_DPI)

        assert len(result.lines) == EXPECTED_LINES
        xs = [p[0] for p in result.lines[0].points]
        assert max(xs) > 1.8 * PAGE_WIDTH

    def test_content_rect_limits_the_search(self) -> None:
        page = straight_text_page()
        top_half = (0, 0, PAGE_WIDTH, PAGE_HEIGHT // 2)

        result = trace_text_lines(page, float(PAGE_DPI), top_half)

        assert 0 < len(result.lines) < EXPECTED_LINES
        for line in result.lines:
            assert max(p[1] for p in line.points) < PAGE_HEIGHT // 2 + 20

    def test_colour_input_gives_the_same_lines(self) -> None:
        page = straight_text_page()
        colour = np.dstack([page] * 3)

        assert len(trace_text_lines(colour, float(PAGE_DPI)).lines) == EXPECTED_LINES

    def test_blank_page_has_no_lines(self) -> None:
        blank = np.full((PAGE_HEIGHT, PAGE_WIDTH), BACKGROUND, dtype=np.uint8)

        assert trace_text_lines(blank, float(PAGE_DPI)).lines == []

    def test_dark_scanner_border_is_not_a_text_line(self) -> None:
        page = straight_text_page()
        page[:, :40] = 10
        page[:25, :] = 10

        result = trace_text_lines(page, float(PAGE_DPI))

        assert len(result.lines) == EXPECTED_LINES


class TestSpeed:
    """Tracing is made of whole-image operations only."""

    def test_a_page_is_traced_in_well_under_a_second(self) -> None:
        page = curved_text_page()
        trace_text_lines(page, float(PAGE_DPI))

        timings = []
        for _ in range(3):
            start = time.perf_counter()
            trace_text_lines(page, float(PAGE_DPI))
            timings.append(time.perf_counter() - start)

        assert min(timings) < 1.0
