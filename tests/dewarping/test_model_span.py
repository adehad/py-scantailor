"""Automatic dewarping refuses a model from two close curves and says what it found.

Upstream stretches whatever pair of curves it keeps over the whole page, so two
neighbouring lines turn a wobble of a pixel into an error of dozens of pixels at
the far end (issue 226). Here a model whose curves span less than a share of the
content height is refused, and the metrics behind every decision come back with
the result.
"""

import numpy as np
import pytest

from scantailor.dewarping import DEFAULT_MIN_SPAN_PERCENT
from scantailor.dewarping.auto_model import (
    DewarpMetrics,
    ModelOutcome,
    build_auto_distortion_model,
    detect_distortion_model,
)
from scantailor.dewarping.distortion_model_builder import DistortionModelBuilder
from scantailor.dewarping.text_line_tracer import trace_text_lines
from tests.dewarping.book_pages import (
    DEFAULT_PAPER,
    BentPage,
    bend_page,
    bent_book_page,
    justified_text_page,
    sagging_edges,
)

DPI = 200
CONTENT = (160.0, 200.0, 890.0, 1090.0)
LEFT = ((100.0, 0.0), (100.0, 1000.0))
RIGHT = ((900.0, 0.0), (900.0, 1000.0))


def _few_lines_page(keep: int = 3, dark_ground: bool = False) -> BentPage:
    """A bent page with only its first ``keep`` lines of text, the rest blank.

    The tracer's filters drop the first line, so three lines leave the two
    neighbouring ones below it. Without a dark ground around the paper it finds
    no page edge either, so those two lines are the only candidate curves.
    """
    flat, baselines = justified_text_page(seed=5)
    for baseline in baselines[keep:]:
        flat[baseline - 32 : baseline + 12, :] = DEFAULT_PAPER
    top, bottom = sagging_edges(100.0, 1100.0, 140.0, 1450.0, 45.0, -25.0)
    ground = 45 if dark_ground else DEFAULT_PAPER
    return bend_page(flat, baselines, 70, top, bottom, (1200, 1600), ground)


@pytest.fixture(scope="module")
def close_lines() -> BentPage:
    return _few_lines_page()


@pytest.fixture(scope="module")
def full_page() -> BentPage:
    return bent_book_page(45.0, -25.0)


def _flat_curve(y: float) -> np.ndarray:
    xs = np.linspace(110.0, 890.0, 30)
    return np.column_stack([xs, np.full(30, y)])


class TestBuilderReportsItsChoice:
    def test_the_chosen_curves_are_named_by_the_order_they_were_added_in(
        self,
    ) -> None:
        builder = DistortionModelBuilder((0.0, 1.0))
        builder.set_vertical_bounds(LEFT, RIGHT)
        builder.add_horizontal_curve(_flat_curve(800.0))
        builder.add_horizontal_curve(_flat_curve(200.0))

        built = builder.build()

        assert built.model.is_valid()
        assert (built.top_index, built.bottom_index) == (1, 0)

    def test_the_span_is_the_distance_between_the_chosen_curves(self) -> None:
        builder = DistortionModelBuilder((0.0, 1.0))
        builder.set_vertical_bounds(LEFT, RIGHT)
        builder.add_horizontal_curve(_flat_curve(200.0))
        builder.add_horizontal_curve(_flat_curve(800.0))

        assert builder.build().span == pytest.approx(600.0, abs=1.0)

    def test_the_span_keeps_the_original_pixels_through_a_transformation(
        self,
    ) -> None:
        builder = DistortionModelBuilder((0.0, 1.0))
        builder.set_vertical_bounds(LEFT, RIGHT)
        builder.add_horizontal_curve(_flat_curve(200.0))
        builder.add_horizontal_curve(_flat_curve(800.0))
        builder.transform(np.array([[3.0, 0.0, 20.0], [0.0, 3.0, -10.0]]))

        assert builder.build().span == pytest.approx(600.0, abs=1.0)

    def test_no_model_means_no_choice_and_no_span(self) -> None:
        builder = DistortionModelBuilder((0.0, 1.0))
        builder.set_vertical_bounds(LEFT, RIGHT)
        builder.add_horizontal_curve(_flat_curve(200.0))

        built = builder.build()

        assert not built.model.is_valid()
        assert (built.top_index, built.bottom_index, built.span) == (None, None, 0.0)


class TestTracerCounts:
    def test_the_tracer_counts_the_lines_it_found_before_filtering(
        self, full_page: BentPage
    ) -> None:
        result = trace_text_lines(full_page.image, float(DPI), CONTENT)

        assert result.traced_count >= len(result.lines) >= 10

    def test_the_count_includes_the_lines_the_filters_dropped(
        self, close_lines: BentPage
    ) -> None:
        result = trace_text_lines(close_lines.image, float(DPI), CONTENT)

        assert result.traced_count == 3
        assert len(result.lines) == 2


class TestRefusingACloseModel:
    def test_two_close_lines_are_refused_by_default(
        self, close_lines: BentPage
    ) -> None:
        detected = detect_distortion_model(close_lines.image, DPI, CONTENT)

        metrics = detected.metrics
        assert detected.model is None
        assert metrics.outcome is ModelOutcome.SPAN_TOO_SMALL
        assert metrics.edges_traced == 0
        assert metrics.lines_kept == 2
        assert metrics.span_percent is not None
        assert 0.0 < metrics.span_percent < DEFAULT_MIN_SPAN_PERCENT
        assert (metrics.top_from_edge, metrics.bottom_from_edge) == (False, False)

    def test_a_minimum_of_zero_accepts_them_as_upstream_does(
        self, close_lines: BentPage
    ) -> None:
        detected = detect_distortion_model(
            close_lines.image, DPI, CONTENT, min_span_percent=0
        )

        assert detected.model is not None and detected.model.is_valid()
        assert detected.metrics.outcome is ModelOutcome.BUILT

    def test_the_minimum_is_compared_with_the_span_itself(
        self, close_lines: BentPage
    ) -> None:
        span = detect_distortion_model(
            close_lines.image, DPI, CONTENT, min_span_percent=0
        ).metrics.span_percent
        assert span is not None

        below = detect_distortion_model(
            close_lines.image, DPI, CONTENT, min_span_percent=span - 0.5
        )
        above = detect_distortion_model(
            close_lines.image, DPI, CONTENT, min_span_percent=span + 0.5
        )

        assert below.model is not None
        assert above.model is None

    def test_the_page_edges_rescue_a_page_of_few_lines(self) -> None:
        page = _few_lines_page(keep=2, dark_ground=True)

        detected = detect_distortion_model(page.image, DPI, CONTENT)

        metrics = detected.metrics
        assert detected.model is not None
        assert metrics.outcome is ModelOutcome.BUILT
        assert metrics.edges_traced == 2
        assert metrics.span_percent is not None and metrics.span_percent > 100.0
        assert metrics.bottom_from_edge is True

    def test_the_convenience_function_refuses_the_same_models(
        self, close_lines: BentPage
    ) -> None:
        assert build_auto_distortion_model(close_lines.image, DPI, CONTENT) is None
        assert (
            build_auto_distortion_model(
                close_lines.image, DPI, CONTENT, min_span_percent=0
            )
            is not None
        )


class TestMetricsOfOtherPages:
    def test_a_page_of_full_lines_spans_most_of_its_content(
        self, full_page: BentPage
    ) -> None:
        metrics = detect_distortion_model(full_page.image, DPI, CONTENT).metrics

        assert metrics.outcome is ModelOutcome.BUILT
        assert metrics.lines_traced >= metrics.lines_kept >= 10
        assert metrics.edges_traced == 2
        assert metrics.span_percent is not None and metrics.span_percent > 90.0
        assert metrics.min_span_percent == DEFAULT_MIN_SPAN_PERCENT

    def test_a_blank_page_reports_no_text(self) -> None:
        blank = np.full((1600, 1200), 235, dtype=np.uint8)

        detected = detect_distortion_model(blank, DPI, CONTENT)

        assert detected.model is None
        assert detected.metrics.outcome is ModelOutcome.NO_TEXT
        assert detected.metrics.span_percent is None

    def test_one_line_and_no_edge_report_too_few_curves(self) -> None:
        page = _few_lines_page(keep=2)

        detected = detect_distortion_model(page.image, DPI, CONTENT)

        assert detected.model is None
        assert detected.metrics.outcome is ModelOutcome.TOO_FEW_CURVES
        assert detected.metrics.lines_kept == 1
        assert detected.metrics.edges_traced == 0


class TestDescribing:
    @staticmethod
    def _metrics(outcome: ModelOutcome, **fields: object) -> DewarpMetrics:
        values: dict = {
            "lines_traced": 33,
            "lines_kept": 2,
            "edges_traced": 0,
            "outcome": outcome,
            "min_span_percent": 40.0,
        }
        values.update(fields)
        return DewarpMetrics(**values)

    def test_a_refused_model_says_why_and_what_would_change_it(self) -> None:
        text = self._metrics(ModelOutcome.SPAN_TOO_SMALL, span_percent=7.4).describe()

        assert text == (
            "2 of 33 lines kept, no page edge. Curves span 7% of the text. Below "
            "the 40% minimum, so the page is left as is; a minimum of 7% or less "
            "would use them."
        )

    def test_a_built_model_names_where_its_curves_came_from(self) -> None:
        text = self._metrics(
            ModelOutcome.BUILT,
            lines_kept=12,
            edges_traced=1,
            span_percent=97.2,
            top_from_edge=False,
            bottom_from_edge=True,
        ).describe()

        assert text == (
            "12 of 33 lines kept, 1 page edge. Curves span 97% of the text, top "
            "from a text line, bottom from the page edge."
        )

    def test_two_edges_are_plural(self) -> None:
        text = self._metrics(
            ModelOutcome.BUILT,
            edges_traced=2,
            span_percent=104.0,
            top_from_edge=True,
            bottom_from_edge=True,
        ).describe()

        assert text.startswith("2 of 33 lines kept, 2 page edges.")

    def test_a_page_without_text_or_edges_says_so(self) -> None:
        text = self._metrics(ModelOutcome.NO_TEXT, lines_traced=0, lines_kept=0)

        assert text.describe() == (
            "No text lines or page edges found, so the page is left as is."
        )

    def test_too_few_curves_and_no_pair_are_told_apart(self) -> None:
        few = self._metrics(ModelOutcome.TOO_FEW_CURVES, lines_kept=1).describe()
        none = self._metrics(ModelOutcome.NO_PAIR, lines_kept=3).describe()

        assert few == (
            "1 of 33 lines kept, no page edge: fewer than two curves, so the page "
            "is left as is."
        )
        assert none == (
            "3 of 33 lines kept, no page edge, but no pair of curves gives a usable "
            "model, so the page is left as is."
        )
