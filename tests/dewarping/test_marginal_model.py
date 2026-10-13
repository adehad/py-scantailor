"""Marginal dewarping model: curves bend to the page edges near the spine.

The reference polyline marked "C++" comes from compiling upstream's ``XSpline``
with g++ 12 and sampling the same control points.
"""

import numpy as np
import pytest

from scantailor.dewarping.marginal_model import (
    _bottom_margin,
    _control_point_xs,
    _curve,
    _top_margin,
    build_marginal_model,
)

HEIGHT, WIDTH = 300, 400
CONTENT = (50.0, 60.0, 300.0, 180.0)
PAPER = 220
BACKGROUND = 10


def _page_with_curled_edge(spine_on_right: bool) -> np.ndarray:
    """Dark background; the paper's top and bottom edges recede near the spine."""
    page = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    for x in range(WIDTH):
        distance = (WIDTH - x) if spine_on_right else x
        recess = max(0, 120 - distance) // 4
        page[20 + recess : HEIGHT - 20 - recess, x] = PAPER
    return page


class TestMarginalModel:
    """Models from a synthetic page."""

    def test_curves_keep_the_content_ends_and_bend_toward_the_edges_at_the_spine(
        self,
    ) -> None:
        model = build_marginal_model(
            _page_with_curled_edge(True), CONTENT, spine_on_right=True
        )

        assert model is not None
        assert model.top_curve is not None and model.bottom_curve is not None
        top = model.top_curve.polyline
        bottom = model.bottom_curve.polyline
        assert top[0, 0] == pytest.approx(CONTENT[0])
        assert top[-1, 0] == pytest.approx(CONTENT[0] + CONTENT[2])
        # Away from the spine the edge is the paper's top edge at row 20.
        assert top[0, 1] == pytest.approx(20.0, abs=1.0)
        assert bottom[0, 1] == pytest.approx(HEIGHT - 21.0, abs=1.0)
        # At the spine end the paper has receded by about 17 px.
        assert top[-1, 1] > top[0, 1] + 10.0
        assert bottom[-1, 1] < bottom[0, 1] - 10.0

    def test_left_page_binding_bends_the_other_end(self) -> None:
        model = build_marginal_model(
            _page_with_curled_edge(False), CONTENT, spine_on_right=False
        )

        assert model is not None and model.top_curve is not None
        top = model.top_curve.polyline
        assert top[0, 1] > top[-1, 1] + 10.0

    def test_flat_page_gives_straight_curves(self) -> None:
        page = np.full((HEIGHT, WIDTH), PAPER, dtype=np.uint8)

        model = build_marginal_model(page, CONTENT, spine_on_right=True)

        assert model is not None
        assert model.top_curve is not None and model.bottom_curve is not None
        assert np.ptp(model.top_curve.polyline[:, 1]) == pytest.approx(0.0, abs=1e-6)
        assert np.ptp(model.bottom_curve.polyline[:, 1]) == pytest.approx(0.0, abs=1e-6)

    def test_empty_content_rect_gives_no_model(self) -> None:
        page = np.full((HEIGHT, WIDTH), PAPER, dtype=np.uint8)

        assert build_marginal_model(page, (10.0, 10.0, 0.0, 50.0), True) is None

    def test_curves_are_monotone_in_x_and_stay_within_the_paper(self) -> None:
        model = build_marginal_model(
            _page_with_curled_edge(True), CONTENT, spine_on_right=True
        )

        assert model is not None and model.top_curve is not None
        top = model.top_curve.polyline
        assert np.all(np.diff(top[:, 0]) > 0)
        assert top[:, 1].min() >= 19.0


class TestBottomCurveKeepsToTheBottomEdge:
    """Its inner control points start on the content's bottom edge, not its top."""

    def test_a_page_without_paper_leaves_the_bottom_curve_on_the_content_bottom(
        self,
    ) -> None:
        dark = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)

        model = build_marginal_model(dark, CONTENT, spine_on_right=True)

        assert model is not None and model.bottom_curve is not None
        bottom = model.bottom_curve.polyline
        np.testing.assert_allclose(bottom[:, 1], CONTENT[1] + CONTENT[3])

    def test_a_search_for_the_bottom_margin_never_reaches_above_the_start(self) -> None:
        paper = np.zeros((100, 10), dtype=bool)
        paper[:30, :] = True  # paper only above the start height
        columns = np.array([2.0, 5.0])
        starts = np.array([60.0, 60.0])

        moved = _bottom_margin(paper, columns, starts)

        np.testing.assert_array_equal(moved, starts)


class TestMarginSearch:
    def _page(self) -> np.ndarray:
        paper = np.zeros((100, 10), dtype=bool)
        paper[20:80, :] = True
        return paper

    def test_the_top_margin_is_the_first_run_of_paper_from_the_top(self) -> None:
        moved = _top_margin(self._page(), np.array([3.0]), np.array([50.0]))

        assert moved[0] == 20.0

    def test_the_bottom_margin_is_the_first_run_of_paper_from_the_bottom(self) -> None:
        moved = _bottom_margin(self._page(), np.array([3.0]), np.array([50.0]))

        assert moved[0] == 79.0

    def test_a_run_shorter_than_sixteen_pixels_is_not_a_margin(self) -> None:
        paper = np.zeros((100, 10), dtype=bool)
        paper[20:30, :] = True  # ten rows of paper, then dark again
        paper[40:80, :] = True

        moved = _top_margin(paper, np.array([3.0]), np.array([60.0]))

        assert moved[0] == 40.0

    def test_points_outside_the_image_are_left_where_they_are(self) -> None:
        moved = _top_margin(
            self._page(), np.array([-5.0, 50.0]), np.array([50.0, 50.0])
        )

        np.testing.assert_array_equal(moved, [50.0, 50.0])

    def test_a_run_cut_off_by_the_image_edge_does_not_count(self) -> None:
        paper = np.zeros((100, 10), dtype=bool)
        paper[90:100, :] = True  # only ten rows of paper at the very bottom

        moved = _top_margin(paper, np.array([3.0]), np.array([99.0]))

        assert moved[0] == 99.0


class TestControlPoints:
    def test_there_are_two_ends_and_five_points_next_to_the_binding(self) -> None:
        right = _control_point_xs(100.0, 290.0, spine_on_right=True)
        left = _control_point_xs(100.0, 290.0, spine_on_right=False)

        assert len(right) == len(left) == 7
        assert (right[0], right[-1]) == (100.0, 390.0)
        np.testing.assert_allclose(right[1:-1], 100 + 290 * np.arange(24, 29) / 29)
        np.testing.assert_allclose(left[1:-1], 100 + 290 * np.arange(1, 6) / 29)

    def test_a_point_on_a_whole_column_is_reached_from_below(self) -> None:
        """145 * (25 / 29) is 124.99999999999999, so the point reads column 124."""
        xs = _control_point_xs(0.0, 145.0, spine_on_right=True)

        assert xs[2] < 125.0
        assert int(xs[2]) == 124


class TestSpline:
    def test_the_curve_through_seven_control_points_matches_the_cpp_polyline(
        self,
    ) -> None:
        xs = np.linspace(150.0, 1050.0, 7)
        ys = np.array([155.0, 160.0, 170.0, 175.0, 168.0, 158.0, 155.0])

        polyline = _curve(xs, ys).polyline

        assert len(polyline) == 16
        np.testing.assert_allclose(polyline[0], [150.0, 155.0], atol=1e-9)
        np.testing.assert_allclose(polyline[8], [635.625, 172.42462], atol=1e-4)
        np.testing.assert_allclose(polyline[-1], [1050.0, 155.0], atol=1e-9)
