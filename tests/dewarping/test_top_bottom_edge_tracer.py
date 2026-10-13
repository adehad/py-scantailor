"""Tracing the page's top and bottom edges between the content bounds.

Reference values marked "C++" come from compiling the upstream sources of
``TopBottomEdgeTracer`` (its priority queue, shortest path search and snakes)
with g++ 12 and running them on the same gradient.
"""

import cv2
import numpy as np
import pytest

import scantailor.dewarping.top_bottom_edge_tracer as edges
from scantailor.dewarping.grid_filters import gauss_blur
from scantailor.dewarping.line_geometry import grid_line_cells
from scantailor.dewarping.text_line_tracer import trace_text_lines
from scantailor.dewarping.top_bottom_edge_tracer import trace_top_bottom_edges
from tests.dewarping.book_pages import BentPage, bent_book_page, justified_text_page

CONTENT = (160, 200, 890, 1090)


@pytest.fixture(scope="module")
def page() -> BentPage:
    return bent_book_page(bow=45.0, squeeze=-25.0)


@pytest.fixture(scope="module")
def bounds(page: BentPage):
    result = trace_text_lines(page.image, 200.0, CONTENT).vertical_bounds
    assert result is not None
    return (result.first, result.second)


def _true_edge_y(edge: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return edge[:, 1] - np.interp(edge[:, 0], truth[:, 0], truth[:, 1])


class TestTracedEdges:
    def test_both_page_edges_are_found(self, page: BentPage, bounds) -> None:
        curves = trace_top_bottom_edges(page.image, bounds)

        assert len(curves) == 2
        heights = sorted(float(curve[:, 1].mean()) for curve in curves)
        assert heights[0] < 400 < 1100 < heights[1]

    def test_curves_follow_the_edges_of_the_bent_page_within_a_couple_of_pixels(
        self, page: BentPage, bounds
    ) -> None:
        curves = sorted(
            trace_top_bottom_edges(page.image, bounds), key=lambda c: c[:, 1].mean()
        )

        assert np.abs(_true_edge_y(curves[0], page.top)).max() < 3.0
        assert np.abs(_true_edge_y(curves[1], page.bottom)).max() < 3.0

    def test_curves_run_from_the_second_bound_towards_the_first(
        self, page: BentPage, bounds
    ) -> None:
        for curve in trace_top_bottom_edges(page.image, bounds):
            assert curve[0, 0] > curve[-1, 0]

    def test_curves_end_near_the_bounds(self, page: BentPage, bounds) -> None:
        for curve in trace_top_bottom_edges(page.image, bounds):
            assert curve[0, 0] == pytest.approx(1032.0, abs=25.0)
            assert curve[-1, 0] == pytest.approx(172.0, abs=25.0)

    def test_a_page_without_edges_between_the_bounds_gives_no_curves(self) -> None:
        flat, _ = justified_text_page()
        blank_ground = np.full((1600, 1200), 235, dtype=np.uint8)
        blank_ground[150:1450, 100:1100] = flat[:1300, :1000][:1300, :]
        bounds = (((172.0, 0.0), (172.0, 1600.0)), ((1032.0, 0.0), (1032.0, 1600.0)))

        assert trace_top_bottom_edges(blank_ground, bounds) == []

    def test_degenerate_bounds_give_no_curves(self, page: BentPage) -> None:
        point = ((200.0, 100.0), (200.0, 100.0))
        line = ((1000.0, 0.0), (1000.0, 1600.0))

        assert trace_top_bottom_edges(page.image, (point, line)) == []
        assert trace_top_bottom_edges(page.image, (line, point)) == []

    def test_bounds_that_miss_the_image_give_no_curves(self, page: BentPage) -> None:
        outside = ((5000.0, 0.0), (5000.0, 1600.0))
        inside = ((1000.0, 0.0), (1000.0, 1600.0))

        assert trace_top_bottom_edges(page.image, (outside, inside)) == []

    def test_a_large_image_is_traced_at_reduced_size_and_mapped_back(
        self, page: BentPage, bounds
    ) -> None:
        large = cv2.resize(
            page.image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR
        )
        scaled_bounds = tuple(
            tuple((1.5 * x, 1.5 * y) for x, y in line) for line in bounds
        )

        curves = sorted(
            trace_top_bottom_edges(large, scaled_bounds), key=lambda c: c[:, 1].mean()
        )

        assert len(curves) == 2
        assert np.abs(_true_edge_y(curves[0] / 1.5, page.top)).max() < 4.0
        assert curves[0][:, 0].max() > 1.5 * 1000

    def test_the_trace_is_repeatable(self, page: BentPage, bounds) -> None:
        first = trace_top_bottom_edges(page.image, bounds)
        second = trace_top_bottom_edges(page.image, bounds)

        for one, other in zip(first, second, strict=True):
            np.testing.assert_array_equal(one, other)


class TestSizes:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            ((2000, 1000), (1000, 500)),
            ((1000, 2000), (500, 1000)),
            ((1600, 1600), (1000, 1000)),
        ],
    )
    def test_a_size_is_fitted_into_a_thousand_pixel_square_keeping_its_ratio(
        self, size: tuple[int, int], expected: tuple[int, int]
    ) -> None:
        assert edges._fit_size(size[0], size[1], 1000) == expected


class TestShortestPaths:
    def test_cost_is_the_weakest_step_along_the_best_path(self) -> None:
        deriv = np.zeros((30, 60), dtype=np.float32)
        deriv[10, :] = 0.6
        deriv[10, 20] = 0.3
        starts = grid_line_cells(((0.0, 0.0), (0.0, 29.0)))

        cost, _ = edges._propagate_shortest_paths(deriv, starts, (1.0, 0.0))

        assert cost[1 + 10, 1 + 59] == pytest.approx(1 - 0.3)
        assert cost[1 + 3, 1 + 59] == pytest.approx(1.0)

    def test_start_cells_cost_nothing_and_padding_is_never_entered(self) -> None:
        deriv = np.full((10, 20), 0.5, dtype=np.float32)
        starts = grid_line_cells(((0.0, 0.0), (0.0, 9.0)))

        cost, previous = edges._propagate_shortest_paths(deriv, starts, (1.0, 0.0))

        assert np.all(cost[1:-1, 1] == 0.0)
        assert np.all(cost[0, :] == -1.0)
        assert np.all(cost[:, -1] == -1.0)
        assert np.all(previous[1:-1, 1] == -1)

    def test_paths_only_advance_towards_the_target_bound(self) -> None:
        deriv = np.full((10, 20), 0.5, dtype=np.float32)
        starts = grid_line_cells(((0.0, 0.0), (0.0, 9.0)))

        cost, _ = edges._propagate_shortest_paths(deriv, starts, (1.0, 0.0))

        assert np.all(cost[1:-1, 2:-1] == pytest.approx(0.5))

    def test_endpoints_are_the_best_ends_a_hundred_pixels_apart(self) -> None:
        cost = np.full((402, 12), 1.0, dtype=np.float32)
        cost[1 + 50, 1 + 5] = 0.2
        cost[1 + 400 - 100, 1 + 5] = 0.4
        cost[1 + 60, 1 + 5] = 0.1
        cells = [(5, y) for y in range(400)]

        endpoints = edges._locate_best_path_endpoints(cost, cells)

        assert endpoints == [(5, 60), (5, 300)]

    def test_ends_that_are_too_expensive_are_not_endpoints(self) -> None:
        cost = np.full((120, 12), 0.97, dtype=np.float32)
        cells = [(5, y) for y in range(100)]

        assert edges._locate_best_path_endpoints(cost, cells) == []


class TestSnakes:
    def test_knots_are_taken_about_fifteen_pixels_apart_and_include_the_end(
        self,
    ) -> None:
        previous = np.full((12, 102), -1, dtype=np.int8)
        previous[1:-1, 2:-1] = 3  # every cell came from its left neighbour

        knots = edges._path_to_snake(previous, (99, 4))

        assert tuple(knots[0]) == (99.0, 4.0)
        assert np.all(np.diff(knots[:, 0]) < 0)
        assert np.hypot(*np.diff(knots, axis=0).T).min() >= 7
        assert knots[-1, 0] <= 16


class TestAgainstCpp:
    """A formula image with strong page edges, run through the C++ search and snakes."""

    @staticmethod
    def _page() -> np.ndarray:
        height, width = 160, 220
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        top = 30 + 10 * np.sin(np.pi * xx / width)
        bottom = 130 - 8 * np.sin(np.pi * xx / width)
        gray = np.where((yy > top) & (yy < bottom), 235.0, 45.0)
        gray += 3 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
        return np.clip(gray, 0, 255).astype(np.uint8)

    def test_search_endpoints_and_snake_match_the_cpp_ones(self) -> None:
        page = self._page()
        first = ((40.0, 0.0), (41.0, 159.0))
        second = ((180.0, 0.0), (178.0, 159.0))
        direction = edges._average_unit_vector(first, second)
        deriv = edges._directional_derivative(page, direction)
        blurred = gauss_blur(np.abs(deriv), 2.0, 2.0)
        toward = edges._direction_from_point_to_line(
            (
                0.5 * (first[0][0] + first[1][0]),
                0.5 * (first[0][1] + first[1][1]),
            ),
            second,
        )

        cost, previous = edges._propagate_shortest_paths(
            deriv, grid_line_cells(first), toward
        )
        endpoints = edges._locate_best_path_endpoints(cost, grid_line_cells(second))
        assert endpoints == [(180, 34)]

        snake = edges._path_to_snake(previous, endpoints[0])
        assert len(snake) == 10
        down = edges._down_the_hill_direction(220, 160, snake, direction)
        valley = edges._run_snake(snake, blurred, down, uphill=False)
        up = edges._down_the_hill_direction(220, 160, valley, direction)
        ridge = edges._run_snake(valley, np.abs(deriv), (-up[0], -up[1]), uphill=True)

        np.testing.assert_allclose(ridge[0], [179.5, 36.0], atol=1e-3)
        np.testing.assert_allclose(ridge[5], [104.5, 39.5], atol=1e-3)
        np.testing.assert_allclose(ridge[-1], [44.5, 36.0], atol=1e-3)
