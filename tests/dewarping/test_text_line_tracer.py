"""Text line tracing on generated pages, and the steps it is made of.

Reference traces marked "C++" come from compiling the upstream
``TowardsLineTracer`` with g++ 12 and running it on the same maps.
"""

import numpy as np
import pytest

import scantailor.dewarping.text_line_tracer as tracer
from scantailor.dewarping.text_line_tracer import (
    TextLineResult,
    trace_text_lines,
)
from tests.dewarping.book_pages import BentPage, bent_book_page
from tests.dewarping.kernel_reference import recorded

DPI = 200
CONTENT = (160, 200, 890, 1090)


@pytest.fixture(scope="module")
def flat_page() -> BentPage:
    return bent_book_page(bow=0.0, squeeze=0.0)


@pytest.fixture(scope="module")
def bent_page() -> BentPage:
    return bent_book_page(bow=45.0, squeeze=-25.0)


@pytest.fixture(scope="module")
def bent_result(bent_page: BentPage) -> TextLineResult:
    return trace_text_lines(bent_page.image, float(DPI), CONTENT)


class TestTracedLines:
    def test_most_text_lines_of_a_flat_page_are_found_and_each_is_level(
        self, flat_page: BentPage
    ) -> None:
        result = trace_text_lines(flat_page.image, float(DPI), CONTENT)

        assert len(result.lines) >= 0.6 * len(flat_page.baselines)
        for line in result.lines:
            assert np.abs(line[:, 1] - np.median(line[:, 1])).mean() < 2.0

    def test_traced_lines_span_most_of_the_distance_between_the_bounds(
        self, bent_result: TextLineResult
    ) -> None:
        assert bent_result.vertical_bounds is not None
        for line in bent_result.lines:
            assert line[-1, 0] - line[0, 0] > 0.7 * 860

    def test_traced_lines_of_a_bent_page_are_straight_once_the_page_is_flat(
        self, bent_page: BentPage, bent_result: TextLineResult
    ) -> None:
        assert len(bent_result.lines) >= 10
        for line in bent_result.lines:
            heights = bent_page.warped_to_flat_height(line)

            assert np.ptp(heights) < 4.0

    def test_traced_lines_run_through_distinct_text_lines(
        self, bent_page: BentPage, bent_result: TextLineResult
    ) -> None:
        heights = sorted(
            float(np.median(bent_page.warped_to_flat_height(line)))
            for line in bent_result.lines
        )

        assert np.min(np.diff(heights)) > 30.0

    def test_vertical_bounds_enclose_the_text_in_input_pixels(
        self, bent_result: TextLineResult
    ) -> None:
        bounds = bent_result.vertical_bounds
        assert bounds is not None

        def x_at(line, y: float) -> float:
            (x1, y1), (x2, y2) = line
            return x1 + (y - y1) * (x2 - x1) / (y2 - y1)

        # The text spans flat x 70 to 930, which is warped x of about 170 to 1030.
        for y in (300, 800, 1300):
            assert x_at(bounds.first, y) == pytest.approx(172.0, abs=12.0)
            assert x_at(bounds.second, y) == pytest.approx(1032.0, abs=12.0)

    def test_the_content_rect_decides_where_the_bounds_lie(
        self, flat_page: BentPage
    ) -> None:
        narrow = (400, 200, 500, 1090)

        bounds = trace_text_lines(flat_page.image, float(DPI), narrow).vertical_bounds

        assert bounds is not None
        assert bounds.first[0][0] == pytest.approx(399.0, abs=15.0)
        assert bounds.second[0][0] == pytest.approx(900.0, abs=15.0)

    def test_colour_input_gives_the_same_lines(self, flat_page: BentPage) -> None:
        colour = np.dstack([flat_page.image] * 3)

        gray_lines = trace_text_lines(flat_page.image, float(DPI), CONTENT).lines
        colour_lines = trace_text_lines(colour, float(DPI), CONTENT).lines

        assert len(colour_lines) == len(gray_lines)

    def test_a_blank_page_has_no_lines_but_still_has_bounds(self) -> None:
        blank = np.full((1600, 1200), 235, dtype=np.uint8)

        result = trace_text_lines(blank, float(DPI), CONTENT)

        assert result.lines == []
        assert result.vertical_bounds is not None

    def test_ink_touching_the_image_border_does_not_move_the_bounds(
        self, flat_page: BentPage
    ) -> None:
        page = flat_page.image.copy()
        page[:, :60] = 10
        page[:40, :] = 10

        with_border = trace_text_lines(page, float(DPI), CONTENT).vertical_bounds
        without = trace_text_lines(flat_page.image, float(DPI), CONTENT).vertical_bounds

        assert with_border is not None and without is not None
        np.testing.assert_allclose(
            np.array(with_border.first), np.array(without.first), atol=1.0
        )
        np.testing.assert_allclose(
            np.array(with_border.second), np.array(without.second), atol=1.0
        )

    def test_a_page_at_another_resolution_is_traced_in_its_own_pixels(
        self, flat_page: BentPage
    ) -> None:
        doubled = np.kron(flat_page.image, np.ones((2, 2), dtype=np.uint8))
        content = tuple(2 * value for value in CONTENT)

        result = trace_text_lines(doubled, 2.0 * DPI, content)

        assert len(result.lines) >= 0.6 * len(flat_page.baselines)
        assert max(line[:, 0].max() for line in result.lines) > 1.8 * 960
        assert max(line[:, 1].max() for line in result.lines) < 2 * 1500

    def test_the_resolution_may_be_given_as_a_pair(self, flat_page: BentPage) -> None:
        result = trace_text_lines(flat_page.image, (float(DPI), float(DPI)), CONTENT)

        assert len(result.lines) >= 0.6 * len(flat_page.baselines)

    def test_a_page_at_nearly_200_dpi_is_not_resampled(
        self, flat_page: BentPage
    ) -> None:
        near = trace_text_lines(flat_page.image, 210.0, CONTENT)
        exact = trace_text_lines(flat_page.image, 200.0, CONTENT)

        assert len(near.lines) == len(exact.lines)
        np.testing.assert_allclose(near.lines[0], exact.lines[0])

    def test_tracing_twice_gives_the_same_lines(self, flat_page: BentPage) -> None:
        first = trace_text_lines(flat_page.image, float(DPI), CONTENT)
        second = trace_text_lines(flat_page.image, float(DPI), CONTENT)

        assert len(first.lines) == len(second.lines)
        for one, other in zip(first.lines, second.lines, strict=True):
            np.testing.assert_array_equal(one, other)


class TestCurvatureFilter:
    XS = np.linspace(0, 400, 21)

    def test_a_straight_line_is_consistent(self) -> None:
        assert tracer._is_curvature_consistent(np.column_stack([self.XS, np.zeros(21)]))

    def test_a_line_bowing_one_way_is_consistent(self) -> None:
        bowed = np.column_stack([self.XS, 30 * np.sin(np.pi * self.XS / 400)])

        assert tracer._is_curvature_consistent(bowed)

    def test_an_s_shaped_line_is_not(self) -> None:
        wave = np.column_stack([self.XS, 40 * np.sin(2 * np.pi * self.XS / 400)])

        assert not tracer._is_curvature_consistent(wave)

    def test_a_zigzag_is_not(self) -> None:
        ys = np.where(np.arange(21) % 2 == 0, 0.0, 20.0)

        assert not tracer._is_curvature_consistent(np.column_stack([self.XS, ys]))

    def test_two_points_are_consistent_and_one_is_rejected(self) -> None:
        assert tracer._is_curvature_consistent(np.array([[0.0, 0.0], [5.0, 5.0]]))
        assert not tracer._is_curvature_consistent(np.array([[0.0, 0.0]]))


class TestBoundFilters:
    BOUNDS = (((100.0, 0.0), (100.0, 1000.0)), ((900.0, 0.0), (900.0, 1000.0)))

    def test_a_line_that_stops_far_short_of_the_bounds_is_dropped(self) -> None:
        short = np.array([[400.0, 300.0], [800.0, 300.0]])
        long = np.array([[110.0, 300.0], [890.0, 300.0]])

        kept = tracer._filter_short_curves([short, long], self.BOUNDS)

        assert len(kept) == 1
        np.testing.assert_array_equal(kept[0], long)

    def test_a_line_outside_the_bounds_at_both_ends_is_dropped(self) -> None:
        outside = np.array([[10.0, 300.0], [60.0, 300.0]])
        inside = np.array([[110.0, 300.0], [890.0, 300.0]])
        straddling = np.array([[50.0, 300.0], [500.0, 300.0]])

        kept = tracer._filter_out_of_bounds_curves(
            [outside, inside, straddling], self.BOUNDS
        )

        assert len(kept) == 2


class TestSanitising:
    def _page(self) -> np.ndarray:
        ink = np.zeros((60, 80), dtype=bool)
        ink[0:10, 0:5] = True
        ink[20:24, 20:40] = True
        ink[40, 40] = True
        ink[30:32, 60:62] = True
        ink[30:33, 66:68] = True
        return ink

    def test_components_touching_the_border_are_removed(self) -> None:
        cleaned = tracer._sanitize_binary_image(self._page(), (0, 0, 80, 60))

        assert not cleaned[0:10, 0:5].any()

    def test_text_like_ink_is_kept(self) -> None:
        cleaned = tracer._sanitize_binary_image(self._page(), (0, 0, 80, 60))

        assert cleaned[20:24, 20:40].all()

    def test_specks_and_two_by_two_blocks_are_removed(self) -> None:
        cleaned = tracer._sanitize_binary_image(self._page(), (0, 0, 80, 60))

        assert not cleaned[40, 40]
        assert not cleaned[30:32, 60:62].any()

    def test_a_block_of_two_by_three_is_the_smallest_thing_kept(self) -> None:
        cleaned = tracer._sanitize_binary_image(self._page(), (0, 0, 80, 60))

        assert cleaned[30:33, 66:68].all()

    def test_everything_outside_the_content_rect_is_cleared(self) -> None:
        cleaned = tracer._sanitize_binary_image(self._page(), (0, 0, 30, 60))

        assert cleaned[20:24, 20:30].all()
        assert not cleaned[20:24, 30:40].any()


class TestHelpers:
    def test_distance_map_holds_squared_distances_to_background_and_border(
        self,
    ) -> None:
        mask = np.zeros((7, 9), dtype=bool)
        mask[2:5, 2:7] = True

        sedm = tracer._squared_distance_map(mask)

        assert sedm[3, 4] == 4
        assert sedm[2, 2] == 1
        assert sedm[0, 0] == 0

    def test_the_image_border_counts_as_background(self) -> None:
        np.testing.assert_array_equal(
            tracer._squared_distance_map(np.ones((3, 3), dtype=bool)),
            [[1, 1, 1], [1, 4, 1], [1, 1, 1]],
        )

    def test_rectangles_are_scaled_with_position_and_size_rounded_separately(
        self,
    ) -> None:
        assert tracer._scale_rect((100, 50, 500, 300), 0.5, 0.5) == (50, 25, 250, 150)
        assert tracer._scale_rect((3, 3, 5, 5), 0.5, 0.5) == (2, 2, 3, 3)

    def test_mid_line_of_parallel_bounds_runs_half_way_between(self) -> None:
        line = tracer._mid_line(((0.0, 0.0), (0.0, 10.0)), ((10.0, 0.0), (10.0, 10.0)))

        assert line == ((5.0, 0.0), (5.0, 10.0))

    def test_mid_line_of_crossing_bounds_starts_at_the_crossing(self) -> None:
        line = tracer._mid_line(((0.0, 0.0), (10.0, 10.0)), ((10.0, 0.0), (0.0, 10.0)))

        assert line[0] == pytest.approx((5.0, 5.0))
        assert line[1][0] == pytest.approx(5.0)

    def test_seeds_lie_where_the_distance_map_peaks_along_the_middle_line(self) -> None:
        sedm = np.zeros((20, 40), dtype=np.int64)
        sedm[8:12, :] = np.array([[1], [4], [4], [1]])
        sedm[:, 10] = 0
        sedm[:, 30] = 0

        seeds = tracer._find_mid_line_seeds(sedm, ((0.0, 9.0), (39.0, 9.0)))

        assert seeds
        assert all(5 <= x <= 35 for x, _ in seeds)


class TestTowardsLineTracer:
    """Compared with the C++ tracer on a ridge map built from formulas."""

    @staticmethod
    def _maps() -> tuple[np.ndarray, np.ndarray]:
        height, width = 60, 160
        yy, xx = np.mgrid[0:height, 0:width].astype(float)
        mask = np.abs(yy - (height / 2 + 6 * np.sin(2 * np.pi * xx / width * 1.5))) < (
            5 + 2 * np.cos(xx / 17)
        )
        ridge = (3.0 + 2.0 * np.sin(xx / 9) * np.cos(yy / 7)).astype(np.float32)
        return tracer._squared_distance_map(mask), ridge

    @pytest.mark.parametrize(
        ("seed", "expected"),
        [
            ((40, 30), [(31, 36), (20, 37), (10, 35), (3, 34)]),
            (
                (100, 31),
                [
                    (92, 24),
                    (82, 23),
                    (72, 25),
                    (62, 28),
                    (52, 30),
                    (42, 34),
                    (32, 36),
                    (22, 37),
                    (12, 34),
                    (3, 34),
                ],
            ),
            ((150, 29), []),
            ((80, 12), []),
        ],
    )
    def test_trace_matches_the_cpp_one(
        self, seed: tuple[int, int], expected: list
    ) -> None:
        sedm, ridge = self._maps()
        line = np.array([[3.0, 0.0], [2.0, 60.0]])

        cells = tracer._trace_towards_line(sedm, ridge, line, seed[0], seed[1], 10.0)

        assert [tuple(map(int, cell)) for cell in cells] == expected

    def test_the_trace_ends_on_the_far_side_of_the_line_it_heads_for(self) -> None:
        sedm, ridge = self._maps()
        line = np.array([[155.0, 0.0], [156.0, 60.0]])

        cells = tracer._trace_towards_line(sedm, ridge, line, 100, 31, 10.0)

        assert len(cells) > 0
        assert cells[-1][0] > 140

    def test_a_ridge_map_of_negative_values_still_leads_the_trace_on(self) -> None:
        sedm, _ = self._maps()
        ridge = np.full(sedm.shape, -3.0, dtype=np.float32)
        line = np.array([[3.0, 0.0], [2.0, 60.0]])

        cells = tracer._trace_towards_line(sedm, ridge, line, 100, 31, 10.0)

        assert len(cells) > 0


def _brute_force_squared_distances(mask: np.ndarray) -> np.ndarray:
    """Squared distance of each ink pixel to the nearest background or border."""
    padded = np.pad(mask, 1, constant_values=False)
    background = np.argwhere(~padded)
    expected = np.zeros(mask.shape, dtype=np.int64)
    for y, x in np.argwhere(mask):
        offsets = background - (y + 1, x + 1)
        expected[y, x] = (offsets**2).sum(axis=1).min()
    return expected


class TestSquaredDistanceMapExactness:
    @pytest.mark.parametrize("seed", range(4))
    def test_it_matches_a_search_over_every_background_pixel(self, seed: int) -> None:
        mask = np.random.default_rng(seed).random((23, 31)) < 0.85

        sedm = tracer._squared_distance_map(mask)

        np.testing.assert_array_equal(sedm, _brute_force_squared_distances(mask))
        assert sedm.dtype == np.int64

    def test_distances_deep_inside_a_large_blob_are_exact(self) -> None:
        mask = np.zeros((420, 640), dtype=bool)
        mask[10:410, 20:620] = True
        yy, xx = np.mgrid[0:420, 0:640]
        nearest_edge = np.minimum.reduce([yy - 9, 410 - yy, xx - 19, 620 - xx])

        sedm = tracer._squared_distance_map(mask)

        np.testing.assert_array_equal(sedm, np.where(mask, nearest_edge**2, 0))


class TestRecordedWalks:
    """Ridge walks on formula maps follow the recorded cells."""

    @staticmethod
    def _ridge() -> np.ndarray:
        yy, xx = np.mgrid[0:90, 0:240]
        return (((xx * 5 + yy * 3) % 29) - 14).astype(np.float32) / np.float32(8.0)

    def test_the_distance_map_of_the_walk_mask_is_as_recorded(self) -> None:
        np.testing.assert_array_equal(
            tracer._squared_distance_map(recorded("walk_mask")),
            recorded("walk_sedm"),
        )

    @pytest.mark.parametrize("case", range(5))
    def test_a_walk_reports_the_recorded_cells(self, case: int) -> None:
        seed_x, seed_y = recorded(f"walk{case}_seed").tolist()

        cells = tracer._trace_towards_line(
            recorded("walk_sedm"),
            self._ridge(),
            recorded(f"walk{case}_line"),
            seed_x,
            seed_y,
            10.0,
        )

        np.testing.assert_array_equal(cells, recorded(f"walk{case}_cells"))
        assert cells.dtype == np.int64
        assert cells.shape[1] == 2


class TestWalksAlongTheBorder:
    """A walk on the first row skips the neighbours outside the image."""

    @staticmethod
    def _maps() -> tuple[np.ndarray, np.ndarray]:
        height, width = 40, 120
        mask = np.zeros((height, width), dtype=bool)
        mask[:2] = True
        yy, xx = np.mgrid[0:height, 0:width]
        ridge = (((xx * 7 + yy * 5) % 23) - 11).astype(np.float32) / np.float32(4.0)
        ridge[0] += 10
        return tracer._squared_distance_map(mask), ridge

    @pytest.mark.parametrize(("line_x", "seed"), [(5.0, (100, 0)), (115.0, (10, 1))])
    def test_it_matches_the_walk_on_a_map_padded_with_cells_it_never_enters(
        self, line_x: float, seed: tuple[int, int]
    ) -> None:
        sedm, ridge = self._maps()
        line = np.array([[line_x, 0.0], [line_x, 40.0]])
        pad = 2
        # Zero distance and the lowest ridge value never win a step.
        padded_sedm = np.pad(sedm, pad)
        padded_ridge = np.pad(ridge, pad, constant_values=np.finfo(np.float32).min)

        cells = tracer._trace_towards_line(sedm, ridge, line, *seed, 10.0)
        padded_cells = tracer._trace_towards_line(
            padded_sedm, padded_ridge, line + pad, seed[0] + pad, seed[1] + pad, 10.0
        )

        assert (cells[:, 1] == 0).all()
        assert abs(int(cells[-1, 0]) - line_x) < 1
        np.testing.assert_array_equal(cells, padded_cells - pad)
