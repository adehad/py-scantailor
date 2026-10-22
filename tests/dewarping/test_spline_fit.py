"""X-spline evaluation, the quadratic forces and the fit of an extended spline.

Reference numbers marked "C++" come from compiling the upstream sources of
``XSpline``, ``spfit`` and ``DistortionModelBuilder::fitExtendedSpline`` with
g++ 12 and running them on the same input.
"""

import numpy as np
import pytest

from scantailor.dewarping.spline_fit import (
    DEFAULT_SAMPLE,
    HEAD_SAMPLE,
    JUNCTION_SAMPLE,
    TAIL_SAMPLE,
    _points_and_derivs,
    control_points_attraction_force,
    fit_extended_spline,
    fit_spline_between_ends,
    junction_points_attraction_force,
    linear_combination_at,
    lu_solve,
    sample_spline,
    spline_point_at,
)
from scantailor.math.spline import XSpline
from tests.dewarping.kernel_reference import recorded

TENSIONS = np.array([1.0, -1.0, 0.3, -0.5, 1.0, 0.0])


def _random_spline(seed: int = 1, count: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, (count, 2)), TENSIONS[:count]


class TestEvaluation:
    @pytest.mark.parametrize("t", [0.0, 0.13, 0.5, 0.77, 1.0])
    def test_points_agree_with_the_general_purpose_xspline(self, t: float) -> None:
        positions, tensions = _random_spline()
        reference = XSpline()
        for position, tension in zip(positions, tensions, strict=True):
            reference.append_control_point(position, tension)

        assert spline_point_at(positions, tensions, t) == pytest.approx(
            tuple(reference.point_at(t)), abs=1e-9
        )

    @pytest.mark.parametrize("t", [0.03, 0.33, 0.5, 0.9, 0.97])
    def test_derivatives_agree_with_finite_differences(self, t: float) -> None:
        positions, tensions = _random_spline()
        step = 1e-5

        def point(at: float) -> np.ndarray:
            return np.array(spline_point_at(positions, tensions, at))

        first = (point(t + step) - point(t - step)) / (2 * step)
        second = (point(t + step) - 2 * point(t) + point(t - step)) / step**2
        derivs = _points_and_derivs(positions, tensions, np.array([t]))[0]

        assert derivs[2:4] == pytest.approx(first, abs=1e-4)
        assert derivs[4:] == pytest.approx(second, abs=1e-2)

    def test_the_weights_of_a_point_sum_to_one(self) -> None:
        for t in (0.0, 0.2, 0.55, 1.0):
            weights = linear_combination_at(TENSIONS, t)

            assert sum(weight for _, weight in weights) == pytest.approx(1.0)

    def test_each_control_point_appears_at_most_once_in_a_combination(self) -> None:
        for t in (0.0, 0.1, 0.9, 1.0):
            indexes = [index for index, _ in linear_combination_at(TENSIONS, t)]

            assert len(indexes) == len(set(indexes))

    def test_an_interpolating_spline_passes_through_its_control_points(self) -> None:
        positions, _ = _random_spline()
        tensions = np.full(len(positions), -1.0)

        for index, position in enumerate(positions):
            at = index / (len(positions) - 1)

            assert spline_point_at(positions, tensions, at) == pytest.approx(
                tuple(position), abs=1e-9
            )


class TestSampling:
    def test_first_and_last_samples_are_flagged_head_and_tail(self) -> None:
        positions, tensions = _random_spline()

        ts, _, flags = sample_spline(positions, tensions)

        assert (ts[0], ts[-1]) == (0.0, 1.0)
        assert flags[0] == HEAD_SAMPLE
        assert flags[-1] == TAIL_SAMPLE

    def test_every_segment_boundary_is_sampled_as_a_junction(self) -> None:
        positions, tensions = _random_spline()

        ts, _, flags = sample_spline(positions, tensions)

        junctions = ts[flags == JUNCTION_SAMPLE]
        np.testing.assert_allclose(
            junctions, np.arange(1, len(positions) - 1) / (len(positions) - 1)
        )
        assert set(flags) <= {DEFAULT_SAMPLE, HEAD_SAMPLE, TAIL_SAMPLE, JUNCTION_SAMPLE}

    def test_parameters_increase_along_the_samples(self) -> None:
        positions, tensions = _random_spline()

        ts, _, _ = sample_spline(positions, tensions)

        assert np.all(np.diff(ts) > 0)

    def test_chords_stay_within_the_requested_distance_of_the_curve(self) -> None:
        positions, tensions = _random_spline()
        _ts, points, _ = sample_spline(positions, tensions, 0.2)
        dense = np.array(
            [spline_point_at(positions, tensions, t) for t in np.linspace(0, 1, 4000)]
        )

        worst = max(np.min(np.hypot(*(dense - point).T)) for point in points)
        # Each sample lies on the curve itself.
        assert worst < 0.05

        midpoints = 0.5 * (points[1:] + points[:-1])
        gaps = [np.min(np.hypot(*(dense - mid).T)) for mid in midpoints]
        assert max(gaps) < 0.2 + 1e-6

    def test_a_distance_cap_between_samples_adds_samples(self) -> None:
        positions, tensions = _random_spline()

        sparse = sample_spline(positions, tensions, 0.2)[0]
        capped = sample_spline(positions, tensions, 0.2, 5.0)[1]

        assert len(capped) > len(sparse)
        assert np.hypot(*np.diff(capped, axis=0).T).max() <= 5.0 + 1e-9


class TestForces:
    def _quadratic(self, force: tuple, displacement: np.ndarray) -> float:
        matrix, linear, constant = force
        flat = displacement.reshape(-1)
        return float(flat @ matrix @ flat + linear @ flat + constant)

    def test_control_point_force_is_the_sum_of_squared_neighbour_distances(
        self,
    ) -> None:
        positions, _ = _random_spline()
        displacement = np.random.default_rng(3).normal(size=positions.shape)
        moved = positions + displacement

        value = self._quadratic(
            control_points_attraction_force(positions), displacement
        )

        assert value == pytest.approx(float((np.diff(moved, axis=0) ** 2).sum()))

    def test_junction_force_is_the_sum_of_squared_junction_distances(self) -> None:
        positions, _ = _random_spline(seed=2, count=5)
        tensions = np.ones(5)
        displacement = np.random.default_rng(4).normal(size=positions.shape)
        moved = positions + displacement
        junctions = np.array(
            [spline_point_at(moved, tensions, i / 4) for i in range(5)]
        )

        value = self._quadratic(
            junction_points_attraction_force(positions, tensions), displacement
        )

        assert value == pytest.approx(float((np.diff(junctions, axis=0) ** 2).sum()))

    def test_forces_vanish_when_the_points_coincide(self) -> None:
        positions = np.tile([[5.0, 7.0]], (5, 1))

        _, _, constant = control_points_attraction_force(positions)

        assert constant == 0.0


class TestLinearSolver:
    def test_solution_matches_numpy_for_a_well_conditioned_system(self) -> None:
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(8, 8)) + 8 * np.eye(8)
        rhs = rng.normal(size=8)

        solution = lu_solve(matrix, rhs)

        assert solution is not None
        np.testing.assert_allclose(solution, np.linalg.solve(matrix, rhs), atol=1e-10)

    def test_a_pivot_below_the_tolerance_means_singular(self) -> None:
        matrix = np.array([[1e-9, 0.0], [0.0, 1e-9]])

        assert lu_solve(matrix, np.ones(2)) is None

    def test_a_rank_deficient_matrix_is_singular(self) -> None:
        matrix = np.array([[1.0, 2.0], [2.0, 4.0]])

        assert lu_solve(matrix, np.ones(2)) is None

    def test_entries_below_the_tolerance_are_taken_as_zero(self) -> None:
        """An entry of 1e-8 or less is not eliminated, even where that matters.

        The matrix is exactly singular, since its determinant is 1e-9 * 1e9 - 1,
        but the C++ solver skips the 1e-9 and goes on to a solution.
        """
        matrix = np.array([[1e-9, 1.0], [1.0, 1e9]])

        solution = lu_solve(matrix, np.array([1.0, 1.0]))

        assert solution is not None
        np.testing.assert_allclose(solution, [1.0 - 1e9, 1.0])


class TestFittingAnExtendedSpline:
    BOUNDS = (((100.0, 0.0), (100.0, 1000.0)), ((900.0, 0.0), (900.0, 1000.0)))

    def _sine_polyline(self) -> np.ndarray:
        xs = np.linspace(100, 900, 60)
        return np.column_stack([xs, 300 + 60 * np.sin(np.pi * (xs - 100) / 800)])

    def test_control_points_match_the_cpp_fit_of_a_sine(self) -> None:
        spline = fit_extended_spline(self._sine_polyline(), *self.BOUNDS)

        assert spline is not None
        np.testing.assert_allclose(
            spline.positions,
            [
                [59.913544526711519, 290.33200852983043],
                [300.43227736644246, 347.99567931494448],
                [499.99999999997635, 366.07234361915414],
                [699.56772263352502, 347.99567931495216],
                [940.08645547329502, 290.33200852982901],
            ],
            atol=1e-6,
        )

    def test_control_points_match_the_cpp_fit_of_a_tilted_asymmetric_curve(
        self,
    ) -> None:
        xs = np.linspace(120, 880, 30)
        ys = 400 + 50 * np.sin(np.pi * (xs - 100) / 800 + 0.4) + 0.05 * (xs - 500)
        bounds = (((100.0, 0.0), (110.0, 1000.0)), ((900.0, 0.0), (895.0, 1000.0)))

        spline = fit_extended_spline(np.column_stack([xs, ys]), *bounds)

        assert spline is not None
        np.testing.assert_allclose(
            spline.positions,
            [
                [66.263216, 392.451664],
                [292.728231, 442.172641],
                [510.52756, 450.794906],
                [699.962576, 430.281701],
                [937.606757, 394.089292],
            ],
            atol=1e-4,
        )
        polyline = spline.to_polyline()
        assert len(polyline) == 21
        np.testing.assert_allclose(polyline[0], [104.007385, 400.738494], atol=1e-4)
        np.testing.assert_allclose(polyline[10], [429.380983, 445.936369], atol=1e-4)
        np.testing.assert_allclose(polyline[-1], [897.999393, 400.12136], atol=1e-4)

    def test_the_curve_ends_on_the_bounds_and_follows_the_data(self) -> None:
        polyline = self._sine_polyline()

        spline = fit_extended_spline(polyline, *self.BOUNDS)

        assert spline is not None
        fitted = spline.to_polyline()
        assert fitted[0, 0] == pytest.approx(100.0, abs=1e-6)
        assert fitted[-1, 0] == pytest.approx(900.0, abs=1e-6)
        truth = 300 + 60 * np.sin(np.pi * (fitted[:, 0] - 100) / 800)
        assert np.abs(fitted[:, 1] - truth).max() < 0.15

    def test_a_curve_that_stops_short_of_the_bounds_is_extended_to_them(self) -> None:
        xs = np.linspace(300, 700, 40)
        polyline = np.column_stack([xs, 300 + 0.05 * (xs - 500)])

        spline = fit_extended_spline(polyline, *self.BOUNDS)

        assert spline is not None
        fitted = spline.to_polyline()
        assert fitted[0, 0] == pytest.approx(100.0, abs=1e-6)
        assert fitted[-1, 0] == pytest.approx(900.0, abs=1e-6)
        # A straight line stays straight when extended.
        assert fitted[0, 1] == pytest.approx(300 + 0.05 * (100 - 500), abs=0.5)

    def test_an_end_on_a_tilted_bound_lies_on_that_bound(self) -> None:
        bounds = (((100.0, 0.0), (140.0, 1000.0)), ((900.0, 0.0), (860.0, 1000.0)))
        xs = np.linspace(150, 850, 40)
        polyline = np.column_stack([xs, 400 + 0 * xs])

        spline = fit_extended_spline(polyline, *bounds)

        assert spline is not None
        start, end = spline.to_polyline()[[0, -1]]
        assert start[0] == pytest.approx(100 + 0.04 * start[1], abs=1e-6)
        assert end[0] == pytest.approx(900 - 0.04 * end[1], abs=1e-6)

    def test_a_steep_short_curve_that_folds_the_spline_back_gives_none(self) -> None:
        """The first round folds the spline back on itself; C++ throws BadCurve."""
        polyline = np.array([[241.73, 108.92], [256.91, 245.72]])
        bounds = (((100.0, 0.0), (191.71, 1000.0)), ((900.0, 0.0), (845.52, 1000.0)))

        assert fit_extended_spline(polyline, *bounds) is None

    def test_a_polyline_whose_ends_coincide_gives_none(self) -> None:
        polyline = np.array([[300.0, 300.0], [400.0, 320.0], [300.0, 300.0]])

        assert fit_extended_spline(polyline, *self.BOUNDS) is None

    def test_the_fit_is_repeatable(self) -> None:
        polyline = self._sine_polyline()

        first = fit_extended_spline(polyline, *self.BOUNDS)
        second = fit_extended_spline(polyline, *self.BOUNDS)

        assert first is not None and second is not None
        np.testing.assert_array_equal(first.positions, second.positions)


class TestFittingASplineBetweenEnds:
    TENSIONS = np.array([0.0, -1.0, -1.0, -1.0, 0.0])

    @staticmethod
    def _sine_polyline() -> np.ndarray:
        xs = np.linspace(100, 900, 60)
        return np.column_stack([xs, 300 + 60 * np.sin(np.pi * (xs - 100) / 800)])

    def test_the_spline_starts_and_ends_on_the_polylines_ends(self) -> None:
        polyline = self._sine_polyline()

        spline = fit_spline_between_ends(polyline, self.TENSIONS)

        assert spline.positions[0] == pytest.approx(polyline[0], abs=1e-9)
        assert spline.positions[-1] == pytest.approx(polyline[-1], abs=1e-9)

    def test_the_fitted_spline_stays_within_a_pixel_of_a_sine(self) -> None:
        spline = fit_spline_between_ends(self._sine_polyline(), self.TENSIONS)

        points = spline.to_polyline()
        expected = 300 + 60 * np.sin(np.pi * (points[:, 0] - 100) / 800)
        assert np.abs(points[:, 1] - expected).max() < 1.0

    def test_the_number_of_tensions_sets_the_number_of_control_points(self) -> None:
        spline = fit_spline_between_ends(self._sine_polyline(), np.zeros(3))

        assert spline.positions.shape == (3, 2)

    def test_a_straight_polyline_gives_control_points_on_it(self) -> None:
        xs = np.linspace(100, 900, 30)
        polyline = np.column_stack([xs, 0.5 * xs + 10])

        spline = fit_spline_between_ends(polyline, self.TENSIONS)

        positions = spline.positions
        assert positions[:, 1] == pytest.approx(0.5 * positions[:, 0] + 10, abs=1e-6)


class TestRecordedOutputs:
    """Fits of traced lines and spline samples come out as recorded."""

    @pytest.mark.parametrize("case", range(10))
    def test_fitted_control_points_and_their_polyline_match_the_recording(
        self, case: int
    ) -> None:
        bounds = recorded(f"fit{case}_bounds")

        spline = fit_extended_spline(
            recorded(f"fit{case}_polyline"), bounds[0], bounds[1]
        )

        assert spline is not None
        np.testing.assert_array_equal(
            spline.positions, recorded(f"fit{case}_positions")
        )
        np.testing.assert_array_equal(
            spline.to_polyline(), recorded(f"fit{case}_polyline_out")
        )

    @pytest.mark.parametrize(
        ("spline", "limits"),
        [(spline, limits) for spline in range(6) for limits in range(3)],
    )
    def test_samples_match_the_recording(self, spline: int, limits: int) -> None:
        max_dist, max_between = [(0.2, np.inf), (0.2, 10.0), (1.5, 25.0)][limits]

        ts, points, flags = sample_spline(
            recorded(f"spline{spline}_positions"),
            recorded(f"spline{spline}_tensions"),
            max_dist,
            max_between,
        )

        np.testing.assert_array_equal(ts, recorded(f"spline{spline}_{limits}_t"))
        np.testing.assert_array_equal(
            points, recorded(f"spline{spline}_{limits}_points")
        )
        np.testing.assert_array_equal(flags, recorded(f"spline{spline}_{limits}_flags"))
        assert flags.dtype == np.int64
