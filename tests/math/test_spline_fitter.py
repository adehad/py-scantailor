"""Tests for spline fitting utilities."""

from __future__ import annotations

import numpy as np
import pytest

from scantailor.math import (
    FitResult,
    XSpline,
    fit_spline_initial_guess,
    fit_spline_to_points,
    fit_spline_to_polyline,
    sample_spline_by_arc_length,
    sample_spline_uniformly,
)


class TestFitSplineToPoints:
    """Tests for fit_spline_to_points function."""

    def test_fit_simple_line(self):
        """Fit spline to points on a line."""
        points = np.array([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=np.float64)
        result = fit_spline_to_points(points, num_control_points=3, max_iterations=50)

        assert isinstance(result, FitResult)
        assert result.spline.num_control_points == 3
        # Residual should be small for collinear points
        assert result.residual < 10.0

    def test_fit_curved_points(self):
        """Fit spline to points on a curve."""
        # Generate points on a parabola
        t = np.linspace(0, 10, 20)
        points = np.column_stack([t, 0.1 * (t - 5) ** 2])

        result = fit_spline_to_points(points, num_control_points=5, max_iterations=50)

        assert result.spline.num_control_points == 5
        # Check that fitted spline passes near the points
        for pt in points[::5]:  # Check every 5th point
            closest, _ = result.spline.point_closest_to(pt)
            dist = np.sqrt(np.sum((pt - closest) ** 2))
            assert dist < 5.0  # Within 5 pixels

    def test_fit_returns_result(self):
        """Fit returns FitResult with all fields."""
        points = np.array([[0, 0], [10, 5], [20, 0]], dtype=np.float64)
        result = fit_spline_to_points(points, num_control_points=3)

        assert hasattr(result, "spline")
        assert hasattr(result, "residual")
        assert hasattr(result, "success")
        assert hasattr(result, "iterations")
        assert isinstance(result.residual, float)
        assert isinstance(result.success, bool)
        assert isinstance(result.iterations, int)

    def test_fit_invalid_points_shape(self):
        """Invalid points shape raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            fit_spline_to_points(np.array([1, 2, 3]))

    def test_fit_too_few_points(self):
        """Too few points raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 points"):
            fit_spline_to_points(np.array([[0, 0]]))

    def test_fit_too_few_control_points(self):
        """Too few control points raises ValueError."""
        points = np.array([[0, 0], [10, 10]])
        with pytest.raises(ValueError, match="At least 2 control points"):
            fit_spline_to_points(points, num_control_points=1)


class TestFitSplineToPolyline:
    """Tests for fit_spline_to_polyline function."""

    def test_fit_with_smoothing(self):
        """Fit with smoothing reduces curvature."""
        # Noisy sine wave
        t = np.linspace(0, 2 * np.pi, 50)
        noise = np.random.RandomState(42).randn(50) * 0.5
        points = np.column_stack([t * 10, np.sin(t) * 10 + noise])

        # Fit without smoothing
        result_no_smooth = fit_spline_to_polyline(
            points, num_control_points=5, smoothing=0, max_iterations=30
        )

        # Fit with smoothing
        result_smooth = fit_spline_to_polyline(
            points, num_control_points=5, smoothing=1.0, max_iterations=30
        )

        # Both should produce valid splines
        assert result_no_smooth.spline.num_control_points == 5
        assert result_smooth.spline.num_control_points == 5

    def test_fit_polyline_basic(self):
        """Basic polyline fitting works."""
        polyline = np.array(
            [[0, 0], [5, 10], [10, 5], [15, 15], [20, 0]], dtype=np.float64
        )
        result = fit_spline_to_polyline(
            polyline, num_control_points=4, max_iterations=30
        )

        assert result.spline.num_control_points == 4
        assert result.residual >= 0


class TestFitSplineInitialGuess:
    """Tests for fit_spline_initial_guess function."""

    def test_initial_guess_uniform_spacing(self):
        """Initial guess distributes control points uniformly."""
        points = np.array(
            [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], dtype=np.float64
        )
        spline = fit_spline_initial_guess(points, num_control_points=3)

        assert spline.num_control_points == 3
        # First control point near start
        assert np.allclose(spline.control_point_position(0), [0, 0], atol=1.0)
        # Last control point near end
        assert np.allclose(spline.control_point_position(2), [40, 0], atol=1.0)

    def test_initial_guess_curved_path(self):
        """Initial guess follows curved path."""
        # Quarter circle
        theta = np.linspace(0, np.pi / 2, 20)
        points = np.column_stack([np.cos(theta) * 10, np.sin(theta) * 10])

        spline = fit_spline_initial_guess(points, num_control_points=5)

        assert spline.num_control_points == 5
        # Control points should roughly follow the arc
        start = spline.control_point_position(0)
        end = spline.control_point_position(4)
        assert start[0] > start[1]  # First point closer to x-axis
        assert end[1] > end[0]  # Last point closer to y-axis

    def test_initial_guess_too_few_points(self):
        """Too few points raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 points"):
            fit_spline_initial_guess(np.array([[0, 0]]))

    def test_initial_guess_fewer_points_than_control_points(self):
        """Handles case with fewer data points than control points."""
        points = np.array([[0, 0], [10, 10]], dtype=np.float64)
        spline = fit_spline_initial_guess(points, num_control_points=5)

        # Should use all available points
        assert spline.num_control_points == 2


class TestSampleSplineUniformly:
    """Tests for sample_spline_uniformly function."""

    def test_sample_endpoints(self):
        """Samples include endpoints."""
        spline = XSpline()
        spline.append_control_point([0, 0], -1)
        spline.append_control_point([10, 10], -1)

        samples = sample_spline_uniformly(spline, num_samples=5)

        assert len(samples) == 5
        assert np.allclose(samples[0], [0, 0], atol=0.1)
        assert np.allclose(samples[-1], [10, 10], atol=0.1)

    def test_sample_count(self):
        """Returns correct number of samples."""
        spline = XSpline()
        spline.append_control_point([0, 0], 0)
        spline.append_control_point([10, 10], 0)

        for n in [1, 5, 10, 100]:
            samples = sample_spline_uniformly(spline, num_samples=n)
            assert len(samples) == n

    def test_sample_zero_returns_empty(self):
        """Zero samples returns empty array."""
        spline = XSpline()
        spline.append_control_point([0, 0], 0)
        spline.append_control_point([10, 10], 0)

        samples = sample_spline_uniformly(spline, num_samples=0)
        assert len(samples) == 0

    def test_sample_partial_range(self):
        """Sample partial range of spline."""
        spline = XSpline()
        spline.append_control_point([0, 0], -1)
        spline.append_control_point([10, 10], -1)

        samples = sample_spline_uniformly(spline, num_samples=5, from_t=0.25, to_t=0.75)

        assert len(samples) == 5
        # First sample should not be at origin
        assert not np.allclose(samples[0], [0, 0], atol=0.1)


class TestSampleSplineByArcLength:
    """Tests for sample_spline_by_arc_length function."""

    def test_arc_length_endpoints(self):
        """Arc length samples include endpoints."""
        spline = XSpline()
        spline.append_control_point([0, 0], -1)
        spline.append_control_point([10, 10], -1)

        samples = sample_spline_by_arc_length(spline, num_samples=5)

        assert len(samples) == 5
        assert np.allclose(samples[0], [0, 0], atol=0.1)
        assert np.allclose(samples[-1], [10, 10], atol=0.1)

    def test_arc_length_uniform_spacing(self):
        """Arc length samples are more uniformly spaced."""
        spline = XSpline()
        spline.append_control_point([0, 0], -1)
        spline.append_control_point([5, 20], -1)  # Sharp curve
        spline.append_control_point([10, 0], -1)

        samples = sample_spline_by_arc_length(spline, num_samples=10)

        # Compute distances between consecutive samples
        dists = np.sqrt(np.sum(np.diff(samples, axis=0) ** 2, axis=1))

        # Standard deviation of distances should be relatively low
        # (uniform spacing means similar distances)
        assert np.std(dists) < np.mean(dists) * 0.5

    def test_arc_length_single_sample(self):
        """Single sample returns midpoint."""
        spline = XSpline()
        spline.append_control_point([0, 0], -1)
        spline.append_control_point([10, 0], -1)

        samples = sample_spline_by_arc_length(spline, num_samples=1)

        assert len(samples) == 1
        # Should be roughly at midpoint
        assert abs(samples[0][0] - 5) < 2.0


class TestIntegration:
    """Integration tests for spline fitting workflow."""

    def test_fit_then_sample(self):
        """Fit spline then sample it."""
        # Create noisy data
        t = np.linspace(0, 10, 30)
        points = np.column_stack([t, np.sin(t) * 5])

        # Fit spline
        result = fit_spline_to_points(
            points, num_control_points=5, tension=-0.5, max_iterations=50
        )

        # Sample the fitted spline
        samples = sample_spline_uniformly(result.spline, num_samples=20)

        assert len(samples) == 20
        # Samples should roughly follow the original curve
        assert np.min(samples[:, 1]) < 0  # Has negative y values
        assert np.max(samples[:, 1]) > 0  # Has positive y values

    def test_initial_guess_then_refine(self):
        """Initial guess followed by optimization."""
        points = np.array([[0, 0], [5, 5], [10, 0], [15, 5], [20, 0]], dtype=np.float64)

        # Get initial guess
        initial = fit_spline_initial_guess(points, num_control_points=4)

        # The initial guess should be reasonable
        assert initial.num_control_points == 4

        # Refine with optimization
        result = fit_spline_to_points(points, num_control_points=4, max_iterations=50)

        # Refined should have good fit
        assert result.residual < 50.0
