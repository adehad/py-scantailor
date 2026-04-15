"""Tests for X-Spline implementation."""

import numpy as np

from scantailor.math.spline import ControlPoint, PointAndDerivatives, XSpline


class TestControlPoint:
    """Tests for ControlPoint dataclass."""

    def test_basic_creation(self):
        """Create a control point with position and tension."""
        cp = ControlPoint(pos=[10, 20], tension=0.5)
        assert np.allclose(cp.pos, [10, 20])
        assert cp.tension == 0.5

    def test_tension_clipping(self):
        """Tension should be clipped to [-1, 1]."""
        cp1 = ControlPoint(pos=[0, 0], tension=2.0)
        assert cp1.tension == 1.0

        cp2 = ControlPoint(pos=[0, 0], tension=-2.0)
        assert cp2.tension == -1.0

    def test_pos_conversion(self):
        """Position should be converted to float64 array."""
        cp = ControlPoint(pos=(1, 2), tension=0)
        assert cp.pos.dtype == np.float64
        assert cp.pos.shape == (2,)


class TestXSplineBasics:
    """Basic tests for XSpline class."""

    def test_empty_spline(self):
        """Empty spline has no control points or segments."""
        spline = XSpline()
        assert spline.num_control_points == 0
        assert spline.num_segments == 0

    def test_single_control_point(self):
        """Spline with single control point."""
        spline = XSpline()
        spline.append_control_point([5, 10], tension=0)
        assert spline.num_control_points == 1
        assert spline.num_segments == 0

        # point_at should return the single point
        pt = spline.point_at(0.5)
        assert np.allclose(pt, [5, 10])

    def test_two_control_points(self):
        """Spline with two control points."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)
        assert spline.num_control_points == 2
        assert spline.num_segments == 1

    def test_append_control_point(self):
        """Append multiple control points."""
        spline = XSpline()
        for i in range(5):
            spline.append_control_point([i * 10, i * 5], tension=0)
        assert spline.num_control_points == 5
        assert spline.num_segments == 4

    def test_insert_control_point(self):
        """Insert control point at specific position."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([20, 20], tension=0)
        spline.insert_control_point(1, [10, 10], tension=0)

        assert spline.num_control_points == 3
        assert np.allclose(spline.control_point_position(1), [10, 10])

    def test_erase_control_point(self):
        """Erase a control point."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)
        spline.append_control_point([20, 20], tension=0)

        spline.erase_control_point(1)
        assert spline.num_control_points == 2
        assert np.allclose(spline.control_point_position(1), [20, 20])

    def test_move_control_point(self):
        """Move a control point to new position."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)

        spline.move_control_point(0, [5, 5])
        assert np.allclose(spline.control_point_position(0), [5, 5])

    def test_tension_accessors(self):
        """Get and set tension values."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0.5)

        assert spline.control_point_tension(0) == 0.5

        spline.set_control_point_tension(0, -0.3)
        assert spline.control_point_tension(0) == -0.3


class TestXSplinePointEvaluation:
    """Tests for point_at and related methods."""

    def test_point_at_endpoints(self):
        """Points at t=0 and t=1 should be at control points."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=-1)  # Interpolating
        spline.append_control_point([10, 20], tension=-1)

        pt_start = spline.point_at(0.0)
        pt_end = spline.point_at(1.0)

        assert np.allclose(pt_start, [0, 0], atol=1e-6)
        assert np.allclose(pt_end, [10, 20], atol=1e-6)

    def test_point_at_clamping(self):
        """Values outside [0, 1] should be clamped."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)

        pt_neg = spline.point_at(-0.5)
        pt_over = spline.point_at(1.5)

        assert np.allclose(pt_neg, spline.point_at(0.0))
        assert np.allclose(pt_over, spline.point_at(1.0))

    def test_point_at_midpoint_linear(self):
        """Midpoint of two-point spline with zero tension."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)

        pt_mid = spline.point_at(0.5)
        # With zero tension, midpoint should be close to geometric midpoint
        assert np.allclose(pt_mid, [5, 5], atol=1.0)

    def test_interpolating_spline(self):
        """Interpolating spline (tension=-1) passes through control points."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=-1)
        spline.append_control_point([5, 10], tension=-1)
        spline.append_control_point([10, 0], tension=-1)

        # Points at control point locations
        t0 = spline.control_point_index_to_t(0)
        t1 = spline.control_point_index_to_t(1)
        t2 = spline.control_point_index_to_t(2)

        assert np.allclose(spline.point_at(t0), [0, 0], atol=1e-6)
        assert np.allclose(spline.point_at(t1), [5, 10], atol=1e-6)
        assert np.allclose(spline.point_at(t2), [10, 0], atol=1e-6)

    def test_approximating_spline(self):
        """Approximating spline (tension>0) does not pass through middle points."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=1)
        spline.append_control_point([5, 10], tension=1)
        spline.append_control_point([10, 0], tension=1)

        t1 = spline.control_point_index_to_t(1)
        pt = spline.point_at(t1)

        # Should NOT be exactly at [5, 10]
        assert not np.allclose(pt, [5, 10], atol=0.1)


class TestXSplineDerivatives:
    """Tests for derivative calculations."""

    def test_point_and_derivs_returns_point(self):
        """point_and_derivs_at returns correct point."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=-1)
        spline.append_control_point([5, 5], tension=-1)
        spline.append_control_point([10, 10], tension=-1)

        # Test at a non-control-point position
        pd = spline.point_and_derivs_at(0.25)
        pt = spline.point_at(0.25)

        assert isinstance(pd, PointAndDerivatives)
        assert np.allclose(pd.point, pt, atol=1e-3)

    def test_first_derivative_direction(self):
        """First derivative should point in direction of travel."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 0], tension=0)  # Horizontal line

        pd = spline.point_and_derivs_at(0.5)

        # For horizontal line, derivative should be mostly in x direction
        assert pd.first_deriv[0] > 0  # Moving right
        assert abs(pd.first_deriv[1]) < abs(pd.first_deriv[0])  # Small y component

    def test_signed_curvature_straight_line(self):
        """Curvature of a straight line should be near zero."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)

        pd = spline.point_and_derivs_at(0.5)
        curvature = pd.signed_curvature()

        assert abs(curvature) < 0.1

    def test_signed_curvature_curved_spline(self):
        """Curved spline should have non-zero curvature."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=-1)
        spline.append_control_point([5, 10], tension=-1)
        spline.append_control_point([10, 0], tension=-1)

        pd = spline.point_and_derivs_at(0.5)
        curvature = pd.signed_curvature()

        # Should have significant curvature
        assert abs(curvature) > 0.01


class TestXSplineLinearCombination:
    """Tests for linear combination coefficients."""

    def test_coefficients_sum_to_one(self):
        """Linear combination coefficients should sum to 1."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)
        spline.append_control_point([20, 0], tension=0)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            coeffs = spline.linear_combination_at(t)
            total = sum(c.coeff for c in coeffs)
            assert abs(total - 1.0) < 1e-6

    def test_coefficients_reconstruct_point(self):
        """Point can be reconstructed from coefficients."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 20], tension=0)
        spline.append_control_point([20, 0], tension=0)

        t = 0.3
        coeffs = spline.linear_combination_at(t)
        expected = spline.point_at(t)

        reconstructed = np.zeros(2)
        for c in coeffs:
            reconstructed += (
                spline.control_point_position(c.control_point_idx) * c.coeff
            )

        assert np.allclose(reconstructed, expected, atol=1e-6)


class TestXSplineClosestPoint:
    """Tests for point_closest_to method."""

    def test_closest_to_control_point(self):
        """Closest point to a control point location."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=-1)
        spline.append_control_point([10, 0], tension=-1)

        closest, t = spline.point_closest_to([0, 1])

        # Should be close to start
        assert t < 0.2
        assert np.allclose(closest, [0, 0], atol=1.0)

    def test_closest_to_midpoint(self):
        """Closest point to middle of spline."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=-1)
        spline.append_control_point([10, 0], tension=-1)

        closest, t = spline.point_closest_to([5, 1])

        # Should be near midpoint
        assert 0.3 < t < 0.7
        assert np.allclose(closest[0], 5, atol=1.0)

    def test_closest_empty_spline(self):
        """Closest point on empty spline returns origin."""
        spline = XSpline()
        closest, t = spline.point_closest_to([5, 5])

        assert np.allclose(closest, [0, 0])
        assert t == 0.0


class TestXSplinePolyline:
    """Tests for to_polyline method."""

    def test_polyline_endpoints(self):
        """Polyline should include endpoints."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)

        polyline = spline.to_polyline()

        assert len(polyline) >= 2
        assert np.allclose(polyline[0], [0, 0], atol=1e-6)
        assert np.allclose(polyline[-1], [10, 10], atol=1e-6)

    def test_polyline_sampling_density(self):
        """More curved splines should have more samples."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=-1)
        spline.append_control_point([5, 20], tension=-1)  # Sharp curve
        spline.append_control_point([10, 0], tension=-1)

        polyline_fine = spline.to_polyline(max_dist_from_spline=0.1)
        polyline_coarse = spline.to_polyline(max_dist_from_spline=5.0)

        assert len(polyline_fine) > len(polyline_coarse)

    def test_polyline_empty_spline(self):
        """Empty spline produces empty polyline."""
        spline = XSpline()
        polyline = spline.to_polyline()
        assert len(polyline) == 0

    def test_polyline_partial_range(self):
        """Polyline can be generated for partial range."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([10, 10], tension=0)
        spline.append_control_point([20, 0], tension=0)

        polyline = spline.to_polyline(from_t=0.25, to_t=0.75)

        # Should not start at [0, 0] or end at [20, 0]
        assert not np.allclose(polyline[0], [0, 0])
        assert not np.allclose(polyline[-1], [20, 0])


class TestXSplineTension:
    """Tests for tension parameter effects."""

    def test_negative_tension_interpolates(self):
        """Negative tension creates interpolating spline."""
        spline = XSpline()
        points = [[0, 0], [5, 5], [10, 0], [15, 5], [20, 0]]

        for pt in points:
            spline.append_control_point(pt, tension=-1)

        # Check that spline passes through middle points
        for i, pt in enumerate(points):
            t = spline.control_point_index_to_t(i)
            spline_pt = spline.point_at(t)
            assert np.allclose(spline_pt, pt, atol=1e-5)

    def test_positive_tension_approximates(self):
        """Positive tension creates approximating spline."""
        spline = XSpline()
        points = [[0, 0], [5, 5], [10, 0], [15, 5], [20, 0]]

        for pt in points:
            spline.append_control_point(pt, tension=1)

        # Middle points should not be exactly on spline
        for i in range(1, len(points) - 1):
            t = spline.control_point_index_to_t(i)
            spline_pt = spline.point_at(t)
            # Allow endpoints to match, middle points shouldn't
            if 0 < i < len(points) - 1:
                assert not np.allclose(spline_pt, points[i], atol=0.5)

    def test_zero_tension_sharp_angle(self):
        """Zero tension creates sharp angle interpolation."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=0)
        spline.append_control_point([5, 10], tension=0)
        spline.append_control_point([10, 0], tension=0)

        # At the middle control point, derivatives can be zero.
        # Test slightly off the control point to verify the curve behavior
        t = spline.control_point_index_to_t(1)
        pd_before = spline.point_and_derivs_at(t - 0.1)
        pd_after = spline.point_and_derivs_at(t + 0.1)

        # The derivatives should exist and curve should change direction
        # (y derivative should go from positive to negative)
        assert pd_before.first_deriv[1] > 0  # Going up before midpoint
        assert pd_after.first_deriv[1] < 0  # Going down after midpoint


class TestControlPointIndexToT:
    """Tests for control_point_index_to_t method."""

    def test_index_to_t_endpoints(self):
        """First and last indices map to 0 and 1."""
        spline = XSpline()
        for i in range(5):
            spline.append_control_point([i * 10, 0], tension=0)

        assert spline.control_point_index_to_t(0) == 0.0
        assert spline.control_point_index_to_t(4) == 1.0

    def test_index_to_t_uniform(self):
        """Indices should map to uniform t values."""
        spline = XSpline()
        for i in range(5):
            spline.append_control_point([i * 10, 0], tension=0)

        assert spline.control_point_index_to_t(1) == 0.25
        assert spline.control_point_index_to_t(2) == 0.5
        assert spline.control_point_index_to_t(3) == 0.75


class TestXSplineComplexCurves:
    """Tests for more complex spline scenarios."""

    def test_s_curve(self):
        """S-shaped curve."""
        spline = XSpline()
        spline.append_control_point([0, 0], tension=-1)
        spline.append_control_point([5, 10], tension=-1)
        spline.append_control_point([15, -10], tension=-1)
        spline.append_control_point([20, 0], tension=-1)

        # Curve should be smooth
        polyline = spline.to_polyline(max_dist_from_spline=0.5)
        assert len(polyline) > 10

        # Check y values change sign
        y_values = [pt[1] for pt in polyline]
        assert min(y_values) < 0
        assert max(y_values) > 0

    def test_closed_shape_approximation(self):
        """Near-closed shape."""
        spline = XSpline()
        # Square-ish shape
        spline.append_control_point([0, 0], tension=-0.5)
        spline.append_control_point([10, 0], tension=-0.5)
        spline.append_control_point([10, 10], tension=-0.5)
        spline.append_control_point([0, 10], tension=-0.5)
        spline.append_control_point([0, 0], tension=-0.5)  # Close the loop

        # Should be able to generate polyline with reasonable sampling
        polyline = spline.to_polyline(
            max_dist_from_spline=0.5, max_dist_between_samples=2.0
        )
        assert len(polyline) >= 2  # At minimum start and end

        # Start and end should be same (or very close)
        assert np.allclose(polyline[0], polyline[-1], atol=1.0)

    def test_many_control_points(self):
        """Spline with many control points."""
        spline = XSpline()

        # Sine wave approximation
        for i in range(20):
            x = i * 5
            y = 10 * np.sin(i * 0.5)
            spline.append_control_point([x, y], tension=-0.5)

        assert spline.num_control_points == 20
        assert spline.num_segments == 19

        # Should be able to generate polyline
        polyline = spline.to_polyline()
        assert len(polyline) >= 20  # At least as many as control points
