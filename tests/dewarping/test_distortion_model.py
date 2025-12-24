"""Tests for distortion model."""

from __future__ import annotations

import numpy as np
import pytest

from scantailor.dewarping import (
    Curve,
    DistortionModel,
    create_distortion_model,
    create_distortion_model_from_splines,
)
from scantailor.math import XSpline


class TestCurve:
    """Tests for Curve class."""

    def test_create_from_polyline(self):
        """Create curve from polyline."""
        polyline = np.array([[0, 0], [50, 10], [100, 0]], dtype=np.float64)
        curve = Curve.from_polyline(polyline)

        assert curve.is_valid()
        assert len(curve.polyline) == 3
        assert curve.spline is None

    def test_create_from_spline(self):
        """Create curve from XSpline."""
        spline = XSpline()
        spline.append_control_point([0, 0], -0.5)
        spline.append_control_point([50, 10], -0.5)
        spline.append_control_point([100, 0], -0.5)

        curve = Curve.from_spline(spline)

        assert curve.is_valid()
        assert curve.spline is not None
        assert len(curve.polyline) >= 2

    def test_invalid_curve_single_point(self):
        """Single point curve is invalid."""
        polyline = np.array([[50, 50]], dtype=np.float64)
        curve = Curve.from_polyline(polyline)
        assert not curve.is_valid()

    def test_invalid_curve_closed(self):
        """Closed curve (same start/end) is invalid."""
        polyline = np.array([[0, 0], [50, 50], [0, 0]], dtype=np.float64)
        curve = Curve.from_polyline(polyline)
        assert not curve.is_valid()

    def test_start_and_end_points(self):
        """Start and end point properties."""
        polyline = np.array([[10, 20], [50, 30], [90, 25]], dtype=np.float64)
        curve = Curve.from_polyline(polyline)

        assert curve.start_point[0] == pytest.approx(10.0)
        assert curve.start_point[1] == pytest.approx(20.0)
        assert curve.end_point[0] == pytest.approx(90.0)
        assert curve.end_point[1] == pytest.approx(25.0)

    def test_curve_matches_identical(self):
        """Identical curves match."""
        polyline = np.array([[0, 0], [100, 0]], dtype=np.float64)
        curve1 = Curve.from_polyline(polyline)
        curve2 = Curve.from_polyline(polyline.copy())

        assert curve1.matches(curve2)

    def test_curve_matches_within_tolerance(self):
        """Curves within tolerance match."""
        polyline1 = np.array([[0, 0], [100, 0]], dtype=np.float64)
        polyline2 = np.array([[0.005, 0.005], [100.005, 0.005]], dtype=np.float64)

        curve1 = Curve.from_polyline(polyline1)
        curve2 = Curve.from_polyline(polyline2)

        assert curve1.matches(curve2, tolerance=0.01)

    def test_curve_does_not_match_different(self):
        """Different curves don't match."""
        curve1 = Curve.from_polyline(np.array([[0, 0], [100, 0]]))
        curve2 = Curve.from_polyline(np.array([[0, 0], [100, 10]]))

        assert not curve1.matches(curve2)

    def test_curve_does_not_match_different_lengths(self):
        """Curves with different number of points don't match."""
        curve1 = Curve.from_polyline(np.array([[0, 0], [100, 0]]))
        curve2 = Curve.from_polyline(np.array([[0, 0], [50, 0], [100, 0]]))

        assert not curve1.matches(curve2)


class TestDistortionModel:
    """Tests for DistortionModel class."""

    def test_create_valid_model(self):
        """Create valid distortion model."""
        model = create_distortion_model(
            top_polyline=np.array([[0, 0], [100, 0]]),
            bottom_polyline=np.array([[0, 100], [100, 100]]),
        )

        assert model.is_valid()
        assert model.top_curve is not None
        assert model.bottom_curve is not None

    def test_create_from_splines(self):
        """Create model from XSplines."""
        top_spline = XSpline()
        top_spline.append_control_point([0, 0], -0.5)
        top_spline.append_control_point([100, 5], -0.5)

        bottom_spline = XSpline()
        bottom_spline.append_control_point([0, 100], -0.5)
        bottom_spline.append_control_point([100, 95], -0.5)

        model = create_distortion_model_from_splines(top_spline, bottom_spline)

        assert model.is_valid()

    def test_empty_model_invalid(self):
        """Empty model is invalid."""
        model = DistortionModel()
        assert not model.is_valid()

    def test_partial_model_invalid(self):
        """Model with only one curve is invalid."""
        model = DistortionModel(
            top_curve=Curve.from_polyline(np.array([[0, 0], [100, 0]]))
        )
        assert not model.is_valid()

    def test_non_convex_corners_invalid(self):
        """Non-convex corner arrangement is invalid."""
        # Crossing lines - not convex
        model = create_distortion_model(
            top_polyline=np.array([[0, 0], [100, 100]]),  # Diagonal
            bottom_polyline=np.array([[100, 0], [0, 100]]),  # Crossing diagonal
        )
        assert not model.is_valid()

    def test_bounding_box(self):
        """Bounding box calculation."""
        model = create_distortion_model(
            top_polyline=np.array([[10, 20], [90, 30]]),
            bottom_polyline=np.array([[15, 80], [85, 90]]),
        )

        bbox = model.bounding_box()
        assert bbox is not None

        left, top, right, bottom = bbox
        assert left == pytest.approx(10.0)
        assert top == pytest.approx(20.0)
        assert right == pytest.approx(90.0)
        assert bottom == pytest.approx(90.0)

    def test_bounding_box_with_transform(self):
        """Bounding box with transformation."""
        model = create_distortion_model(
            top_polyline=np.array([[0, 0], [100, 0]]),
            bottom_polyline=np.array([[0, 100], [100, 100]]),
        )

        # Scale by 2
        transform = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        bbox = model.bounding_box(transform)
        assert bbox is not None

        left, top, right, bottom = bbox
        assert left == pytest.approx(0.0)
        assert top == pytest.approx(0.0)
        assert right == pytest.approx(200.0)
        assert bottom == pytest.approx(200.0)

    def test_model_matches_identical(self):
        """Identical models match."""
        model1 = create_distortion_model(
            top_polyline=np.array([[0, 0], [100, 0]]),
            bottom_polyline=np.array([[0, 100], [100, 100]]),
        )
        model2 = create_distortion_model(
            top_polyline=np.array([[0, 0], [100, 0]]),
            bottom_polyline=np.array([[0, 100], [100, 100]]),
        )

        assert model1.matches(model2)

    def test_invalid_models_match(self):
        """Two invalid models match."""
        model1 = DistortionModel()
        model2 = DistortionModel()

        assert model1.matches(model2)

    def test_valid_invalid_dont_match(self):
        """Valid and invalid models don't match."""
        model1 = create_distortion_model(
            top_polyline=np.array([[0, 0], [100, 0]]),
            bottom_polyline=np.array([[0, 100], [100, 100]]),
        )
        model2 = DistortionModel()

        assert not model1.matches(model2)


class TestConvexityValidation:
    """Tests for corner convexity validation."""

    def test_rectangular_is_convex(self):
        """Rectangular corners are convex."""
        model = create_distortion_model(
            top_polyline=np.array([[0, 0], [100, 0]]),
            bottom_polyline=np.array([[0, 100], [100, 100]]),
        )
        assert model.is_valid()

    def test_trapezoidal_is_convex(self):
        """Trapezoidal corners are convex."""
        model = create_distortion_model(
            top_polyline=np.array([[10, 0], [90, 0]]),
            bottom_polyline=np.array([[0, 100], [100, 100]]),
        )
        assert model.is_valid()

    def test_self_intersecting_not_convex(self):
        """Self-intersecting quadrilateral is not convex."""
        # Bowtie shape
        model = create_distortion_model(
            top_polyline=np.array([[0, 0], [100, 0]]),
            bottom_polyline=np.array([[100, 100], [0, 100]]),  # Reversed
        )
        assert not model.is_valid()
