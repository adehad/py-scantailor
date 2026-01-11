"""Spline fitting utilities using scipy optimization.

This module provides functionality to fit X-splines to data points,
commonly used for tracing text lines during dewarping.

SIMPLIFICATIONS FROM C++ IMPLEMENTATION
========================================

The original C++ ScanTailor spline fitting system (src/math/spfit/) consists of:

1. **Optimizer** - Custom quadratic programming solver with:
   - Linear equality constraints (b^T * x + c = 0)
   - External forces (attraction to data points)
   - Internal forces (smoothness/regularization)
   - Incremental updates with undo capability

2. **SplineFitter** - Coordinates spline sampling and optimization:
   - Uses QuadraticFunction to model squared distances
   - SqDistApproximant for local quadratic approximation of distance
   - Sparse mapping for efficient variable handling
   - Support for constraints (e.g., fixing endpoints)

3. **ModelShape** - Abstract interface for target shapes:
   - PolylineModelShape for fitting to polylines
   - FrenetFrame for local coordinate systems
   - localSqDistApproximant() for distance approximation

4. **ConstraintSet** - Linear constraints on control points

PYTHON SIMPLIFICATION:
- Uses scipy.optimize.minimize (L-BFGS-B) instead of custom optimizer
- Direct point-to-spline distance calculation instead of quadratic approximation
- No constraint support (endpoints are free to move)
- Simpler regularization via curvature penalty instead of quadratic internal forces

TRADE-OFFS:
- Pros: Much simpler code, leverages well-tested scipy optimization
- Cons: May be slower for large splines, no constraint support
- For typical dewarping use cases (< 20 control points), performance is adequate

FUTURE IMPROVEMENTS (if needed):
- Add constraint support for fixed endpoints
- Implement analytical gradients for faster convergence
- Port PolylineModelShape for more accurate distance calculation
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from scantailor.math.spline import XSpline

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class FitResult:
    """Result of spline fitting operation.

    Attributes:
        spline: The fitted XSpline.
        residual: Sum of squared distances from data points to spline.
        success: Whether the optimization converged.
        iterations: Number of optimization iterations.
    """

    spline: XSpline
    residual: float
    success: bool
    iterations: int


def fit_spline_to_points(
    points: NDArray[np.float64],
    num_control_points: int = 5,
    tension: float = -0.5,
    max_iterations: int = 100,
) -> FitResult:
    """Fit an X-spline to a set of data points.

    Uses scipy optimization to find control point positions that minimize
    the sum of squared distances from the data points to the spline.

    Args:
        points: Array of shape (N, 2) with data points to fit.
        num_control_points: Number of control points for the spline.
        tension: Tension value for all control points (default -0.5 for smooth fit).
        max_iterations: Maximum optimization iterations.

    Returns:
        FitResult with the fitted spline and optimization info.

    Raises:
        ValueError: If fewer than 2 points provided or invalid num_control_points.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        msg = "Points must be array of shape (N, 2)"
        raise ValueError(msg)
    if len(points) < 2:
        msg = "At least 2 points required for fitting"
        raise ValueError(msg)
    if num_control_points < 2:
        msg = "At least 2 control points required"
        raise ValueError(msg)

    # Initialize control points by uniformly distributing along the point range
    t_values = np.linspace(0, 1, num_control_points)
    point_indices = (t_values * (len(points) - 1)).astype(int)
    initial_control_points = points[point_indices].flatten()

    def objective(control_point_coords: NDArray[np.float64]) -> float:
        """Compute sum of squared distances from points to spline."""
        spline = _create_spline_from_coords(control_point_coords, tension)
        total_sq_dist = 0.0
        for pt in points:
            closest, _ = spline.point_closest_to(pt, accuracy=0.5)
            total_sq_dist += float(np.sum((pt - closest) ** 2))
        return total_sq_dist

    # Run optimization
    result = minimize(
        objective,
        initial_control_points,
        method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )

    # Create final spline from optimized positions
    final_spline = _create_spline_from_coords(result.x, tension)

    return FitResult(
        spline=final_spline,
        residual=float(result.fun),
        success=bool(result.success),
        iterations=int(result.nit),
    )


def fit_spline_to_polyline(
    polyline: NDArray[np.float64],
    num_control_points: int = 5,
    tension: float = -0.5,
    smoothing: float = 0.1,
    max_iterations: int = 100,
) -> FitResult:
    """Fit an X-spline to a polyline with smoothing regularization.

    Similar to fit_spline_to_points but adds a smoothing term that penalizes
    curvature, useful for fitting to noisy polylines from traced text lines.

    Args:
        polyline: Array of shape (N, 2) with polyline vertices.
        num_control_points: Number of control points for the spline.
        tension: Tension value for all control points.
        smoothing: Weight for smoothing regularization (0 = no smoothing).
        max_iterations: Maximum optimization iterations.

    Returns:
        FitResult with the fitted spline and optimization info.
    """
    polyline = np.asarray(polyline, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[1] != 2:
        msg = "Polyline must be array of shape (N, 2)"
        raise ValueError(msg)
    if len(polyline) < 2:
        msg = "At least 2 points required for fitting"
        raise ValueError(msg)
    if num_control_points < 2:
        msg = "At least 2 control points required"
        raise ValueError(msg)

    # Initialize control points
    t_values = np.linspace(0, 1, num_control_points)
    point_indices = (t_values * (len(polyline) - 1)).astype(int)
    initial_control_points = polyline[point_indices].flatten()

    def objective(control_point_coords: NDArray[np.float64]) -> float:
        """Compute objective with data fit and smoothing terms."""
        spline = _create_spline_from_coords(control_point_coords, tension)

        # Data fitting term: sum of squared distances
        data_term = 0.0
        for pt in polyline:
            closest, _ = spline.point_closest_to(pt, accuracy=0.5)
            data_term += float(np.sum((pt - closest) ** 2))

        # Smoothing term: penalize second derivatives (curvature)
        smooth_term = 0.0
        if smoothing > 0:
            for t in np.linspace(0.1, 0.9, 9):
                pd = spline.point_and_derivs_at(t)
                curvature = pd.signed_curvature()
                smooth_term += curvature**2

        return data_term + smoothing * smooth_term

    result = minimize(
        objective,
        initial_control_points,
        method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )

    final_spline = _create_spline_from_coords(result.x, tension)

    return FitResult(
        spline=final_spline,
        residual=float(result.fun),
        success=bool(result.success),
        iterations=int(result.nit),
    )


def fit_spline_initial_guess(
    points: NDArray[np.float64],
    num_control_points: int = 5,
    tension: float = -0.5,
) -> XSpline:
    """Create an initial spline guess by uniformly distributing control points.

    This provides a reasonable starting point without optimization, useful
    when a quick approximation is needed or as initialization for manual
    adjustment.

    Args:
        points: Array of shape (N, 2) with data points.
        num_control_points: Number of control points to create.
        tension: Tension value for all control points.

    Returns:
        XSpline with control points distributed along the data points.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        msg = "At least 2 points required"
        raise ValueError(msg)

    # Distribute control points uniformly by arc length
    if len(points) <= num_control_points:
        # Use all points as control points
        spline = XSpline()
        for pt in points:
            spline.append_control_point(pt, tension)
        return spline

    # Calculate cumulative arc length
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length < 1e-10:
        # Degenerate case: all points at same location
        spline = XSpline()
        spline.append_control_point(points[0], tension)
        spline.append_control_point(points[0], tension)
        return spline

    # Sample at uniform arc lengths
    target_lengths = np.linspace(0, total_length, num_control_points)

    spline = XSpline()
    for target in target_lengths:
        # Find segment containing target length
        idx = np.searchsorted(cumulative_length, target, side="right") - 1
        idx = max(0, min(idx, len(points) - 2))

        # Interpolate within segment
        seg_start = cumulative_length[idx]
        seg_length = segment_lengths[idx] if idx < len(segment_lengths) else 1e-10
        if seg_length < 1e-10:
            pt = points[idx]
        else:
            alpha = (target - seg_start) / seg_length
            alpha = max(0, min(1, alpha))
            pt = points[idx] * (1 - alpha) + points[idx + 1] * alpha

        spline.append_control_point(pt, tension)

    return spline


def _create_spline_from_coords(coords: NDArray[np.float64], tension: float) -> XSpline:
    """Create XSpline from flattened coordinate array."""
    spline = XSpline()
    num_points = len(coords) // 2
    for i in range(num_points):
        x = coords[i * 2]
        y = coords[i * 2 + 1]
        spline.append_control_point([x, y], tension)
    return spline


def sample_spline_uniformly(
    spline: XSpline,
    num_samples: int,
    from_t: float = 0.0,
    to_t: float = 1.0,
) -> NDArray[np.float64]:
    """Sample points uniformly along a spline by parameter.

    Args:
        spline: The spline to sample.
        num_samples: Number of samples to generate.
        from_t: Starting parameter value.
        to_t: Ending parameter value.

    Returns:
        Array of shape (num_samples, 2) with sampled points.
    """
    if num_samples < 1:
        return np.zeros((0, 2), dtype=np.float64)

    t_values = np.linspace(from_t, to_t, num_samples)
    points = np.array([spline.point_at(t) for t in t_values], dtype=np.float64)
    return points


def sample_spline_by_arc_length(
    spline: XSpline,
    num_samples: int,
    from_t: float = 0.0,
    to_t: float = 1.0,
) -> NDArray[np.float64]:
    """Sample points uniformly along a spline by arc length.

    This provides more evenly spaced samples than uniform parameter sampling
    when the spline has varying curvature.

    Args:
        spline: The spline to sample.
        num_samples: Number of samples to generate.
        from_t: Starting parameter value.
        to_t: Ending parameter value.

    Returns:
        Array of shape (num_samples, 2) with sampled points.
    """
    if num_samples < 1:
        return np.zeros((0, 2), dtype=np.float64)

    if num_samples == 1:
        return np.array([spline.point_at((from_t + to_t) / 2)], dtype=np.float64)

    # First, sample densely to estimate arc length
    dense_samples = 100
    t_dense = np.linspace(from_t, to_t, dense_samples)
    points_dense = np.array([spline.point_at(t) for t in t_dense], dtype=np.float64)

    # Compute cumulative arc length
    diffs = np.diff(points_dense, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]

    if total_length < 1e-10:
        return np.tile(points_dense[0], (num_samples, 1))

    # Find t values for uniform arc length spacing
    target_lengths = np.linspace(0, total_length, num_samples)

    result = []
    for target in target_lengths:
        # Find segment containing target length
        idx = np.searchsorted(cumulative, target, side="right") - 1
        idx = max(0, min(idx, dense_samples - 2))

        # Interpolate t value
        seg_length = segment_lengths[idx] if idx < len(segment_lengths) else 1e-10
        if seg_length < 1e-10:
            t = t_dense[idx]
        else:
            alpha = (target - cumulative[idx]) / seg_length
            alpha = max(0, min(1, alpha))
            t = t_dense[idx] * (1 - alpha) + t_dense[idx + 1] * alpha

        result.append(spline.point_at(t))

    return np.array(result, dtype=np.float64)
