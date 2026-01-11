"""X-Spline implementation for smooth curve fitting.

X-splines are a spline model with tension control, allowing both interpolating
and approximating behavior. They are particularly useful for dewarping text lines.

Reference:
    Blanc, C., Schlick, C.: X-splines: a spline model designed for the end-user.
    http://scholar.google.com/scholar?cluster=2002168279173394147
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ControlPoint:
    """A control point with position and tension.

    Attributes:
        pos: Position as (x, y) array.
        tension: Tension value in [-1, 1].
            - tension < 0: interpolating patches (curve passes through point)
            - tension == 0: sharp angle interpolating patches
            - tension > 0: approximating patches (curve passes near point)
    """

    pos: NDArray[np.float64]
    tension: float = 0.0

    def __post_init__(self) -> None:
        """Validate and convert position and tension values."""
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.tension = float(np.clip(self.tension, -1.0, 1.0))


@dataclass
class PointAndDerivatives:
    """Point on spline with first and second derivatives.

    Attributes:
        point: Position on the spline.
        first_deriv: First derivative with respect to t.
        second_deriv: Second derivative with respect to t.
    """

    point: NDArray[np.float64]
    first_deriv: NDArray[np.float64]
    second_deriv: NDArray[np.float64]

    def signed_curvature(self) -> float:
        """Calculate signed curvature at this point.

        The sign indicates curving direction. In image coordinates (Y down),
        positive curvature means the curve bends clockwise.

        Returns:
            Signed curvature value.
        """
        cross = (
            self.first_deriv[0] * self.second_deriv[1]
            - self.first_deriv[1] * self.second_deriv[0]
        )
        tangent_length = np.sqrt(self.first_deriv[0] ** 2 + self.first_deriv[1] ** 2)
        if tangent_length < 1e-10:
            return 0.0
        return float(cross / (tangent_length**3))


@dataclass
class LinearCoefficient:
    """Linear combination coefficient for a control point.

    Attributes:
        control_point_idx: Index of the control point.
        coeff: Coefficient value.
    """

    control_point_idx: int
    coeff: float


class _GBlendFunc:
    """G-blend function for X-spline evaluation (formula 20 in paper)."""

    def __init__(self, q: float, p: float) -> None:
        self._c1 = q
        self._c2 = 2 * q
        self._c3 = 10 - 12 * q - p
        self._c4 = 2 * p + 14 * q - 15
        self._c5 = 6 - 5 * q - p

    def value(self, u: float) -> float:
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        u5 = u4 * u
        return (
            self._c1 * u + self._c2 * u2 + self._c3 * u3 + self._c4 * u4 + self._c5 * u5
        )

    def first_derivative(self, u: float) -> float:
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        return (
            self._c1
            + 2 * self._c2 * u
            + 3 * self._c3 * u2
            + 4 * self._c4 * u3
            + 5 * self._c5 * u4
        )

    def second_derivative(self, u: float) -> float:
        u2 = u * u
        u3 = u2 * u
        return 2 * self._c2 + 6 * self._c3 * u + 12 * self._c4 * u2 + 20 * self._c5 * u3


class _HBlendFunc:
    """H-blend function for X-spline evaluation (formula 20 in paper)."""

    def __init__(self, q: float) -> None:
        self._c1 = q
        self._c2 = 2 * q
        self._c4 = -2 * q
        self._c5 = -q

    def value(self, u: float) -> float:
        u2 = u * u
        u4 = u2 * u2
        u5 = u4 * u
        return self._c1 * u + self._c2 * u2 + self._c4 * u4 + self._c5 * u5

    def first_derivative(self, u: float) -> float:
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        return self._c1 + 2 * self._c2 * u + 4 * self._c4 * u3 + 5 * self._c5 * u4

    def second_derivative(self, u: float) -> float:
        u2 = u * u
        u3 = u2 * u
        return 2 * self._c2 + 12 * self._c4 * u2 + 20 * self._c5 * u3


@dataclass
class _TensionDerivedParams:
    """Parameters derived from tension values for a segment."""

    # Fixed knot values for t in [0, 1] within a segment
    T0: float = -1.0
    T1: float = 0.0
    T2: float = 1.0
    T3: float = 2.0

    # Tension-modified knots
    T0p: float = 0.0
    T1p: float = 0.0
    T2m: float = 0.0
    T3m: float = 0.0

    # q and p parameters for blend functions
    q: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    p: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    @classmethod
    def from_tensions(cls, tension1: float, tension2: float) -> _TensionDerivedParams:
        """Create parameters from two adjacent control point tensions."""
        params = cls()

        # Tk+ = t(k+1) + s(k+1), Tk- = t(k-1) - s(k-1)
        s1 = max(tension1, 0.0)
        s2 = max(tension2, 0.0)
        params.T0p = params.T1 + s1
        params.T1p = params.T2 + s2
        params.T2m = params.T1 - s1
        params.T3m = params.T2 - s2

        # q's lie in [0, 0.5]
        s1_ = -0.5 * min(tension1, 0.0)
        s2_ = -0.5 * min(tension2, 0.0)
        params.q = [s1_, s2_, s1_, s2_]

        # Formula 17: p = 2 * (tk - Tk)^2
        params.p = [
            2.0 * (params.T0 - params.T0p) ** 2,
            2.0 * (params.T1 - params.T1p) ** 2,
            2.0 * (params.T2 - params.T2m) ** 2,
            2.0 * (params.T3 - params.T3m) ** 2,
        ]

        return params


@dataclass
class _DecomposedDerivs:
    """Decomposed derivative coefficients for a point on the spline."""

    zero_deriv_coeffs: list[float] = field(default_factory=lambda: [0.0] * 4)
    first_deriv_coeffs: list[float] = field(default_factory=lambda: [0.0] * 4)
    second_deriv_coeffs: list[float] = field(default_factory=lambda: [0.0] * 4)
    control_points: list[int] = field(default_factory=lambda: [0] * 4)
    num_control_points: int = 0


class XSpline:
    """An open X-Spline curve.

    X-splines allow smooth curves with controllable tension at each control point.
    Negative tension creates interpolating splines (passing through control points),
    positive tension creates approximating splines (smooth but not passing through).

    Example:
        >>> spline = XSpline()
        >>> spline.append_control_point([0, 0], tension=-1)  # Interpolating
        >>> spline.append_control_point([5, 10], tension=0)
        >>> spline.append_control_point([10, 0], tension=-1)  # Interpolating
        >>> point = spline.point_at(0.5)  # Get midpoint
    """

    def __init__(self) -> None:
        """Initialize an empty X-Spline."""
        self._control_points: list[ControlPoint] = []

    @property
    def num_control_points(self) -> int:
        """Return the number of control points."""
        return len(self._control_points)

    @property
    def num_segments(self) -> int:
        """Return the number of segments (spans between adjacent control points)."""
        return max(0, len(self._control_points) - 1)

    def control_point_index_to_t(self, idx: int) -> float:
        """Convert a control point index to its t parameter value.

        Args:
            idx: Control point index.

        Returns:
            The t value in [0, 1] corresponding to this control point.
        """
        if len(self._control_points) <= 1:
            return 0.0
        return idx / (len(self._control_points) - 1)

    def append_control_point(
        self,
        pos: NDArray[np.float64] | tuple[float, float] | list[float],
        tension: float = 0.0,
    ) -> None:
        """Append a control point to the end of the spline.

        Args:
            pos: Position as (x, y).
            tension: Tension value in [-1, 1].
        """
        self._control_points.append(
            ControlPoint(pos=np.asarray(pos, dtype=np.float64), tension=tension)
        )

    def insert_control_point(
        self,
        idx: int,
        pos: NDArray[np.float64] | tuple[float, float] | list[float],
        tension: float = 0.0,
    ) -> None:
        """Insert a control point at a specified position.

        Args:
            idx: Position where the new control point will be inserted.
            pos: Position as (x, y).
            tension: Tension value in [-1, 1].
        """
        self._control_points.insert(
            idx, ControlPoint(pos=np.asarray(pos, dtype=np.float64), tension=tension)
        )

    def erase_control_point(self, idx: int) -> None:
        """Remove a control point.

        Args:
            idx: Index of the control point to remove.
        """
        del self._control_points[idx]

    def control_point_position(self, idx: int) -> NDArray[np.float64]:
        """Get the position of a control point.

        Args:
            idx: Control point index.

        Returns:
            Position as (x, y) array.
        """
        return self._control_points[idx].pos.copy()

    def move_control_point(
        self,
        idx: int,
        pos: NDArray[np.float64] | tuple[float, float] | list[float],
    ) -> None:
        """Move a control point to a new position.

        Args:
            idx: Control point index.
            pos: New position as (x, y).
        """
        self._control_points[idx].pos = np.asarray(pos, dtype=np.float64)

    def control_point_tension(self, idx: int) -> float:
        """Get the tension of a control point.

        Args:
            idx: Control point index.

        Returns:
            Tension value in [-1, 1].
        """
        return self._control_points[idx].tension

    def set_control_point_tension(self, idx: int, tension: float) -> None:
        """Set the tension of a control point.

        Args:
            idx: Control point index.
            tension: Tension value in [-1, 1].
        """
        self._control_points[idx].tension = float(np.clip(tension, -1.0, 1.0))

    def point_at(self, t: float) -> NDArray[np.float64]:
        """Calculate a point on the spline at position t.

        Args:
            t: Position on the spline in [0, 1].

        Returns:
            Point on the spline as (x, y) array.

        Raises:
            ValueError: If the spline has fewer than 2 control points.
        """
        num_segs = self.num_segments
        if num_segs == 0:
            if len(self._control_points) == 1:
                return self._control_points[0].pos.copy()
            msg = "Spline must have at least 2 control points"
            raise ValueError(msg)

        t = float(np.clip(t, 0.0, 1.0))

        if t == 1.0:
            return self._point_at_impl(num_segs - 1, 1.0)

        t2 = t * num_segs
        segment = int(np.floor(t2))
        return self._point_at_impl(segment, t2 - segment)

    def point_and_derivs_at(self, t: float) -> PointAndDerivatives:
        """Calculate point and derivatives at position t.

        Args:
            t: Position on the spline in [0, 1].

        Returns:
            PointAndDerivatives with point, first and second derivatives.
        """
        t = float(np.clip(t, 0.0, 1.0))
        derivs = self._decomposed_derivs(t)

        point = np.zeros(2, dtype=np.float64)
        first_deriv = np.zeros(2, dtype=np.float64)
        second_deriv = np.zeros(2, dtype=np.float64)

        for i in range(derivs.num_control_points):
            cp = self._control_points[derivs.control_points[i]].pos
            point += cp * derivs.zero_deriv_coeffs[i]
            first_deriv += cp * derivs.first_deriv_coeffs[i]
            second_deriv += cp * derivs.second_deriv_coeffs[i]

        return PointAndDerivatives(
            point=point, first_deriv=first_deriv, second_deriv=second_deriv
        )

    def linear_combination_at(self, t: float) -> list[LinearCoefficient]:
        """Get the linear combination coefficients at position t.

        The point at t is a weighted sum of control point positions:
        point(t) = sum(coeff[i] * control_point[i].pos)

        Args:
            t: Position on the spline in [0, 1].

        Returns:
            List of LinearCoefficient with control point indices and weights.
        """
        t = float(np.clip(t, 0.0, 1.0))
        num_segs = self.num_segments
        if num_segs == 0:
            return []

        if t == 1.0:
            return self._linear_combination_for(num_segs - 1, 1.0)

        t2 = t * num_segs
        segment = int(np.floor(t2))
        return self._linear_combination_for(segment, t2 - segment)

    def point_closest_to(
        self,
        target: NDArray[np.float64] | tuple[float, float] | list[float],
        accuracy: float = 0.2,
    ) -> tuple[NDArray[np.float64], float]:
        """Find the point on the spline closest to a target point.

        Args:
            target: The point to find the closest spline point to.
            accuracy: Maximum distance from the found point to the spline.

        Returns:
            Tuple of (closest_point, t_value).
        """
        target = np.asarray(target, dtype=np.float64)

        if not self._control_points:
            return np.zeros(2, dtype=np.float64), 0.0

        num_segs = self.num_segments
        if num_segs == 0:
            return self._control_points[0].pos.copy(), 0.0

        # Find the closest segment
        prev_pt = self._point_at_impl(0, 0.0)
        best_segment = 0
        best_sqdist = float(np.sum((target - prev_pt) ** 2))

        for seg in range(num_segs):
            next_pt = self._point_at_impl(seg, 1.0)
            sqdist = self._sq_dist_to_line_segment(target, prev_pt, next_pt)
            if sqdist < best_sqdist:
                best_segment = seg
                best_sqdist = sqdist
            prev_pt = next_pt

        # Binary search within the best segment
        sq_accuracy = accuracy * accuracy
        prev_t = 0.0
        next_t = 1.0
        prev_pt = self._point_at_impl(best_segment, prev_t)
        next_pt = self._point_at_impl(best_segment, next_t)

        while np.sum((prev_pt - next_pt) ** 2) > sq_accuracy:
            mid_t = 0.5 * (prev_t + next_t)
            mid_pt = self._point_at_impl(best_segment, mid_t)

            # Project target and midpoint onto the line from prev to next
            line_vec = next_pt - prev_pt
            line_len_sq = np.sum(line_vec**2)
            if line_len_sq < 1e-10:
                break

            pt = np.dot(target - prev_pt, line_vec) / line_len_sq
            pm = np.dot(mid_pt - prev_pt, line_vec) / line_len_sq

            if pt < pm:
                next_t = mid_t
                next_pt = mid_pt
            else:
                prev_t = mid_t
                prev_pt = mid_pt

        # Return the closer of prev_pt and next_pt
        if np.sum((target - prev_pt) ** 2) < np.sum((target - next_pt) ** 2):
            result_t = (best_segment + prev_t) / num_segs
            return prev_pt, result_t
        else:
            result_t = (best_segment + next_t) / num_segs
            return next_pt, result_t

    def to_polyline(
        self,
        max_dist_from_spline: float = 1.0,
        max_dist_between_samples: float = 10.0,
        from_t: float = 0.0,
        to_t: float = 1.0,
    ) -> NDArray[np.float64]:
        """Convert spline to a polyline by adaptive sampling.

        Args:
            max_dist_from_spline: Maximum distance from sample points to spline.
            max_dist_between_samples: Maximum distance between consecutive samples.
            from_t: Start parameter value.
            to_t: End parameter value.

        Returns:
            Array of shape (N, 2) with polyline vertices.
        """
        if not self._control_points:
            return np.zeros((0, 2), dtype=np.float64)

        points: list[NDArray[np.float64]] = []

        def add_sample(pt: NDArray[np.float64], _t: float) -> None:
            points.append(pt)

        from_pt = self.point_at(from_t)
        to_pt = self.point_at(to_t)

        add_sample(from_pt, from_t)

        if self.num_segments > 0:
            self._maybe_add_more_samples(
                add_sample,
                max_dist_from_spline**2,
                max_dist_between_samples**2,
                self.num_segments,
                from_t,
                from_pt,
                to_t,
                to_pt,
            )

        add_sample(to_pt, to_t)

        return np.array(points, dtype=np.float64)

    def _point_at_impl(self, segment: int, t: float) -> NDArray[np.float64]:
        """Calculate point within a specific segment.

        Args:
            segment: Segment index.
            t: Position within segment in [0, 1].

        Returns:
            Point on the spline.
        """
        coeffs = self._linear_combination_for(segment, t)
        point = np.zeros(2, dtype=np.float64)
        for c in coeffs:
            point += self._control_points[c.control_point_idx].pos * c.coeff
        return point

    def _linear_combination_for(
        self, segment: int, t: float
    ) -> list[LinearCoefficient]:
        """Get linear combination coefficients for a segment.

        Args:
            segment: Segment index.
            t: Position within segment in [0, 1].

        Returns:
            List of LinearCoefficient.
        """
        n = len(self._control_points)

        # Get the 4 control points affecting this segment
        idxs = [
            max(0, segment - 1),
            segment,
            segment + 1,
            min(segment + 2, n - 1),
        ]

        pts = [self._control_points[i] for i in idxs]
        tdp = _TensionDerivedParams.from_tensions(pts[1].tension, pts[2].tension)

        # Compute blend function values
        a = [0.0, 0.0, 0.0, 0.0]

        # Control point 0
        u0 = (
            (t - tdp.T0p) / (tdp.T0 - tdp.T0p) if abs(tdp.T0 - tdp.T0p) > 1e-10 else 0.0
        )
        if t <= tdp.T0p:
            a[0] = _GBlendFunc(tdp.q[0], tdp.p[0]).value(u0)
        else:
            a[0] = _HBlendFunc(tdp.q[0]).value(u0)

        # Control point 1
        u1 = (
            (t - tdp.T1p) / (tdp.T1 - tdp.T1p) if abs(tdp.T1 - tdp.T1p) > 1e-10 else 0.0
        )
        a[1] = _GBlendFunc(tdp.q[1], tdp.p[1]).value(u1)

        # Control point 2
        u2 = (
            (t - tdp.T2m) / (tdp.T2 - tdp.T2m) if abs(tdp.T2 - tdp.T2m) > 1e-10 else 0.0
        )
        a[2] = _GBlendFunc(tdp.q[2], tdp.p[2]).value(u2)

        # Control point 3
        u3 = (
            (t - tdp.T3m) / (tdp.T3 - tdp.T3m) if abs(tdp.T3 - tdp.T3m) > 1e-10 else 0.0
        )
        if t >= tdp.T3m:
            a[3] = _GBlendFunc(tdp.q[3], tdp.p[3]).value(u3)
        else:
            a[3] = _HBlendFunc(tdp.q[3]).value(u3)

        # Normalize
        total = float(sum(a))
        if abs(total) > 1e-10:
            a = [x / total for x in a]

        # Merge coefficients for same control points
        coeffs: list[LinearCoefficient] = []
        if idxs[0] == idxs[1]:
            coeffs.append(LinearCoefficient(idxs[0], a[0] + a[1]))
        else:
            coeffs.append(LinearCoefficient(idxs[0], a[0]))
            coeffs.append(LinearCoefficient(idxs[1], a[1]))

        if idxs[2] == idxs[3]:
            coeffs.append(LinearCoefficient(idxs[2], a[2] + a[3]))
        else:
            coeffs.append(LinearCoefficient(idxs[2], a[2]))
            coeffs.append(LinearCoefficient(idxs[3], a[3]))

        return coeffs

    def _decomposed_derivs(self, t: float) -> _DecomposedDerivs:
        """Compute decomposed derivatives at position t."""
        num_segs = self.num_segments
        if num_segs == 0:
            return _DecomposedDerivs()

        if t == 1.0:
            return self._decomposed_derivs_impl(num_segs - 1, 1.0)

        t2 = t * num_segs
        segment = int(np.floor(t2))
        return self._decomposed_derivs_impl(segment, t2 - segment)

    def _decomposed_derivs_impl(self, segment: int, t: float) -> _DecomposedDerivs:
        """Compute decomposed derivatives within a segment."""
        derivs = _DecomposedDerivs()
        n = len(self._control_points)

        derivs.num_control_points = 4
        derivs.control_points = [
            max(0, segment - 1),
            segment,
            segment + 1,
            min(segment + 2, n - 1),
        ]

        pts = [self._control_points[i] for i in derivs.control_points]
        tdp = _TensionDerivedParams.from_tensions(pts[1].tension, pts[2].tension)

        # dt/dT = numSegments (derivative scaling factor)
        dt_dT = self.num_segments

        a = [0.0, 0.0, 0.0, 0.0]
        da = [0.0, 0.0, 0.0, 0.0]
        dda = [0.0, 0.0, 0.0, 0.0]

        # Control point 0
        denom0 = tdp.T0 - tdp.T0p
        if abs(denom0) > 1e-10:
            ta = 1.0 / denom0
            u = ta * t + (-tdp.T0p * ta)
            if t <= tdp.T0p:
                g = _GBlendFunc(tdp.q[0], tdp.p[0])
                a[0] = g.value(u)
                da[0] = g.first_derivative(u) * (ta * dt_dT)
                dda[0] = g.second_derivative(u) * (ta * dt_dT) ** 2
            else:
                h = _HBlendFunc(tdp.q[0])
                a[0] = h.value(u)
                da[0] = h.first_derivative(u) * (ta * dt_dT)
                dda[0] = h.second_derivative(u) * (ta * dt_dT) ** 2

        # Control point 1
        denom1 = tdp.T1 - tdp.T1p
        if abs(denom1) > 1e-10:
            ta = 1.0 / denom1
            u = ta * t + (-tdp.T1p * ta)
            g = _GBlendFunc(tdp.q[1], tdp.p[1])
            a[1] = g.value(u)
            da[1] = g.first_derivative(u) * (ta * dt_dT)
            dda[1] = g.second_derivative(u) * (ta * dt_dT) ** 2

        # Control point 2
        denom2 = tdp.T2 - tdp.T2m
        if abs(denom2) > 1e-10:
            ta = 1.0 / denom2
            u = ta * t + (-tdp.T2m * ta)
            g = _GBlendFunc(tdp.q[2], tdp.p[2])
            a[2] = g.value(u)
            da[2] = g.first_derivative(u) * (ta * dt_dT)
            dda[2] = g.second_derivative(u) * (ta * dt_dT) ** 2

        # Control point 3
        denom3 = tdp.T3 - tdp.T3m
        if abs(denom3) > 1e-10:
            ta = 1.0 / denom3
            u = ta * t + (-tdp.T3m * ta)
            if t >= tdp.T3m:
                g = _GBlendFunc(tdp.q[3], tdp.p[3])
                a[3] = g.value(u)
                da[3] = g.first_derivative(u) * (ta * dt_dT)
                dda[3] = g.second_derivative(u) * (ta * dt_dT) ** 2
            else:
                h = _HBlendFunc(tdp.q[3])
                a[3] = h.value(u)
                da[3] = h.first_derivative(u) * (ta * dt_dT)
                dda[3] = h.second_derivative(u) * (ta * dt_dT) ** 2

        # Normalize and compute derivative coefficients
        total = float(sum(a))
        total2 = total * total
        total4 = total2 * total2
        d_total = sum(da)
        dd_total = sum(dda)

        for i in range(4):
            if abs(total) > 1e-10:
                derivs.zero_deriv_coeffs[i] = a[i] / total

                d1 = da[i] * total - a[i] * d_total
                derivs.first_deriv_coeffs[i] = (
                    d1 / total2 if abs(total2) > 1e-10 else 0.0
                )

                dd1 = dda[i] * total + da[i] * d_total
                dd2 = da[i] * d_total + a[i] * dd_total
                if abs(total4) > 1e-10:
                    dd3 = ((dd1 - dd2) * total2 - d1 * (2 * total * d_total)) / total4
                else:
                    dd3 = 0.0
                derivs.second_deriv_coeffs[i] = dd3

        # Merge control points with same index
        write_idx = 0
        merge_idx = 0
        read_idx = 1

        while read_idx < 4:
            # Merge all consecutive points with same index
            while (
                read_idx < 4
                and derivs.control_points[read_idx] == derivs.control_points[merge_idx]
            ):
                derivs.zero_deriv_coeffs[merge_idx] += derivs.zero_deriv_coeffs[
                    read_idx
                ]
                derivs.first_deriv_coeffs[merge_idx] += derivs.first_deriv_coeffs[
                    read_idx
                ]
                derivs.second_deriv_coeffs[merge_idx] += derivs.second_deriv_coeffs[
                    read_idx
                ]
                read_idx += 1

            # Check if coefficients are non-zero
            coeff_sum = (
                abs(derivs.zero_deriv_coeffs[merge_idx])
                + abs(derivs.first_deriv_coeffs[merge_idx])
                + abs(derivs.second_deriv_coeffs[merge_idx])
            )
            if coeff_sum > 1e-10:
                derivs.zero_deriv_coeffs[write_idx] = derivs.zero_deriv_coeffs[
                    merge_idx
                ]
                derivs.first_deriv_coeffs[write_idx] = derivs.first_deriv_coeffs[
                    merge_idx
                ]
                derivs.second_deriv_coeffs[write_idx] = derivs.second_deriv_coeffs[
                    merge_idx
                ]
                derivs.control_points[write_idx] = derivs.control_points[merge_idx]
                write_idx += 1

            if read_idx >= 4:
                break

            merge_idx = read_idx
            read_idx += 1

        # Handle the last merge group
        if merge_idx < 4:
            coeff_sum = (
                abs(derivs.zero_deriv_coeffs[merge_idx])
                + abs(derivs.first_deriv_coeffs[merge_idx])
                + abs(derivs.second_deriv_coeffs[merge_idx])
            )
            if coeff_sum > 1e-10:
                derivs.zero_deriv_coeffs[write_idx] = derivs.zero_deriv_coeffs[
                    merge_idx
                ]
                derivs.first_deriv_coeffs[write_idx] = derivs.first_deriv_coeffs[
                    merge_idx
                ]
                derivs.second_deriv_coeffs[write_idx] = derivs.second_deriv_coeffs[
                    merge_idx
                ]
                derivs.control_points[write_idx] = derivs.control_points[merge_idx]
                write_idx += 1

        derivs.num_control_points = write_idx
        return derivs

    def _maybe_add_more_samples(
        self,
        sink: Callable,
        max_sqdist_to_spline: float,
        max_sqdist_between_samples: float,
        num_segments: int,
        prev_t: float,
        prev_pt: NDArray[np.float64],
        next_t: float,
        next_pt: NDArray[np.float64],
    ) -> None:
        """Recursively add samples to ensure accuracy."""
        prev_next_sqdist = float(np.sum((next_pt - prev_pt) ** 2))
        if prev_next_sqdist < 1e-6:
            return

        mid_t = 0.5 * (prev_t + next_t)

        # Check if there's a junction point nearby
        r_num_segments = 1.0 / num_segments if num_segments > 0 else 1.0
        nearby_junction_t = round(mid_t * num_segments) * r_num_segments

        is_junction = False
        if (nearby_junction_t - prev_t) * (next_t - prev_t) > 0 and (
            nearby_junction_t - next_t
        ) * (prev_t - next_t) > 0:
            mid_t = nearby_junction_t
            is_junction = True

        mid_pt = self.point_at(mid_t)

        if not is_junction:
            # Check if we need more samples
            line_vec = next_pt - prev_pt
            line_len_sq = np.sum(line_vec**2)
            if line_len_sq > 1e-10:
                proj_scalar = np.dot(mid_pt - prev_pt, line_vec) / line_len_sq
                proj_scalar = np.clip(proj_scalar, 0.0, 1.0)
                projection = prev_pt + proj_scalar * line_vec
                dist_to_line_sq = float(np.sum((mid_pt - projection) ** 2))

                if (
                    prev_next_sqdist <= max_sqdist_between_samples
                    and dist_to_line_sq <= max_sqdist_to_spline
                ):
                    return

        # Add more samples recursively
        self._maybe_add_more_samples(
            sink,
            max_sqdist_to_spline,
            max_sqdist_between_samples,
            num_segments,
            prev_t,
            prev_pt,
            mid_t,
            mid_pt,
        )

        sink(mid_pt, mid_t)

        self._maybe_add_more_samples(
            sink,
            max_sqdist_to_spline,
            max_sqdist_between_samples,
            num_segments,
            mid_t,
            mid_pt,
            next_t,
            next_pt,
        )

    @staticmethod
    def _sq_dist_to_line_segment(
        pt: NDArray[np.float64],
        line_p1: NDArray[np.float64],
        line_p2: NDArray[np.float64],
    ) -> float:
        """Calculate squared distance from a point to a line segment."""
        line_vec = line_p2 - line_p1
        line_len_sq = float(np.sum(line_vec**2))

        if line_len_sq < 1e-10:
            return float(np.sum((pt - line_p1) ** 2))

        proj_scalar = np.dot(pt - line_p1, line_vec) / line_len_sq

        if proj_scalar <= 0:
            closest = line_p1
        elif proj_scalar >= 1:
            closest = line_p2
        else:
            closest = line_p1 + proj_scalar * line_vec

        return float(np.sum((pt - closest) ** 2))
