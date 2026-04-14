"""Arc length mapper for dewarping.

This module provides the ArcLengthMapper class that maps between x coordinates
and arc lengths along a discrete function, essential for cylindrical surface dewarping.

SIMPLIFICATIONS FROM C++ IMPLEMENTATION
========================================

The original C++ ArcLengthMapper (src/math/ArcLengthMapper.cpp) uses:
- A Hint class that tracks last segment and search direction for O(1) sequential access
- Binary search fallback for random access
- Separate interpolation functions for x->arclen and arclen->x

PYTHON SIMPLIFICATION:
- Uses numpy for vectorized operations
- Simpler linear interpolation using np.interp
- Hint mechanism replaced with numpy searchsorted (binary search)
- Performance is comparable for typical polyline sizes (< 1000 points)

The mapping assumes adjacent samples are connected by straight lines,
so arc length between samples is monotonically increasing.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ArcLengthMapper:
    """Maps between x coordinates and arc lengths along a discrete function.

    This class handles a discrete function where we only know values at
    specific x coordinates. It computes cumulative arc lengths and provides
    bidirectional mapping between x and arc length.

    The arc length is computed assuming adjacent samples are connected by
    straight lines in (x, f(x)) space.

    Attributes:
        _samples_x: Array of x coordinates for samples.
        _samples_arclen: Array of cumulative arc lengths at each sample.
    """

    _samples_x: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    _samples_arclen: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    _prev_fx: float = field(default=0.0, repr=False)

    def add_sample(self, x: float, fx: float) -> None:
        """Add an x -> f(x) sample.

        Note that the x value of every sample must be greater than the previous one.

        Args:
            x: The x coordinate of the sample.
            fx: The function value at x (used for arc length calculation).
        """
        if len(self._samples_x) == 0:
            arc_len = 0.0
        else:
            dx = x - self._samples_x[-1]
            dy = fx - self._prev_fx
            arc_len = self._samples_arclen[-1] + np.sqrt(dx * dx + dy * dy)

        self._samples_x = np.append(self._samples_x, x)
        self._samples_arclen = np.append(self._samples_arclen, arc_len)
        self._prev_fx = fx

    def total_arc_length(self) -> float:
        """Return the total arc length from first to last sample.

        Returns:
            Total arc length, or 0.0 if fewer than 2 samples.
        """
        if len(self._samples_arclen) < 2:
            return 0.0
        return float(self._samples_arclen[-1])

    def normalize_range(self, total_arc_len: float) -> None:
        """Scale arc lengths so total equals the given value.

        This should be called after all samples have been added.

        Args:
            total_arc_len: The desired total arc length.
        """
        if len(self._samples_arclen) <= 1:
            return

        if self._samples_arclen[-1] == 0:
            return

        scale = total_arc_len / self._samples_arclen[-1]
        self._samples_arclen = self._samples_arclen * scale

    def arc_len_to_x(self, arc_len: float) -> float:
        """Map from arc length to the corresponding x coordinate.

        Works for arc lengths beyond the first or last sample via extrapolation.

        Args:
            arc_len: Arc length to convert.

        Returns:
            The x coordinate corresponding to the arc length.
        """
        n = len(self._samples_x)
        if n == 0:
            return 0.0
        if n == 1:
            return float(self._samples_x[0])

        # Handle extrapolation beyond bounds
        if arc_len <= self._samples_arclen[0]:
            return self._interpolate_arc_len_in_segment(arc_len, 0)
        if arc_len >= self._samples_arclen[-1]:
            return self._interpolate_arc_len_in_segment(arc_len, n - 2)

        # Binary search for the segment
        idx = int(np.searchsorted(self._samples_arclen, arc_len, side="right")) - 1
        idx = max(0, min(idx, n - 2))
        return self._interpolate_arc_len_in_segment(arc_len, idx)

    def x_to_arc_len(self, x: float) -> float:
        """Map from x coordinate to arc length.

        Works for x values beyond the first or last sample via extrapolation.

        Args:
            x: X coordinate to convert.

        Returns:
            The arc length corresponding to the x coordinate.
        """
        n = len(self._samples_x)
        if n == 0:
            return 0.0
        if n == 1:
            return float(self._samples_arclen[0])

        # Handle extrapolation beyond bounds
        if x <= self._samples_x[0]:
            return self._interpolate_x_in_segment(x, 0)
        if x >= self._samples_x[-1]:
            return self._interpolate_x_in_segment(x, n - 2)

        # Binary search for the segment
        idx = int(np.searchsorted(self._samples_x, x, side="right")) - 1
        idx = max(0, min(idx, n - 2))
        return self._interpolate_x_in_segment(x, idx)

    def _interpolate_arc_len_in_segment(self, arc_len: float, segment: int) -> float:
        """Interpolate x value from arc length within a segment.

        Formula: x = x0 + (arcLen - a0) * (x1 - x0) / (a1 - a0)

        Args:
            arc_len: Arc length to convert.
            segment: Segment index.

        Returns:
            Interpolated x coordinate.
        """
        x0 = self._samples_x[segment]
        a0 = self._samples_arclen[segment]
        x1 = self._samples_x[segment + 1]
        a1 = self._samples_arclen[segment + 1]

        if abs(a1 - a0) < 1e-10:
            return float(x0)

        return float(x0 + (arc_len - a0) * (x1 - x0) / (a1 - a0))

    def _interpolate_x_in_segment(self, x: float, segment: int) -> float:
        """Interpolate arc length from x within a segment.

        Formula: a = a0 + (a1 - a0) * (x - x0) / (x1 - x0)

        Args:
            x: X coordinate to convert.
            segment: Segment index.

        Returns:
            Interpolated arc length.
        """
        x0 = self._samples_x[segment]
        a0 = self._samples_arclen[segment]
        x1 = self._samples_x[segment + 1]
        a1 = self._samples_arclen[segment + 1]

        if abs(x1 - x0) < 1e-10:
            return float(a0)

        return float(a0 + (a1 - a0) * (x - x0) / (x1 - x0))

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""
        return len(self._samples_x)

    @property
    def samples(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the samples as (x_array, arclen_array)."""
        return self._samples_x.copy(), self._samples_arclen.copy()
