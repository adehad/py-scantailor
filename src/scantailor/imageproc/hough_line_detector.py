"""Hough transform line finder for nearly vertical lines.

Lines are parameterised by the angle of their unit normal and their distance
from the origin along it. Every point votes into one bin per angle, and the
lines reported are the local maxima of the vote histogram.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

_DEG2RAD = math.pi / 180.0
_PEAK_WINDOW = 5
_VOTE_CHUNK_ELEMENTS = 1 << 19


@dataclass(frozen=True)
class HoughLine:
    """A line as a unit normal, its distance from the origin and its vote sum.

    Attributes:
        norm_x (float): X component of the unit normal.
        norm_y (float): Y component of the unit normal.
        distance (float): Signed distance of the line from the origin along the normal.
        quality (int): Sum of the weights of the points that voted for the line.
    """

    norm_x: float
    norm_y: float
    distance: float
    quality: int

    def point_at_y(self, y: float) -> tuple[float, float]:
        """Return the point of the line at height ``y``."""
        return ((self.distance - y * self.norm_y) / self.norm_x, y)


class HoughLineDetector:
    """Accumulates weighted points and finds the lines they line up on."""

    def __init__(
        self,
        width: int,
        height: int,
        distance_resolution: float,
        start_angle: float,
        angle_delta: float,
        num_angles: int,
    ) -> None:
        """Prepare an empty vote histogram.

        Args:
            width (int): Width of the input plane; x ranges over 0..width-1.
            height (int): Height of the input plane; y ranges over 0..height-1.
            distance_resolution (float): Width of the lines sought, in input
                units; points this close to a line count towards it.
            start_angle (float): First normal angle, degrees from the x axis.
            angle_delta (float): Degrees between consecutive angles.
            num_angles (int): How many angles to test.
        """
        max_x, max_y = width - 1, height - 1
        radians = [
            (start_angle + angle_delta * i) * _DEG2RAD for i in range(num_angles)
        ]
        self._cos = np.array([math.cos(a) for a in radians], dtype=np.float64)
        self._sin = np.array([math.sin(a) for a in radians], dtype=np.float64)

        max_distance = 0.0
        min_distance = 0.0
        for px, py in ((max_x, max_y), (max_x, 0), (0, max_y)):
            for c, s in zip(self._cos.tolist(), self._sin.tolist(), strict=True):
                distance = c * px + s * py
                max_distance = max(max_distance, distance)
                min_distance = min(min_distance, distance)

        self._resolution = distance_resolution
        self._reciprocal_resolution = 1.0 / distance_resolution
        self._bias = -min_distance
        max_bin = int((max_distance + self._bias) * self._reciprocal_resolution + 0.5)
        self._histogram = np.zeros((num_angles, max_bin + 1), dtype=np.uint32)

    def process(
        self,
        xs: NDArray[np.int64],
        ys: NDArray[np.int64],
        weights: NDArray[np.uint32],
    ) -> None:
        """Add weighted points to the histogram.

        Args:
            xs (NDArray[np.int64]): X coordinates, inside the input plane.
            ys (NDArray[np.int64]): Y coordinates, inside the input plane.
            weights (NDArray[np.uint32]): Vote weight of each point.
        """
        if xs.size == 0:
            return
        angles, bins = self._histogram.shape
        x = xs.astype(np.float64)
        y = ys.astype(np.float64)
        # Angles go in chunks whose (angles, points) buffers are allocated once
        # and reused; fresh multi-megabyte arrays cost more than the arithmetic.
        chunk = min(angles, max(1, _VOTE_CHUNK_ELEMENTS // xs.size))
        distance = np.empty((chunk, xs.size), dtype=np.float64)
        term = np.empty_like(distance)
        index = np.empty(distance.shape, dtype=np.int64)
        row_offsets = np.arange(chunk, dtype=np.int64)[:, None] * bins
        votes_per_row = np.tile(weights.astype(np.float64), chunk)
        for first in range(0, angles, chunk):
            rows = min(chunk, angles - first)
            d, t, i = distance[:rows], term[:rows], index[:rows]
            np.multiply(self._cos[first : first + rows, None], x, out=d)
            np.multiply(self._sin[first : first + rows, None], y, out=t)
            d += t
            d += self._bias
            d *= self._reciprocal_resolution
            d += 0.5
            i[...] = d
            i += row_offsets[:rows]
            # bincount sums in float64, exact while a bin stays below 2**53.
            votes = np.bincount(
                i.ravel(),
                weights=votes_per_row[: i.size],
                minlength=rows * bins,
            )
            self._histogram[first : first + rows] += votes.reshape(rows, bins).astype(
                np.uint32
            )

    def visualize_hough_space(self, quality_lower_bound: int) -> NDArray[np.uint8]:
        """Draw the vote histogram with its peaks, for debug images.

        Args:
            quality_lower_bound (int): The bound ``find_lines`` would get.

        Returns:
            NDArray[np.uint8]: ``(angles, distance bins, 3)`` BGR image: votes
                as gray scaled to the highest bin, peaks tinted half red.
        """
        highest = int(self._histogram.max(initial=0))
        if highest == 0:
            gray = np.zeros(self._histogram.shape, dtype=np.uint8)
        else:
            gray = np.floor(self._histogram * (255.0 / highest) + 0.5).astype(np.uint8)
        visual = np.asarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
        peaks = _find_histogram_peaks(self._histogram, quality_lower_bound)
        red = np.array([0, 0, 255], dtype=np.float64)
        visual[peaks] = np.round((visual[peaks] + red) / 2).astype(np.uint8)
        return visual

    def find_lines(self, quality_lower_bound: int) -> list[HoughLine]:
        """Return the lines whose vote sum reaches the bound, best first.

        Args:
            quality_lower_bound (int): Smallest acceptable vote sum.

        Returns:
            list[HoughLine]: Lines ordered by falling quality; equal qualities
            keep the raster order of their peaks.
        """
        peaks = _find_histogram_peaks(self._histogram, quality_lower_bound)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            peaks.view(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )
        # The first pixel of each peak in raster order seeds it and fixes its
        # place among peaks of equal quality.
        found, first_pixels = np.unique(labels.ravel(), return_index=True)
        in_raster_order = np.argsort(first_pixels)
        lines: list[HoughLine] = []
        for label, first_pixel in zip(
            found[in_raster_order].tolist(),
            first_pixels[in_raster_order].tolist(),
            strict=True,
        ):
            if label == 0:
                continue
            left, top, width, height = stats[label, : cv2.CC_STAT_AREA].tolist()
            centre_row = (2 * top + height - 1) // 2
            centre_col = (2 * left + width - 1) // 2
            level = int(self._histogram.flat[first_pixel])
            distance = (centre_col + 0.5) * self._resolution - self._bias
            lines.append(
                HoughLine(
                    float(self._cos[centre_row]),
                    float(self._sin[centre_row]),
                    distance,
                    level,
                )
            )
        lines.sort(key=lambda line: -line.quality)
        return lines


def _peak_candidates(
    histogram: NDArray[np.uint32], lower_bound: int
) -> NDArray[np.bool_]:
    """Bins at or above the bound that no bin within 2 steps exceeds."""
    # float64 holds every uint32 count exactly; cv2.dilate does not take uint32.
    window_max = cv2.dilate(
        histogram.astype(np.float64),
        np.ones((_PEAK_WINDOW, _PEAK_WINDOW), np.uint8),
        borderType=cv2.BORDER_REPLICATE,
    )
    return (histogram >= lower_bound) & (histogram == window_max)


def _find_histogram_peaks(
    histogram: NDArray[np.uint32], lower_bound: int
) -> NDArray[np.bool_]:
    """Keep only plateaus of equal candidate bins that are strict local maxima.

    A plateau of equal bins survives when no neighbour within 2 steps is higher.
    Raising every bin that touches a plateau by one makes any plateau with a
    higher neighbour change its candidate status, which exposes it.
    """
    candidates = _peak_candidates(histogram, lower_bound)
    grown = cv2.dilate(
        candidates.astype(np.uint8), np.ones((_PEAK_WINDOW, _PEAK_WINDOW), np.uint8)
    ).astype(np.bool_)
    neighbourhood = grown & ~candidates
    raised = histogram + neighbourhood.astype(np.uint32)
    changed = _peak_candidates(raised, lower_bound) ^ candidates

    count, labels = cv2.connectedComponents(
        candidates.view(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    spoiled = np.zeros(count, dtype=np.bool_)
    spoiled[labels[changed & candidates]] = True
    spoiled[0] = False
    return candidates & ~spoiled[labels]
