"""Skew of a black-and-white page found by shearing it and scoring the rows.

A coarse scan over whole degrees is refined by a binary search. A shear that
lines the text up makes row counts jump between text lines and gaps, which
shows as a large sum of squared differences between neighbouring rows.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from scantailor.imageproc.binary_shear import shift_blocks
from scantailor.imageproc.reduce_threshold import reduce_threshold

_DEG2RAD = math.pi / 180.0
_COARSE_STEP = 1.0
_LOW_SCORE = 1000.0


@dataclass(frozen=True)
class Skew:
    """A skew angle and how far it can be trusted.

    Attributes:
        angle (float): Degrees; positive means the page is skewed clockwise.
        confidence (float): 0 is the worst; ``GOOD_CONFIDENCE`` and above is high.
    """

    angle: float
    confidence: float

    GOOD_CONFIDENCE: ClassVar[float] = 2.0


def find_skew_binary(
    image: NDArray[np.bool_],
    *,
    max_angle: float = 7.0,
    min_angle: float = 0.1,
    accuracy: float = 0.1,
    resolution_ratio: float = 1.0,
    coarse_reduction: int = 2,
    fine_reduction: int = 1,
) -> Skew:
    """Find the skew of a page whose text lines are roughly horizontal.

    Args:
        image (NDArray[np.bool_]): (H, W) page, True for black.
        max_angle (float): Largest skew searched for, both ways, in degrees.
        min_angle (float): Smaller angles are reported as no skew.
        accuracy (float): Allowed error of the angle in degrees.
        resolution_ratio (float): Horizontal over vertical resolution.
        coarse_reduction (int): Halvings of the image before the coarse scan.
        fine_reduction (int): Halvings of the image before the fine search.

    Returns:
        Skew: The angle that deskews the page when applied as a rotation, and
        its confidence.

    Raises:
        ValueError: If a reduction is negative or the image is empty.
    """
    if coarse_reduction < 0 or fine_reduction < 0:
        raise ValueError("reductions must not be negative")
    if image.size == 0:
        raise ValueError("image is empty")

    shared = min(coarse_reduction, fine_reduction)
    coarse = image
    for i in range(shared):
        coarse = reduce_threshold(coarse, 1 if i == 0 else 2)
    fine = coarse
    for i in range(shared, coarse_reduction):
        coarse = reduce_threshold(coarse, 1 if i == 0 else 2)

    coarse_scorer = _ShearScorer(coarse, resolution_ratio)
    scores: list[float] = []
    best_coarse_score = 0.0
    best_coarse_angle = -max_angle
    angle = -max_angle
    while angle <= max_angle:
        score = coarse_scorer.score(angle)
        scores.append(score)
        if score > best_coarse_score:
            best_coarse_angle = angle
            best_coarse_score = score
        angle += _COARSE_STEP
    sum_coarse = sum(scores)
    num_coarse = len(scores)

    if accuracy >= _COARSE_STEP:
        confidence = 0.0
        if num_coarse > 1 and sum_coarse > 0:
            confidence = best_coarse_score / sum_coarse * num_coarse
        return Skew(-best_coarse_angle, confidence - 1.0)

    for i in range(shared, fine_reduction):
        fine = reduce_threshold(fine, 1 if i == 0 else 2)
    fine_scorer = (
        coarse_scorer if fine is coarse else _ShearScorer(fine, resolution_ratio)
    )

    angle_plus = best_coarse_angle + 0.5 * _COARSE_STEP
    angle_minus = best_coarse_angle - 0.5 * _COARSE_STEP
    score_plus = fine_scorer.score(angle_plus)
    score_minus = fine_scorer.score(angle_minus)
    fine_score_1, fine_score_2 = score_plus, score_minus
    while angle_plus - angle_minus > accuracy:
        if score_plus > score_minus:
            angle_minus = 0.5 * (angle_plus + angle_minus)
            score_minus = fine_scorer.score(angle_minus)
        elif score_plus < score_minus:
            angle_plus = 0.5 * (angle_plus + angle_minus)
            score_plus = fine_scorer.score(angle_plus)
        else:
            break

    if score_plus > score_minus:
        best_angle, best_score = angle_plus, score_plus
    else:
        best_angle, best_score = angle_minus, score_minus

    if abs(best_angle) < min_angle:
        return Skew(0.0, Skew.GOOD_CONFIDENCE)
    if best_score <= _LOW_SCORE:
        return Skew(-best_angle, 0.0)

    if num_coarse > 1:
        confidence = best_score / sum_coarse * num_coarse
    else:
        confidence = (best_score / (sum_coarse + fine_score_1 + fine_score_2)) * (
            num_coarse + 2
        )
    return Skew(-best_angle, confidence - 1.0)


class _ShearScorer:
    """Scores how sharply a vertical shear of one image separates its rows.

    The sheared image is never built: shearing moves whole column blocks along
    y, so the row counts of the result follow from per-row prefix sums.
    """

    def __init__(self, image: NDArray[np.bool_], resolution_ratio: float) -> None:
        height, width = image.shape
        self._height = height
        self._width = width
        self._ratio = resolution_ratio
        self._prefix = np.zeros((height, width + 1), dtype=np.int32)
        np.cumsum(image, axis=1, dtype=np.int32, out=self._prefix[:, 1:])

    def score(self, angle: float) -> float:
        """Sum of squared differences between neighbouring rows after the shear.

        Args:
            angle (float): Shear angle in degrees.

        Returns:
            float: The score; larger means better aligned rows.
        """
        shear = math.tan(angle * _DEG2RAD) / self._ratio
        counts = np.zeros(self._height, dtype=np.int64)
        for start, stop, shift in shift_blocks(self._width, shear, 0.5 * self._width):
            if abs(shift) >= self._height:
                continue
            block = self._prefix[:, stop] - self._prefix[:, start]
            if shift > 0:
                counts[shift:] += block[: self._height - shift]
            elif shift < 0:
                counts[: self._height + shift] += block[-shift:]
            else:
                counts += block
        diffs = np.diff(counts).astype(np.float64)
        return float(np.dot(diffs, diffs))
