"""Skew of a black-and-white page, searched by shearing and scoring rows."""

import cv2
import numpy as np
import pytest

from scantailor.imageproc.skew_finder import Skew, find_skew_binary
from tests.filters.page_split import cpp_reference as cpp


def _text_page(
    angle: float, seed: int = 0, height: int = 360, width: int = 280
) -> np.ndarray:
    """Word boxes in lines with uneven line spacing, rotated by ``angle`` degrees."""
    rng = np.random.default_rng(seed)
    page = np.zeros((height, width), dtype=np.uint8)
    y = 30
    while y < height - 40:
        x = 25
        while x < width - 30:
            word = int(rng.integers(10, 30))
            page[y : y + 7, x : min(x + word, width - 25)] = 255
            x += word + int(rng.integers(4, 9))
        y += 16 + int(rng.integers(0, 6))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(page, matrix, (width, height), flags=cv2.INTER_NEAREST)
    return rotated > 127


@pytest.mark.parametrize("angle", [0.0, 1.7, -2.4, 4.0, -6.5])
@pytest.mark.parametrize(
    ("coarse", "fine", "accuracy"),
    [(0, 0, 0.5), (0, 0, 0.1), (2, 1, 0.1), (1, 1, 0.25)],
)
def test_matches_upstream(
    angle: float, coarse: int, fine: int, accuracy: float
) -> None:
    page = _text_page(angle, seed=abs(int(angle * 10)) + 7)

    got = find_skew_binary(
        page, coarse_reduction=coarse, fine_reduction=fine, accuracy=accuracy
    )

    want_angle, want_confidence = cpp.find_skew(page, coarse, fine, accuracy)
    assert got.angle == pytest.approx(want_angle, abs=1e-9)
    assert got.confidence == pytest.approx(want_confidence, rel=1e-9)


def test_a_coarse_only_search_matches_upstream() -> None:
    page = _text_page(3.0)

    got = find_skew_binary(page, accuracy=1.0, coarse_reduction=0, fine_reduction=0)

    want_angle, want_confidence = cpp.find_skew(page, 0, 0, 1.0)
    assert got.angle == pytest.approx(want_angle)
    assert got.confidence == pytest.approx(want_confidence)


def test_finds_the_rotation_that_straightens_a_skewed_page() -> None:
    page = _text_page(2.0)

    skew = find_skew_binary(page, coarse_reduction=0, fine_reduction=0, accuracy=0.1)

    # OpenCV's positive angle is counter-clockwise; skew is positive clockwise.
    assert skew.angle == pytest.approx(-2.0, abs=0.35)
    assert skew.confidence >= Skew.GOOD_CONFIDENCE


def test_a_blank_page_has_no_confidence() -> None:
    blank = np.zeros((100, 100), dtype=bool)

    skew = find_skew_binary(blank, coarse_reduction=0, fine_reduction=0)

    assert skew.confidence == 0.0


def _aligned_bars(rng: np.random.Generator) -> np.ndarray:
    """Horizontal lines of words whose tops and heights vary, as glyphs do."""
    page = np.zeros((200, 200), dtype=bool)
    for y in range(20, 180, 14):
        x = int(rng.integers(8, 30))
        while x < 170:
            word = int(rng.integers(8, 25))
            top = y + int(rng.integers(-2, 3))
            page[top : top + int(rng.integers(4, 8)), x : x + word] = True
            x += word + int(rng.integers(3, 7))
    return page


def test_a_page_with_no_skew_reports_zero_with_good_confidence() -> None:
    page = _aligned_bars(np.random.default_rng(5))

    skew = find_skew_binary(page, coarse_reduction=0, fine_reduction=0, accuracy=0.1)

    assert skew.angle == pytest.approx(0.0, abs=0.1)
    assert skew.confidence >= Skew.GOOD_CONFIDENCE


def test_equal_scores_either_side_of_the_best_coarse_angle_pick_the_lower_one() -> None:
    """Upstream stops the bisection on a tie and keeps the angle below the optimum.

    Perfectly mirrored bars score the same at +0.5 and -0.5 degrees, so the
    result is a half degree skew although the bars are exactly horizontal.
    """
    page = np.zeros((200, 200), dtype=bool)
    for i, y in enumerate(range(20, 180, 14)):
        page[y : y + 6, 10 : 120 + i * 5] = True

    skew = find_skew_binary(page, coarse_reduction=0, fine_reduction=0, accuracy=0.1)
    want_angle, _ = cpp.find_skew(page, 0, 0, 0.1)

    assert skew.angle == want_angle == 0.5


def test_negative_reductions_are_rejected() -> None:
    with pytest.raises(ValueError, match="reduction"):
        find_skew_binary(np.zeros((10, 10), dtype=bool), coarse_reduction=-1)


def test_an_empty_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        find_skew_binary(np.zeros((0, 10), dtype=bool))
