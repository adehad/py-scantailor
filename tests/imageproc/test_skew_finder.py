"""Skew of a black-and-white page, searched by shearing and scoring rows."""

import cv2
import numpy as np
import pytest

from scantailor.imageproc.skew_finder import Skew, _ShearScorer, find_skew_binary
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


@pytest.mark.parametrize("angle", [1.7, -2.4, 4.0, -6.5])
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


@pytest.mark.parametrize("coarse", [0, 1])
def test_a_level_text_page_is_not_reported_as_skewed(coarse: int) -> None:
    page = _text_page(0.0, seed=7)

    got = find_skew_binary(page, coarse_reduction=coarse, fine_reduction=0)

    assert got.angle == 0.0
    assert got.confidence >= Skew.GOOD_CONFIDENCE


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


def test_a_flat_top_the_bisection_narrows_onto_one_side_of_is_centred() -> None:
    """The aligned bars score the same from -0.25 to 0.25 degrees but less at
    -0.5 than at 0.5, so the bisection ties on [0, 0.25]. The answer is the
    centre of the whole flat stretch, not of that bracket.
    """
    page = _aligned_bars(np.random.default_rng(5))
    scorer = _ShearScorer(page, 1.0)
    assert scorer.score(-0.5) < scorer.score(0.5) < scorer.score(0.0)
    assert scorer.score(-0.25) == scorer.score(0.0) == scorer.score(0.25)

    skew = find_skew_binary(page, coarse_reduction=0, fine_reduction=0, accuracy=0.1)

    assert skew.angle == 0.0
    assert skew.confidence >= Skew.GOOD_CONFIDENCE


def _mirrored_bars() -> np.ndarray:
    """11 level bars, 6 rows tall, bar i being 120 + 5i pixels long."""
    page = np.zeros((200, 200), dtype=bool)
    for i, y in enumerate(range(20, 180, 14)):
        page[y : y + 6, 10 : 120 + i * 5] = True
    return page


def test_equal_scores_either_side_of_the_best_coarse_angle_pick_the_middle() -> None:
    """Shearing level bars by +0.5 and -0.5 degrees scores the same, so the
    coarse winner, 0 degrees, is the answer.
    """
    skew = find_skew_binary(
        _mirrored_bars(), coarse_reduction=0, fine_reduction=0, accuracy=0.1
    )

    assert skew.angle == 0.0
    assert skew.confidence >= Skew.GOOD_CONFIDENCE


def test_the_tie_differs_from_upstream_which_reports_half_a_degree() -> None:
    upstream_angle, _ = cpp.find_skew(_mirrored_bars(), 0, 0, 0.1)

    assert upstream_angle == 0.5


def test_negative_reductions_are_rejected() -> None:
    with pytest.raises(ValueError, match="reduction"):
        find_skew_binary(np.zeros((10, 10), dtype=bool), coarse_reduction=-1)


def test_an_empty_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        find_skew_binary(np.zeros((0, 10), dtype=bool))
