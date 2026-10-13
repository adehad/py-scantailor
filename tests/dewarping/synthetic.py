"""Synthetic pages with curved text lines, shared by dewarping and pipeline tests."""

import cv2
import numpy as np
from numpy.typing import NDArray

PAGE_WIDTH = 1200
PAGE_HEIGHT = 1600
PAGE_DPI = 200
TEXT_LEFT = 110
FIRST_BASELINE = 170
LINE_SPACING = 48
BACKGROUND = 235
INK = 20
_TEXT = "The quick brown fox jumps over the lazy dog and keeps running far away"


def straight_text_page() -> NDArray[np.uint8]:
    """A page of evenly spaced, perfectly horizontal text lines."""
    page = np.full((PAGE_HEIGHT, PAGE_WIDTH), BACKGROUND, dtype=np.uint8)
    y = FIRST_BASELINE
    while y < PAGE_HEIGHT - FIRST_BASELINE:
        cv2.putText(
            page,
            _TEXT,
            (TEXT_LEFT, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            INK,
            2,
            cv2.LINE_AA,
        )
        y += LINE_SPACING
    return page


_WORDS = (
    "the quick brown fox jumps over lazy dog and keeps running far away from "
    "every single tall tree while others watch quietly near old stone wall"
).split()


def justified_text_page() -> NDArray[np.uint8]:
    """Evenly spaced, perfectly horizontal lines of text with a margin on both sides.

    Every line is justified between the left margin and its mirror image, so that
    no text touches the image border, as on a scanned page. The words are drawn at
    random, because a repeated sentence would stack the same words above each
    other, which no real page does.
    """
    page = np.full((PAGE_HEIGHT, PAGE_WIDTH), BACKGROUND, dtype=np.uint8)
    rng = np.random.default_rng(0)
    text_width = PAGE_WIDTH - 2 * TEXT_LEFT
    y = FIRST_BASELINE
    while y < PAGE_HEIGHT - FIRST_BASELINE:
        line: list[tuple[str, int]] = []
        used = 0
        while True:
            word = _WORDS[int(rng.integers(len(_WORDS)))]
            width = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)[0][0]
            if used + width + 14 * len(line) > text_width:
                break
            line.append((word, width))
            used += width
        gap = (text_width - used) / max(1, len(line) - 1)
        x = float(TEXT_LEFT)
        for word, width in line:
            cv2.putText(
                page,
                word,
                (round(x), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.05,
                INK,
                2,
                cv2.LINE_AA,
            )
            x += width + gap
        y += LINE_SPACING
    return page


def curved_text_page(bow: float = 60.0) -> NDArray[np.uint8]:
    """The justified page bent so that every line sags by ``bow`` pixels mid-page."""
    page = justified_text_page()
    ys, xs = np.mgrid[0:PAGE_HEIGHT, 0:PAGE_WIDTH].astype(np.float32)
    sag = bow * np.sin(np.pi * xs / PAGE_WIDTH)
    return cv2.remap(page, xs, ys - sag, cv2.INTER_LINEAR, borderValue=BACKGROUND)


def row_profile_variance(page: NDArray[np.uint8], x_range: tuple[int, int]) -> float:
    """Variance of the ink per row over a column range.

    Straight text lines concentrate ink in few rows, which makes the variance
    large; curved or tilted lines smear it across rows.
    """
    ink = 255.0 - page[:, x_range[0] : x_range[1]].astype(np.float64)
    return float(ink.sum(axis=1).var())
