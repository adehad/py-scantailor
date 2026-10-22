"""Generated gray scans for page split tests: word-like blocks on paper."""

import cv2
import numpy as np
from numpy.typing import NDArray

PAPER = 235
INK = 30
LINE_PITCH = 34
WORD_HEIGHT = 14


def blank(width: int, height: int) -> NDArray[np.uint8]:
    """Return an all-paper page."""
    return np.full((height, width), PAPER, dtype=np.uint8)


def add_text(
    page: NDArray[np.uint8],
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    seed: int,
) -> None:
    """Fill a column with lines of word boxes that run to both of its edges."""
    rng = np.random.default_rng(seed)
    for y in range(y0, y1 - WORD_HEIGHT, LINE_PITCH):
        x = x0
        while x < x1:
            word = int(rng.integers(18, 56))
            # Tops and heights vary like glyphs do. Boxes of one height make the
            # page score the same for opposite shears, which stalls the skew search.
            top = y + int(rng.integers(-2, 3))
            height = WORD_HEIGHT + int(rng.integers(-3, 3))
            page[top : top + height, x : min(x + word, x1)] = INK
            x += word + int(rng.integers(6, 12))


def add_vertical_band(
    page: NDArray[np.uint8],
    x0: int,
    x1: int,
    level: int,
    y0: int = 0,
    y1: int | None = None,
) -> None:
    """Darken the columns ``x0:x1``."""
    page[y0:y1, x0:x1] = level


def two_page_spread(width: int = 1500, height: int = 1000) -> NDArray[np.uint8]:
    """Two text columns with a clean gutter between them."""
    page = blank(width, height)
    half = width // 2
    add_text(page, 110, half - 100, 100, height - 100, seed=1)
    add_text(page, half + 100, width - 110, 100, height - 100, seed=2)
    return page


def spread_with_folding_line(
    width: int = 1500, height: int = 1000
) -> NDArray[np.uint8]:
    """A spread with a thin dark line down the gutter, a little right of centre."""
    page = two_page_spread(width, height)
    add_vertical_band(page, width // 2 + 14, width // 2 + 22, 45)
    return page


def spread_with_black_gutter(
    width: int = 1500, height: int = 1000
) -> NDArray[np.uint8]:
    """A spread whose gutter is a wide black shadow."""
    page = two_page_spread(width, height)
    add_vertical_band(page, width // 2 - 40, width // 2 + 40, 15)
    return page


def spread_with_dark_borders(
    width: int = 1500, height: int = 1000
) -> NDArray[np.uint8]:
    """A spread photographed on a dark scanner bed: black strips at every edge."""
    page = two_page_spread(width, height)
    add_vertical_band(page, 0, 36, 8)
    add_vertical_band(page, width - 36, width, 8)
    page[:28, :] = 8
    page[-28:, :] = 8
    return page


def skewed_spread(
    angle: float, width: int = 1500, height: int = 1000
) -> NDArray[np.uint8]:
    """A clean spread rotated by ``angle`` degrees, paper filling the corners."""
    page = two_page_spread(width, height)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return np.asarray(
        cv2.warpAffine(
            page,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=PAPER,
        ),
        dtype=np.uint8,
    )


def single_page(width: int = 1000, height: int = 1400) -> NDArray[np.uint8]:
    """One page of text with ample margins."""
    page = blank(width, height)
    add_text(page, 130, width - 130, 130, height - 130, seed=3)
    return page


def single_page_with_left_offcut(
    width: int = 1000, height: int = 1400
) -> NDArray[np.uint8]:
    """One page with fragments of its neighbour along the left edge."""
    page = single_page(width, height)
    page[:, :200] = PAPER
    add_text(page, 6, 90, 150, height - 150, seed=4)
    return page


def single_page_with_right_offcut(
    width: int = 1000, height: int = 1400
) -> NDArray[np.uint8]:
    """One page with fragments of its neighbour along the right edge."""
    page = single_page(width, height)
    page[:, width - 200 :] = PAPER
    add_text(page, width - 92, width - 6, 150, height - 150, seed=5)
    return page


def single_page_missing_inside_edge(
    width: int = 1000, height: int = 1400
) -> NDArray[np.uint8]:
    """A page whose inside margin the scanner cut off: text reaches the left edge."""
    page = blank(width, height)
    add_text(page, 0, width - 130, 130, height - 130, seed=6)
    return page


def single_page_with_neighbours_letters_on_left_edge(
    width: int = 1000, height: int = 1400
) -> NDArray[np.uint8]:
    """One page with a strip of its neighbour's text running off the left edge."""
    page = single_page(width, height)
    page[:, :200] = PAPER
    add_text(page, 0, 70, 150, height - 150, seed=4)
    return page


def single_page_with_neighbours_letters_on_right_edge(
    width: int = 1000, height: int = 1400
) -> NDArray[np.uint8]:
    """One page with a strip of its neighbour's text running off the right edge."""
    page = single_page(width, height)
    page[:, width - 200 :] = PAPER
    add_text(page, width - 70, width, 150, height - 150, seed=5)
    return page


def single_page_with_dark_gutter_strip(
    width: int = 1000, height: int = 1400
) -> NDArray[np.uint8]:
    """One page with a dark vertical strip of the spine along its left edge."""
    page = single_page(width, height)
    add_vertical_band(page, 0, 60, 20)
    return page


def spread_with_empty_page(width: int = 1500, height: int = 1000) -> NDArray[np.uint8]:
    """A spread where only the left page has text."""
    page = blank(width, height)
    add_text(page, 110, width // 2 - 100, 100, height - 100, seed=7)
    return page


def single_page_with_edge_lines(
    width: int = 1000, height: int = 1400
) -> NDArray[np.uint8]:
    """One page with a thin dark line near each side edge."""
    page = single_page(width, height)
    add_vertical_band(page, 78, 84, 45)
    add_vertical_band(page, width - 84, width - 78, 45)
    return page


def single_page_with_right_edge_line(
    width: int = 1000, height: int = 1400
) -> NDArray[np.uint8]:
    """One page with a thin dark line near its right edge only."""
    page = single_page(width, height)
    add_vertical_band(page, width - 84, width - 78, 45)
    return page


def spread_of_one_repeated_sentence(dpi: float) -> NDArray[np.uint8]:
    """Two 6 x 8 inch pages that repeat one sentence on every line, at ``dpi``.

    Every line holds the same letters at the same places, so the letters stack
    into columns down the page. The gutter runs from 0.43 to 0.56 of the width.

    Args:
        dpi (float): Resolution to scale the spread to from the 300 dpi drawing.

    Returns:
        NDArray[np.uint8]: The (H, W) gray spread.
    """
    page = blank(1800, 2400)
    for baseline in range(300, 2100, 72):
        cv2.putText(
            page,
            "The quick brown fox jumps over the lazy dog",
            (225, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            20,
            4,
            cv2.LINE_AA,
        )
    spread = np.hstack([page, page])
    factor = dpi / 300.0
    size = (round(spread.shape[1] * factor), round(spread.shape[0] * factor))
    return np.asarray(
        cv2.resize(spread, size, interpolation=cv2.INTER_AREA), dtype=np.uint8
    )
