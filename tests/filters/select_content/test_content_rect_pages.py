"""Content detection on seeded pages gives the boxes it gave before the speed-up."""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.select_content.content_box_finder import (
    Rect,
    _distance_to_ink,
    find_content_rect,
)

_HEIGHT = 1754
_WIDTH = 1240
_PAPER = 235
_INK = 40
_SHADOW = 20

# Seed -> box found by the finder with whole-page distance transforms in the
# trimming step.
_EXPECTED_BOXES = {
    0: Rect(left=95, top=316, right=1109, bottom=1725),
    1: Rect(left=185, top=130, right=1181, bottom=1185),
    4: Rect(left=663, top=755, right=1140, bottom=1697),
    6: Rect(left=383, top=780, right=835, bottom=1670),
}


def _page(seed: int) -> NDArray[np.uint8]:
    """Text blocks of varying size, optional shadows along the edges and specks."""
    rng = np.random.default_rng(seed)
    page = np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)
    for _ in range(int(rng.integers(1, 5))):
        left = int(rng.integers(60, _WIDTH - 300))
        top = int(rng.integers(60, _HEIGHT - 300))
        width = int(rng.integers(100, _WIDTH - left - 30))
        height = int(rng.integers(60, _HEIGHT - top - 30))
        for y in range(top, top + height, 16):
            for x in range(left, left + width, 9):
                if rng.random() < 0.7:
                    page[y : y + 9, x : x + 5] = _INK
    if seed % 3 == 0:
        page[:, : int(rng.integers(5, 40))] = _SHADOW
    if seed % 3 == 1:
        page[-int(rng.integers(5, 40)) :, :] = _SHADOW
        page[: int(rng.integers(3, 20)), :] = _SHADOW
    for _ in range(int(rng.integers(0, 30))):
        y = int(rng.integers(0, _HEIGHT - 3))
        x = int(rng.integers(0, _WIDTH - 3))
        page[y : y + 3, x : x + 3] = _INK
    return page


@pytest.mark.parametrize("seed", sorted(_EXPECTED_BOXES))
def test_the_box_found_on_a_seeded_page_is_unchanged(seed: int) -> None:
    """Trimming measures distances inside the area only, with the same outcome."""
    found = find_content_rect(_page(seed), None)

    assert found == _EXPECTED_BOXES[seed]


def test_distance_to_ink_inside_a_crop_matches_the_whole_image() -> None:
    """A distance to ink does not change when the image is cut to hold all the ink."""
    rng = np.random.default_rng(3)
    ink = np.zeros((200, 150), dtype=bool)
    ink[60:140, 40:100] = rng.random((80, 60)) < 0.05
    rows, columns = slice(30, 170), slice(10, 130)

    whole = _distance_to_ink(ink)[rows, columns]
    cropped = _distance_to_ink(ink[rows, columns])

    assert np.array_equal(whole, cropped)
