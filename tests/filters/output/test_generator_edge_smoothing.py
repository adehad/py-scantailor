"""What Output's edge smoothing changes on an edge, and what it must leave alone."""

import cv2
import numpy as np
import pytest

from scantailor.core.models import Dpi
from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.generator import _morphological_smooth, generate_output
from scantailor.filters.output.params import Params

# A slab of ink whose lower edge carries the defect, drawn with # for ink.
_SLAB = (
    "...............",
    ".#############.",
    ".#############.",
    ".#############.",
)
_FLAT_EDGE = (*_SLAB, "...............", "...............")
_FILLED_EDGE = (*_SLAB, ".#############.", "...............")

_BUMPS = {
    1: (*_SLAB, ".......#.......", "..............."),
    2: (*_SLAB, ".......##......", "..............."),
    3: (*_SLAB, "......###......", "..............."),
}
_NOTCHES = {
    1: (*_SLAB, ".######.######.", "..............."),
    2: (*_SLAB, ".######..#####.", "..............."),
    3: (*_SLAB, ".#####...#####.", "..............."),
}
_KEPT = {
    "bump four long": (*_SLAB, ".....####......", "..............."),
    "bump two high": (*_SLAB, ".......#.......", ".......#......."),
    "notch four long": (*_SLAB, ".#####....####.", "..............."),
    "notch two deep": (
        *_SLAB[:3],
        ".######.######.",
        ".######.######.",
        "...............",
    ),
}


def _picture(rows: tuple[str, ...], quarter_turns: int) -> np.ndarray:
    """A binary page drawn row by row, # for ink, turned counter-clockwise.

    Args:
        rows (tuple[str, ...]): The page's rows, top to bottom, of equal length.
        quarter_turns (int): Counter-clockwise quarter turns, so each edge of
            the slab gets a turn at holding the defect.

    Returns:
        np.ndarray: (H, W) uint8 page, 0 for ink and 255 for paper.
    """
    ink = np.array([[cell == "#" for cell in row] for row in rows])
    return np.where(np.rot90(ink, quarter_turns), np.uint8(0), np.uint8(255))


def _strokes(width: int) -> np.ndarray:
    """Draw a vertical, a horizontal and a diagonal stroke of one width.

    Args:
        width (int): Stroke width in pixels.

    Returns:
        np.ndarray: (80, 80) uint8 page, 0 for ink and 255 for paper.
    """
    page = np.full((80, 80), 255, dtype=np.uint8)
    page[10:70, 20 : 20 + width] = 0
    page[50 : 50 + width, 30:75] = 0
    for i in range(40):
        page[5 + i, 35 + i : 35 + i + width] = 0
    return page


def _ink_pieces_match_one_to_one(before: np.ndarray, after: np.ndarray) -> bool:
    """Tell whether every piece of ink overlaps exactly one piece on the other page.

    A split and a join elsewhere leave the number of pieces unchanged, so the
    pieces are matched by the pixels they share rather than counted.

    Args:
        before (np.ndarray): (H, W) uint8 binary page, 0 for ink.
        after (np.ndarray): Page of the same shape and encoding.

    Returns:
        bool: True when the 8-connected pieces of ink pair up one to one.
    """
    count_before, labels_before = cv2.connectedComponents(
        (before == 0).astype(np.uint8), connectivity=8
    )
    count_after, labels_after = cv2.connectedComponents(
        (after == 0).astype(np.uint8), connectivity=8
    )
    shared = (labels_before > 0) & (labels_after > 0)
    pairs = np.unique(np.stack([labels_before[shared], labels_after[shared]]), axis=1)
    return (
        count_before == count_after
        and pairs.shape[1] == count_before - 1
        and len(np.unique(pairs[0])) == pairs.shape[1]
        and len(np.unique(pairs[1])) == pairs.shape[1]
    )


def _edge_length(page: np.ndarray) -> int:
    """Count the sides where an ink pixel meets a paper pixel.

    Args:
        page (np.ndarray): (H, W) uint8 binary page, 0 for ink.

    Returns:
        int: Horizontal plus vertical ink-paper boundaries, in pixel sides.
    """
    ink = page == 0
    return int(
        np.count_nonzero(ink[:, 1:] != ink[:, :-1])
        + np.count_nonzero(ink[1:] != ink[:-1])
    )


def _black_and_white(morphological_smoothing: bool) -> Params:
    """Bilevel output at 300 DPI, despeckling off so only the smoothing differs.

    Args:
        morphological_smoothing (bool): Whether Output smooths the edges.

    Returns:
        Params: The output parameters.
    """
    return Params(
        color_mode=ColorMode.BLACK_AND_WHITE,
        output_dpi=Dpi.uniform(300),
        binarization=BinarizationOptions(
            morphological_smoothing=morphological_smoothing
        ),
        despeckle_level=DespeckleLevel.OFF,
    )


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
@pytest.mark.parametrize("length", sorted(_BUMPS))
def test_a_bump_one_pixel_high_and_up_to_three_long_is_shaved_off(
    length, quarter_turns
):
    page = _picture(_BUMPS[length], quarter_turns)

    got = _morphological_smooth(page)

    np.testing.assert_array_equal(got, _picture(_FLAT_EDGE, quarter_turns))


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
@pytest.mark.parametrize("length", sorted(_NOTCHES))
def test_a_notch_one_pixel_deep_and_up_to_three_long_is_filled(length, quarter_turns):
    page = _picture(_NOTCHES[length], quarter_turns)

    got = _morphological_smooth(page)

    np.testing.assert_array_equal(got, _picture(_FILLED_EDGE, quarter_turns))


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", sorted(_KEPT))
def test_a_longer_bump_or_deeper_notch_is_kept_as_drawn(shape, quarter_turns):
    """Detail that size may be a serif, a dot or the gap between two letters."""
    page = _picture(_KEPT[shape], quarter_turns)

    np.testing.assert_array_equal(_morphological_smooth(page), page)


@pytest.mark.parametrize("width", [1, 2])
def test_smoothing_keeps_a_thin_stroke(width):
    page = _strokes(width)

    np.testing.assert_array_equal(_morphological_smooth(page), page)


def test_black_and_white_output_keeps_a_two_pixel_stroke():
    gray = np.full((100, 100), 230, dtype=np.uint8)
    gray[10:90, 48:50] = 30

    result = generate_output(gray, _black_and_white(morphological_smoothing=True))

    ink_per_row = (result.image[10:90, 46:52] == 0).sum(axis=1, dtype=np.int64)
    assert (ink_per_row >= 2).all()


def test_smoothing_a_book_page_shortens_its_edges_without_splitting_or_joining_ink(
    sample_book_tool,
):
    """The sample book's first page, scaled to 300 DPI as a bitonal book is scanned."""
    rgb = sample_book_tool.render_sample_page("01_single_page").image
    scale = 300 / sample_book_tool.SAMPLE_DPI
    gray = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    rough = generate_output(gray, _black_and_white(morphological_smoothing=False))
    smooth = generate_output(gray, _black_and_white(morphological_smoothing=True))

    assert _ink_pieces_match_one_to_one(rough.image, smooth.image)
    assert _edge_length(smooth.image) < 0.98 * _edge_length(rough.image)
    ink = np.count_nonzero(rough.image == 0)
    assert abs(np.count_nonzero(smooth.image == 0) - ink) < 0.02 * ink
