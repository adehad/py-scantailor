"""Tests for the individual steps of the content box finder."""

import numpy as np
import pytest

from scantailor.filters.select_content import content_box_finder as finder
from scantailor.filters.select_content.content_box_finder import Rect
from scantailor.imageproc.binarize import binarize_wolf


def _mask(shape: tuple[int, int], *boxes: tuple[int, int, int, int]) -> np.ndarray:
    """Return a mask that is True in each (top, bottom, left, right) box."""
    mask = np.zeros(shape, dtype=bool)
    for top, bottom, left, right in boxes:
        mask[top:bottom, left:right] = True
    return mask


class TestRect:
    def test_size_counts_both_edges(self) -> None:
        rect = Rect(left=2, top=3, right=5, bottom=9)

        assert (rect.width, rect.height, rect.area) == (4, 7, 28)

    def test_rect_with_crossed_edges_is_empty(self) -> None:
        rect = Rect(left=4, top=0, right=3, bottom=9)

        assert rect.is_empty
        assert rect.area == 0

    def test_window_selects_the_covered_pixels(self) -> None:
        image = np.arange(30).reshape(5, 6)

        window = Rect(left=1, top=2, right=3, bottom=3).window()

        assert image[window].tolist() == [[13, 14, 15], [19, 20, 21]]


class TestFindShadows:
    def test_long_bar_touching_the_border_is_a_shadow(self) -> None:
        ink = _mask((400, 300), (0, 400, 0, 20))

        shadows = finder._find_shadows(ink)

        assert np.array_equal(shadows, ink)

    def test_narrow_bar_is_a_shadow_only_when_it_touches_the_border(self) -> None:
        touching = _mask((400, 300), (0, 400, 0, 10))
        detached = _mask((400, 300), (0, 400, 3, 13))

        assert finder._find_shadows(touching).all(axis=0)[:10].all()
        assert not finder._find_shadows(detached).any()

    def test_text_sized_marks_are_not_shadows(self) -> None:
        ink = _mask((400, 300), (100, 112, 50, 90), (140, 152, 50, 120))

        assert not finder._find_shadows(ink).any()

    def test_border_shadow_keeps_small_marks_beside_it_out(self) -> None:
        ink = _mask((400, 300), (0, 400, 0, 20), (100, 112, 100, 140))

        shadows = finder._find_shadows(ink)

        assert shadows[:, :20].all()
        assert not shadows[100:112, 100:140].any()


class TestTrimBlocksToContent:
    def test_block_shrinks_to_its_content(self) -> None:
        blocks = _mask((40, 40), (5, 35, 5, 35))
        content = _mask((40, 40), (10, 12, 8, 20))

        trimmed = finder._trim_blocks_to_content(content, blocks)

        assert np.array_equal(trimmed, _mask((40, 40), (10, 12, 8, 20)))

    def test_block_without_content_disappears(self) -> None:
        blocks = _mask((40, 40), (5, 15, 5, 15), (25, 35, 25, 35))
        content = _mask((40, 40), (28, 30, 28, 30))

        trimmed = finder._trim_blocks_to_content(content, blocks)

        assert not trimmed[:20].any()
        assert trimmed[28:30, 28:30].all()

    def test_no_blocks_gives_no_blocks(self) -> None:
        content = _mask((10, 10), (2, 4, 2, 4))

        assert not finder._trim_blocks_to_content(
            content, np.zeros((10, 10), dtype=bool)
        ).any()


class TestRemoveBlocksTouchingBorders:
    def test_block_on_the_border_is_removed(self) -> None:
        blocks = _mask((80, 80), (0, 10, 30, 50), (30, 40, 30, 50))

        kept = finder._remove_blocks_touching_borders(blocks)

        assert not kept[:10].any()
        assert kept[30:40, 30:50].all()

    def test_flood_stops_a_quarter_of_the_image_in(self) -> None:
        blocks = _mask((80, 80), (0, 80, 10, 14))  # Runs the full height.

        kept = finder._remove_blocks_touching_borders(blocks)

        # Reach is 20 rows from each end, so the middle survives.
        assert not kept[:20].any()
        assert not kept[-20:].any()
        assert kept[30:50, 10:14].all()

    def test_blocks_away_from_the_border_are_untouched(self) -> None:
        blocks = _mask((80, 80), (20, 40, 20, 40))

        assert np.array_equal(finder._remove_blocks_touching_borders(blocks), blocks)


class TestDistanceToInk:
    def test_distance_is_euclidean(self) -> None:
        ink = np.zeros((11, 11), dtype=bool)
        ink[5, 5] = True

        distance = finder._distance_to_ink(ink)

        assert distance[5, 8] == pytest.approx(3.0, abs=1e-4)
        assert distance[8, 9] == pytest.approx(5.0, abs=1e-4)

    def test_selected_borders_lie_one_pixel_outside(self) -> None:
        ink = np.zeros((6, 20), dtype=bool)

        distance = finder._distance_to_ink(ink, left=True)

        assert distance[2, 0] == pytest.approx(1.0)
        assert distance[2, 9] == pytest.approx(10.0)

    def test_nothing_to_measure_to_gives_a_huge_distance(self) -> None:
        distance = finder._distance_to_ink(np.zeros((5, 5), dtype=bool))

        assert (distance > 1000).all()


class TestUltimateErodedPoints:
    def test_hole_inside_a_mark_has_a_peak_at_its_centre(self) -> None:
        content = _mask((31, 31), (5, 26, 5, 26))
        content[10:21, 10:21] = False

        peaks = finder._ultimate_eroded_points(content)

        assert peaks[15, 15]
        assert peaks.sum() == 1

    def test_open_gap_between_marks_has_no_peak(self) -> None:
        content = _mask((21, 41), (5, 16, 5, 9), (5, 16, 31, 35))

        assert not finder._ultimate_eroded_points(content).any()

    def test_no_ink_gives_no_peaks(self) -> None:
        assert not finder._ultimate_eroded_points(np.zeros((9, 9), dtype=bool)).any()


class TestTextLines:
    def test_valley_between_two_lines_splits_them(self) -> None:
        rows = np.array([0, 8, 9, 10, 9, 8, 1, 8, 9, 10, 9, 8, 0], dtype=np.int64)

        ranges = finder._split_text_lines(rows)

        assert len(ranges) == 2
        assert sorted(ranges)[0][1] < 6 < sorted(ranges)[1][0]

    def test_flat_histogram_is_not_split(self) -> None:
        rows = np.full(14, 9, dtype=np.int64)

        assert finder._split_text_lines(rows) == [(0, 13)]

    def test_short_range_is_dropped(self) -> None:
        assert finder._split_text_lines(np.array([3, 4, 3], dtype=np.int64)) == []

    def test_row_of_ringed_words_is_text(self) -> None:
        content = np.zeros((30, 260), dtype=bool)
        for start in range(4, 250, 24):
            content[8:22, start : start + 16] = True
            content[11:19, start + 3 : start + 13] = False
        blocks = np.ones_like(content)

        text = finder._estimate_text_mask(content, blocks)

        assert text[10:20].any()

    def test_solid_block_is_not_text(self) -> None:
        content = np.ones((30, 260), dtype=bool)

        assert not finder._estimate_text_mask(content, np.ones_like(content)).any()

    def test_ink_without_blocks_is_not_text(self) -> None:
        content = np.ones((30, 60), dtype=bool)

        assert not finder._estimate_text_mask(content, np.zeros_like(content)).any()


class TestSegmentGarbage:
    def test_vertical_bar_belongs_to_the_vertical_class(self) -> None:
        garbage = _mask((400, 300), (0, 400, 0, 20))

        vertical, horizontal = finder._segment_garbage(garbage)

        assert vertical[1:-1, :20].all()
        assert not horizontal[100:300, :20].any()

    def test_horizontal_bar_belongs_to_the_horizontal_class(self) -> None:
        garbage = _mask((300, 400), (0, 20, 0, 400))

        vertical, horizontal = finder._segment_garbage(garbage)

        assert horizontal[:20, 1:-1].all()
        assert not vertical[:20, 100:300].any()

    def test_unattached_garbage_joins_both_classes(self) -> None:
        garbage = _mask((300, 300), (100, 110, 100, 110))

        vertical, horizontal = finder._segment_garbage(garbage)

        assert vertical[100:110, 100:110].all()
        assert horizontal[100:110, 100:110].all()


class TestTrim:
    def test_far_blob_beside_a_shadow_is_trimmed_away(self) -> None:
        content = _mask((200, 300), (50, 150, 100, 280), (50, 150, 0, 8))
        blocks = content.copy()
        garbage = finder._Garbage(_mask((200, 300), (0, 200, 0, 8)), False)
        text = _mask((200, 300), (50, 150, 100, 280))

        area = Rect(0, 50, 279, 149)
        trimmed = finder._trim_side("left", content, blocks, text, area, garbage)

        assert trimmed.left == 100

    def test_rectangle_that_would_lose_most_of_its_area_is_kept(self) -> None:
        content = _mask((200, 300), (50, 150, 0, 150), (50, 150, 200, 210))
        blocks = content.copy()
        garbage = finder._Garbage(np.zeros((200, 300), dtype=bool), False)
        text = np.zeros_like(content)

        area = Rect(0, 50, 209, 149)
        trimmed = finder._trim_side("right", content, blocks, text, area, garbage)

        assert trimmed == area or trimmed.right == 209


class TestBinarize:
    def test_uses_the_wolf_defaults_of_the_c_plus_plus_finder(self) -> None:
        """Window 51, bounds 50 and 254 and coefficient 0.3."""
        rng = np.random.default_rng(3)
        gray = rng.integers(40, 256, size=(120, 140), dtype=np.uint8)
        gray[30:50, 20:100] = 90

        expected = binarize_wolf(gray, 51, k=0.3, lower_bound=50, upper_bound=254) == 0

        assert np.array_equal(finder._binarize(gray), expected)
