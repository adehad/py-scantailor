"""Tests for the content box finder, including gutter shadows along a page edge."""

import numpy as np
import pytest

from scantailor.filters.select_content.content_box import ContentBox
from scantailor.filters.select_content.content_box_finder import (
    Rect,
    find_content_rect,
)
from scantailor.filters.select_content.content_finder import find_content_box

PAGE_HEIGHT = 1200
PAGE_WIDTH = 900
TEXT_LEFT = 260
TEXT_TOP = 180
LINE_PITCH = 70
LINE_HEIGHT = 18
LINE_COUNT = 12
TEXT_RIGHT_LIMIT = 700
PAPER = 245
INK = 30


def _text_page() -> np.ndarray:
    """Return a light page with lines of word-like dark blocks."""
    page = np.full((PAGE_HEIGHT, PAGE_WIDTH), PAPER, dtype=np.uint8)
    rng = np.random.default_rng(1)
    for row in range(LINE_COUNT):
        top = TEXT_TOP + row * LINE_PITCH
        x = TEXT_LEFT
        while x < TEXT_RIGHT_LIMIT:
            word = int(rng.integers(30, 70))
            page[top : top + LINE_HEIGHT, x : x + word] = INK
            x += word + 14
    return page


def _with_gutter_wedge(page: np.ndarray, widest: int = 130) -> np.ndarray:
    """Darken the left edge with a gradient wedge that narrows towards the bottom."""
    rows = np.arange(page.shape[0])[:, None]
    cols = np.arange(page.shape[1])[None, :]
    width = widest - 100 * rows / page.shape[0]
    shade = np.clip(40 + 205 * cols / np.maximum(width, 1), 0, 255)
    return np.where(cols < width, np.minimum(page, shade), page).astype(np.uint8)


def _text_extent(page: np.ndarray) -> tuple[int, int, int, int]:
    """Return left, top, right and bottom (exclusive) of the ink in a text page."""
    ys, xs = np.nonzero(page == INK)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class TestTextOnly:
    """A plain text page needs no shadow handling."""

    def test_box_hugs_the_text(self) -> None:
        page = _text_page()
        left, top, right, bottom = _text_extent(page)

        box = find_content_box(page, dpi=150.0).content_box

        assert box.x == pytest.approx(left, abs=3)
        assert box.y == pytest.approx(top, abs=3)
        assert box.right == pytest.approx(right, abs=3)
        assert box.bottom == pytest.approx(bottom, abs=3)

    def test_input_is_not_modified(self) -> None:
        page = _text_page()
        before = page.copy()

        find_content_box(page, dpi=150.0)

        assert np.array_equal(page, before)

    def test_result_scales_with_dpi(self) -> None:
        page = _text_page()
        left, _top, right, bottom = _text_extent(page)

        # Twice the dpi, so the processing image is half the size.
        box = find_content_box(page, dpi=300.0).content_box

        assert box.x == pytest.approx(left, abs=6)
        assert box.right == pytest.approx(right, abs=6)
        assert box.bottom == pytest.approx(bottom, abs=6)

    def test_confidence_is_full_for_a_page_sized_box(self) -> None:
        result = find_content_box(_text_page(), dpi=150.0)

        assert result.confidence == 1.0


class TestGutterShadow:
    """A dark wedge along one edge must stay out of the content box."""

    def test_wedge_at_the_left_edge_is_excluded(self) -> None:
        page = _with_gutter_wedge(_text_page())
        left, top, right, bottom = _text_extent(_text_page())

        box = find_content_box(page, dpi=150.0).content_box

        assert box.x == pytest.approx(left, abs=3)
        assert box.y == pytest.approx(top, abs=3)
        assert box.right == pytest.approx(right, abs=3)
        assert box.bottom == pytest.approx(bottom, abs=3)

    def test_wedge_with_a_stray_rule_at_the_bottom_is_excluded(self) -> None:
        page = _with_gutter_wedge(_text_page())
        page[1160:1163, 600:900] = INK
        left, _, right, _ = _text_extent(_text_page())

        box = find_content_box(page, dpi=150.0).content_box

        assert box.x == pytest.approx(left, abs=3)
        assert box.right == pytest.approx(right, abs=3)
        assert box.bottom < 1100

    def test_wedge_at_the_right_edge_is_excluded(self) -> None:
        page = _with_gutter_wedge(_text_page())[:, ::-1].copy()
        left, _top, right, _bottom = _text_extent(_text_page()[:, ::-1])

        box = find_content_box(page, dpi=150.0).content_box

        assert box.x == pytest.approx(left, abs=3)
        assert box.right == pytest.approx(right, abs=3)

    def test_wedge_at_the_top_edge_is_excluded(self) -> None:
        page = _with_gutter_wedge(_text_page()).T.copy()
        left, top, _right, bottom = _text_extent(_text_page().T)

        box = find_content_box(page, dpi=150.0).content_box

        assert box.x == pytest.approx(left, abs=3)
        assert box.y == pytest.approx(top, abs=3)
        assert box.bottom == pytest.approx(bottom, abs=3)

    def test_darker_than_the_text_does_not_matter(self) -> None:
        page = _with_gutter_wedge(_text_page())
        page[:, :20] = 0
        left, _, right, _ = _text_extent(_text_page())

        box = find_content_box(page, dpi=150.0).content_box

        assert box.x == pytest.approx(left, abs=3)
        assert box.right == pytest.approx(right, abs=3)


class TestOutsideMask:
    """Pixels with no source behind them are treated as the darkest shadow."""

    def test_exposed_corner_next_to_shadow_does_not_hide_it(self) -> None:
        page = _with_gutter_wedge(_text_page())
        # A rotation exposes a white strip between the border and the shadow.
        page = np.concatenate(
            [np.full((PAGE_HEIGHT, 3), 255, dtype=np.uint8), page[:, :-3]], axis=1
        )
        outside = np.zeros(page.shape, dtype=bool)
        outside[:, :3] = True
        left, _, right, _ = _text_extent(_text_page())

        box = find_content_box(page, dpi=150.0, outside=outside).content_box

        assert box.x == pytest.approx(left + 3, abs=3)
        assert box.right == pytest.approx(right + 3, abs=3)

    def test_exposed_pixels_are_painted_with_the_darkest_remaining_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[np.ndarray] = []

        def spy(gray: np.ndarray, page_rect: Rect | None = None) -> None:
            seen.append(gray.copy())

        monkeypatch.setattr(
            "scantailor.filters.select_content.content_finder.find_content_rect", spy
        )
        page = np.full((40, 60), 200, dtype=np.uint8)
        page[10:20, 30:40] = 70
        outside = np.zeros(page.shape, dtype=bool)
        outside[:, :6] = True
        page[outside] = 255

        find_content_box(page, dpi=150.0, outside=outside)

        assert (seen[0][:, :6] == 70).all()
        assert (seen[0][:, 6:] == page[:, 6:]).all()

    def test_all_outside_finds_nothing(self) -> None:
        page = _text_page()

        result = find_content_box(
            page, dpi=150.0, outside=np.ones(page.shape, dtype=bool)
        )

        assert result.confidence == 0.0

    def test_mask_is_not_modified(self) -> None:
        page = _text_page()
        outside = np.zeros(page.shape, dtype=bool)
        outside[:, :5] = True
        before = outside.copy()

        find_content_box(page, dpi=150.0, outside=outside)

        assert np.array_equal(outside, before)


class TestPageBox:
    """Everything outside the page box counts as shadow."""

    def test_content_beyond_the_page_box_is_ignored(self) -> None:
        page = _text_page()
        left, _top, _, _bottom = _text_extent(page)
        page_box = ContentBox(x=0, y=0, width=500, height=PAGE_HEIGHT)

        box = find_content_box(page, dpi=150.0, page_box=page_box).content_box

        assert box.x == pytest.approx(left, abs=3)
        assert box.right <= 505

    def test_page_box_covering_the_image_changes_nothing(self) -> None:
        page = _text_page()
        whole = ContentBox(x=0, y=0, width=PAGE_WIDTH, height=PAGE_HEIGHT)

        with_box = find_content_box(page, dpi=150.0, page_box=whole).content_box
        without = find_content_box(page, dpi=150.0).content_box

        assert with_box == without


class TestNoContent:
    """Pages without content get an empty box, as upstream's finder returns."""

    @pytest.mark.parametrize("level", [0, 255])
    def test_flat_page_gets_an_empty_box_with_zero_confidence(self, level: int) -> None:
        page = np.full((300, 200), level, dtype=np.uint8)

        result = find_content_box(page, dpi=150.0)

        assert result.confidence == 0.0
        assert result.content_box.is_empty()

    def test_a_dark_band_along_the_edge_is_no_content(self) -> None:
        page = np.full((300, 200), 230, dtype=np.uint8)
        page[:, :10] = 30

        assert find_content_box(page, dpi=150.0).content_box.is_empty()


class TestFindContentRect:
    """The 150 dpi core returns an inclusive rectangle with a pixel of slack."""

    def test_returns_none_without_ink(self) -> None:
        assert find_content_rect(np.full((200, 200), 255, dtype=np.uint8)) is None

    def test_rect_surrounds_the_text(self) -> None:
        page = _text_page()
        left, top, right, bottom = _text_extent(page)

        rect = find_content_rect(page)

        assert isinstance(rect, Rect)
        assert rect.left <= left <= rect.right
        assert rect.top <= top <= rect.bottom
        assert rect.right >= right - 1
        assert rect.bottom >= bottom - 1
