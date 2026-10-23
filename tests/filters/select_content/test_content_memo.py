"""Content detection is remembered per processed image and changes no result."""

import numpy as np
import pytest

from scantailor.filters.select_content import content_finder
from scantailor.filters.select_content.content_box import ContentBox
from scantailor.filters.select_content.content_finder import find_content_box

DPI = 150


def _page(seed: int = 0, shift: int = 0) -> np.ndarray:
    """A 150 dpi page with a block of text-like stripes."""
    rng = np.random.default_rng(seed)
    page = np.full((700, 500), 235, dtype=np.uint8)
    for top in range(120 + shift, 560, 14):
        page[top : top + 5, 90:400] = 25
    page += rng.integers(0, 4, page.shape, dtype=np.uint8)
    return page


@pytest.fixture(autouse=True)
def empty_memo() -> None:
    content_finder._CONTENT_MEMO.clear()


@pytest.fixture
def detections(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Shapes of the images the block search really ran on."""
    seen: list[tuple[int, int]] = []
    real = content_finder.find_content_rect

    def counting(gray, page_rect, dbg=None):
        seen.append(gray.shape)
        return real(gray, page_rect, dbg)

    monkeypatch.setattr(content_finder, "find_content_rect", counting)
    return seen


def test_same_image_is_analysed_once(detections: list) -> None:
    page = _page()

    first = find_content_box(page, DPI)
    second = find_content_box(page.copy(), DPI)

    assert second == first
    assert len(detections) == 1


def test_remembered_result_equals_a_fresh_one() -> None:
    page = _page()

    remembered = find_content_box(page, DPI)
    content_finder._CONTENT_MEMO.clear()

    assert find_content_box(page, DPI) == remembered


def test_changed_pixels_are_analysed_again(detections: list) -> None:
    find_content_box(_page(shift=0), DPI)
    moved = find_content_box(_page(shift=30), DPI)

    assert len(detections) == 2
    assert moved.content_box.y > 100


def test_a_different_page_box_is_analysed_again(detections: list) -> None:
    page = _page()

    find_content_box(page, DPI)
    find_content_box(page, DPI, page_box=ContentBox(x=0, y=0, width=500, height=400))

    assert len(detections) == 2


def test_a_different_dpi_is_analysed_again(detections: list) -> None:
    page = _page()

    find_content_box(page, 150)
    find_content_box(page, 300)

    assert len(detections) == 2


def test_pixels_outside_the_source_take_part_in_the_key(detections: list) -> None:
    page = _page()
    outside = np.zeros(page.shape, dtype=bool)
    outside[:, :60] = True

    find_content_box(page, DPI)
    find_content_box(page, DPI, outside=outside)
    find_content_box(page, DPI, outside=outside.copy())

    assert len(detections) == 2


class TestPaintOutsideDarkest:
    def test_outside_pixels_take_the_darkest_level_inside(self) -> None:
        rng = np.random.default_rng(4)
        gray = rng.integers(10, 250, (60, 80), dtype=np.uint8)
        outside = np.zeros(gray.shape, dtype=bool)
        outside[:10, :] = True
        outside[:, 70:] = True
        gray[:5, :5] = 0  # darker than anything inside

        painted = content_finder.paint_outside_darkest(gray, outside)

        expected = np.where(outside, gray[~outside].min(), gray).astype(np.uint8)
        assert np.array_equal(painted, expected)

    def test_input_is_not_modified(self) -> None:
        gray = np.full((20, 20), 200, dtype=np.uint8)
        outside = np.zeros(gray.shape, dtype=bool)
        outside[:3] = True
        before = gray.copy()

        content_finder.paint_outside_darkest(gray, outside)

        assert np.array_equal(gray, before)


def _corner_mask() -> np.ndarray:
    mask = np.zeros((700, 500), dtype=bool)
    mask[:40, :40] = True
    return mask


def test_a_deferred_outside_mask_is_not_made_when_the_result_is_remembered() -> None:
    page = _page()
    made: list[int] = []

    def mask() -> np.ndarray:
        made.append(1)
        return _corner_mask()

    first = find_content_box(page, DPI, outside=mask, outside_key="corner")
    second = find_content_box(page.copy(), DPI, outside=mask, outside_key="corner")

    assert second == first
    assert len(made) == 1


def test_a_deferred_outside_mask_gives_the_result_of_the_eager_one() -> None:
    page = _page()

    eager = find_content_box(page, DPI, outside=_corner_mask())
    content_finder._CONTENT_MEMO.clear()
    deferred = find_content_box(page, DPI, outside=_corner_mask, outside_key="c")

    assert deferred == eager


def test_a_different_outside_key_is_analysed_again(detections: list) -> None:
    page = _page()

    find_content_box(page, DPI, outside=_corner_mask, outside_key="a")
    find_content_box(page, DPI, outside=_corner_mask, outside_key="b")

    assert len(detections) == 2
