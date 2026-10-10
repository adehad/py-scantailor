"""Tests for automatic picture detection."""

import cv2
import numpy as np

from scantailor.imageproc.pictures import (
    PICTURE_THRESHOLD,
    detect_pictures,
    estimate_picture_mask,
    find_picture_rectangles,
)


def _text_page(height=400, width=300) -> np.ndarray:
    page = np.full((height, width), 235, dtype=np.uint8)
    for top in range(30, height - 30, 30):
        for left in range(30, width - 30, 12):
            page[top : top + 16, left : left + 3] = 25
            page[top + 6 : top + 9, left : left + 9] = 25
    return page


def _add_photo(page: np.ndarray, rows: slice, cols: slice, seed=0) -> None:
    height, width = rows.stop - rows.start, cols.stop - cols.start
    ramp = np.linspace(70, 190, width)[None, :] + np.linspace(0, 25, height)[:, None]
    noise = np.random.default_rng(seed).integers(-22, 23, size=(height, width))
    page[rows, cols] = np.clip(ramp + noise, 40, 215).astype(np.uint8)


class TestDetectPictures:
    """The picture-likeness map of a 300 dpi page."""

    def test_text_only_page_has_no_picture(self):
        detected = detect_pictures(_text_page())

        assert (detected < PICTURE_THRESHOLD).all()

    def test_textured_block_is_detected_and_text_is_not(self):
        page = _text_page()
        page[250:370, 60:240] = 235
        _add_photo(page, slice(250, 370), slice(60, 240))

        detected = detect_pictures(page) >= PICTURE_THRESHOLD

        assert detected[262:358, 72:228].all()
        assert not detected[20:230, :].any()

    def test_blank_page_has_no_picture(self):
        page = np.full((100, 100), 240, dtype=np.uint8)

        assert (detect_pictures(page) < PICTURE_THRESHOLD).all()

    def test_input_is_not_modified(self):
        page = _text_page()
        before = page.copy()

        detect_pictures(page)

        assert np.array_equal(page, before)

    def test_higher_sensitivity_never_detects_less(self):
        page = _text_page()
        _add_photo(page, slice(250, 370), slice(60, 240))

        normal = detect_pictures(page) >= PICTURE_THRESHOLD
        sensitive = (
            detect_pictures(page, higher_search_sensitivity=True) >= PICTURE_THRESHOLD
        )

        assert sensitive.sum() >= normal.sum()


class TestEstimatePictureMask:
    """Picture detection at any resolution."""

    def _page_with_photo(self) -> np.ndarray:
        page = _text_page()
        page[250:370, 60:240] = 235
        _add_photo(page, slice(250, 370), slice(60, 240))
        return page

    def test_mask_has_the_shape_of_the_input(self):
        page = self._page_with_photo()

        mask = estimate_picture_mask(page, (300, 300))

        assert mask.shape == page.shape
        assert mask.dtype == bool

    def test_photo_is_found_at_double_resolution(self):
        page = self._page_with_photo()
        doubled = cv2.resize(page, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        mask = estimate_picture_mask(doubled, (600, 600))

        assert mask[2 * 262 : 2 * 358, 2 * 72 : 2 * 228].all()
        assert not mask[2 * 20 : 2 * 230, :].any()

    def test_non_square_resolution_is_handled(self):
        page = self._page_with_photo()
        stretched = cv2.resize(page, None, fx=1, fy=2, interpolation=cv2.INTER_NEAREST)

        mask = estimate_picture_mask(stretched, (300, 600))

        assert mask[2 * 262 : 2 * 358, 72:228].all()

    def test_tiny_image_does_not_fail(self):
        mask = estimate_picture_mask(np.full((5, 7), 200, dtype=np.uint8), (600, 600))

        assert mask.shape == (5, 7)


class TestFindPictureRectangles:
    """Bounding boxes of picture clusters."""

    def test_two_separate_blocks_give_two_rectangles(self):
        mask = np.zeros((200, 300), dtype=bool)
        mask[10:60, 10:80] = True
        mask[120:180, 150:290] = True

        rectangles = sorted(find_picture_rectangles(mask))

        assert rectangles == [(10, 10, 79, 59), (150, 120, 289, 179)]

    def test_blocks_closer_than_the_join_distance_merge(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:60, 10:80] = True
        mask[10:60, 90:150] = True

        assert find_picture_rectangles(mask) == [(10, 10, 149, 59)]

    def test_blocks_far_apart_stay_separate(self):
        mask = np.zeros((100, 300), dtype=bool)
        mask[10:60, 10:80] = True
        mask[10:60, 140:200] = True

        assert len(find_picture_rectangles(mask)) == 2

    def test_empty_mask_gives_no_rectangles(self):
        assert find_picture_rectangles(np.zeros((20, 20), dtype=bool)) == []

    def test_low_sensitivity_trims_a_sparse_border(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        mask[10, 20:25] = True

        full = find_picture_rectangles(mask, 100)[0]
        trimmed = find_picture_rectangles(mask, 50)[0]

        assert full[1] == 10
        assert trimmed[1] == 20

    def test_full_sensitivity_keeps_the_whole_box(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        assert find_picture_rectangles(mask, 100) == [(20, 20, 79, 79)]
