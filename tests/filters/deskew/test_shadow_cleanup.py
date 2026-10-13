"""Horizontal shadows are removed before Deskew measures the skew."""

from pathlib import Path

import numpy as np

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.shadow_cleanup import remove_horizontal_shadows
from scantailor.imageproc.analysis import SkewResult
from tests.filters.deskew.synthetic_pages import text_page

PAGE = PageId(
    image_id=ImageId(file_path=Path("/shadow.tiff")), sub_page=SubPage.SINGLE_PAGE
)
INK = 0
WHITE = 255
DPI_300 = Dpi.uniform(300)


def _blank(height: int = 600, width: int = 800) -> np.ndarray:
    return np.full((height, width), WHITE, dtype=np.uint8)


class TestRemoveHorizontalShadows:
    def test_a_full_width_band_thicker_than_the_brick_is_removed(self):
        # At 300 DPI the page is halved once, so the brick is 200x14 reduced
        # pixels: 400x28 pixels of the page. A 40 pixel band is thicker.
        page = _blank()
        page[100:140, :] = INK

        cleaned = remove_horizontal_shadows(page, DPI_300)

        assert (cleaned == WHITE).all()

    def test_a_band_thinner_than_the_brick_is_kept(self):
        page = _blank()
        page[100:110, :] = INK

        cleaned = remove_horizontal_shadows(page, DPI_300)

        assert np.array_equal(cleaned, page)

    def test_a_band_narrower_than_the_brick_is_kept(self):
        page = _blank()
        page[100:140, 100:450] = INK

        cleaned = remove_horizontal_shadows(page, DPI_300)

        assert np.array_equal(cleaned, page)

    def test_ink_touching_a_shadow_goes_with_it(self):
        page = _blank()
        page[100:140, :] = INK
        page[140:180, 300:320] = INK  # a descender joined to the band
        page[400:420, 300:340] = INK  # separate text

        cleaned = remove_horizontal_shadows(page, DPI_300)

        assert (cleaned[100:180] == WHITE).all()
        assert (cleaned[400:420, 300:340] == INK).all()

    def test_the_input_is_not_modified(self):
        page = _blank()
        page[100:140, :] = INK
        before = page.copy()

        remove_horizontal_shadows(page, DPI_300)

        assert np.array_equal(page, before)

    def test_the_brick_follows_the_resolution(self):
        # At 150 DPI nothing is reduced, so the brick is 200x14 page pixels and
        # a 20 pixel band 250 pixels wide is a shadow; at 300 DPI it is not.
        page = _blank()
        page[100:120, 100:350] = INK

        assert (remove_horizontal_shadows(page, Dpi.uniform(150)) == WHITE).all()
        assert np.array_equal(remove_horizontal_shadows(page, DPI_300), page)

    def test_a_page_without_shadows_is_unchanged(self):
        page = text_page()

        assert np.array_equal(remove_horizontal_shadows(page, DPI_300), page)

    def test_a_tiny_page_does_not_fail(self):
        page = _blank(1, 1)

        assert np.array_equal(remove_horizontal_shadows(page, DPI_300), page)


class TestDetectSkewWithShadows:
    @staticmethod
    def _page_with_shadow() -> np.ndarray:
        page = text_page()
        page[-45:-5, :] = INK
        return page

    @staticmethod
    def _seen_by_find_skew(monkeypatch, run) -> np.ndarray:
        seen: list[np.ndarray] = []

        def spy(binary, **_kwargs):
            seen.append(binary.copy())
            return SkewResult(angle=0.0, confidence=0.0)

        monkeypatch.setattr("scantailor.filters.deskew.filter.find_skew", spy)
        run()
        return seen[0]

    def test_the_page_given_to_the_skew_finder_has_no_shadow_when_the_dpi_is_known(
        self, monkeypatch
    ):
        page = self._page_with_shadow()

        seen = self._seen_by_find_skew(
            monkeypatch,
            lambda: Filter().detect_skew(page, PAGE, store_result=False, dpi=DPI_300),
        )

        assert (seen[-45:-5] == WHITE).all()
        assert (seen[: len(page) - 45] == 0).any()

    def test_the_page_is_given_as_it_is_without_a_dpi(self, monkeypatch):
        page = self._page_with_shadow()

        seen = self._seen_by_find_skew(
            monkeypatch,
            lambda: Filter().detect_skew(page, PAGE, store_result=False),
        )

        assert (seen[-45:-5] == INK).all()

    def test_the_oblique_detection_also_sees_the_cleaned_page(self, monkeypatch):
        page = self._page_with_shadow()
        seen: list[np.ndarray] = []

        def spy(mask, *_args, **_kwargs):
            seen.append(mask.copy())
            return mask

        monkeypatch.setattr("scantailor.filters.deskew.filter.rotate_mask", spy)
        monkeypatch.setattr(
            "scantailor.filters.deskew.filter.find_oblique_degrees", lambda _m: None
        )
        filter_ = Filter()
        filter_.auto_oblique_default = True

        filter_.detect_skew(page, PAGE, dpi=DPI_300)

        assert (seen[0][-45:-5] == WHITE).all()
