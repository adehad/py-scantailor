"""The filter hands the exposed-corner mask and the page box on to content detection."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.select_content import filter as filter_module
from scantailor.filters.select_content.detection import PageDetectionMode
from scantailor.filters.select_content.filter import Filter
from scantailor.filters.select_content.params import Params

HEIGHT = 600
WIDTH = 440
STRIP = 3


def _page_id() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("/outside.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _page_with_shadow_behind_an_exposed_strip() -> np.ndarray:
    """Return a text page with a dark wedge on the left, behind a white strip."""
    page = np.full((HEIGHT, WIDTH), 245, dtype=np.uint8)
    rng = np.random.default_rng(2)
    for row in range(7):
        top = 90 + row * 60
        x = 150
        while x < 380:
            word = int(rng.integers(25, 55))
            page[top : top + 16, x : x + word] = 30
            x += word + 12
    rows = np.arange(HEIGHT)[:, None]
    cols = np.arange(WIDTH)[None, :]
    width = 110 - 80 * rows / HEIGHT
    shade = np.clip(40 + 205 * cols / np.maximum(width, 1), 0, 255)
    page = np.where(cols < width, np.minimum(page, shade), page).astype(np.uint8)
    page[:, :STRIP] = 255
    return page


def _exposed_strip() -> np.ndarray:
    outside = np.zeros((HEIGHT, WIDTH), dtype=bool)
    outside[:, :STRIP] = True
    return outside


class TestOutsideMask:
    def test_process_excludes_the_shadow_when_given_the_mask(self) -> None:
        page = _page_with_shadow_behind_an_exposed_strip()
        filter_ = Filter()

        params = filter_.process(page, _page_id(), dpi=150.0, outside=_exposed_strip())

        assert params.content_box.x == pytest.approx(150, abs=4)

    def test_detect_content_excludes_the_shadow_when_given_the_mask(self) -> None:
        page = _page_with_shadow_behind_an_exposed_strip()

        result = Filter().detect_content(
            page, _page_id(), dpi=150.0, outside=_exposed_strip()
        )

        assert result.content_box.x == pytest.approx(150, abs=4)

    def test_mask_is_optional(self) -> None:
        page = _page_with_shadow_behind_an_exposed_strip()

        params = Filter().process(page, _page_id(), dpi=150.0)

        assert params.content_box.width > 0


class TestPageBoxIsPassedOn:
    def test_finder_receives_the_stored_page_box(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        real_finder = filter_module.find_content_box

        def spy(*args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return real_finder(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(filter_module, "find_content_box", spy)
        page = np.full((HEIGHT, WIDTH), 245, dtype=np.uint8)
        page[200:216, 100:300] = 30
        filter_ = Filter()
        page_id = _page_id()
        filter_.set_params(
            page_id, Params(page_detection_mode=PageDetectionMode.DISABLED)
        )

        filter_.process(page, page_id, dpi=150.0)

        page_box = captured["page_box"]
        assert (page_box.x, page_box.y) == (0, 0)  # type: ignore[attr-defined]
        assert (page_box.width, page_box.height) == (WIDTH, HEIGHT)  # type: ignore[attr-defined]
