"""Select Content detects again only when its stored boxes are stale."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.select_content import filter as content_filter
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)
from scantailor.filters.select_content.filter import Filter

PAGE = PageId(image_id=ImageId(file_path=Path("/p.tiff")), sub_page=SubPage.SINGLE_PAGE)
OUTLINE = [(0.0, 0.0), (400.0, 0.0), (400.0, 500.0), (0.0, 500.0)]


def _image() -> np.ndarray:
    image = np.full((500, 400), 255, dtype=np.uint8)
    image[100:140, 80:320] = 0
    image[200:240, 80:320] = 0
    return image


@pytest.fixture
def counted(monkeypatch):
    """Count how often content and page detection actually run."""
    calls = {"content": 0, "page": 0}
    content, page = content_filter.find_content_box, content_filter.find_page_box

    def count_content(*args, **kwargs):
        calls["content"] += 1
        return content(*args, **kwargs)

    def count_page(*args, **kwargs):
        calls["page"] += 1
        return page(*args, **kwargs)

    monkeypatch.setattr(content_filter, "find_content_box", count_content)
    monkeypatch.setattr(content_filter, "find_page_box", count_page)
    return calls


class TestUpdateNeeds:
    def test_an_unknown_outline_makes_every_box_stale(self):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)

        assert filter_.update_needs(PAGE, None) == (True, True)

    def test_a_page_never_processed_needs_both(self):
        assert Filter().update_needs(PAGE, OUTLINE) == (True, True)

    def test_a_processed_page_with_the_same_outline_needs_nothing(self):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)

        assert filter_.update_needs(PAGE, OUTLINE) == (False, False)

    def test_float_noise_in_the_outline_is_ignored(self):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)
        jittered = [(x + 1e-6, y - 1e-6) for x, y in OUTLINE]

        assert filter_.update_needs(PAGE, jittered) == (False, False)

    def test_a_moved_outline_makes_both_stale(self):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)
        shifted = [(x + 5.0, y) for x, y in OUTLINE]

        assert filter_.update_needs(PAGE, shifted) == (True, True)

    def test_a_different_polygon_size_makes_both_stale(self):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)

        assert filter_.update_needs(PAGE, OUTLINE[:3]) == (True, True)

    def test_switching_the_content_mode_makes_only_the_content_stale(self):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)
        filter_.disable_content_detection(PAGE)

        assert filter_.update_needs(PAGE, OUTLINE) == (True, False)

    def test_switching_the_page_mode_makes_the_page_stale(self):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)
        filter_.reset_page_to_auto(PAGE)

        assert filter_.update_needs(PAGE, OUTLINE)[1] is True

    def test_fine_tuning_only_matters_while_the_page_is_detected(self):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)
        filter_.set_fine_tune(PAGE, True)

        assert filter_.update_needs(PAGE, OUTLINE) == (False, False)

        filter_.reset_page_to_auto(PAGE)
        filter_.process(_image(), PAGE, page_outline=OUTLINE)
        filter_.set_fine_tune(PAGE, False)
        assert filter_.update_needs(PAGE, OUTLINE)[1] is True


class TestProcessDetectsOnlyWhenStale:
    def test_the_second_run_with_the_same_outline_runs_no_detection(self, counted):
        filter_ = Filter()
        filter_.reset_page_to_auto(PAGE)

        filter_.process(_image(), PAGE, page_outline=OUTLINE)
        first = dict(counted)
        again = filter_.process(_image(), PAGE, page_outline=OUTLINE)

        assert first == {"content": 1, "page": 1}
        assert counted == first
        assert again.content_box == filter_.get_params(PAGE).content_box

    def test_without_an_outline_every_run_detects_as_before(self, counted):
        filter_ = Filter()

        filter_.process(_image(), PAGE)
        filter_.process(_image(), PAGE)

        assert counted["content"] == 2

    def test_a_moved_outline_detects_again(self, counted):
        filter_ = Filter()
        filter_.process(_image(), PAGE, page_outline=OUTLINE)

        filter_.process(_image(), PAGE, page_outline=[(x + 5.0, y) for x, y in OUTLINE])

        assert counted["content"] == 2

    def test_the_stored_dependencies_describe_the_modes_and_outline_used(self):
        filter_ = Filter()

        params = filter_.process(_image(), PAGE, page_outline=OUTLINE)

        assert params.dependencies.rotated_page_outline == OUTLINE
        assert params.dependencies.content_detection_mode is ContentDetectionMode.AUTO
        assert params.dependencies.page_detection_mode is PageDetectionMode.DISABLED
