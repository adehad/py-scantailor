"""Tests for Settings class."""

from __future__ import annotations

from pathlib import Path

from scantailor.core import ImageId, PageId, SubPage
from scantailor.filters.select_content import (
    ContentBox,
    ContentDetectionMode,
    Params,
    Settings,
)


def make_page_id(name: str, sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    """Helper to create PageId instances."""
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}.tiff")),
        sub_page=sub_page,
    )


class TestSettings:
    """Tests for Settings class."""

    def test_default_params(self):
        """Default params should have empty boxes."""
        settings = Settings()
        page_id = make_page_id("test")

        params = settings.get_params(page_id)

        assert params.content_box.is_empty()
        assert params.page_box.is_empty()

    def test_params_not_set_by_default(self):
        """is_params_set should return False for unset pages."""
        settings = Settings()
        page_id = make_page_id("test")

        assert settings.is_params_set(page_id) is False

    def test_set_params(self):
        """set_params should store parameters."""
        settings = Settings()
        page_id = make_page_id("test")
        params = Params(
            content_box=ContentBox(x=10, y=20, width=100, height=50),
            content_detection_mode=ContentDetectionMode.MANUAL,
        )

        settings.set_params(page_id, params)

        stored = settings.get_params(page_id)
        assert stored.content_box.x == 10
        assert stored.content_detection_mode == ContentDetectionMode.MANUAL
        assert settings.is_params_set(page_id) is True

    def test_different_pages_independent(self):
        """Different pages should have independent parameters."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        settings.set_params(
            page1,
            Params(content_box=ContentBox(x=10, y=10, width=100, height=100)),
        )
        settings.set_params(
            page2,
            Params(content_box=ContentBox(x=50, y=50, width=200, height=200)),
        )

        assert settings.get_params(page1).content_box.x == 10
        assert settings.get_params(page2).content_box.x == 50

    def test_sub_pages_independent(self):
        """Different sub-pages of same image should be independent."""
        settings = Settings()
        image_id = ImageId(file_path=Path("/spread.tiff"))
        left_page = PageId(image_id=image_id, sub_page=SubPage.LEFT_PAGE)
        right_page = PageId(image_id=image_id, sub_page=SubPage.RIGHT_PAGE)

        settings.set_params(
            left_page,
            Params(content_box=ContentBox(x=10, y=10, width=100, height=100)),
        )
        settings.set_params(
            right_page,
            Params(content_box=ContentBox(x=400, y=10, width=100, height=100)),
        )

        assert settings.get_params(left_page).content_box.x == 10
        assert settings.get_params(right_page).content_box.x == 400

    def test_clear_page(self):
        """clear_page should remove parameters for one page."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        settings.set_params(page1, Params())
        settings.set_params(page2, Params())

        settings.clear_page(page1)

        assert settings.is_params_set(page1) is False
        assert settings.is_params_set(page2) is True

    def test_clear(self):
        """clear should remove all parameters."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        settings.set_params(page1, Params())
        settings.set_params(page2, Params())

        settings.clear()

        assert settings.is_params_set(page1) is False
        assert settings.is_params_set(page2) is False

    def test_apply_params_to_pages(self):
        """apply_params_to_pages should set params for all pages."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        params = Params(
            content_box=ContentBox(x=20, y=20, width=150, height=150),
            content_detection_mode=ContentDetectionMode.MANUAL,
        )

        settings.apply_params_to_pages([page1, page2], params)

        assert settings.get_params(page1).content_box.x == 20
        assert settings.get_params(page2).content_box.x == 20

    def test_page_detection_box(self):
        """get/set page detection box should work."""
        settings = Settings()

        settings.set_page_detection_box(200.0, 300.0)
        width, height = settings.get_page_detection_box()

        assert width == 200.0
        assert height == 300.0

    def test_page_detection_box_clamps_negative(self):
        """set_page_detection_box should clamp negative values."""
        settings = Settings()

        settings.set_page_detection_box(-100.0, -200.0)
        width, height = settings.get_page_detection_box()

        assert width == 0.0
        assert height == 0.0

    def test_page_detection_tolerance(self):
        """get/set page detection tolerance should work."""
        settings = Settings()

        settings.set_page_detection_tolerance(0.5)

        assert settings.get_page_detection_tolerance() == 0.5

    def test_page_detection_tolerance_clamps(self):
        """set_page_detection_tolerance should clamp to 0-1."""
        settings = Settings()

        settings.set_page_detection_tolerance(-0.5)
        assert settings.get_page_detection_tolerance() == 0.0

        settings.set_page_detection_tolerance(1.5)
        assert settings.get_page_detection_tolerance() == 1.0

    def test_get_all_content_boxes(self):
        """get_all_content_boxes should return all valid boxes."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        page3 = make_page_id("scan3")

        settings.set_params(
            page1,
            Params(content_box=ContentBox(x=10, y=10, width=100, height=100)),
        )
        settings.set_params(
            page2,
            Params(content_box=ContentBox(x=20, y=20, width=200, height=200)),
        )
        settings.set_params(page3, Params())  # Empty box

        boxes = settings.get_all_content_boxes()

        # Should only include valid (non-empty) boxes
        assert len(boxes) == 2
