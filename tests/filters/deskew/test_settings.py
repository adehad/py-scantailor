"""Tests for Deskew filter settings."""

from pathlib import Path

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.deskew.params import AutoManualMode, Params
from scantailor.filters.deskew.settings import Settings


def make_page_id(name: str, sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    """Helper to create PageId instances."""
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}.tiff")),
        sub_page=sub_page,
    )


class TestSettings:
    """Tests for Settings class."""

    def test_default_params(self):
        """Default params should be 0 angle, AUTO mode."""
        settings = Settings()
        page_id = make_page_id("test")

        params = settings.get_params(page_id)

        assert params.deskew_angle_deg == 0.0
        assert params.mode == AutoManualMode.AUTO

    def test_params_not_set_by_default(self):
        """is_params_set should return False for unset pages."""
        settings = Settings()
        page_id = make_page_id("test")

        assert settings.is_params_set(page_id) is False

    def test_set_params(self):
        """set_params should store parameters."""
        settings = Settings()
        page_id = make_page_id("test")
        params = Params(deskew_angle_deg=5.0, mode=AutoManualMode.MANUAL)

        settings.set_params(page_id, params)

        assert settings.get_params(page_id).deskew_angle_deg == 5.0
        assert settings.is_params_set(page_id) is True

    def test_set_angle_default_manual(self):
        """set_angle without mode should default to MANUAL."""
        settings = Settings()
        page_id = make_page_id("test")

        settings.set_angle(page_id, 3.0)

        params = settings.get_params(page_id)
        assert params.deskew_angle_deg == 3.0
        assert params.mode == AutoManualMode.MANUAL

    def test_set_angle_with_mode(self):
        """set_angle with mode should use that mode."""
        settings = Settings()
        page_id = make_page_id("test")

        settings.set_angle(page_id, 3.0, mode="auto")

        params = settings.get_params(page_id)
        assert params.mode == AutoManualMode.AUTO

    def test_different_pages_independent(self):
        """Different pages should have independent parameters."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")

        settings.set_params(page1, Params(deskew_angle_deg=2.0))
        settings.set_params(page2, Params(deskew_angle_deg=-3.0))

        assert settings.get_params(page1).deskew_angle_deg == 2.0
        assert settings.get_params(page2).deskew_angle_deg == -3.0

    def test_sub_pages_independent(self):
        """Different sub-pages of same image should be independent."""
        settings = Settings()
        image_id = ImageId(file_path=Path("/spread.tiff"))
        left_page = PageId(image_id=image_id, sub_page=SubPage.LEFT_PAGE)
        right_page = PageId(image_id=image_id, sub_page=SubPage.RIGHT_PAGE)

        settings.set_params(left_page, Params(deskew_angle_deg=1.5))
        settings.set_params(right_page, Params(deskew_angle_deg=-2.0))

        assert settings.get_params(left_page).deskew_angle_deg == 1.5
        assert settings.get_params(right_page).deskew_angle_deg == -2.0

    def test_apply_params_to_pages(self):
        """apply_params_to_pages should set params for all pages."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        params = Params(deskew_angle_deg=4.0, mode=AutoManualMode.MANUAL)

        settings.apply_params_to_pages([page1, page2], params)

        assert settings.get_params(page1).deskew_angle_deg == 4.0
        assert settings.get_params(page2).deskew_angle_deg == 4.0

    def test_clear(self):
        """clear should remove all parameters."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        settings.set_params(page1, Params(deskew_angle_deg=1.0))
        settings.set_params(page2, Params(deskew_angle_deg=2.0))

        settings.clear()

        assert settings.is_params_set(page1) is False
        assert settings.is_params_set(page2) is False

    def test_clear_page(self):
        """clear_page should remove parameters for one page."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        settings.set_params(page1, Params(deskew_angle_deg=1.0))
        settings.set_params(page2, Params(deskew_angle_deg=2.0))

        settings.clear_page(page1)

        assert settings.is_params_set(page1) is False
        assert settings.is_params_set(page2) is True

    def test_get_all_angles(self):
        """get_all_angles should return all stored angles."""
        settings = Settings()
        page1 = make_page_id("scan1")
        page2 = make_page_id("scan2")
        settings.set_params(page1, Params(deskew_angle_deg=1.5))
        settings.set_params(page2, Params(deskew_angle_deg=-2.5))

        angles = settings.get_all_angles()

        assert angles[page1] == 1.5
        assert angles[page2] == -2.5
