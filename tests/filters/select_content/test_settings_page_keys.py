"""Select Content params are stored per page, even within a multi-page file."""

from pathlib import Path

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.select_content.content_box import ContentBox
from scantailor.filters.select_content.params import Params
from scantailor.filters.select_content.settings import Settings

TIFF = Path("/scans/book.tiff")


def _page(page: int, sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    return PageId(image_id=ImageId(file_path=TIFF, page=page), sub_page=sub_page)


def _params(x: float) -> Params:
    return Params().with_manual_content(ContentBox(x=x, y=10, width=100, height=100))


def test_two_pages_of_one_multi_page_tiff_keep_separate_content_boxes() -> None:
    settings = Settings()

    settings.set_params(_page(1), _params(11))
    settings.set_params(_page(2), _params(22))

    assert settings.get_params(_page(1)).content_box.x == 11
    assert settings.get_params(_page(2)).content_box.x == 22


def test_unset_page_of_a_multi_page_tiff_is_not_set_by_its_sibling() -> None:
    settings = Settings()

    settings.set_params(_page(1), _params(11))

    assert not settings.is_params_set(_page(2))


def test_sub_pages_of_one_image_page_keep_separate_content_boxes() -> None:
    settings = Settings()

    settings.set_params(_page(1, SubPage.LEFT_PAGE), _params(11))
    settings.set_params(_page(1, SubPage.RIGHT_PAGE), _params(22))

    assert settings.get_params(_page(1, SubPage.LEFT_PAGE)).content_box.x == 11
    assert settings.get_params(_page(1, SubPage.RIGHT_PAGE)).content_box.x == 22


def test_clearing_one_page_leaves_its_siblings() -> None:
    settings = Settings()
    settings.set_params(_page(1), _params(11))
    settings.set_params(_page(2), _params(22))

    settings.clear_page(_page(1))

    assert not settings.is_params_set(_page(1))
    assert settings.get_params(_page(2)).content_box.x == 22


def test_apply_to_pages_writes_each_page_separately() -> None:
    settings = Settings()

    settings.apply_params_to_pages([_page(1), _page(2)], _params(5))
    settings.set_params(_page(2), _params(7))

    assert settings.get_params(_page(1)).content_box.x == 5
    assert settings.get_params(_page(2)).content_box.x == 7


def test_json_round_trip_keeps_pages_of_one_file_apart() -> None:
    settings = Settings()
    settings.set_params(_page(1), _params(11))
    settings.set_params(_page(2, SubPage.RIGHT_PAGE), _params(22))

    loaded = Settings.model_validate_json(settings.model_dump_json())

    assert loaded.get_params(_page(1)).content_box.x == 11
    assert loaded.get_params(_page(2, SubPage.RIGHT_PAGE)).content_box.x == 22
    assert not loaded.is_params_set(_page(2))


def test_json_written_with_path_based_keys_still_loads() -> None:
    legacy = {
        "params_by_page": {
            f"{TIFF}:0": _params(11).model_dump(mode="json"),
            f"{TIFF}:2": _params(22).model_dump(mode="json"),
        }
    }

    loaded = Settings.model_validate(legacy)

    assert loaded.get_params(_page(0)).content_box.x == 11
    assert loaded.get_params(_page(0, SubPage.RIGHT_PAGE)).content_box.x == 22


def test_legacy_key_with_a_drive_letter_loads() -> None:
    legacy = {
        "params_by_page": {
            "C:\\scans\\book.tiff:1": _params(11).model_dump(mode="json")
        }
    }

    loaded = Settings.model_validate(legacy)

    page = PageId(
        image_id=ImageId(file_path=Path("C:\\scans\\book.tiff")),
        sub_page=SubPage.LEFT_PAGE,
    )
    assert loaded.get_params(page).content_box.x == 11
