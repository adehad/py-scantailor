"""Scope resolution for the Split mode dialog."""

from pathlib import Path

from scantailor.core.models import ImageId, PageId
from scantailor.core.propagation import ApplyScope, select_pages


def _pages(count: int) -> list[PageId]:
    return [PageId(image_id=ImageId(file_path=Path(f"/{i}.tif"))) for i in range(count)]


class TestEveryOtherSelected:
    def test_starts_from_the_current_page_not_the_first_selected_page(self):
        pages = _pages(6)
        selected = pages[1:6]

        result = select_pages(
            ApplyScope.EVERY_OTHER_SELECTED, pages[2], pages, selected
        )

        assert result == [pages[2], pages[4]]

    def test_current_page_at_the_start_of_the_selection_takes_the_first_pages(self):
        pages = _pages(6)
        selected = pages[1:6]

        result = select_pages(
            ApplyScope.EVERY_OTHER_SELECTED, pages[1], pages, selected
        )

        assert result == [pages[1], pages[3], pages[5]]

    def test_current_page_outside_the_selection_starts_from_the_first_selected(self):
        pages = _pages(6)
        selected = pages[2:6]

        result = select_pages(
            ApplyScope.EVERY_OTHER_SELECTED, pages[0], pages, selected
        )

        assert result == [pages[2], pages[4]]
