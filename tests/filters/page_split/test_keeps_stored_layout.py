"""Filter.keeps_stored_layout answers settle's question without the image's pixels."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId
from scantailor.filters.page_split.filter import Filter
from scantailor.filters.page_split.layout_type import LayoutType

PAGE = PageId(image_id=ImageId(file_path=Path("/spread.tif")))


def _spread() -> np.ndarray:
    pixels = np.full((400, 600), 235, dtype=np.uint8)
    for left in (40, 340):
        for top in range(60, 340, 20):
            pixels[top : top + 8, left : left + 220] = 25
    return pixels


@pytest.fixture
def settled() -> Filter:
    """A filter that has settled the spread's layout once."""
    split = Filter()
    split.settle(_spread(), PAGE)
    return split


def test_a_settled_layout_is_kept_for_the_same_size(settled: Filter) -> None:
    assert settled.keeps_stored_layout(PAGE, (600, 400), 0)
    assert not settled.settle(_spread(), PAGE).detected


def test_a_page_without_a_layout_is_not_kept() -> None:
    assert not Filter().keeps_stored_layout(PAGE, (600, 400), 0)


def test_a_new_rotation_is_not_kept(settled: Filter) -> None:
    assert not settled.keeps_stored_layout(PAGE, (400, 600), 90)


def test_a_new_requested_layout_type_is_not_kept(settled: Filter) -> None:
    settled.set_layout_type(PAGE, LayoutType.SINGLE_PAGE_UNCUT)

    assert not settled.keeps_stored_layout(PAGE, (600, 400), 0)
