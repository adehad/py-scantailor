"""Select Content records upstream's intermediate images in debug mode."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.select_content import content_finder
from scantailor.filters.select_content.content_finder import find_content_box
from scantailor.filters.select_content.filter import Filter
from scantailor.filters.select_content.page_finder import find_page_box

DPI = 300.0

# Upstream's ContentBoxFinder order up to the trimming, which adds one image
# per decision; the despeckler's own images come between "content" and
# "despeckled" and "ueps" also comes from the shadow filter when it runs.
_CONTENT_STEPS = [
    "gray150",
    "bw150",
    "page_mask_applied",
    "horShadowsSeed",
    "verShadowsSeed",
    "shadowsSeed",
    "dilated",
    "shadowsDilated",
    "shadows",
    "non_border_shadows",
    "filtered_shadows",
    "content",
    "despeckled",
    "contentBlocks",
    "initial_trimming",
    "more_whitespace",
    "more_trimming",
    "except_bordering",
    "ueps",
    "textMask",
    "initial_hor_garbage",
    "initial_vert_garbage",
]
_TRIM_LABELS = {"trim_too_much", "trimmed", "not_trimmed"}
_PAGE_STEPS = [
    "gray150",
    "peakThreshold",
    "OtsuThreshold",
    "MokjiThreshold",
    "SauvolaThreshold",
    "WolfThreshold",
]


def _text_page() -> NDArray[np.uint8]:
    """A 300 dpi page with text-like stripes and a dark shadow along the left edge."""
    rng = np.random.default_rng(3)
    page = np.full((1400, 1000), 235, dtype=np.uint8)
    for top in range(240, 1100, 28):
        page[top : top + 10, 180:800] = 25
    page[:, :40] = 20
    page += rng.integers(0, 4, page.shape, dtype=np.uint8)
    return page


def _paper_on_bed() -> NDArray[np.uint8]:
    """Light paper on a dark scanner bed, 300 dpi."""
    page = np.full((1400, 1000), 30, dtype=np.uint8)
    page[100:1300, 120:880] = 230
    return page


def _labels(dbg: DebugImages) -> list[str]:
    return [image.label for image in dbg.images]


def _in_order(wanted: list[str], labels: list[str]) -> bool:
    """Whether ``wanted`` occurs in ``labels`` in this order, others between."""
    remaining = iter(labels)
    return all(label in remaining for label in wanted)


@pytest.fixture(autouse=True)
def empty_memo() -> None:
    content_finder._CONTENT_MEMO.clear()


@pytest.fixture
def dbg() -> Iterator[DebugImages]:
    with DebugImages() as images:
        yield images


def test_content_detection_records_upstream_steps_in_order_then_trims(
    dbg: DebugImages,
) -> None:
    find_content_box(_text_page(), DPI, dbg=dbg)

    labels = _labels(dbg)
    assert _in_order(_CONTENT_STEPS, labels)
    after_garbage = labels[labels.index("initial_vert_garbage") + 1 :]
    assert after_garbage
    assert set(after_garbage) <= _TRIM_LABELS


def test_content_debug_images_are_150_dpi_pictures_of_the_page(
    dbg: DebugImages,
) -> None:
    find_content_box(_text_page(), DPI, dbg=dbg)

    by_label = {image.label: image.path for image in dbg.images}
    gray = cv2.imread(str(by_label["gray150"]), cv2.IMREAD_UNCHANGED)
    trim = next(image.path for image in dbg.images if image.label in _TRIM_LABELS)
    trim_picture = cv2.imread(str(trim), cv2.IMREAD_UNCHANGED)
    assert gray is not None
    assert trim_picture is not None
    assert gray.shape == (700, 500)
    assert trim_picture.shape == (700, 500, 3)


def test_remembered_content_detection_still_records_when_debugging(
    dbg: DebugImages,
) -> None:
    page = _text_page()
    remembered = find_content_box(page, DPI)

    debugged = find_content_box(page, DPI, dbg=dbg)

    assert debugged == remembered
    assert "gray150" in _labels(dbg)


def test_remembered_detection_with_a_deferred_mask_still_records_when_debugging(
    dbg: DebugImages,
) -> None:
    page = _text_page()
    no_outside = np.zeros(page.shape, dtype=bool)
    find_content_box(page, DPI, outside=lambda: no_outside, outside_key="k")

    find_content_box(page, DPI, outside=lambda: no_outside, outside_key="k", dbg=dbg)

    assert "textMask" in _labels(dbg)


def test_content_box_is_the_same_with_and_without_debug_images(
    dbg: DebugImages,
) -> None:
    page = _text_page()
    plain = find_content_box(page, DPI)
    content_finder._CONTENT_MEMO.clear()

    assert find_content_box(page, DPI, dbg=dbg) == plain


def test_page_detection_records_the_gray_page_and_all_five_binarisations(
    dbg: DebugImages,
) -> None:
    find_page_box(_paper_on_bed(), DPI, dbg=dbg)

    assert _labels(dbg) == _PAGE_STEPS


def test_page_box_is_the_same_with_and_without_debug_images(
    dbg: DebugImages,
) -> None:
    page = _paper_on_bed()

    assert find_page_box(page, DPI, dbg=dbg) == find_page_box(page, DPI)


def test_filter_detections_forward_debug_images_without_storing(
    dbg: DebugImages,
) -> None:
    select_content = Filter()
    page_id = PageId(
        image_id=ImageId(file_path=Path("page.png")), sub_page=SubPage.SINGLE_PAGE
    )
    before = select_content.get_params(page_id)

    select_content.detect_page(_paper_on_bed(), page_id, DPI, False, dbg=dbg)
    select_content.detect_content(_text_page(), page_id, DPI, False, dbg=dbg)

    labels = _labels(dbg)
    assert labels[: len(_PAGE_STEPS)] == _PAGE_STEPS
    assert _in_order(_CONTENT_STEPS, labels[len(_PAGE_STEPS) :])
    assert select_content.get_params(page_id) == before
