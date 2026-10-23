"""Debug images recorded by Page Split's layout detection."""

import cv2
import numpy as np

from scantailor.core.debug_images import DebugImages
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.split_finder import detect_split

from . import scan_pages as pages

LINE_FINDER_LABELS = [
    "gray100",
    "vert_raster_lines",
    "short_segments_removed",
    "hough_space",
    "vector_lines",
]
WHITESPACE_LABELS = [
    "bw300",
    "reduced",
    "garbage_removed",
    "shadows_seed",
    "shadowsDilated",
    "no_garbage",
    "ccImg",
    "spans",
    "spans_refined",
]


def test_spread_without_folding_line_records_line_search_then_whitespace_analysis():
    with DebugImages() as dbg:
        detect_split(
            pages.two_page_spread(), LayoutType.AUTO_LAYOUT_TYPE, 300.0, dbg=dbg
        )
        labels = [image.label for image in dbg.images]

    assert labels == LINE_FINDER_LABELS + WHITESPACE_LABELS


def test_spread_with_folding_line_records_only_the_line_search():
    with DebugImages() as dbg:
        detect_split(
            pages.spread_with_folding_line(),
            LayoutType.AUTO_LAYOUT_TYPE,
            300.0,
            dbg=dbg,
        )
        labels = [image.label for image in dbg.images]

    assert labels == LINE_FINDER_LABELS


def test_page_plus_offcut_request_records_spans_but_not_refined_spans():
    with DebugImages() as dbg:
        detect_split(
            pages.two_page_spread(), LayoutType.PAGE_PLUS_OFFCUT, 300.0, dbg=dbg
        )
        labels = [image.label for image in dbg.images]

    assert labels[-1] == "spans"
    assert "spans_refined" not in labels


def test_recorded_images_are_readable_pngs_at_their_working_resolutions():
    page = pages.two_page_spread(width=1500, height=1000)
    with DebugImages() as dbg:
        detect_split(page, LayoutType.AUTO_LAYOUT_TYPE, 300.0, dbg=dbg)
        decoded = {
            image.label: cv2.imread(str(image.path), cv2.IMREAD_UNCHANGED)
            for image in dbg.images
        }
    shapes = {
        label: pixels.shape for label, pixels in decoded.items() if pixels is not None
    }

    assert shapes.keys() == decoded.keys()

    assert shapes["gray100"] == (333, 500)
    assert shapes["bw300"] == (1000, 1500)
    assert shapes["no_garbage"] == (500, 750)
    assert shapes["spans"] == (500, 750, 3)
    assert shapes["vector_lines"][2] == 3


def test_detection_with_debug_images_finds_the_same_layout_as_without():
    for page in (pages.two_page_spread(), pages.spread_with_folding_line()):
        plain = detect_split(page, LayoutType.AUTO_LAYOUT_TYPE, 300.0)
        with DebugImages() as dbg:
            debugged = detect_split(page, LayoutType.AUTO_LAYOUT_TYPE, 300.0, dbg=dbg)

        assert debugged.layout == plain.layout


def test_detection_with_debug_images_leaves_the_input_page_unchanged():
    page = pages.two_page_spread()
    original = page.copy()

    with DebugImages() as dbg:
        detect_split(page, LayoutType.AUTO_LAYOUT_TYPE, 300.0, dbg=dbg)

    assert np.array_equal(page, original)
