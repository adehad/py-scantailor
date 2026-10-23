"""Debug images of automatic dewarping, on a generated bent book page."""

from collections.abc import Iterator

import cv2
import numpy as np
import pytest

from scantailor.core.debug_images import DebugImage, DebugImages
from scantailor.dewarping.auto_model import AutoModelResult, detect_distortion_model
from scantailor.dewarping.marginal_model import build_marginal_model
from scantailor.dewarping.text_line_tracer import trace_text_lines
from scantailor.dewarping.top_bottom_edge_tracer import trace_top_bottom_edges
from tests.dewarping.book_pages import BentPage, bent_book_page

DPI = 200
CONTENT = (160.0, 200.0, 890.0, 1090.0)

TEXT_LINE_LABELS = [
    "downscaled",
    "binarized",
    "sanitized",
    "left_envelope",
    "right_envelope",
    "left_ransac_model",
    "right_ransac_model",
    "vertBounds",
    "first_dir_deriv",
    "first_dir_deriv_blurred",
    "second_dir_deriv",
    "initialBinarization",
    "abs",
    "blurred",
    "+= diff",
    "postBinarization",
    "obstacles",
    "initial_closed",
    "post &&= initial",
    "seeds",
    "traced",
    "filtered1",
    "initial_snakes",
    "evolved_snakes1",
    "evolved_snakes2",
    "filtered2",
]
EDGE_LABELS = [
    "gradient",
    "best_paths_ltr",
    "down_the_hill_snakes",
    "up_the_hill_snakes",
]
MODEL_LABELS = ["trimmed_polylines", "distortion_model"]


def _read(image: DebugImage) -> np.ndarray:
    pixels = cv2.imread(str(image.path), cv2.IMREAD_UNCHANGED)
    assert pixels is not None
    return pixels


@pytest.fixture(scope="module")
def page() -> BentPage:
    return bent_book_page(bow=45.0, squeeze=-25.0)


@pytest.fixture(scope="module")
def recorded(page: BentPage) -> Iterator[tuple[AutoModelResult, list[DebugImage]]]:
    with DebugImages() as dbg:
        result = detect_distortion_model(page.image, DPI, CONTENT, dbg=dbg)
        yield result, list(dbg.images)


def test_detecting_a_bent_page_records_upstreams_labels_in_upstreams_order(
    recorded: tuple[AutoModelResult, list[DebugImage]],
) -> None:
    _, images = recorded

    labels = [image.label for image in images]

    assert labels == [*TEXT_LINE_LABELS, "downscaled", *EDGE_LABELS, *MODEL_LABELS]


def test_the_edge_tracer_images_are_at_its_working_size_and_the_rest_at_the_pages(
    page: BentPage, recorded: tuple[AutoModelResult, list[DebugImage]]
) -> None:
    _, images = recorded

    shapes = [_read(image).shape[:2] for image in images]

    # The page is 1200 x 1600. The text tracer keeps 200 dpi as it is, the edge
    # tracer fits the page into 1000 pixels and the builder draws on the page.
    edge_start = len(TEXT_LINE_LABELS)
    edge_end = edge_start + 1 + len(EDGE_LABELS)
    assert set(shapes[:edge_start]) == {page.image.shape[:2]}
    assert set(shapes[edge_start:edge_end]) == {(1000, 750)}
    assert set(shapes[edge_end:]) == {page.image.shape[:2]}


def test_the_model_is_the_same_with_and_without_debug_images(
    page: BentPage, recorded: tuple[AutoModelResult, list[DebugImage]]
) -> None:
    with_debug, _ = recorded

    plain = detect_distortion_model(page.image, DPI, CONTENT)

    assert plain.metrics == with_debug.metrics
    assert plain.model is not None and with_debug.model is not None
    for plain_curve, debug_curve in (
        (plain.model.top_curve, with_debug.model.top_curve),
        (plain.model.bottom_curve, with_debug.model.bottom_curve),
    ):
        assert plain_curve is not None and debug_curve is not None
        np.testing.assert_array_equal(plain_curve.polyline, debug_curve.polyline)


def test_a_small_edge_image_is_traced_without_a_downscaled_image(
    page: BentPage,
) -> None:
    small = cv2.resize(page.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    bounds = trace_text_lines(page.image, DPI, (160, 200, 890, 1090)).vertical_bounds
    assert bounds is not None
    halved = tuple(
        tuple((0.5 * x, 0.5 * y) for x, y in line)
        for line in (bounds.first, bounds.second)
    )

    with DebugImages() as dbg:
        trace_top_bottom_edges(small, halved, dbg)  # type: ignore[arg-type]

        labels = [image.label for image in dbg.images]

    assert labels == EDGE_LABELS


def test_a_page_without_text_records_only_the_steps_before_tracing_lines() -> None:
    blank = np.full((600, 400), 255, dtype=np.uint8)

    with DebugImages() as dbg:
        detect_distortion_model(blank, DPI, (0.0, 0.0, 400.0, 600.0), dbg=dbg)

        labels = [image.label for image in dbg.images]

    # No line is traced, so the refiner has no snakes and the builder has fewer
    # than two curves. Neither records anything.
    assert "initial_snakes" not in labels
    assert "distortion_model" not in labels
    assert labels[:3] == ["downscaled", "binarized", "sanitized"]


def test_a_marginal_model_records_the_paper_mask_with_its_control_points(
    page: BentPage,
) -> None:
    with DebugImages() as dbg:
        build_marginal_model(page.image, CONTENT, spine_on_right=True, dbg=dbg)

        (image,) = dbg.images
        pixels = _read(image)

    assert image.label == "marginal dewarping"
    assert pixels.shape[:2] == page.image.shape[:2]
    red = (pixels[:, :, 2] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 0] == 0)
    # Seven control points on each curve, a 20 pixel square each, some may overlap.
    assert 7 * 400 <= int(red.sum()) <= 14 * 400
