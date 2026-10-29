"""Detection reads views resampled straight to the size each finder works at."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.core import transformation
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, _PageState, process_page
from scantailor.core.settings import BlackOnWhiteDetection
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation, render_transformed
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.select_content.content_finder import find_content_box
from scantailor.filters.select_content.page_finder import find_page_box
from scantailor.imageproc import filters
from tests.core.dark_pages import printed_page


def _page_id() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("page.png")), sub_page=SubPage.SINGLE_PAGE
    )


def _page_on_a_bed(angle: float = 2.0) -> NDArray[np.uint8]:
    """A printed page turned ``angle`` degrees on a dark scanner bed, as gray."""
    page = printed_page(angle=angle)
    bed = np.full((page.shape[0] + 200, page.shape[1] + 200), 60, dtype=np.uint8)
    bed[100:-100, 100:-100] = page
    return bed


def test_detection_resamples_no_more_than_deskew_needs_at_full_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _page_on_a_bed()
    full_size: list[tuple[int, int]] = []
    warp = transformation.warp_area_averaged
    crop = transformation._crop_window

    def counting_warp(image, dst_to_src, size, **options):
        if size[0] * size[1] >= page.size // 2:
            full_size.append(size)
        return warp(image, dst_to_src, size, **options)

    def counting_crop(image, shift, width, height):
        window = crop(image, shift, width, height)
        if window is not None and width * height >= page.size // 2:
            full_size.append((width, height))
        return window

    monkeypatch.setattr(transformation, "warp_area_averaged", counting_warp)
    monkeypatch.setattr(transformation, "_crop_window", counting_crop)

    process_page(
        page,
        _page_id(),
        StageSequence(),
        PipelineOptions(
            dpi=600.0, end_stage=FilterStage.SELECT_CONTENT, render_image=False
        ),
    )

    assert len(full_size) <= 1


@pytest.mark.parametrize("black_on_white", [True, False])
def test_a_detection_view_is_kept_until_the_geometry_changes(
    monkeypatch: pytest.MonkeyPatch, black_on_white: bool
) -> None:
    state = _PageState(_page_on_a_bed(), 600.0)
    monkeypatch.setattr(state, "is_black_on_white", lambda dbg=None: black_on_white)
    detection = BlackOnWhiteDetection()

    first = state.detection_view(detection, dpi=150)
    again = state.detection_view(detection, dpi=150)
    state.apply_deskew(_DeskewStub(), DeskewParams(deskew_angle_deg=1.0))
    after_deskew = state.detection_view(detection, dpi=150)

    assert again is first
    assert after_deskew is not first
    assert after_deskew.shape != first.shape


class _DeskewStub:
    """Applies deskew parameters to a transformation, as the filter does."""

    def apply_geometry(
        self, transformation: ImageTransformation, params: DeskewParams
    ) -> None:
        """Rotate ``transformation`` by the parameters' angle.

        Args:
            transformation (ImageTransformation): Transformation to update.
            params (DeskewParams): Angle to apply.
        """
        transformation.set_post_rotation(params.deskew_angle_deg)


def test_a_view_rendered_at_a_quarter_size_is_the_full_view_averaged_by_area() -> None:
    page = _page_on_a_bed(angle=0.0)
    height, width = page.shape
    xform = ImageTransformation.from_size(width, height, Dpi.uniform(600))
    size = (round(width / 4), round(height / 4))

    shrunk = render_transformed(page, xform, size)

    expected = cv2.resize(page, size, interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(shrunk, expected)


@pytest.mark.parametrize(
    ("angle", "shrink"),
    [(0.0, 1), (0.0, 4), (2.0, 1), (2.0, 3.7)],
    ids=["slice", "whole-factor", "rotated-full-size", "rotated-shrunk"],
)
def test_rendering_inverted_gives_the_pixels_of_the_inverted_image_rendered(
    angle: float, shrink: float
) -> None:
    page = _page_on_a_bed(angle=0.0)
    height, width = page.shape
    xform = ImageTransformation.from_size(width, height, Dpi.uniform(600))
    xform.set_pre_crop_area(
        np.array([(20.0, 10.0), (width - 60.0, 0.0), (width - 10.0, height - 30.0)])
    )
    xform.set_post_rotation(angle)
    full_width, full_height = (
        round(xform.resulting_rect.width),
        round(xform.resulting_rect.height),
    )
    size = (round(full_width / shrink), round(full_height / shrink))

    inverted = render_transformed(page, xform, size, invert=True)

    np.testing.assert_array_equal(inverted, render_transformed(255 - page, xform, size))


def test_a_view_rendered_small_keeps_the_crop_white_where_the_full_view_does() -> None:
    page = np.zeros((800, 600), dtype=np.uint8)
    xform = ImageTransformation.from_size(600, 800, Dpi.uniform(600))
    # The slanted right edge leaves a wedge of the view's rectangle outside it.
    xform.set_pre_crop_area(
        np.array([(0.0, 0.0), (300.0, 0.0), (200.0, 800.0), (0.0, 800.0)])
    )

    shrunk = render_transformed(page, xform, (75, 200))

    assert shrunk.shape == (200, 75)
    assert shrunk[190, 70] == 255
    assert shrunk[10, 10] == 0


def test_the_content_finder_finds_the_same_box_on_a_page_shrunk_ahead() -> None:
    page = _page_on_a_bed()
    height, width = page.shape
    shrunk = cv2.resize(
        page, (round(width / 4), round(height / 4)), interpolation=cv2.INTER_AREA
    )

    found = find_content_box(page, 600.0)
    found_shrunk = find_content_box(shrunk, 600.0, frame_size=(width, height))

    assert found_shrunk == found


def test_the_page_finder_finds_the_same_box_on_a_page_shrunk_ahead() -> None:
    page = _page_on_a_bed()
    height, width = page.shape
    shrunk = cv2.resize(
        page, (round(width / 4), round(height / 4)), interpolation=cv2.INTER_AREA
    )

    found = find_page_box(page, 600.0)
    found_shrunk = find_page_box(shrunk, 600.0, frame_size=(width, height))

    assert found is not None
    assert found_shrunk == found


def test_a_page_whose_geometry_is_settled_is_not_converted_to_gray_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = cv2.cvtColor(_page_on_a_bed(), cv2.COLOR_GRAY2BGR)
    stages = StageSequence()
    options = PipelineOptions(
        dpi=600.0, end_stage=FilterStage.PAGE_LAYOUT, render_image=False
    )
    process_page(page, _page_id(), stages, options)
    converted: list[tuple[int, ...]] = []
    qgray = filters.qgray

    def counting_qgray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        converted.append(image.shape)
        return qgray(image)

    monkeypatch.setattr(filters, "qgray", counting_qgray)

    process_page(page, _page_id(), stages, options)

    assert converted == []
