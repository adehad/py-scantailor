"""Hand-drawn boxes stay on their content when the split cutters move.

The boxes are stored in the cropped page's coordinates, so moving a cutter
shifts the page under them. Each box is mapped from the view it was drawn in
into the new view, then clipped to the page box (or, for the page box, to the
image).
"""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.page_geometry import build_transformation
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.filters.select_content.dependencies import Dependencies
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)
from scantailor.filters.select_content.filter import Filter
from scantailor.filters.select_content.params import Params

WIDTH = 400
HEIGHT = 300
DPI = 254.0
PAGE = PageId(
    image_id=ImageId(file_path=Path("offcut.tiff")), sub_page=SubPage.SINGLE_PAGE
)
MANUAL_BOX = ContentBox(x=50.0, y=60.0, width=110.0, height=140.0)


def _image() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), 200, dtype=np.uint8)


def _sequence(left: float, right: float) -> StageSequence:
    sequence = StageSequence()
    sequence.page_split_filter.set_manual_cutters(PAGE, left, right, WIDTH, HEIGHT)
    sequence.deskew_filter.set_manual_angle(PAGE, 0.0)
    sequence.select_content_filter.set_manual_content(PAGE, MANUAL_BOX)
    return sequence


def _run(sequence: StageSequence):
    options = PipelineOptions(end_stage=FilterStage.SELECT_CONTENT, dpi=DPI)
    return process_page(_image(), PAGE, sequence, options)


def _move_cutters(sequence: StageSequence, left: float, right: float) -> None:
    sequence.page_split_filter.set_manual_cutters(PAGE, left, right, WIDTH, HEIGHT)


def _content(sequence: StageSequence) -> ContentBox:
    return sequence.select_content_filter.get_params(PAGE).content_box


def _moved_by(dx: float, box: ContentBox = MANUAL_BOX) -> ContentBox:
    return box.model_copy(update={"x": box.x + dx})


class TestManualContentBoxStaysOnItsContent:
    def test_the_box_is_left_alone_while_the_cutters_stay(self) -> None:
        sequence = _sequence(30, 350)
        _run(sequence)

        _run(sequence)

        assert _content(sequence) == MANUAL_BOX

    def test_moving_the_left_cutter_shifts_the_box_by_the_whole_move(self) -> None:
        """The page's left edge moves 40 px right; its content is 40 px further left."""
        sequence = _sequence(30, 350)
        _run(sequence)

        _move_cutters(sequence, 70, 350)
        _run(sequence)

        assert _content(sequence) == _moved_by(-40.0)

    def test_moving_the_right_cutter_leaves_the_box_where_it_is(self) -> None:
        sequence = _sequence(30, 350)
        _run(sequence)

        _move_cutters(sequence, 30, 310)
        _run(sequence)

        assert _content(sequence) == MANUAL_BOX

    def test_moving_both_cutters_inwards_follows_the_left_one_only(self) -> None:
        sequence = _sequence(30, 350)
        _run(sequence)

        _move_cutters(sequence, 70, 310)
        _run(sequence)

        assert _content(sequence) == _moved_by(-40.0)

    def test_shifting_both_cutters_together_shifts_the_box_the_other_way(self) -> None:
        """The page outline keeps its size, yet the content under it has moved."""
        sequence = _sequence(30, 350)
        _run(sequence)

        _move_cutters(sequence, 50, 370)
        _run(sequence)

        assert _content(sequence) == _moved_by(-20.0)

    def test_the_box_is_clipped_where_the_cutter_cuts_into_it(self) -> None:
        """The left cutter passes 20 px into the box, which loses that strip."""
        sequence = _sequence(30, 350)
        _run(sequence)

        _move_cutters(sequence, 100, 350)
        _run(sequence)

        assert _content(sequence) == ContentBox(x=0.0, y=60.0, width=90.0, height=140.0)

    def test_an_automatic_box_is_detected_again_rather_than_moved(self) -> None:
        sequence = _sequence(30, 350)
        sequence.select_content_filter.reset_content_to_auto(PAGE)
        _run(sequence)

        _move_cutters(sequence, 70, 350)
        _run(sequence)

        params = sequence.select_content_filter.get_params(PAGE)
        assert params.is_content_auto()

    def test_a_manual_page_box_follows_the_content_too(self) -> None:
        sequence = _sequence(30, 350)
        sequence.select_content_filter.set_manual_page(
            PAGE, PageBox(x=20, y=0, width=300, height=HEIGHT)
        )
        _run(sequence)

        _move_cutters(sequence, 70, 350)
        _run(sequence)

        params = sequence.select_content_filter.get_params(PAGE)
        assert params.page_detection_mode is PageDetectionMode.MANUAL
        assert params.page_box == PageBox(x=0.0, y=0.0, width=280.0, height=HEIGHT)
        assert _content(sequence) == _moved_by(-40.0)

    @pytest.mark.parametrize("moves", [1, 2])
    def test_repeated_runs_move_the_box_only_when_the_view_changed(
        self, moves: int
    ) -> None:
        sequence = _sequence(30, 350)
        _run(sequence)
        _move_cutters(sequence, 70, 350)

        for _ in range(moves):
            _run(sequence)

        assert _content(sequence) == _moved_by(-40.0)

    def test_a_change_of_deskew_angle_keeps_the_box_around_its_content(self) -> None:
        sequence = _sequence(30, 350)
        _run(sequence)
        before = build_transformation(
            sequence, PAGE, (WIDTH, HEIGHT), DPI, FilterStage.SELECT_CONTENT
        )

        sequence.deskew_filter.set_manual_angle(PAGE, 2.0)
        _run(sequence)

        after = build_transformation(
            sequence, PAGE, (WIDTH, HEIGHT), DPI, FilterStage.SELECT_CONTENT
        )
        expected_centre = after.transform_point(
            *before.transform_point_back(MANUAL_BOX.center_x, MANUAL_BOX.center_y)
        )
        box = _content(sequence)
        assert (box.center_x, box.center_y) == pytest.approx(expected_centre, abs=0.01)
        assert box.width == pytest.approx(MANUAL_BOX.width, abs=5.0)
        assert box.height == pytest.approx(MANUAL_BOX.height, abs=5.0)


class TestBoxesDrawnInAnUnknownView:
    """Boxes stored without the view they were drawn in stay where they are."""

    def test_the_box_is_only_clipped(self) -> None:
        filter_ = Filter()
        filter_.set_params(
            PAGE,
            Params(
                content_detection_mode=ContentDetectionMode.MANUAL,
                content_box=MANUAL_BOX,
                dependencies=Dependencies(
                    content_detection_mode=ContentDetectionMode.MANUAL,
                    rotated_page_outline=[(0, 0), (320, 0), (320, 300), (0, 300)],
                ),
            ),
        )
        narrower = np.full((HEIGHT, 280), 200, dtype=np.uint8)

        filter_.process(
            narrower,
            PAGE,
            page_outline=[(0, 0), (280, 0), (280, 300), (0, 300)],
            view_transform=np.array([[1, 0, -70], [0, 1, 0], [0, 0, 1]], dtype=float),
        )

        assert filter_.get_params(PAGE).content_box == MANUAL_BOX
