"""A manual content box follows the page box when the split offcuts move.

The box is stored in the cropped page's coordinates, so moving a cutter shifts
the page under it. The rule is upstream's: translate the box by how far the page
box's centre moved, then clip it to the page box.
"""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.filters.select_content.detection import PageDetectionMode

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


class TestManualContentBoxFollowsThePage:
    def test_the_box_is_left_alone_while_the_offcuts_stay(self) -> None:
        sequence = _sequence(30, 350)
        _run(sequence)

        _run(sequence)

        assert _content(sequence) == MANUAL_BOX

    def test_the_box_moves_with_the_page_boxs_centre_when_an_offcut_moves(self) -> None:
        """The page shrinks from 320 to 280 px, so its centre moves 20 px left."""
        sequence = _sequence(30, 350)
        _run(sequence)

        _move_cutters(sequence, 70, 350)
        _run(sequence)

        assert _content(sequence) == ContentBox(
            x=30.0, y=60.0, width=110.0, height=140.0
        )

    def test_the_box_is_clipped_to_the_new_page_box(self) -> None:
        """Growing the left offcut shrinks the page below the box's right edge."""
        sequence = _sequence(30, 350)
        _run(sequence)

        _move_cutters(sequence, 30, 150)
        _run(sequence)

        page = sequence.select_content_filter.get_params(PAGE).page_box
        box = _content(sequence)
        assert page == PageBox(x=0.0, y=0.0, width=120.0, height=float(HEIGHT))
        assert box.x >= page.x and box.right <= page.right

    def test_an_automatic_box_is_not_moved(self) -> None:
        sequence = _sequence(30, 350)
        sequence.select_content_filter.reset_content_to_auto(PAGE)
        _run(sequence)

        _move_cutters(sequence, 70, 350)
        _run(sequence)

        params = sequence.select_content_filter.get_params(PAGE)
        assert params.is_content_auto()

    def test_a_manual_page_box_keeps_the_box_where_it_is(self) -> None:
        sequence = _sequence(30, 350)
        sequence.select_content_filter.set_manual_page(
            PAGE, PageBox(x=0, y=0, width=320, height=HEIGHT)
        )
        _run(sequence)

        _move_cutters(sequence, 70, 350)
        _run(sequence)

        params = sequence.select_content_filter.get_params(PAGE)
        assert params.page_detection_mode is PageDetectionMode.MANUAL
        assert _content(sequence) == MANUAL_BOX

    @pytest.mark.parametrize("moves", [1, 2])
    def test_repeated_runs_move_the_box_only_when_the_page_moved(
        self, moves: int
    ) -> None:
        sequence = _sequence(30, 350)
        _run(sequence)
        _move_cutters(sequence, 70, 350)

        for _ in range(moves):
            _run(sequence)

        assert _content(sequence).x == pytest.approx(30.0)
