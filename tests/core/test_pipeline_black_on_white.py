"""Detection and Output on pages of light content on a dark background."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.content_mask import compute_content_mask
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, PipelineResult, process_page
from scantailor.core.settings import BlackOnWhiteDetection
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation
from tests.core.dark_pages import dark_page, dark_sheet_on_light_bed, printed_page

_ANGLE = 3.0
_OFF = BlackOnWhiteDetection(enabled=False)
_ON = BlackOnWhiteDetection()


def _page_id(name: str = "page.png") -> PageId:
    return PageId(image_id=ImageId(file_path=Path(name)), sub_page=SubPage.SINGLE_PAGE)


def _detect(
    image: np.ndarray, detection: BlackOnWhiteDetection, page_auto: bool = False
) -> PipelineResult:
    """Detect Deskew and Select Content on a page that Page Split leaves whole."""
    page = _page_id()
    sequence = StageSequence()
    if page_auto:
        sequence.select_content_filter.reset_page_to_auto(page)
    options = PipelineOptions(
        start_stage=FilterStage.DESKEW,
        end_stage=FilterStage.SELECT_CONTENT,
        dpi=300,
        black_on_white_detection=detection,
    )
    return process_page(image, page, sequence, options)


def _box(result: PipelineResult, name: str) -> tuple[int, int, int, int]:
    box = result.metadata["SELECT_CONTENT"][name]
    return tuple(round(box[key]) for key in ("x", "y", "width", "height"))


@pytest.fixture(scope="module")
def printed() -> PipelineResult:
    """The same page printed black on white, the result to match."""
    return _detect(printed_page(angle=_ANGLE), _ON)


class TestDetectionOnDarkSheetOnLightBed:
    """A dark sheet whose light bed defeats detection unless it is inverted."""

    def test_deskew_finds_the_text_angle_once_the_page_is_inverted(self, printed):
        before = _detect(dark_sheet_on_light_bed(angle=_ANGLE), _OFF)
        after = _detect(dark_sheet_on_light_bed(angle=_ANGLE), _ON)

        assert not before.metadata["DESKEW"]["confident"]
        assert after.metadata["DESKEW"]["confident"]
        assert after.metadata["DESKEW"]["skew_angle"] == pytest.approx(
            printed.metadata["DESKEW"]["skew_angle"], abs=0.05
        )

    def test_content_box_lands_on_the_text_once_the_page_is_inverted(self, printed):
        before = _detect(dark_sheet_on_light_bed(angle=_ANGLE), _OFF)
        after = _detect(dark_sheet_on_light_bed(angle=_ANGLE), _ON)

        width, height = _box(printed, "content_box")[2:]
        assert _box(before, "content_box")[2:] != pytest.approx((width, height), abs=6)
        assert _box(after, "content_box")[2:] == pytest.approx((width, height), abs=6)

    def test_page_box_finds_the_sheet_once_the_page_is_inverted(self):
        after = _detect(dark_sheet_on_light_bed(), _ON, page_auto=True)

        assert _box(after, "page_box") == pytest.approx((150, 150, 1100, 1500), abs=10)


def test_content_box_of_a_dark_page_matches_the_printed_one(printed):
    after = _detect(dark_page(angle=_ANGLE), _ON)

    assert _box(after, "content_box") == _box(printed, "content_box")
    assert after.metadata["DESKEW"]["skew_angle"] == pytest.approx(
        printed.metadata["DESKEW"]["skew_angle"]
    )


def test_page_finder_takes_a_dark_page_for_its_surround_without_inversion(printed):
    before = _detect(dark_page(angle=_ANGLE), _OFF, page_auto=True)

    assert _box(before, "page_box") != (0, 0, 1100, 1500)


def test_detection_switched_off_leaves_a_printed_page_as_it_was(printed):
    unchanged = _detect(printed_page(angle=_ANGLE), _OFF)

    assert unchanged.metadata == printed.metadata


def test_content_mask_of_an_inverted_dark_page_matches_the_printed_one():
    page = dark_page()
    xform = ImageTransformation.from_size(1100, 1500, Dpi.uniform(300))

    dark = compute_content_mask(page, xform, black_on_white=False)
    light = compute_content_mask(255 - page, xform)

    assert np.array_equal(dark.image, light.image)


class TestOutputPolarity:
    """Output takes the measured polarity unless the user chose one."""

    @staticmethod
    def _render(
        sequence: StageSequence,
        detection: BlackOnWhiteDetection = _ON,
        detect_only: bool = False,
    ) -> PipelineResult:
        options = PipelineOptions(
            dpi=300, black_on_white_detection=detection, detect_only=detect_only
        )
        return process_page(dark_page(400, 500), _page_id(), sequence, options)

    def test_a_dark_page_is_stored_and_rendered_white_on_black(self):
        sequence = StageSequence()

        result = self._render(sequence)

        params = sequence.output_filter.get_params(_page_id())
        assert params.black_on_white is False
        assert not params.black_on_white_set_manually
        assert result.metadata["OUTPUT"]["black_on_white"] is False
        assert np.mean(result.image) < 128

    def test_a_polarity_chosen_by_hand_is_kept(self):
        sequence = StageSequence()
        output = sequence.output_filter
        output.set_params(
            _page_id(), output.get_params(_page_id()).with_manual_black_on_white(True)
        )

        self._render(sequence)

        assert output.get_params(_page_id()).black_on_white is True

    @pytest.mark.parametrize(
        "detection",
        [_OFF, BlackOnWhiteDetection(at_output=False)],
        ids=["detection off", "not at output"],
    )
    def test_output_keeps_the_stored_polarity_when_it_does_not_measure(self, detection):
        sequence = StageSequence()

        self._render(sequence, detection)

        assert sequence.output_filter.get_params(_page_id()).black_on_white is True
        assert not sequence.output_filter.is_params_set(_page_id())

    def test_detect_only_run_stores_no_polarity(self):
        sequence = StageSequence()

        self._render(sequence, detect_only=True)

        assert not sequence.output_filter.is_params_set(_page_id())
