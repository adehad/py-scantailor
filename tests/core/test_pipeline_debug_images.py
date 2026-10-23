"""Which intermediate images a render records, and that it records none by default."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from scantailor.core.debug_images import DebugImages
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import ColorCommonOptions
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.page_split import filter as page_split_filter
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from tests.dewarping.synthetic import PAGE_DPI, curved_text_page

SKEW_DEGREES = 3.0
MANUAL_ANGLE = 2.0


def _page() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("scan.tiff")), sub_page=SubPage.SINGLE_PAGE
    )


def _skewed_text_page() -> np.ndarray:
    """White page with rows of dark word-like blocks, rotated by SKEW_DEGREES."""
    image = np.full((500, 400), 255, dtype=np.uint8)
    for row in range(60, 440, 24):
        for column in range(40, 360, 30):
            image[row : row + 10, column : column + 22] = 0
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), SKEW_DEGREES, 1.0)
    return np.asarray(
        cv2.warpAffine(image, matrix, (width, height), borderValue=255),
        dtype=np.uint8,
    )


def _labels_of_render(
    sequence: StageSequence, stage: FilterStage, image: np.ndarray | None = None
) -> list[str]:
    with DebugImages() as dbg:
        process_page(
            _skewed_text_page() if image is None else image,
            _page(),
            sequence,
            PipelineOptions(end_stage=stage, dpi=300.0, debug_images=dbg),
        )
        return [item.label for item in dbg.images]


def test_deskew_detection_records_the_binarised_page_then_the_cleaned_page():
    assert _labels_of_render(StageSequence(), FilterStage.DESKEW) == [
        "bw_rotated",
        "after_cleanup",
    ]


def test_deskew_with_a_stored_angle_records_its_images_without_changing_it():
    sequence = StageSequence()
    _labels_of_render(sequence, FilterStage.DESKEW)
    stored = sequence.deskew_filter.get_params(_page())

    labels = _labels_of_render(sequence, FilterStage.DESKEW)

    assert labels == ["bw_rotated", "after_cleanup"]
    assert sequence.deskew_filter.get_params(_page()) == stored


def test_deskew_with_a_manual_angle_records_its_images_and_keeps_the_angle():
    sequence = StageSequence()
    sequence.deskew_filter.set_manual_angle(_page(), MANUAL_ANGLE)

    labels = _labels_of_render(sequence, FilterStage.DESKEW)

    assert labels == ["bw_rotated", "after_cleanup"]
    assert sequence.deskew_filter.get_params(_page()).deskew_angle_deg == MANUAL_ANGLE


def test_render_records_only_the_images_of_its_last_stage():
    labels = _labels_of_render(StageSequence(), FilterStage.SELECT_CONTENT)

    assert "gray150" in labels
    assert "bw_rotated" not in labels
    assert "after_cleanup" not in labels
    assert "gray100" not in labels


@pytest.mark.parametrize(
    "stage", [FilterStage.FIX_ORIENTATION, FilterStage.PAGE_LAYOUT]
)
def test_stage_without_detection_records_nothing(stage: FilterStage):
    assert _labels_of_render(StageSequence(), stage) == []


def test_render_without_a_collector_detects_the_same_angle_as_with_one():
    plain = StageSequence()
    process_page(
        _skewed_text_page(),
        _page(),
        plain,
        PipelineOptions(end_stage=FilterStage.DESKEW),
    )
    debugged = StageSequence()
    _labels_of_render(debugged, FilterStage.DESKEW)

    assert (
        plain.deskew_filter.get_params(_page()).deskew_angle_deg
        == debugged.deskew_filter.get_params(_page()).deskew_angle_deg
    )


def test_pipeline_options_carry_no_collector_by_default():
    assert PipelineOptions().debug_images is None


def test_select_content_with_stored_boxes_records_the_finders_and_keeps_the_boxes():
    sequence = StageSequence()
    _labels_of_render(sequence, FilterStage.SELECT_CONTENT)
    stored = sequence.select_content_filter.get_params(_page())

    labels = _labels_of_render(sequence, FilterStage.SELECT_CONTENT)

    assert labels[0] == "gray150"
    assert "content" in labels
    assert sequence.select_content_filter.get_params(_page()) == stored


def test_output_render_records_the_generators_steps():
    labels = _labels_of_render(StageSequence(), FilterStage.OUTPUT)

    assert "binarized_and_cropped" in labels
    assert "despeckled" in labels


def test_dark_page_records_the_polarity_estimate_at_deskew():
    dark = 255 - _skewed_text_page()

    labels = _labels_of_render(StageSequence(), FilterStage.DESKEW, dark)

    assert labels[:2] == ["bw150", "content_mask"]
    assert labels[2:] == ["bw_rotated", "after_cleanup"]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("mode", "label"),
    [
        (DewarpingMode.AUTO, "distortion_model"),
        (DewarpingMode.MARGINAL, "marginal dewarping"),
    ],
)
def test_output_render_records_the_dewarping_models_images(
    mode: DewarpingMode, label: str
):
    sequence = StageSequence()
    params = sequence.output_filter.get_params(_page()).model_copy(
        update={"dewarping": DewarpingOptions(mode=mode, post_deskew=False)}
    )
    sequence.output_filter.set_params(_page(), params)

    with DebugImages() as dbg:
        process_page(
            curved_text_page(),
            _page(),
            sequence,
            PipelineOptions(dpi=float(PAGE_DPI), debug_images=dbg),
        )
        labels = [item.label for item in dbg.images]

    assert label in labels


def test_page_split_with_a_manual_split_records_its_images_and_keeps_the_split():
    sequence = StageSequence()
    height, width = _skewed_text_page().shape
    sequence.page_split_filter.set_manual_split(_page(), width / 2, width, height)
    stored = sequence.page_split_filter.get_params(_page())

    labels = _labels_of_render(sequence, FilterStage.PAGE_SPLIT)

    assert labels[0] == "gray100"
    assert sequence.page_split_filter.get_params(_page()) == stored


def test_page_split_detecting_a_single_uncut_page_runs_the_detector_once(monkeypatch):
    calls: list[object] = []
    real = page_split_filter.detect_split

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(page_split_filter, "detect_split", counting)
    sequence = StageSequence()
    sequence.page_split_filter.set_layout_type(_page(), LayoutType.SINGLE_PAGE_UNCUT)

    _labels_of_render(sequence, FilterStage.PAGE_SPLIT)

    assert len(calls) == 1


def test_select_content_with_manual_boxes_records_the_finders_and_keeps_the_boxes():
    sequence = StageSequence()
    sequence.select_content_filter.set_manual_page(
        _page(), PageBox(x=10, y=10, width=380, height=480)
    )
    sequence.select_content_filter.set_manual_content(
        _page(), ContentBox(x=40, y=60, width=320, height=380)
    )
    _labels_of_render(sequence, FilterStage.SELECT_CONTENT)
    stored = sequence.select_content_filter.get_params(_page())

    labels = _labels_of_render(sequence, FilterStage.SELECT_CONTENT)

    assert "gray150" in labels
    assert "content" in labels
    assert sequence.select_content_filter.get_params(_page()) == stored


def _normalised_gray(sequence: StageSequence) -> None:
    params = sequence.output_filter.get_params(_page())
    sequence.output_filter.set_params(
        _page(),
        params.model_copy(
            update={
                "color_mode": ColorMode.COLOR_GRAYSCALE,
                "color_common": ColorCommonOptions(normalize_illumination=True),
            }
        ),
    )


@pytest.mark.parametrize("configure", [None, _normalised_gray], ids=["bw", "gray"])
def test_output_pixels_are_the_same_with_and_without_debug_images(configure):
    shaded = _skewed_text_page().astype(np.float64)
    shaded *= np.linspace(0.75, 1.0, shaded.shape[1])[None, :]
    page = shaded.astype(np.uint8)

    def render(dbg: DebugImages | None) -> np.ndarray:
        sequence = StageSequence()
        if configure is not None:
            configure(sequence)
        return process_page(
            page, _page(), sequence, PipelineOptions(dpi=300.0, debug_images=dbg)
        ).image

    plain = render(None)
    with DebugImages() as dbg:
        debugged = render(dbg)
        labels = [item.label for item in dbg.images]

    assert "background" in labels
    np.testing.assert_array_equal(debugged, plain)


def test_select_content_with_a_manual_page_box_still_records_the_page_finder():
    sequence = StageSequence()
    sequence.select_content_filter.set_manual_page(
        _page(), PageBox(x=10, y=10, width=380, height=480)
    )

    labels = _labels_of_render(sequence, FilterStage.SELECT_CONTENT)

    assert "peakThreshold" in labels
    assert "content" in labels
