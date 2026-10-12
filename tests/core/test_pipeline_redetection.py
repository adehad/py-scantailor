"""Switching Page Split and Deskew to auto detects again, stale results too."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets import deskew as deskew_widget
from scantailor.app.ui.options_widgets import page_split as page_split_widget
from scantailor.core.models import ImageId, OrthogonalRotation, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.deskew.dependencies import Dependencies as DeskewDependencies
from scantailor.filters.deskew.params import AutoManualMode
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayoutType

# OpenCV rotates counter-clockwise for positive angles, so the correction is positive.
SKEW_DEGREES = 3.0
STALE_ANGLE = 4.0
MANUAL_ANGLE = 10.0
DETECTION_TOLERANCE_DEG = 1.0

SPREAD_WIDTH = 600
SPREAD_HEIGHT = 400
GUTTER_X = 300
MANUAL_SPLIT_X = 150


def _page(sub_page: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    return PageId(image_id=ImageId(file_path=Path("scan.tiff")), sub_page=sub_page)


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


def _spread() -> np.ndarray:
    """Two pages of different tone with a dark gutter between them."""
    image = np.full((SPREAD_HEIGHT, SPREAD_WIDTH), 220, dtype=np.uint8)
    image[:, GUTTER_X:] = 190
    image[:, GUTTER_X - 4 : GUTTER_X + 4] = 40
    return image


def _run_to(
    image: np.ndarray, page: PageId, sequence: StageSequence, stage: FilterStage
) -> None:
    process_page(image, page, sequence, PipelineOptions(end_stage=stage, dpi=300.0))


def test_deskew_auto_after_manual_redetects_the_angle(qtbot: QtBot) -> None:
    sequence = StageSequence()
    page = _page()
    image = _skewed_text_page()
    sequence.deskew_filter.set_manual_angle(page, MANUAL_ANGLE)
    widget = deskew_widget.OptionsWidget(sequence.deskew_filter)
    qtbot.addWidget(widget)
    widget.set_current_page(page)

    widget._auto_btn.click()
    _run_to(image, page, sequence, FilterStage.DESKEW)

    params = sequence.deskew_filter.get_params(page)
    assert params.mode is AutoManualMode.AUTO
    assert params.deskew_angle_deg != MANUAL_ANGLE
    assert params.deskew_angle_deg == pytest.approx(
        SKEW_DEGREES, abs=DETECTION_TOLERANCE_DEG
    )


def test_deskew_auto_click_forgets_the_stored_angle(qtbot: QtBot) -> None:
    sequence = StageSequence()
    page = _page()
    sequence.deskew_filter.set_manual_angle(page, MANUAL_ANGLE)
    widget = deskew_widget.OptionsWidget(sequence.deskew_filter)
    qtbot.addWidget(widget)
    widget.set_current_page(page)

    widget._auto_btn.click()

    assert not sequence.deskew_filter.settings.is_params_set(page)


def test_deskew_angle_detected_under_other_rotation_is_recomputed() -> None:
    sequence = StageSequence()
    page = _page()
    image = _skewed_text_page()
    height, width = image.shape
    outline = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=float)
    sequence.deskew_filter.set_params(
        page,
        DeskewParams(
            deskew_angle_deg=STALE_ANGLE,
            mode=AutoManualMode.AUTO,
            dependencies=DeskewDependencies.of(outline, rotation_degrees=90),
        ),
    )

    _run_to(image, page, sequence, FilterStage.DESKEW)

    angle = sequence.deskew_filter.get_params(page).deskew_angle_deg
    assert angle == pytest.approx(SKEW_DEGREES, abs=DETECTION_TOLERANCE_DEG)


def test_deskew_angle_detected_under_same_geometry_is_kept() -> None:
    sequence = StageSequence()
    page = _page()
    image = _skewed_text_page()
    _run_to(image, page, sequence, FilterStage.DESKEW)
    detected = sequence.deskew_filter.get_params(page)
    kept = detected.model_copy(update={"deskew_angle_deg": STALE_ANGLE})
    sequence.deskew_filter.set_params(page, kept)

    _run_to(image, page, sequence, FilterStage.DESKEW)

    assert sequence.deskew_filter.get_params(page).deskew_angle_deg == STALE_ANGLE


def test_deskew_angle_is_recomputed_when_the_orientation_changes() -> None:
    sequence = StageSequence()
    page = _page()
    image = _skewed_text_page()
    _run_to(image, page, sequence, FilterStage.DESKEW)
    sequence.deskew_filter.set_params(
        page,
        sequence.deskew_filter.get_params(page).model_copy(
            update={"deskew_angle_deg": STALE_ANGLE}
        ),
    )

    sequence.fix_orientation_filter.set_rotation(
        page.image_id, OrthogonalRotation(degrees=180)
    )
    _run_to(image, page, sequence, FilterStage.DESKEW)

    assert sequence.deskew_filter.get_params(page).deskew_angle_deg != STALE_ANGLE


def test_page_split_auto_after_manual_detects_the_split_line_again(
    qtbot: QtBot,
) -> None:
    sequence = StageSequence()
    page = _page(SubPage.LEFT_PAGE)
    image = _spread()
    split = sequence.page_split_filter
    split.set_params(
        page, split.get_params(page).with_layout_type(LayoutType.TWO_PAGES)
    )
    split.set_manual_split(page, MANUAL_SPLIT_X, SPREAD_WIDTH, SPREAD_HEIGHT)
    widget = page_split_widget.OptionsWidget(split)
    qtbot.addWidget(widget)
    widget.set_current_page(page)

    widget._auto_btn.click()
    _run_to(image, page, sequence, FilterStage.PAGE_SPLIT)

    params = split.get_params(page)
    assert params.is_auto()
    assert params.layout_type is LayoutType.TWO_PAGES
    assert params.page_layout is not None
    split_x = params.page_layout.get_split_line_x()
    assert split_x is not None
    assert split_x != MANUAL_SPLIT_X
    assert split_x == pytest.approx(GUTTER_X, abs=SPREAD_WIDTH * 0.1)


def test_page_split_layout_change_to_two_pages_detects_for_the_new_type(
    qtbot: QtBot,
) -> None:
    sequence = StageSequence()
    page = _page(SubPage.LEFT_PAGE)
    split = sequence.page_split_filter
    split.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)
    assert split.get_params(page).page_layout is not None
    assert (
        split.get_params(page).page_layout.layout_type  # type: ignore[union-attr]
        is PageLayoutType.SINGLE_PAGE_UNCUT
    )
    widget = page_split_widget.OptionsWidget(split)
    qtbot.addWidget(widget)
    widget.set_current_page(page)

    with qtbot.waitSignal(widget.page_count_requested) as counted:
        widget._two_pages_btn.click()
    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)

    assert counted.args == [page.image_id, 2]
    layout = split.get_params(page).page_layout
    assert layout is not None
    assert layout.layout_type is PageLayoutType.TWO_PAGES


def test_page_split_layout_back_to_single_page_drops_the_split_line() -> None:
    sequence = StageSequence()
    page = _page(SubPage.LEFT_PAGE)
    split = sequence.page_split_filter
    split.set_layout_type(page, LayoutType.TWO_PAGES)
    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)

    split.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)

    layout = split.get_params(page).page_layout
    assert layout is not None
    assert layout.layout_type is PageLayoutType.SINGLE_PAGE_UNCUT


def test_page_split_layout_detected_for_another_image_size_is_recomputed() -> None:
    sequence = StageSequence()
    page = _page(SubPage.LEFT_PAGE)
    split = sequence.page_split_filter
    split.set_layout_type(page, LayoutType.TWO_PAGES)
    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)
    first = split.get_params(page).page_layout

    _run_to(_spread()[:, : SPREAD_WIDTH // 2], page, sequence, FilterStage.PAGE_SPLIT)

    second = split.get_params(page).page_layout
    assert first is not None
    assert second is not None
    assert second.outline[2] == SPREAD_WIDTH // 2
    assert second != first


def test_manual_two_page_split_survives_when_nothing_upstream_changed() -> None:
    sequence = StageSequence()
    page = _page(SubPage.LEFT_PAGE)
    split = sequence.page_split_filter
    split.set_layout_type(page, LayoutType.TWO_PAGES)
    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)
    split.set_params(
        page,
        split.get_params(page).with_manual_layout(
            split.get_params(page).page_layout.with_split_line(MANUAL_SPLIT_X)  # type: ignore[union-attr]
        ),
    )

    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)

    layout = split.get_params(page).page_layout
    assert layout is not None
    assert layout.get_split_line_x() == MANUAL_SPLIT_X


def test_apply_auto_page_split_to_other_pages_clears_their_layout() -> None:
    sequence = StageSequence()
    source = _page(SubPage.LEFT_PAGE)
    target = PageId(image_id=ImageId(file_path=Path("other.tiff")))
    split = sequence.page_split_filter
    split.set_layout_type(source, LayoutType.TWO_PAGES)
    _run_to(_spread(), source, sequence, FilterStage.PAGE_SPLIT)

    split.apply_to_pages([target], split.get_params(source))

    copied = split.get_params(target)
    assert copied.layout_type is LayoutType.TWO_PAGES
    assert copied.page_layout is None


def test_manual_cutters_survive_after_an_earlier_auto_detection() -> None:
    sequence = StageSequence()
    page = _page()
    split = sequence.page_split_filter
    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)
    split.set_manual_cutters(page, 100.0, 500.0, SPREAD_WIDTH, SPREAD_HEIGHT)

    _run_to(_spread(), page, sequence, FilterStage.PAGE_SPLIT)

    params = split.get_params(page)
    assert params.is_manual()
    assert params.page_layout is not None
    assert params.page_layout.layout_type is PageLayoutType.SINGLE_PAGE_CUT
