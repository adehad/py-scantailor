"""Regression tests for the upstream reports on the oblique feature.

Upstream ScanTailor-Advanced issues 145, 156, 169 and 170 describe how the C++
implementation applies a correction nobody asked for, or ignores a request to
turn detection back on. Each test reproduces a report against the port.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from scantailor.core.default_params import (
    DefaultParams,
    DeskewDefaults,
    apply_default_params,
)
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.propagation import apply_deskew
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.deskew.dependencies import Dependencies
from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.params import AutoManualMode

from .synthetic_pages import sheared_page, text_page

AUTO = AutoManualMode.AUTO
MANUAL = AutoManualMode.MANUAL
SHEAR = 4.0
DEPENDENCIES = Dependencies(
    page_outline=((0.0, 0.0), (700.0, 0.0), (700.0, 900.0), (0.0, 900.0)),
    rotation_degrees=0,
)


def _page(name: str) -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _tilted_page(degrees: float) -> np.ndarray:
    """Text lines tilted counter-clockwise on screen by ``degrees``."""
    page = text_page()
    height, width = page.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return np.asarray(
        cv2.warpAffine(page, matrix, (width, height), borderValue=255), dtype=np.uint8
    )


def _geometry(filter_: Filter, page: PageId) -> ImageTransformation:
    xform = ImageTransformation.from_size(700, 900)
    filter_.apply_geometry(xform, filter_.get_params(page))
    return xform


def _process(filter_: Filter, page: PageId, image: np.ndarray) -> None:
    """Run what the pipeline runs for the Deskew stage of one page."""
    if filter_.needs_detection(page, DEPENDENCIES):
        filter_.detect_skew(image, page, dependencies=DEPENDENCIES)


class TestUpstream145:
    """New project, Deskew left at its defaults: oblique is applied although
    "Automatic oblique correction" is off by default."""

    def test_a_new_page_is_not_sheared_when_automatic_oblique_is_off(self):
        filter_ = Filter()
        page = _page("new")

        _process(filter_, page, sheared_page(SHEAR))

        params = filter_.get_params(page)
        assert params.oblique_mode is MANUAL
        assert params.oblique_angle_deg == 0.0
        assert _geometry(filter_, page).post_oblique == 0.0

    def test_a_new_project_from_the_default_profile_is_not_sheared(self):
        stages = StageSequence()
        page = _page("new")
        apply_default_params(stages, [page.image_id], DefaultParams())

        _process(stages.deskew_filter, page, sheared_page(SHEAR))

        params = stages.deskew_filter.get_params(page)
        assert params.oblique_mode is MANUAL
        assert params.oblique_angle_deg == 0.0

    def test_the_default_decides_for_pages_processed_afterwards(self):
        filter_ = Filter()
        before, after = _page("before"), _page("after")
        _process(filter_, before, sheared_page(SHEAR))

        filter_.auto_oblique_default = True
        _process(filter_, after, sheared_page(SHEAR))

        assert filter_.get_params(before).oblique_angle_deg == 0.0
        assert filter_.get_params(after).oblique_mode is AUTO
        assert filter_.get_params(after).oblique_angle_deg == pytest.approx(
            SHEAR, abs=1.0
        )


class TestUpstream156:
    """Defaults have the oblique correction off, yet projects force it on and
    show "Oblique mode" Auto until it is set to 0 and applied to all pages."""

    def test_every_page_stays_unsheared_with_the_default_off(self):
        stages = StageSequence()
        pages = [_page(f"p{number}") for number in range(3)]
        defaults = DefaultParams(deskew=DeskewDefaults(mode=MANUAL, auto_oblique=False))
        apply_default_params(stages, [page.image_id for page in pages], defaults)

        for page in pages:
            _process(stages.deskew_filter, page, sheared_page(SHEAR))

        for page in pages:
            params = stages.deskew_filter.get_params(page)
            assert params.oblique_angle_deg == 0.0
            assert params.oblique_mode is MANUAL

    def test_the_oblique_mode_agrees_with_the_default(self):
        for enabled, mode in ((False, MANUAL), (True, AUTO)):
            stages = StageSequence()
            page = _page("page")
            defaults = DefaultParams(
                deskew=DeskewDefaults(mode=MANUAL, auto_oblique=enabled)
            )
            apply_default_params(stages, [page.image_id], defaults)

            assert stages.deskew_filter.get_params(page).oblique_mode is mode
            assert stages.deskew_filter.auto_oblique_default is enabled

    def test_a_manual_deskew_default_with_oblique_on_is_detected_not_left_at_zero(self):
        stages = StageSequence()
        page = _page("page")
        defaults = DefaultParams(deskew=DeskewDefaults(mode=MANUAL, auto_oblique=True))
        apply_default_params(stages, [page.image_id], defaults)

        _process(stages.deskew_filter, page, sheared_page(SHEAR))

        params = stages.deskew_filter.get_params(page)
        assert params.oblique_angle_deg == pytest.approx(SHEAR, abs=1.0)
        assert params.mode is MANUAL


class TestUpstream169:
    """Once Auto is disabled in Deskew it stays disabled: pressing Auto has no
    effect and the mode switches back to Manual."""

    def test_pressing_auto_after_manual_detects_the_rotation_again(self):
        filter_ = Filter()
        page = _page("page")
        image = _tilted_page(2.0)
        filter_.set_manual_angle(page, 0.0)

        filter_.redetect_rotation(page)
        assert filter_.get_params(page).mode is AUTO
        assert filter_.needs_detection(page, DEPENDENCIES)
        _process(filter_, page, image)

        params = filter_.get_params(page)
        assert params.mode is AUTO
        assert params.deskew_angle_deg != 0.0
        assert not filter_.needs_detection(page, DEPENDENCIES)

    def test_pressing_oblique_auto_after_manual_detects_the_oblique_again(self):
        filter_ = Filter()
        page = _page("page")
        filter_.set_manual_oblique(page, 0.0)
        filter_.set_manual_angle(page, 0.0)

        filter_.redetect_oblique(page)
        _process(filter_, page, sheared_page(SHEAR))

        params = filter_.get_params(page)
        assert params.oblique_mode is AUTO
        assert params.oblique_angle_deg == pytest.approx(SHEAR, abs=1.0)
        assert params.mode is MANUAL

    def test_a_manual_rotation_does_not_stop_an_automatic_oblique(self):
        filter_ = Filter()
        filter_.auto_oblique_default = True
        page = _page("page")
        filter_.set_manual_angle(page, 0.0)
        filter_.redetect_oblique(page)

        _process(filter_, page, sheared_page(SHEAR))

        assert filter_.get_params(page).oblique_angle_deg != 0.0


class TestUpstream170:
    """Auto deskew reactivates the oblique correction that was switched off by
    unchecking automatic oblique."""

    def test_auto_deskew_does_not_turn_the_oblique_back_on(self):
        filter_ = Filter()
        page = _page("page")
        _process(filter_, page, sheared_page(SHEAR))
        assert filter_.get_params(page).oblique_mode is MANUAL

        filter_.redetect_rotation(page)
        _process(filter_, page, sheared_page(SHEAR))

        params = filter_.get_params(page)
        assert params.oblique_mode is MANUAL
        assert params.oblique_angle_deg == 0.0
        assert _geometry(filter_, page).post_oblique == 0.0

    def test_auto_deskew_keeps_a_manual_oblique_angle(self):
        filter_ = Filter()
        page = _page("page")
        filter_.set_manual_oblique(page, 2.0)

        filter_.redetect_rotation(page)
        _process(filter_, page, sheared_page(SHEAR))

        params = filter_.get_params(page)
        assert params.oblique_angle_deg == 2.0
        assert params.oblique_mode is MANUAL

    def test_auto_deskew_does_not_follow_a_default_switched_on_later(self):
        filter_ = Filter()
        page = _page("page")
        _process(filter_, page, sheared_page(SHEAR))

        filter_.auto_oblique_default = True
        filter_.redetect_rotation(page)
        _process(filter_, page, sheared_page(SHEAR))

        assert filter_.get_params(page).oblique_mode is MANUAL

    def test_applying_only_the_rotation_leaves_the_oblique_of_other_pages(self):
        stages = StageSequence()
        source, target = _page("source"), _page("target")
        stages.deskew_filter.set_manual_angle(source, 1.0)
        stages.deskew_filter.set_manual_oblique(source, 3.0)
        stages.deskew_filter.set_manual_oblique(target, 0.0)

        apply_deskew(stages, source, [target], apply_oblique=False)

        params = stages.deskew_filter.get_params(target)
        assert params.deskew_angle_deg == 1.0
        assert params.oblique_angle_deg == 0.0


class TestPipelineDetection:
    """The pipeline's Deskew stage runs the same detection."""

    @staticmethod
    def _run(stages: StageSequence, page: PageId) -> None:
        process_page(
            sheared_page(SHEAR),
            page,
            stages,
            PipelineOptions(end_stage=FilterStage.DESKEW, dpi=300.0),
        )

    def test_the_default_on_detects_the_oblique_of_a_page(self):
        stages = StageSequence()
        stages.deskew_filter.auto_oblique_default = True
        page = _page("page")

        self._run(stages, page)

        params = stages.deskew_filter.get_params(page)
        assert params.oblique_mode is AUTO
        assert params.oblique_angle_deg == pytest.approx(SHEAR, abs=1.0)

    def test_the_default_off_leaves_the_page_unsheared(self):
        stages = StageSequence()
        page = _page("page")

        self._run(stages, page)

        params = stages.deskew_filter.get_params(page)
        assert params.oblique_mode is MANUAL
        assert params.oblique_angle_deg == 0.0
