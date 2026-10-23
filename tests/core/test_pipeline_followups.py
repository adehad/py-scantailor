"""Post-dewarp skew, despeckle strength, Select Content dependencies, aggregate flag."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.params import DistortionModelSpec
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox
from tests.dewarping.synthetic import (
    BACKGROUND,
    PAGE_DPI,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    justified_text_page,
    row_profile_variance,
    straight_text_page,
)

_CONTENT = ContentBox(x=100, y=100, width=PAGE_WIDTH - 200, height=PAGE_HEIGHT - 200)
_FLAT_EDGES = DistortionModelSpec(
    top=((100.0, 200.0), (600.0, 200.0), (1100.0, 200.0)),
    bottom=((100.0, 1400.0), (600.0, 1400.0), (1100.0, 1400.0)),
)
_COLUMNS = (150, 1050)


def _page(name: str = "page") -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _rotated_text_page(degrees: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((PAGE_WIDTH / 2, PAGE_HEIGHT / 2), degrees, 1.0)
    return np.asarray(
        cv2.warpAffine(
            straight_text_page(),
            matrix,
            (PAGE_WIDTH, PAGE_HEIGHT),
            borderValue=BACKGROUND,
        ),
        dtype=np.uint8,
    )


def _dewarp_sequence(
    page: PageId, mode: DewarpingMode, *, post_deskew: bool = True
) -> StageSequence:
    sequence = StageSequence()
    sequence.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    sequence.deskew_filter.set_manual_angle(page, 0.0)
    sequence.select_content_filter.set_manual_content(page, _CONTENT)
    params = sequence.output_filter.get_params(page).model_copy(
        update={
            "color_mode": ColorMode.COLOR_GRAYSCALE,
            "output_dpi": Dpi.uniform(PAGE_DPI),
            "dewarping": DewarpingOptions(mode=mode, post_deskew=post_deskew),
            "distortion_model": _FLAT_EDGES,
        }
    )
    sequence.output_filter.set_params(page, params)
    return sequence


def _run(image: np.ndarray, page: PageId, sequence: StageSequence, **options):
    return process_page(
        image, page, sequence, PipelineOptions(dpi=float(PAGE_DPI), **options)
    )


class TestPostDewarpSkew:
    @pytest.mark.parametrize("degrees", [2.0, -3.0])
    def test_manual_dewarping_levels_the_text_lines_left_by_the_page_skew(
        self, degrees: float
    ):
        page = _page()
        warped = _rotated_text_page(degrees)
        measured = _run(warped, page, _dewarp_sequence(page, DewarpingMode.MANUAL))
        unmeasured = _run(
            warped,
            page,
            _dewarp_sequence(page, DewarpingMode.MANUAL, post_deskew=False),
        )

        # Rows of text have the most contrast between them once they are level.
        assert row_profile_variance(
            measured.image, _COLUMNS
        ) > 2 * row_profile_variance(unmeasured.image, _COLUMNS)

    @pytest.mark.parametrize("degrees", [2.0, -3.0])
    def test_the_correction_is_the_skew_with_the_sign_flipped(self, degrees: float):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.MANUAL)

        result = _run(_rotated_text_page(degrees), page, sequence)

        # cv2 turns counter-clockwise for positive angles; the output turns the
        # page clockwise for positive ones.
        assert result.metadata["OUTPUT"]["post_deskew_angle"] == pytest.approx(
            degrees, abs=0.3
        )

    def test_the_angle_is_kept_in_the_page_parameters(self):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.MANUAL)

        result = _run(_rotated_text_page(2.0), page, sequence)

        stored = sequence.output_filter.get_params(page).dewarping.post_deskew_angle
        assert stored == result.metadata["OUTPUT"]["post_deskew_angle"] != 0.0

    def test_a_stale_stored_angle_is_replaced_by_the_measured_one(self):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.MANUAL)
        stale = sequence.output_filter.get_params(page)
        sequence.output_filter.set_params(
            page,
            stale.with_dewarping(
                stale.dewarping.model_copy(update={"post_deskew_angle": 9.0})
            ),
        )

        _run(straight_text_page(), page, sequence)

        stored = sequence.output_filter.get_params(page).dewarping.post_deskew_angle
        assert abs(stored) < 0.5

    def test_a_straight_page_is_not_rotated(self):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.MANUAL)

        result = _run(straight_text_page(), page, sequence)

        assert abs(result.metadata["OUTPUT"]["post_deskew_angle"]) < 0.5

    def test_the_option_switched_off_measures_nothing(self):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.MANUAL, post_deskew=False)

        result = _run(_rotated_text_page(2.0), page, sequence)

        assert result.metadata["OUTPUT"]["post_deskew_angle"] == 0.0

    def test_automatic_dewarping_does_not_post_deskew_like_c_plus_plus(self):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.AUTO)

        result = _run(_rotated_text_page(2.0), page, sequence)

        assert result.metadata["OUTPUT"]["post_deskew_angle"] == 0.0
        assert (
            sequence.output_filter.get_params(page).dewarping.post_deskew_angle == 0.0
        )

    def test_without_dewarping_the_stored_angle_is_left_alone(self):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.OFF)
        before = sequence.output_filter.get_params(page)

        _run(_rotated_text_page(2.0), page, sequence)

        assert sequence.output_filter.get_params(page) == before


class TestAutoDewarpAcceptance:
    """Upstream needs two traced lines for a model and declines with fewer."""

    @staticmethod
    def _first_lines(count: int) -> np.ndarray:
        source = justified_text_page()
        page = np.full_like(source, BACKGROUND)
        for index in range(count):
            baseline = 170 + 48 * index
            page[baseline - 30 : baseline + 12] = source[baseline - 30 : baseline + 12]
        return page

    def test_a_page_with_a_single_text_line_is_left_flat(self):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.AUTO)

        result = _run(self._first_lines(1), page, sequence)

        assert result.metadata["OUTPUT"]["was_dewarped"] is False
        assert result.distortion_model is None

    def test_three_text_lines_give_the_two_traced_lines_a_model_needs(self):
        """Three neighbouring lines span little, so the minimum span must allow them."""
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.AUTO)
        output = sequence.output_filter
        output.set_params(page, output.get_params(page).with_dewarp_min_span_percent(0))

        result = _run(self._first_lines(3), page, sequence)

        assert result.distortion_model is not None

    def test_three_text_lines_are_refused_by_the_default_minimum_span(self):
        page = _page()
        sequence = _dewarp_sequence(page, DewarpingMode.AUTO)

        result = _run(self._first_lines(3), page, sequence)

        assert result.distortion_model is None
        assert result.dewarp_metrics is not None
        assert result.dewarp_metrics.span_percent is not None
        assert result.dewarp_metrics.span_percent < 40.0


class TestDespeckleStrength:
    @staticmethod
    def _levels_seen_by_the_despeckler(monkeypatch, params_update: dict) -> list:
        seen: list[float] = []

        def spy(binary, dpi, level, dbg=None):
            seen.append(level)
            return binary.copy()

        monkeypatch.setattr("scantailor.filters.output.generator.despeckle_image", spy)
        page = _page()
        sequence = StageSequence()
        sequence.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
        sequence.deskew_filter.set_manual_angle(page, 0.0)
        sequence.select_content_filter.set_manual_content(page, _CONTENT)
        params = sequence.output_filter.get_params(page).model_copy(
            update={
                "color_mode": ColorMode.BLACK_AND_WHITE,
                "output_dpi": Dpi.uniform(PAGE_DPI),
                **params_update,
            }
        )
        sequence.output_filter.set_params(page, params)
        _run(straight_text_page(), page, sequence)
        return seen

    def test_an_in_between_strength_reaches_the_despeckler(self, monkeypatch):
        from scantailor.filters.output.params import Params

        update = Params().with_despeckle_factor(1.7).model_dump()
        keep = {k: update[k] for k in ("despeckle_level", "despeckle_factor")}

        seen = self._levels_seen_by_the_despeckler(monkeypatch, keep)

        assert seen == [1.7]

    def test_a_named_level_still_gives_its_preset(self, monkeypatch):
        from scantailor.filters.output.despeckle import DespeckleLevel

        seen = self._levels_seen_by_the_despeckler(
            monkeypatch, {"despeckle_level": DespeckleLevel.NORMAL}
        )

        assert seen == [2.0]

    def test_a_zero_factor_skips_despeckling(self, monkeypatch):
        from scantailor.filters.output.despeckle import DespeckleLevel

        seen = self._levels_seen_by_the_despeckler(
            monkeypatch,
            {"despeckle_level": DespeckleLevel.OFF, "despeckle_factor": 0.0},
        )

        assert seen == []

    def test_the_metadata_reports_the_strength(self, monkeypatch):
        monkeypatch.setattr(
            "scantailor.filters.output.generator.despeckle_image",
            lambda binary, dpi, level, dbg=None: binary.copy(),
        )
        page = _page()
        sequence = StageSequence()
        params = sequence.output_filter.get_params(page).with_despeckle_factor(2.3)
        sequence.output_filter.set_params(page, params)

        result = _run(
            straight_text_page(),
            page,
            sequence,
            start_stage=FilterStage.OUTPUT,
            end_stage=FilterStage.OUTPUT,
            detect_only=True,
        )

        assert result.metadata["OUTPUT"]["despeckle_strength"] == 2.3


class TestSelectContentDependencies:
    @staticmethod
    def _count_content_detections(monkeypatch) -> list[int]:
        from scantailor.filters.select_content import filter as content_filter

        calls = [0]
        real = content_filter.find_content_box

        def counting(*args, **kwargs):
            calls[0] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(content_filter, "find_content_box", counting)
        return calls

    @staticmethod
    def _sequence(page: PageId) -> StageSequence:
        sequence = StageSequence()
        sequence.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
        sequence.deskew_filter.set_manual_angle(page, 0.0)
        return sequence

    def test_content_is_detected_once_while_its_inputs_stay_the_same(self, monkeypatch):
        calls = self._count_content_detections(monkeypatch)
        page = _page()
        sequence = self._sequence(page)

        for _ in range(3):
            _run(
                straight_text_page(),
                page,
                sequence,
                end_stage=FilterStage.SELECT_CONTENT,
            )

        assert calls[0] == 1

    def test_a_new_deskew_angle_moves_the_outline_and_detects_again(self, monkeypatch):
        calls = self._count_content_detections(monkeypatch)
        page = _page()
        sequence = self._sequence(page)
        options = {"end_stage": FilterStage.SELECT_CONTENT}
        _run(straight_text_page(), page, sequence, **options)

        sequence.deskew_filter.set_manual_angle(page, 3.0)
        _run(straight_text_page(), page, sequence, **options)

        assert calls[0] == 2

    def test_switching_the_content_back_to_automatic_detects_again(self, monkeypatch):
        calls = self._count_content_detections(monkeypatch)
        page = _page()
        sequence = self._sequence(page)
        options = {"end_stage": FilterStage.SELECT_CONTENT}
        _run(straight_text_page(), page, sequence, **options)
        sequence.select_content_filter.set_manual_content(page, _CONTENT)
        _run(straight_text_page(), page, sequence, **options)
        assert calls[0] == 1

        sequence.select_content_filter.reset_content_to_auto(page)
        _run(straight_text_page(), page, sequence, **options)

        assert calls[0] == 2

    def test_stages_before_the_first_detecting_one_never_detect_content(
        self, monkeypatch
    ):
        calls = self._count_content_detections(monkeypatch)
        page = _page()

        _run(
            straight_text_page(),
            page,
            self._sequence(page),
            start_stage=FilterStage.PAGE_LAYOUT,
            end_stage=FilterStage.PAGE_LAYOUT,
        )

        assert calls[0] == 0


class TestAggregateSizeFlag:
    @staticmethod
    def _layout_metadata(
        sequence: StageSequence, page: PageId, content: ContentBox
    ) -> dict:
        sequence.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
        sequence.deskew_filter.set_manual_angle(page, 0.0)
        sequence.select_content_filter.set_manual_content(page, content)
        result = _run(
            straight_text_page(), page, sequence, end_stage=FilterStage.PAGE_LAYOUT
        )
        return result.metadata["PAGE_LAYOUT"]

    def test_the_flag_tells_whether_a_page_changed_the_size_others_align_to(self):
        sequence = StageSequence()
        small = ContentBox(x=100, y=100, width=400, height=400)
        large = ContentBox(x=100, y=100, width=800, height=900)

        first = self._layout_metadata(sequence, _page("a"), small)
        smaller = self._layout_metadata(sequence, _page("b"), small)
        larger = self._layout_metadata(sequence, _page("c"), large)

        assert first["aggregate_size_changed"] is True
        assert smaller["aggregate_size_changed"] is False
        assert larger["aggregate_size_changed"] is True


class TestDeskewShadowCleanupWiring:
    def test_pipeline_gives_deskew_detection_the_page_resolution(self, monkeypatch):
        from scantailor.filters.deskew.filter import Filter as DeskewFilter

        seen: list[Dpi | None] = []
        original = DeskewFilter.detect_skew

        def spy(self, *args, **kwargs):
            seen.append(kwargs.get("dpi"))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(DeskewFilter, "detect_skew", spy)
        page = _page("shadow_wiring")
        sequence = StageSequence()
        sequence.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
        process_page(
            straight_text_page(),
            page,
            sequence,
            PipelineOptions(dpi=float(PAGE_DPI), end_stage=FilterStage.DESKEW),
        )
        assert seen == [Dpi.uniform(PAGE_DPI)]
