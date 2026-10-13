"""The minimum span of automatic dewarping is a parameter, and its metrics come back.

The parameter lives in the output parameters of every page and in Default
Parameters, round-trips through JSON, and travels with the other dewarping
options under Apply To. The pipeline returns what the detection found with the
result and with the pictures for the Output view.
"""

import json
from pathlib import Path

import numpy as np

from scantailor.core.default_params import DefaultParams, OutputDefaults
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import StageSequence
from scantailor.dewarping import DEFAULT_MIN_SPAN_PERCENT
from scantailor.dewarping.auto_model import ModelOutcome
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.params import ApplyGroup, Params
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox
from tests.dewarping.book_pages import (
    DEFAULT_PAPER,
    BentPage,
    bend_page,
    justified_text_page,
    sagging_edges,
)

DPI = 200
CONTENT = ContentBox(x=160, y=200, width=890, height=1090)


def _page_of_two_close_lines() -> BentPage:
    """A bent page with only its first three lines of text, on no dark ground.

    The tracer keeps two neighbouring lines of the three; nothing else offers a
    curve, so the model can only come from that close pair.
    """
    flat, baselines = justified_text_page(seed=5)
    for baseline in baselines[3:]:
        flat[baseline - 32 : baseline + 12, :] = DEFAULT_PAPER
    top, bottom = sagging_edges(100.0, 1100.0, 140.0, 1450.0, 45.0, -25.0)
    return bend_page(
        flat, baselines, 70, top, bottom, (1200, 1600), background=DEFAULT_PAPER
    )


def _page_id() -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path("span.tiff")), sub_page=SubPage.SINGLE_PAGE
    )


def _sequence(page_id: PageId, mode: DewarpingMode, min_span: int) -> StageSequence:
    sequence = StageSequence()
    sequence.page_split_filter.set_layout_type(page_id, LayoutType.SINGLE_PAGE_UNCUT)
    sequence.deskew_filter.set_manual_angle(page_id, 0.0)
    sequence.select_content_filter.set_manual_content(page_id, CONTENT)
    params = sequence.output_filter.get_params(page_id).model_copy(
        update={
            "color_mode": ColorMode.COLOR_GRAYSCALE,
            "output_dpi": Dpi.uniform(DPI),
            "dewarping": DewarpingOptions(mode=mode, post_deskew=False),
            "dewarp_min_span_percent": min_span,
        }
    )
    sequence.output_filter.set_params(page_id, params)
    return sequence


class TestParameter:
    def test_the_default_is_the_measured_one(self) -> None:
        assert Params().dewarp_min_span_percent == DEFAULT_MIN_SPAN_PERCENT
        assert OutputDefaults().dewarp_min_span_percent == DEFAULT_MIN_SPAN_PERCENT

    def test_it_round_trips_through_json_per_page(self) -> None:
        params = Params().with_dewarp_min_span_percent(25)

        restored = Params.model_validate(json.loads(params.model_dump_json()))

        assert restored.dewarp_min_span_percent == 25

    def test_it_round_trips_through_json_in_default_parameters(self) -> None:
        defaults = DefaultParams(output=OutputDefaults(dewarp_min_span_percent=55))

        restored = DefaultParams.model_validate(json.loads(defaults.model_dump_json()))

        assert restored.output.dewarp_min_span_percent == 55
        assert restored.output.to_params().dewarp_min_span_percent == 55

    def test_stored_parameters_without_it_take_the_default(self) -> None:
        restored = Params.model_validate({"color_mode": "bw"})

        assert restored.dewarp_min_span_percent == DEFAULT_MIN_SPAN_PERCENT

    def test_apply_to_copies_it_with_the_dewarping_options(self) -> None:
        source = Params(dewarping=DewarpingOptions(mode=DewarpingMode.AUTO))
        source = source.with_dewarp_min_span_percent(15)

        merged = Params().with_group_from(source, ApplyGroup.DEWARPING)
        untouched = Params().with_group_from(source, ApplyGroup.DEPTH_PERCEPTION)

        assert merged.dewarp_min_span_percent == 15
        assert merged.dewarping.mode is DewarpingMode.AUTO
        assert untouched.dewarp_min_span_percent == DEFAULT_MIN_SPAN_PERCENT


class TestPipeline:
    def test_a_refused_model_leaves_the_page_undewarped_and_says_why(self) -> None:
        page = _page_of_two_close_lines()
        page_id = _page_id()

        result = process_page(
            page.image,
            page_id,
            _sequence(page_id, DewarpingMode.AUTO, DEFAULT_MIN_SPAN_PERCENT),
            PipelineOptions(dpi=DPI, keep_output_extras=True),
        )

        metrics = result.dewarp_metrics
        assert metrics is not None
        assert metrics.outcome is ModelOutcome.SPAN_TOO_SMALL
        assert metrics.min_span_percent == DEFAULT_MIN_SPAN_PERCENT
        assert result.distortion_model is None
        assert result.metadata["OUTPUT"]["was_dewarped"] is False
        assert result.output_extras is not None
        assert result.output_extras.dewarp_metrics is metrics

    def test_a_minimum_of_zero_dewarps_the_same_page(self) -> None:
        page = _page_of_two_close_lines()
        page_id = _page_id()

        result = process_page(
            page.image,
            page_id,
            _sequence(page_id, DewarpingMode.AUTO, 0),
            PipelineOptions(dpi=DPI),
        )

        assert result.dewarp_metrics is not None
        assert result.dewarp_metrics.outcome is ModelOutcome.BUILT
        assert result.distortion_model is not None
        assert result.metadata["OUTPUT"]["was_dewarped"] is True

    def test_pages_not_dewarped_automatically_carry_no_metrics(self) -> None:
        page = _page_of_two_close_lines()
        page_id = _page_id()

        result = process_page(
            page.image,
            page_id,
            _sequence(page_id, DewarpingMode.OFF, 0),
            PipelineOptions(dpi=DPI, keep_output_extras=True),
        )

        assert result.dewarp_metrics is None
        assert result.output_extras is not None
        assert result.output_extras.dewarp_metrics is None

    def test_the_metrics_count_what_the_page_offered(self) -> None:
        page = _page_of_two_close_lines()
        page_id = _page_id()

        result = process_page(
            page.image,
            page_id,
            _sequence(page_id, DewarpingMode.AUTO, 0),
            PipelineOptions(dpi=DPI),
        )

        metrics = result.dewarp_metrics
        assert metrics is not None
        assert metrics.lines_traced >= metrics.lines_kept >= 2
        assert metrics.edges_traced == 0
        assert metrics.span_percent is not None
        assert 0.0 < metrics.span_percent < 40.0
        assert np.isfinite(metrics.span_percent)
