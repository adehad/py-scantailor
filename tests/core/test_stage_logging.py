"""The log says what each stage decided for a page, and which fallback ran."""

from pathlib import Path
from typing import Any

import numpy as np

from scantailor.core.models import ImageId, PageId
from scantailor.core.pipeline import PipelineOptions, process_batch, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.stage_summary import summarize_stage
from scantailor.dewarping.auto_model import DewarpMetrics, ModelOutcome
from scantailor.filters.deskew.filter import Filter as DeskewFilter
from scantailor.filters.select_content.page_finder import find_page_box


def _page_pixels() -> np.ndarray:
    pixels = np.full((1200, 900), 225, dtype=np.uint8)
    for top in range(200, 1000, 30):
        pixels[top : top + 10, 150:750] = 30
    return pixels


def _page(name: str = "scan01.tif") -> PageId:
    return PageId(image_id=ImageId(file_path=Path("books") / name))


def _stage_lines(records: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["level"] == level and record["message"].endswith(" s")
    ]


def test_a_full_run_logs_one_info_line_per_stage_with_page_stage_and_time(
    log_records: list[dict[str, Any]],
) -> None:
    process_batch([(_page_pixels(), _page())], StageSequence(), PipelineOptions())

    info = _stage_lines(log_records, "INFO")
    assert [record["extra"]["stage"] for record in info] == [
        "fix_orientation",
        "page_split",
        "deskew",
        "select_content",
        "page_layout",
        "output",
    ]
    assert all(record["extra"]["page"] == "scan01.tif" for record in info)
    deskew = info[2]["message"]
    assert deskew.startswith("auto: correction ")
    assert "(content)" in deskew
    assert info[1]["message"].startswith("auto: single-uncut")


def test_a_render_after_the_geometry_pass_does_not_repeat_earlier_stages(
    log_records: list[dict[str, Any]],
) -> None:
    process_batch([(_page_pixels(), _page())], StageSequence(), PipelineOptions())

    summaries = [
        record["extra"]["stage"]
        for record in log_records
        if record["message"].endswith(" s")
    ]
    assert len(summaries) == len(set(summaries)) == len(FilterStage)
    assert _stage_lines(log_records, "DEBUG") == []


def test_a_view_of_one_stage_logs_only_that_stage_at_info(
    log_records: list[dict[str, Any]],
) -> None:
    process_page(
        _page_pixels(),
        _page(),
        StageSequence(),
        PipelineOptions(end_stage=FilterStage.DESKEW),
    )

    info = _stage_lines(log_records, "INFO")
    assert [record["extra"]["stage"] for record in info] == ["deskew"]


def test_a_manual_angle_is_reported_as_manual_stored_and_counter_clockwise() -> None:
    line = summarize_stage(
        FilterStage.DESKEW,
        {"skew_angle": 1.2, "oblique_angle": 0.0, "is_manual": True},
        None,
    )

    assert line == "manual: correction -1.20 deg, oblique 0.00 deg (stored)"


def test_a_weak_detection_says_it_was_not_confident() -> None:
    line = summarize_stage(
        FilterStage.DESKEW,
        {
            "skew_angle": 0.0,
            "oblique_angle": 0.0,
            "is_manual": False,
            "detected_by": "content",
            "confident": False,
        },
        None,
    )

    assert line.endswith("(content, not confident)")


def test_a_two_page_split_names_the_cutter_ends_and_the_image_width() -> None:
    line = summarize_stage(
        FilterStage.PAGE_SPLIT,
        {
            "layout_type": "auto-detect",
            "page_layout": "two-pages",
            "is_manual": False,
            "cutters_x": [(1612.4, 1598.0)],
            "image_width": 3270,
        },
        None,
    )

    assert line == "auto: two-pages, cutter x=1612 (top) to 1598 (bottom) of 3270"


def test_automatic_dewarping_reports_its_outcome_and_kept_lines() -> None:
    metrics = DewarpMetrics(
        lines_traced=22,
        lines_kept=14,
        edges_traced=0,
        outcome=ModelOutcome.BUILT,
        min_span_percent=25.0,
    )

    line = summarize_stage(
        FilterStage.OUTPUT,
        {
            "color_mode": "bw",
            "binarization_method": "sauvola",
            "dewarping_mode": "auto",
            "output_dpi": (600, 600),
            "was_dewarped": True,
        },
        metrics,
    )

    assert line == "bw sauvola, 600x600 dpi; dewarp auto: built, 14 of 22 lines kept"


def test_deskew_logs_the_fallback_from_an_unconfident_top_edge(
    log_messages: list[str],
) -> None:
    filter_ = DeskewFilter()
    filter_.content_based = False

    result = filter_.detect_skew(_page_pixels(), _page(), store_result=False)

    assert result.method == "content"
    assert any(message.startswith("top edge not confident") for message in log_messages)


def test_the_page_finder_logs_the_line_edge_fallback_on_a_light_background(
    log_messages: list[str],
) -> None:
    find_page_box(_page_pixels(), dpi=300.0)

    assert "dark-bed finder found the whole image; line-edge fallback" in log_messages
