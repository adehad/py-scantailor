"""One line per page and stage saying what the stage settled on, for the log.

Each line is built from the metadata dict the pipeline already returns for the
stage, so the log shows the same values a caller of ``process_page`` sees. A
line says whether the values were detected or set by hand and, where detection
has more than one way to decide, which one did.
"""

from collections.abc import Callable, Mapping
from typing import Any

from scantailor.core.stage_sequence import FilterStage
from scantailor.dewarping.auto_model import DewarpMetrics
from scantailor.filters.deskew.params import shown_deskew_angle

Metadata = Mapping[str, Any]


def summarize_stage(
    stage: FilterStage, metadata: Metadata, dewarp_metrics: DewarpMetrics | None
) -> str:
    """Describe in one line what a stage decided for a page.

    Args:
        stage (FilterStage): The stage that just ran.
        metadata (Mapping[str, Any]): The stage's entry in
            ``PipelineResult.metadata``; keys a stage leaves out, such as the
            Output values of a detect-only run, are skipped.
        dewarp_metrics (DewarpMetrics | None): What automatic dewarping found;
            only read for the Output stage.

    Returns:
        str: Plain words without the page or stage name, which the log
            format adds.
    """
    if stage == FilterStage.OUTPUT:
        return _output(metadata, dewarp_metrics)
    return _SUMMARIES[stage](metadata)


def _mode(metadata: Metadata) -> str:
    return "manual" if metadata.get("is_manual") else "auto"


def _box(box: Mapping[str, float]) -> str:
    return (
        f"{round(box['x'])},{round(box['y'])} "
        f"{round(box['width'])}x{round(box['height'])}"
    )


def _fix_orientation(metadata: Metadata) -> str:
    text = f"rotation {metadata.get('rotation_degrees', 0)} deg"
    if metadata.get("trim_area") is not None:
        text += ", trimmed"
    return text


def _page_split(metadata: Metadata) -> str:
    layout = metadata.get("page_layout", metadata.get("layout_type"))
    text = f"{_mode(metadata)}: {layout}"
    cutters = metadata.get("cutters_x") or []
    for top, bottom in cutters:
        text += f", cutter x={top:.0f} (top) to {bottom:.0f} (bottom)"
    if cutters and "image_width" in metadata:
        text += f" of {metadata['image_width']}"
    return text


def _deskew(metadata: Metadata) -> str:
    detected_by = metadata.get("detected_by")
    if detected_by is None:
        source = "stored"
    elif metadata.get("confident", True):
        source = detected_by
    else:
        source = f"{detected_by}, not confident"
    # The sign the Deskew panel shows, so a log line matches what the user sees.
    rotation = shown_deskew_angle(metadata.get("skew_angle", 0.0))
    return (
        f"{_mode(metadata)}: correction {rotation:.2f} deg, "
        f"oblique {metadata.get('oblique_angle', 0.0):.2f} deg ({source})"
    )


def _select_content(metadata: Metadata) -> str:
    content = metadata.get("content_box")
    page = metadata.get("page_box")
    content_text = "none" if content is None else _box(content)
    page_text = "none" if page is None else _box(page)
    return (
        f"content {content_text} ({metadata.get('content_detection_mode')}); "
        f"page {page_text} ({metadata.get('page_detection_mode')})"
    )


def _page_layout(metadata: Metadata) -> str:
    width, height = metadata.get("page_size_px", (0, 0))
    margins = metadata.get("margins", {})
    alignment = metadata.get("alignment", {})
    text = (
        f"page {width}x{height} px; margins "
        f"{margins.get('left', 0):g}/{margins.get('top', 0):g}/"
        f"{margins.get('right', 0):g}/{margins.get('bottom', 0):g} mm (l/t/r/b); "
        f"align {alignment.get('horizontal')} {alignment.get('vertical')}"
    )
    if metadata.get("aggregate_size_changed"):
        text += "; aggregate size changed"
    return text


def _output(metadata: Metadata, metrics: DewarpMetrics | None) -> str:
    text = f"{metadata.get('color_mode')} {metadata.get('binarization_method')}"
    if "output_dpi" in metadata:
        horizontal, vertical = metadata["output_dpi"]
        text += f", {horizontal}x{vertical} dpi"
    dewarping = metadata.get("dewarping_mode", "off")
    if dewarping == "off":
        return text
    text += f"; dewarp {dewarping}"
    if metrics is not None:
        text += (
            f": {metrics.outcome.value}, {metrics.lines_kept} of "
            f"{metrics.lines_traced} lines kept"
        )
    if "was_dewarped" in metadata and not metadata["was_dewarped"]:
        text += ", not applied"
    return text


_SUMMARIES: dict[FilterStage, Callable[[Metadata], str]] = {
    FilterStage.FIX_ORIENTATION: _fix_orientation,
    FilterStage.PAGE_SPLIT: _page_split,
    FilterStage.DESKEW: _deskew,
    FilterStage.SELECT_CONTENT: _select_content,
    FilterStage.PAGE_LAYOUT: _page_layout,
}
