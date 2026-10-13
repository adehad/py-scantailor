"""Persistence of per-page filter parameters inside a project file.

Every stage stores parameters that were set explicitly (by the user or by a
previous processing run). Only those are written out, so unset pages keep
following the stage defaults after a project is reopened.
"""

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.core.project import StageParamsEntry
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.page_layout.guide import Guide
from scantailor.filters.page_layout.params import ContentSize
from scantailor.filters.page_split.layout_type import LayoutType

STAGE_KEYS: dict[FilterStage, str] = {
    FilterStage.FIX_ORIENTATION: "fix_orientation",
    FilterStage.PAGE_SPLIT: "page_split",
    FilterStage.DESKEW: "deskew",
    FilterStage.SELECT_CONTENT: "select_content",
    FilterStage.PAGE_LAYOUT: "page_layout",
    FilterStage.OUTPUT: "output",
}


def _params_type(stage: FilterStage) -> type[BaseModel]:
    """Return the Params model class of a per-page stage.

    Args:
        stage (FilterStage): A stage other than Fix Orientation.

    Returns:
        type[BaseModel]: The stage's Params class.
    """
    match stage:
        case FilterStage.PAGE_SPLIT:
            from scantailor.filters.page_split.params import Params
        case FilterStage.DESKEW:
            from scantailor.filters.deskew.params import Params
        case FilterStage.SELECT_CONTENT:
            from scantailor.filters.select_content.params import Params
        case FilterStage.PAGE_LAYOUT:
            from scantailor.filters.page_layout.params import Params
        case FilterStage.OUTPUT:
            from scantailor.filters.output.params import Params
        case _:
            msg = f"Stage {stage!r} has no per-page Params"
            raise ValueError(msg)
    return Params


def _dump(model: BaseModel) -> dict[str, Any]:
    """Dump a pydantic model to JSON-compatible data."""
    return model.model_dump(mode="json")


def _distinct_images(pages: Iterable[PageId]) -> list[ImageId]:
    """Return the images of the pages in first-seen order, without repeats."""
    return list(dict.fromkeys(page.image_id for page in pages))


def collect_stage_params(
    stages: StageSequence, pages: list[PageId]
) -> dict[str, list[StageParamsEntry]]:
    """Gather the explicitly set parameters of every stage.

    Args:
        stages (StageSequence): The stages whose settings are read.
        pages (list[PageId]): Pages of the project; parameters of other pages
            are not exported.

    Returns:
        dict[str, list[StageParamsEntry]]: Entries per stage key, omitting
            stages that have none.
    """
    result: dict[str, list[StageParamsEntry]] = {}

    orientation = stages.fix_orientation_filter.settings
    orientations = [
        StageParamsEntry(
            page=PageId(image_id=image_id),
            params=orientation.dump_params(image_id),
        )
        for image_id in _distinct_images(pages)
        if orientation.has_params(image_id)
    ]
    if orientations:
        result[STAGE_KEYS[FilterStage.FIX_ORIENTATION]] = orientations

    for stage in FilterStage:
        if stage is FilterStage.FIX_ORIENTATION:
            continue
        settings: Any = stages.filter_at(stage).settings
        entries = [
            StageParamsEntry(page=page, params=_dump(settings.get_params(page)))
            for page in pages
            if settings.is_params_set(page)
        ]
        if entries:
            result[STAGE_KEYS[stage]] = entries
    return result


def apply_stage_params(
    stages: StageSequence, stage_params: dict[str, list[StageParamsEntry]]
) -> None:
    """Load exported parameters into the stages' settings.

    Args:
        stages (StageSequence): The stages to populate.
        stage_params (dict[str, list[StageParamsEntry]]): Output of
            :func:`collect_stage_params`, typically read from a project file.
            Unknown stage keys are ignored.

    Raises:
        pydantic.ValidationError: If an entry does not match its stage's Params.
    """
    orientation = stages.fix_orientation_filter.settings
    for entry in stage_params.get(STAGE_KEYS[FilterStage.FIX_ORIENTATION], []):
        orientation.load_params(entry.page.image_id, entry.params)

    for stage in FilterStage:
        if stage is FilterStage.FIX_ORIENTATION:
            continue
        entries = stage_params.get(STAGE_KEYS[stage], [])
        if not entries:
            continue
        params_type = _params_type(stage)
        settings: Any = stages.filter_at(stage).settings
        set_params: Callable[[PageId, BaseModel], None] = settings.set_params
        for entry in entries:
            set_params(entry.page, params_type.model_validate(entry.params))


def collect_stage_defaults(stages: StageSequence) -> dict[str, Any]:
    """Read the stage-wide values that pages without their own parameters use.

    Returns:
        dict[str, Any]: JSON-compatible values for :func:`apply_stage_defaults`.
    """
    output = stages.output_filter.settings
    select_content = stages.select_content_filter.settings
    layout = stages.page_layout_filter.settings
    frozen = layout.get_frozen_aggregate_hard_size_mm()
    return {
        "page_split_layout_type": (
            stages.page_split_filter.settings.get_default_layout_type().value
        ),
        "output_dpi": output.get_default_dpi().model_dump(mode="json"),
        "output_color_mode": output.get_default_color_mode().value,
        "output_despeckle_level": output.get_default_despeckle_level().value,
        "page_detection_box": list(select_content.get_page_detection_box()),
        "page_detection_tolerance": select_content.get_page_detection_tolerance(),
        "deskew_auto_oblique": stages.deskew_filter.auto_oblique_default,
        "deskew_content_based": stages.deskew_filter.content_based,
        "page_layout_frozen_aggregate_size": (
            None if frozen is None else frozen.model_dump(mode="json")
        ),
        "page_layout_middle_rect_shown": layout.is_middle_rect_shown(),
        "page_layout_guides": [
            guide.model_dump(mode="json") for guide in layout.get_guides()
        ],
    }


def apply_stage_defaults(stages: StageSequence, defaults: dict[str, Any]) -> None:
    """Install stage-wide values into the stages' settings.

    Args:
        stages (StageSequence): The stages to populate.
        defaults (dict[str, Any]): Output of :func:`collect_stage_defaults`. A
            missing key leaves that default as it is, so partial data (from a
            C++ project, or a file saved before a key existed) is accepted.

    Raises:
        ValueError: If a stored value is not valid for its setting.
    """
    output = stages.output_filter.settings
    select_content = stages.select_content_filter.settings
    if (layout_type := defaults.get("page_split_layout_type")) is not None:
        stages.page_split_filter.settings.set_default_layout_type(
            LayoutType(layout_type)
        )
    if (dpi := defaults.get("output_dpi")) is not None:
        output.set_default_dpi(Dpi.model_validate(dpi))
    if (color_mode := defaults.get("output_color_mode")) is not None:
        output.set_default_color_mode(ColorMode(color_mode))
    if (despeckle := defaults.get("output_despeckle_level")) is not None:
        output.set_default_despeckle_level(DespeckleLevel(despeckle))
    if (box := defaults.get("page_detection_box")) is not None:
        width, height = box
        select_content.set_page_detection_box(width, height)
    if (tolerance := defaults.get("page_detection_tolerance")) is not None:
        select_content.set_page_detection_tolerance(tolerance)
    if (auto_oblique := defaults.get("deskew_auto_oblique")) is not None:
        stages.deskew_filter.auto_oblique_default = bool(auto_oblique)
    if (content_based := defaults.get("deskew_content_based")) is not None:
        stages.deskew_filter.content_based = bool(content_based)
    if (frozen := defaults.get("page_layout_frozen_aggregate_size")) is not None:
        stages.page_layout_filter.settings.restore_frozen_aggregate_hard_size_mm(
            ContentSize.model_validate(frozen)
        )
    if (shown := defaults.get("page_layout_middle_rect_shown")) is not None:
        stages.page_layout_filter.settings.set_middle_rect_shown(bool(shown))
    if (guides := defaults.get("page_layout_guides")) is not None:
        stages.page_layout_filter.settings.set_guides(
            Guide.model_validate(guide) for guide in guides
        )
