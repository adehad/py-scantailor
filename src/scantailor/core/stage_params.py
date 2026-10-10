"""Persistence of per-page filter parameters inside a project file.

Every stage stores parameters that were set explicitly (by the user or by a
previous processing run). Only those are written out, so unset pages keep
following the stage defaults after a project is reopened.
"""

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from scantailor.core.models import ImageId, OrthogonalRotation, PageId
from scantailor.core.project import StageParamsEntry
from scantailor.core.stage_sequence import FilterStage, StageSequence

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
    rotations = [
        StageParamsEntry(
            page=PageId(image_id=image_id),
            params=_dump(orientation.get_rotation_for(image_id)),
        )
        for image_id in _distinct_images(pages)
        if orientation.is_rotation_set(image_id)
    ]
    if rotations:
        result[STAGE_KEYS[FilterStage.FIX_ORIENTATION]] = rotations

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
        orientation.apply_rotation(
            entry.page.image_id, OrthogonalRotation.model_validate(entry.params)
        )

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
