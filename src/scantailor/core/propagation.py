"""Carrying a page's settings to other pages, and between stages.

Two propagators run when the user leaves a stage: page orientations flow from
Fix Orientation into the page counts that Page Split works with, and content
boxes flow from Select Content into the Page Layout sizes that Output needs.
The ``apply_*`` functions implement "Apply to..." for each stage.
"""

from collections.abc import Collection, Iterable, Sequence
from enum import Enum, auto
from typing import Any

import numpy as np

from scantailor.core.batch import DEFAULT_DPI
from scantailor.core.default_params import DefaultParams
from scantailor.core.models import ImageId, OrthogonalRotation, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, fix_aggregate_page_size
from scantailor.core.project import ImageInfo, ImageMetadata, Project
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.deskew.params import AutoManualMode as DeskewMode
from scantailor.filters.output.params import ApplyGroup
from scantailor.filters.page_layout.settings import MARGIN_SIDES
from scantailor.filters.page_split.dependencies import (
    Dependencies as SplitDependencies,
)
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout_adapter import (
    correct_page_layout_type,
)
from scantailor.filters.select_content.dependencies import Dependencies


class ApplyScope(Enum):
    """Which pages an "Apply to..." request reaches."""

    THIS_PAGE = auto()
    ALL_PAGES = auto()
    THIS_PAGE_AND_FOLLOWERS = auto()
    THIS_PAGE_EVERY_OTHER = auto()
    EVERY_OTHER = auto()
    SELECTED_PAGES = auto()
    EVERY_OTHER_SELECTED = auto()


def select_pages(
    scope: ApplyScope,
    current: PageId,
    pages: Sequence[PageId],
    selected: Sequence[PageId] = (),
) -> list[PageId]:
    """Resolve an apply scope to concrete pages.

    ``EVERY_OTHER_SELECTED`` deliberately differs from the C++ application:
    it starts from the current page when that page is selected, as the Split
    mode dialog hint promises, whereas C++ always starts from the first
    selected page.

    Args:
        scope (ApplyScope): The requested scope.
        current (PageId): The page whose settings are being copied.
        pages (Sequence[PageId]): All pages of the project in display order.
        selected (Sequence[PageId]): The selected pages in display order.

    Returns:
        list[PageId]: The target pages in display order. The current page is
            included whenever the scope covers it.
    """
    position = pages.index(current) if current in pages else -1
    match scope:
        case ApplyScope.THIS_PAGE:
            return [current]
        case ApplyScope.ALL_PAGES:
            return list(pages)
        case ApplyScope.THIS_PAGE_AND_FOLLOWERS:
            return list(pages[position:]) if position >= 0 else [current]
        case ApplyScope.THIS_PAGE_EVERY_OTHER:
            return list(pages[position::2]) if position >= 0 else [current]
        case ApplyScope.EVERY_OTHER:
            return list(pages[position % 2 :: 2]) if position >= 0 else [current]
        case ApplyScope.SELECTED_PAGES:
            return list(selected)
        case ApplyScope.EVERY_OTHER_SELECTED:
            # The hint in the dialog promises that the current page is included.
            start = selected.index(current) % 2 if current in selected else 0
            return list(selected[start::2])


def advise_logical_page_count(
    metadata: ImageMetadata, rotation: OrthogonalRotation
) -> int:
    """Guess whether an oriented image holds one page or a two-page spread.

    An image that is wider than tall in physical size is taken to be a spread.

    Args:
        metadata (ImageMetadata): Pixel size and resolution of the image.
        rotation (OrthogonalRotation): Orientation chosen for the image.

    Returns:
        int: 1 or 2.
    """
    width, height = rotation.rotate_dimensions(metadata.width, metadata.height)
    dpi_x, dpi_y = rotation.rotate_dimensions(
        metadata.dpi.horizontal, metadata.dpi.vertical
    )
    return 2 if width * dpi_y > height * dpi_x else 1


def _sub_pages_of(info: ImageInfo) -> list[PageId]:
    return [PageId(image_id=info.id, sub_page=sub) for sub in SubPage]


def propagate_page_orientations(
    project: Project, stages: StageSequence
) -> list[ImageId]:
    """Update every image's page count from its chosen orientation.

    Images whose layout type was set by hand keep their page count. When the
    count changes, an automatically detected split is dropped so it is detected
    again for the new orientation.

    Args:
        project (Project): The open project; page counts are updated in place.
        stages (StageSequence): Source of the orientations and the split settings.

    Returns:
        list[ImageId]: Images whose page count changed.
    """
    rotations = stages.fix_orientation_filter.settings
    split = stages.page_split_filter.settings
    changed: list[ImageId] = []
    for info in project.images:
        candidates = _sub_pages_of(info)
        if any(
            split.is_params_set(page)
            and split.get_params(page).layout_type != LayoutType.AUTO_LAYOUT_TYPE
            for page in candidates
        ):
            continue
        count = advise_logical_page_count(
            info.metadata, rotations.get_rotation_for(info.id)
        )
        if count == 2 and info.left_half_removed != info.right_half_removed:
            count = 1
        if count == info.num_sub_pages:
            continue
        info.num_sub_pages = count
        for page in candidates:
            split.clear_params(page)
        changed.append(info.id)
    return changed


def set_page_count(project: Project, image_id: ImageId, count: int) -> bool:
    """Make an image hold one or two logical pages.

    An image with exactly one half removed keeps a single page whatever is asked.

    Args:
        project (Project): The open project; updated in place.
        image_id (ImageId): The image to change.
        count (int): The wanted number of pages, 1 or 2.

    Returns:
        bool: True if the image's page count changed.
    """
    for info in project.images:
        if info.id != image_id:
            continue
        if count == 2 and info.left_half_removed != info.right_half_removed:
            count = 1
        if count == info.num_sub_pages:
            return False
        info.num_sub_pages = count
        return True
    return False


def adopt_requested_page_counts(
    project: Project, stages: StageSequence, images: Iterable[ImageId]
) -> list[ImageId]:
    """Give images the page count their requested Page Split layout type implies.

    Images whose layout type is automatic are left alone.

    Args:
        project (Project): The open project; page counts are updated in place.
        stages (StageSequence): Source of the requested layout types.
        images (Iterable[ImageId]): The images to look at.

    Returns:
        list[ImageId]: Images whose page count changed.
    """
    split = stages.page_split_filter
    changed: list[ImageId] = []
    for image_id in images:
        count = split.get_layout_type(PageId(image_id=image_id)).sub_page_count()
        if count is not None and set_page_count(project, image_id, count):
            changed.append(image_id)
    return changed


def adopt_stored_page_counts(
    project: Project, stages: StageSequence, images: Iterable[ImageId]
) -> list[ImageId]:
    """Give images the page count of the Page Split layout stored for them.

    Used after detection ran, and when entering a later stage, so the page list
    shows two pages for a detected spread. Layouts that no longer match the
    image's size or orientation, and images without a layout, are left alone.
    A layout whose cutters cut nothing counts as a single page.

    Args:
        project (Project): The open project; page counts are updated in place.
        stages (StageSequence): Source of the stored layouts and orientations.
        images (Iterable[ImageId]): The images to look at.

    Returns:
        list[ImageId]: Images whose page count changed.
    """
    split = stages.page_split_filter
    rotations = stages.fix_orientation_filter.settings
    metadata_of = {info.id: info.metadata for info in project.images}
    changed: list[ImageId] = []
    for image_id in images:
        metadata = metadata_of.get(image_id)
        params = split.get_params(PageId(image_id=image_id))
        if metadata is None or params.page_layout is None:
            continue
        rotation = rotations.get_rotation_for(image_id)
        width, height = rotation.rotate_dimensions(metadata.width, metadata.height)
        current = SplitDependencies(
            image_size=(round(width), round(height)),
            rotation_degrees=rotation.degrees,
            layout_type=params.layout_type,
        )
        if not params.dependencies.is_compatible_with(
            current, manual_split_line=params.is_manual()
        ):
            continue
        layout = correct_page_layout_type(params.page_layout)
        if set_page_count(project, image_id, layout.num_sub_pages):
            changed.append(image_id)
    return changed


def propagate_content_boxes(project: Project, stages: StageSequence) -> None:
    """Record every page's content geometry in the Page Layout settings.

    Pages that Select Content has already measured have their geometry worked
    out from the stored parameters, without detection and without reading
    pixels. Pages it has not measured lose any geometry recorded earlier, so
    the output size is known to be incomplete.

    Args:
        project (Project): The open project.
        stages (StageSequence): Source of the stored parameters; the Page
            Layout settings are updated.
    """
    select_content = stages.select_content_filter
    metadata_of = {info.id: info.metadata for info in project.images}
    for page in project.get_pages():
        metadata = metadata_of[page.image_id]
        if not select_content.is_params_set(page) or not (
            metadata.width and metadata.height
        ):
            _forget_geometry(stages, page)
            continue
        dpi = float(metadata.dpi.horizontal) if metadata.is_dpi_ok() else DEFAULT_DPI
        stored_only = PipelineOptions(start_stage=FilterStage.OUTPUT, dpi=dpi)
        # Only the image size matters when nothing is detected.
        blank = np.zeros((metadata.height, metadata.width), dtype=np.uint8)
        fix_aggregate_page_size([(blank, page)], stages, stored_only)


def seed_new_pages(
    stages: StageSequence, images: Iterable[ImageId], defaults: DefaultParams
) -> None:
    """Start the pages of images added to an open project from the defaults.

    C++ gives a page the default parameters of every stage when it is first
    processed. Pages that already have parameters keep them, and the stage-wide
    values of the project (page detection box, default layout type, oblique
    default) are left as they are. A stage whose defaults are the factory ones
    is not seeded, so detection keeps running by itself.

    Args:
        stages (StageSequence): Stages of the open project.
        images (Iterable[ImageId]): The images that were added.
        defaults (DefaultParams): The selected default parameters.
    """
    factory = DefaultParams()
    image_list = list(images)
    pages = [
        PageId(image_id=image_id, sub_page=sub_page)
        for image_id in image_list
        for sub_page in SubPage
    ]

    if defaults.fix_orientation != factory.fix_orientation:
        orientation = stages.fix_orientation_filter.settings
        for image_id in image_list:
            if not orientation.is_rotation_set(image_id):
                orientation.apply_rotation(
                    image_id, defaults.fix_orientation.image_rotation
                )

    seeds: list[tuple[Any, Any]] = []
    if defaults.deskew.mode is DeskewMode.MANUAL:
        seeds.append((stages.deskew_filter.settings, defaults.get_deskew_params()))
    if defaults.select_content != factory.select_content:
        seeds.append(
            (
                stages.select_content_filter.settings,
                defaults.get_select_content_params(),
            )
        )
    if defaults.page_layout != factory.page_layout:
        seeds.append(
            (stages.page_layout_filter.settings, defaults.get_page_layout_params())
        )
    if defaults.output != factory.output:
        seeds.append((stages.output_filter.settings, defaults.get_output_params()))
    for settings, params in seeds:
        for page in pages:
            if not settings.is_params_set(page):
                settings.set_params(page, params)


def _forget_geometry(stages: StageSequence, page: PageId) -> None:
    stages.page_layout_filter.settings.invalidate_content_size(page)


def apply_orientation(
    stages: StageSequence, source: PageId, targets: Iterable[PageId]
) -> None:
    """Give the targets' images the rotation and trim of the source page's image.

    Args:
        stages (StageSequence): The stages whose settings are updated.
        source (PageId): The page whose rotation and trim are copied.
        targets (Iterable[PageId]): Pages that receive them.
    """
    stages.fix_orientation_filter.apply_current_to_pages(source.image_id, targets)


def apply_page_split(
    stages: StageSequence,
    source: PageId,
    targets: Iterable[PageId],
    layout_type: LayoutType,
    *,
    apply_cut: bool,
) -> None:
    """Give the targets a layout type, and optionally the source's cutters.

    Cutters are adapted to each target's own outline instead of being copied as
    coordinates; see ``Filter.apply_layout_type``.

    Args:
        stages (StageSequence): The stages whose settings are updated.
        source (PageId): The page whose layout is copied.
        targets (Iterable[PageId]): Pages that receive it.
        layout_type (LayoutType): The layout type to request; automatic makes
            every target detect its own.
        apply_cut (bool): Copy the source's cutter lines.
    """
    stages.page_split_filter.apply_layout_type(
        source, list(targets), layout_type, apply_cut=apply_cut
    )


def apply_deskew(
    stages: StageSequence,
    source: PageId,
    targets: Iterable[PageId],
    *,
    apply_deskew: bool = True,
    apply_oblique: bool = True,
) -> None:
    """Copy the source page's deskew setting.

    A manual angle is copied as it is. An automatic one is not: the targets
    detect their own. A part that is not copied stays as the target has it.

    Args:
        stages (StageSequence): The stages whose settings are updated.
        source (PageId): The page whose setting is copied.
        targets (Iterable[PageId]): Pages that receive it.
        apply_deskew (bool): Copy the rotation angle and mode.
        apply_oblique (bool): Copy the oblique angle and mode.
    """
    settings = stages.deskew_filter.settings
    settings.apply_parts_to_pages(
        settings.get_params(source),
        targets,
        rotation=apply_deskew,
        oblique=apply_oblique,
    )


def apply_select_content(
    stages: StageSequence,
    source: PageId,
    targets: Iterable[PageId],
    *,
    apply_content_box: bool = True,
    apply_page_box: bool = True,
) -> None:
    """Copy the source page's content and page detection settings.

    A target that already has a manual box keeps it when the matching option is
    off. Cached dependencies are reset so detection runs again where it is
    automatic.

    Args:
        stages (StageSequence): The stages whose settings are updated.
        source (PageId): The page whose settings are copied.
        targets (Iterable[PageId]): Pages that receive them.
        apply_content_box (bool): Copy a manually set content box.
        apply_page_box (bool): Copy a manually set page box.
    """
    settings = stages.select_content_filter.settings
    template = settings.get_params(source).model_copy(
        update={"dependencies": Dependencies()}
    )
    for page in targets:
        if page == source:
            continue
        params = template
        if settings.is_params_set(page):
            old = settings.get_params(page)
            if params.is_page_manual() and not apply_page_box:
                params = params.model_copy(update={"page_box": old.page_box})
            if params.is_content_manual() and not apply_content_box:
                params = params.model_copy(update={"content_box": old.content_box})
        settings.set_params(page, params)


def apply_margins(
    stages: StageSequence,
    source: PageId,
    targets: Iterable[PageId],
    sides: Collection[str] = MARGIN_SIDES,
) -> None:
    """Copy the source page's margins, and whether they follow the content.

    Margins that follow the content are worked out again when a target is
    processed, so the targets keep their own hard margins and lose their
    measured size until then; the chosen sides do not apply to them. The source
    page already holds its own margins, so it is left as it is even when it is
    among the targets.

    Args:
        stages (StageSequence): The stages whose settings are updated.
        source (PageId): The page whose margins are copied.
        targets (Iterable[PageId]): Pages that receive them.
        sides (Collection[str]): Names of the sides copied (``"top"``,
            ``"bottom"``, ``"left"``, ``"right"``); every other side of a target
            keeps its own margin.
    """
    settings = stages.page_layout_filter.settings
    params = settings.get_params(source)
    pages = [page for page in targets if page != source]
    if params.auto_margins:
        for page in pages:
            settings.set_params(page, settings.get_params(page).with_auto_margins())
            settings.invalidate_content_size(page)
    else:
        settings.apply_margins_to_pages(pages, params.hard_margins_mm, sides)


def apply_alignment(
    stages: StageSequence, source: PageId, targets: Iterable[PageId]
) -> None:
    """Copy the source page's alignment.

    Args:
        stages (StageSequence): The stages whose settings are updated.
        source (PageId): The page whose alignment is copied.
        targets (Iterable[PageId]): Pages that receive it.
    """
    settings = stages.page_layout_filter.settings
    settings.apply_alignment_to_pages(
        list(targets), settings.get_params(source).alignment
    )


def apply_output(
    stages: StageSequence,
    source: PageId,
    targets: Iterable[PageId],
    group: ApplyGroup,
) -> None:
    """Copy one group of the source page's output parameters.

    Zones and the manual distortion model belong to no group and stay as they are.

    Args:
        stages (StageSequence): The stages whose settings are updated.
        source (PageId): The page whose parameters are copied.
        targets (Iterable[PageId]): Pages that receive them.
        group (ApplyGroup): The parameter group to copy.
    """
    filter_ = stages.output_filter
    source_params = filter_.get_params(source)
    for target in targets:
        if target == source:
            continue
        filter_.set_params(
            target, filter_.get_params(target).with_group_from(source_params, group)
        )
