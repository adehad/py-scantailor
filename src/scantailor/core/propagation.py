"""Carrying a page's settings to other pages, and between stages.

Two propagators run when the user leaves a stage: page orientations flow from
Fix Orientation into the page counts that Page Split works with, and content
boxes flow from Select Content into the Page Layout sizes that Output needs.
The ``apply_*`` functions implement "Apply to..." for each stage.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from scantailor.core.batch import DEFAULT_DPI
from scantailor.core.default_params import DefaultParams
from scantailor.core.models import ImageId, OrthogonalRotation, PageId, SubPage
from scantailor.core.page_geometry import build_transformation
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
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.filters.select_content.dependencies import Dependencies
from scantailor.filters.select_content.params import Params as SelectContentParams


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
        count = _allowed_page_count(info, count)
        if count == info.num_sub_pages:
            return False
        info.num_sub_pages = count
        return True
    return False


def _allowed_page_count(info: ImageInfo, count: int) -> int:
    """An image with exactly one half removed holds one page whatever is asked."""
    if count == 2 and info.left_half_removed != info.right_half_removed:
        return 1
    return count


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
    metadata_of = {info.id: info.metadata for info in project.images}
    changed: list[ImageId] = []
    for image_id in images:
        metadata = metadata_of.get(image_id)
        count = (
            None if metadata is None else _stored_page_count(stages, image_id, metadata)
        )
        if count is not None and set_page_count(project, image_id, count):
            changed.append(image_id)
    return changed


def pages_after_split(
    project: Project, stages: StageSequence, pages: Sequence[PageId]
) -> list[PageId]:
    """List the pages a batch carries past Page Split, from the stored layouts.

    This is the page list ``adopt_stored_page_counts`` would give, worked out
    without changing the project, so a batch thread can call it while the
    project belongs to the GUI thread. An image whose page count changes is
    replaced, where it first appears in ``pages``, by all of its new pages;
    every other page stays as and where it is.

    Args:
        project (Project): The project the pages belong to; left unchanged.
        stages (StageSequence): Holds the layouts Page Split stored.
        pages (Sequence[PageId]): The batch's pages, in order.

    Returns:
        list[PageId]: The pages to process through the later stages, in order.
    """
    info_of = {info.id: info for info in project.images}
    regrown: dict[ImageId, list[PageId]] = {}
    for image_id in dict.fromkeys(page.image_id for page in pages):
        info = info_of.get(image_id)
        if info is None:
            continue
        stored = _stored_page_count(stages, image_id, info.metadata)
        if stored is None:
            continue
        count = _allowed_page_count(info, stored)
        if count != info.num_sub_pages:
            resized = info.model_copy(update={"num_sub_pages": count})
            regrown[image_id] = project.pages_of_image(resized)
    listed: list[PageId] = []
    placed: set[ImageId] = set()
    for page in pages:
        if page.image_id not in regrown:
            listed.append(page)
        elif page.image_id not in placed:
            placed.add(page.image_id)
            listed.extend(regrown[page.image_id])
    return listed


def split_is_settled(
    project: Project, stages: StageSequence, image_id: ImageId
) -> bool:
    """Whether Page Split would keep an image's stored layout without its pixels.

    The image's size comes from the project, so a replaced file whose new
    size the project has not recorded yet counts by its old size.

    Args:
        project (Project): Holds the image's size.
        stages (StageSequence): Holds its orientation, trim and stored layout.
        image_id (ImageId): The image to check.

    Returns:
        bool: True when Page Split would neither detect nor adapt the layout.
    """
    info = next((info for info in project.images if info.id == image_id), None)
    if info is None or not (info.metadata.width and info.metadata.height):
        return False
    orientation = stages.fix_orientation_filter
    rotation = orientation.get_rotation(image_id)
    width, height = rotation.rotate_dimensions(
        info.metadata.width, info.metadata.height
    )
    oriented = (round(width), round(height))
    return stages.page_split_filter.keeps_stored_layout(
        PageId(image_id=image_id),
        oriented,
        rotation.degrees,
        orientation.trim_crop_area(image_id, oriented),
    )


@dataclass(frozen=True)
class ProjectPageLister:
    """Answers a batch's Page Split questions from a project, without changing it.

    Attributes:
        project: The project the batch's pages belong to.
        stages: The project's stages, which the batch updates.
    """

    project: Project
    stages: StageSequence

    def is_settled(self, image_id: ImageId) -> bool:
        """Whether the image can skip the batch's Page Split pass.

        Args:
            image_id (ImageId): The image to check.

        Returns:
            bool: As ``split_is_settled``.
        """
        return split_is_settled(self.project, self.stages, image_id)

    def pages_after_split(self, pages: Sequence[PageId]) -> list[PageId]:
        """List the pages past Page Split.

        Args:
            pages (Sequence[PageId]): The batch's pages, in order.

        Returns:
            list[PageId]: As ``pages_after_split``.
        """
        return pages_after_split(self.project, self.stages, pages)


def _stored_page_count(
    stages: StageSequence, image_id: ImageId, metadata: ImageMetadata
) -> int | None:
    """Pages in the image's stored Page Split layout; None without a valid one."""
    split = stages.page_split_filter
    params = split.get_params(PageId(image_id=image_id))
    if params.page_layout is None:
        return None
    rotation = stages.fix_orientation_filter.settings.get_rotation_for(image_id)
    width, height = rotation.rotate_dimensions(metadata.width, metadata.height)
    current = SplitDependencies(
        image_size=(round(width), round(height)),
        rotation_degrees=rotation.degrees,
        layout_type=params.layout_type,
    )
    if not params.dependencies.is_compatible_with(
        current, manual_split_line=params.is_manual()
    ):
        return None
    return correct_page_layout_type(params.page_layout).num_sub_pages


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
        dpi = metadata.dpi if metadata.is_dpi_ok() else DEFAULT_DPI
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


def _polygon_bounds(points: Sequence[Sequence[float]]) -> ContentBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return ContentBox.from_ltrb(min(xs), min(ys), max(xs), max(ys))


def _select_content_image(
    stages: StageSequence,
    page: PageId,
    params: SelectContentParams,
    metadata_of: Mapping[ImageId, ImageMetadata],
) -> ContentBox:
    """Bounds of the image Select Content shows for a page; empty if unknown.

    A page Select Content has processed recorded its outline. For any other
    page the outline is worked out from the image size the project stores and
    the page's Fix Orientation, Page Split and Deskew settings, the same
    transformation the pipeline builds.
    """
    if params.dependencies.rotated_page_outline:
        return _polygon_bounds(params.dependencies.rotated_page_outline)
    metadata = metadata_of.get(page.image_id)
    if metadata is None or not (metadata.width and metadata.height):
        return ContentBox()
    rotation = stages.fix_orientation_filter.get_rotation(page.image_id)
    oriented = rotation.rotate_dimensions(metadata.width, metadata.height)
    dpi = metadata.dpi if metadata.is_dpi_ok() else DEFAULT_DPI
    xform = build_transformation(
        stages, page, (int(oriented[0]), int(oriented[1])), dpi, FilterStage.DESKEW
    )
    return _polygon_bounds(xform.resulting_pre_crop_area.tolist())


def _centre_shift(source: ContentBox, target: ContentBox) -> tuple[float, float]:
    """How far the centre of ``target`` lies from that of ``source``."""
    return target.center_x - source.center_x, target.center_y - source.center_y


def _shifted(box: ContentBox, shift: tuple[float, float]) -> ContentBox:
    return box.model_copy(update={"x": box.x + shift[0], "y": box.y + shift[1]})


def _page_box_for_target(
    box: PageBox, source_image: ContentBox, target_image: ContentBox
) -> PageBox:
    """Centre a copied page box on the target image as it was on the source's.

    The box is then clipped to the target image, or becomes the whole image
    when it misses it. Unchanged when either image is unknown.
    """
    if source_image.is_empty() or target_image.is_empty():
        return box
    moved = _shifted(box.to_content_box(), _centre_shift(source_image, target_image))
    clipped = moved.intersection(target_image)
    if clipped.is_empty():
        clipped = target_image
    return PageBox(**clipped.model_dump())


def apply_select_content(
    stages: StageSequence,
    source: PageId,
    targets: Iterable[PageId],
    *,
    apply_content_box: bool = True,
    apply_page_box: bool = True,
    project: Project | None = None,
) -> None:
    """Copy the source page's content and page detection settings.

    A target that already has a manual box keeps it when the matching option is
    off. Cached dependencies are reset so detection runs again where it is
    automatic.

    Copied manual boxes are placed by centres, so they stay put on images of
    the source's size. A manual page box keeps its offset from the centre of
    the image and is clipped to it. A manual content box keeps its offset
    from the centre of the target's page box: the copied one, the whole image
    when page detection is off, or else the box the target last had. Without
    the project, a target Select Content has not processed has no known image
    size and keeps the source's coordinates.

    Args:
        stages (StageSequence): The stages whose settings are updated.
        source (PageId): The page whose settings are copied.
        targets (Iterable[PageId]): Pages that receive them.
        apply_content_box (bool): Copy a manually set content box.
        apply_page_box (bool): Copy a manually set page box.
        project (Project | None): Supplies the image sizes of targets Select
            Content has not processed.
    """
    settings = stages.select_content_filter.settings
    metadata_of = (
        {} if project is None else {info.id: info.metadata for info in project.images}
    )
    original = settings.get_params(source)
    source_image = _select_content_image(stages, source, original, metadata_of)
    template = original.model_copy(update={"dependencies": Dependencies()})
    for page in targets:
        if page == source:
            continue
        params = template
        known = settings.is_params_set(page)
        old = settings.get_params(page)
        target_image = _select_content_image(stages, page, old, metadata_of)
        if params.is_page_manual() and apply_page_box:
            params = params.with_page_box(
                _page_box_for_target(params.page_box, source_image, target_image)
            )
        elif params.is_page_disabled() and not target_image.is_empty():
            params = params.with_page_box(PageBox(**target_image.model_dump()))
        elif not old.page_box.is_empty():
            params = params.with_page_box(old.page_box)
        if params.is_content_manual() and not apply_content_box:
            if known:
                params = params.with_content_box(old.content_box)
        elif (
            params.is_content_manual()
            and not original.page_box.is_empty()
            and not params.page_box.is_empty()
        ):
            shift = _centre_shift(
                original.page_box.to_content_box(),
                params.page_box.to_content_box(),
            )
            params = params.with_content_box(_shifted(params.content_box, shift))
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
