"""Sort orders and completeness checks of the thumbnail list for each stage.

Each stage offers the orders of the original program: Page Split by split type,
Deskew by deviation of the angle, Select Content and Page Layout by size and
deviation, Output by completeness. A page is incomplete at a stage when the
stage holds no result for it yet.
"""

import math
from collections.abc import Callable, Sequence

from scantailor.app.ui.thumbnails.model import (
    PageEntry,
    PageOrderOption,
    natural_order,
)
from scantailor.core.models import PageId
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType

_AUTO_SPLIT_RANK = 100  # Pages whose type is still undecided come after the others
_SPLIT_RANK = {
    LayoutType.SINGLE_PAGE_UNCUT: 1,
    LayoutType.PAGE_PLUS_OFFCUT: 2,
    LayoutType.TWO_PAGES: 3,
}
_UNKNOWN_DEVIATION = -1.0


def is_page_incomplete(
    sequence: StageSequence, stage: FilterStage, page: PageId
) -> bool:
    """Tell whether a stage holds no result for a page.

    Args:
        sequence (StageSequence): The stages and their settings; only read.
        stage (FilterStage): The stage shown.
        page (PageId): The page.

    Returns:
        bool: True if the stage has not processed the page, so its thumbnail is
        only a guess.
    """
    match stage:
        case FilterStage.FIX_ORIENTATION:
            return False
        case FilterStage.PAGE_SPLIT:
            return not sequence.page_split_filter.is_params_set(page)
        case FilterStage.DESKEW:
            return not sequence.deskew_filter.settings.is_params_set(page)
        case FilterStage.SELECT_CONTENT:
            return not sequence.select_content_filter.is_params_set(page)
        case FilterStage.PAGE_LAYOUT:
            return not sequence.page_layout_filter.settings.is_params_set(page)
        case FilterStage.OUTPUT:
            return not sequence.output_filter.is_params_set(page)
    return False


def completeness_provider(sequence: StageSequence) -> Callable[[PageId, str], bool]:
    """Return the incompleteness check the thumbnail list takes.

    Args:
        sequence (StageSequence): The stages to ask.

    Returns:
        Callable[[PageId, str], bool]: Takes a page and a stage key, the integer
        value of a ``FilterStage`` as text.
    """

    def check(page: PageId, stage_key: str) -> bool:
        return is_page_incomplete(sequence, FilterStage(int(stage_key)), page)

    return check


def stage_orders_provider(
    sequence: StageSequence,
) -> Callable[[str], list[PageOrderOption]]:
    """Return the sort option provider the thumbnail list takes.

    Args:
        sequence (StageSequence): The stages whose settings the orders read.

    Returns:
        Callable[[str], list[PageOrderOption]]: Takes a stage key, the integer
        value of a ``FilterStage`` as text.
    """

    def options(stage_key: str) -> list[PageOrderOption]:
        return stage_page_orders(sequence, FilterStage(int(stage_key)))

    return options


def stage_page_orders(
    sequence: StageSequence, stage: FilterStage
) -> list[PageOrderOption]:
    """Return the sort options a stage offers, the natural order first.

    Args:
        sequence (StageSequence): The stages whose settings the orders read.
        stage (FilterStage): The stage shown.

    Returns:
        list[PageOrderOption]: The options; empty for Fix Orientation, which
        has none.
    """
    match stage:
        case FilterStage.FIX_ORIENTATION:
            return []
        case FilterStage.PAGE_SPLIT:
            return [natural_order(), _order_by_split_type(sequence)]
        case FilterStage.DESKEW:
            return [
                natural_order(),
                _order_by_deviation(sequence, _deskew_angle),
            ]
        case FilterStage.SELECT_CONTENT:
            return [
                natural_order(),
                *_size_orders(sequence, _content_size),
                _order_by_deviation(sequence, _content_diagonal),
            ]
        case FilterStage.PAGE_LAYOUT:
            return [
                natural_order(),
                *_size_orders(sequence, _layout_size),
                _order_by_deviation(sequence, _hard_margins_diagonal),
            ]
        case FilterStage.OUTPUT:
            return [natural_order(), _order_by_completeness()]
    return [natural_order()]


def _order_by_completeness() -> PageOrderOption:
    return PageOrderOption("Order by completeness", lambda entry: entry.incomplete)


def _order_by_split_type(sequence: StageSequence) -> PageOrderOption:
    # Pages of one type are told apart by position, so reversing reverses them too.
    def rank(entry: PageEntry) -> tuple[bool, int, int]:
        if entry.incomplete:
            return True, 0, entry.position
        params = sequence.page_split_filter.get_params(entry.page)
        layout = params.page_layout
        layout_type = (
            layout.to_layout_type() if layout is not None else params.layout_type
        )
        return False, _SPLIT_RANK.get(layout_type, _AUTO_SPLIT_RANK), entry.position

    return PageOrderOption("Order by split type", rank)


def _size_orders(
    sequence: StageSequence,
    size: Callable[[StageSequence, PageId], tuple[float, float] | None],
) -> list[PageOrderOption]:
    def by(axis: int) -> Callable[[PageEntry], tuple[bool, float]]:
        def key(entry: PageEntry) -> tuple[bool, float]:
            known = None if entry.incomplete else size(sequence, entry.page)
            # Unknown sizes go to the back.
            return (True, 0.0) if known is None else (False, known[axis])

        return key

    return [
        PageOrderOption("Order by increasing width", by(0)),
        PageOrderOption("Order by increasing height", by(1)),
    ]


def _order_by_deviation(
    sequence: StageSequence,
    value: Callable[[StageSequence, PageId], float],
) -> PageOrderOption:
    """Order pages by how far their value lies from the mean, the largest first.

    Pages whose value is unknown, and incomplete pages, go to the back.
    """
    deviations: dict[PageId, float] = {}

    def prepare(entries: Sequence[PageEntry]) -> None:
        values = {
            entry.page: value(sequence, entry.page)
            for entry in entries
            if not entry.incomplete
        }
        known = [v for v in values.values() if not math.isnan(v)]
        deviations.clear()
        if len(values) < 2 or not known:
            deviations.update({page: 0.0 for page in values})
            return
        mean = sum(known) / len(known)
        for page, page_value in values.items():
            deviations[page] = (
                _UNKNOWN_DEVIATION if math.isnan(page_value) else abs(page_value - mean)
            )

    def key(entry: PageEntry) -> tuple[bool, float]:
        deviation = deviations.get(entry.page, _UNKNOWN_DEVIATION)
        return entry.incomplete, -deviation

    return PageOrderOption("Order by decreasing deviation", key, prepare)


def _deskew_angle(sequence: StageSequence, page: PageId) -> float:
    return sequence.deskew_filter.get_params(page).deskew_angle_deg


def _content_size(sequence: StageSequence, page: PageId) -> tuple[float, float] | None:
    size = sequence.select_content_filter.get_params(page).content_size_mm
    return None if size.is_empty() else (size.width_mm, size.height_mm)


def _content_diagonal(sequence: StageSequence, page: PageId) -> float:
    size = _content_size(sequence, page)
    return math.nan if size is None else math.hypot(*size)


def _layout_size(sequence: StageSequence, page: PageId) -> tuple[float, float] | None:
    params = sequence.page_layout_filter.get_params(page)
    if params.content_size_mm.is_empty():
        return None
    margins = params.hard_margins_mm
    return (
        params.content_size_mm.width_mm + margins.left + margins.right,
        params.content_size_mm.height_mm + margins.top + margins.bottom,
    )


def _hard_margins_diagonal(sequence: StageSequence, page: PageId) -> float:
    margins = sequence.page_layout_filter.get_params(page).hard_margins_mm
    return math.hypot(margins.left + margins.right, margins.top + margins.bottom)
