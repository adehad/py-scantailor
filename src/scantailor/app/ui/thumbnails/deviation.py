"""Which pages stand out from the others at Deskew, Select Content and Margins.

Each of these stages reduces a page to one number: the deskew angle, the
diagonal of the content box, the diagonal of the hard margins. A page deviates
when its number lies further from the median of all pages than both the
stage's coefficient times the spread and its threshold, a percentage of the
median's size. The spread is the median absolute deviation scaled to estimate
the standard deviation, so the coefficient still counts standard deviations,
but one bad page cannot widen it or pull the centre towards itself. The
coefficients and thresholds are application settings.
"""

import math
import statistics
from collections.abc import Callable, Mapping, Sequence

from scantailor.app.ui.thumbnails.model import DeviantPagesProvider, PageEntry
from scantailor.core.models import PageId
from scantailor.core.settings import ApplicationSettings, DeviationSettings
from scantailor.core.stage_sequence import FilterStage, StageSequence

# Fewer pages than this have no meaningful spread, so none of them deviates.
_MIN_PAGES = 3
# Makes the median absolute deviation of normally distributed values an
# estimate of their standard deviation: 1 / Phi^-1(3/4).
_MAD_TO_STD = 1.4826

PageValue = Callable[[StageSequence, PageId], float]
"""Reduces a page to the number its stage compares; NaN when it has none."""


def deskew_angle(sequence: StageSequence, page: PageId) -> float:
    """Return the page's deskew angle in degrees."""
    return sequence.deskew_filter.get_params(page).deskew_angle_deg


def content_size(sequence: StageSequence, page: PageId) -> tuple[float, float] | None:
    """Return the width and height in millimetres of the page's content box.

    Returns:
        tuple[float, float] | None: The size, or None when the box is empty.
    """
    size = sequence.select_content_filter.get_params(page).content_size_mm
    return None if size.is_empty() else (size.width_mm, size.height_mm)


def content_diagonal(sequence: StageSequence, page: PageId) -> float:
    """Return the diagonal in millimetres of the content box; NaN when it is empty."""
    size = content_size(sequence, page)
    return math.nan if size is None else math.hypot(*size)


def hard_margins_diagonal(sequence: StageSequence, page: PageId) -> float:
    """Return the diagonal of the summed horizontal and vertical hard margins, in mm."""
    margins = sequence.page_layout_filter.get_params(page).hard_margins_mm
    return math.hypot(margins.left + margins.right, margins.top + margins.bottom)


def distances_from_median(values: Mapping[PageId, float]) -> dict[PageId, float]:
    """Return how far each page's value lies from the median of the known values.

    The deviation marks and the deviation sort order both use this distance, so
    the marked pages are always the furthest.

    Args:
        values (Mapping[PageId, float]): Page values; NaN where a page has none.

    Returns:
        dict[PageId, float]: The absolute distance per page; NaN for a page
        without a value, and for every page when no value is known.
    """
    known = [value for value in values.values() if not math.isnan(value)]
    centre = statistics.median(known) if known else math.nan
    return {page: abs(value - centre) for page, value in values.items()}


def deviant_pages(
    values: Mapping[PageId, float],
    rule: DeviationSettings,
    unknown_deviates: bool = False,
) -> frozenset[PageId]:
    """Return the pages whose value lies too far from the median.

    Args:
        values (Mapping[PageId, float]): The value of every page that has a
            result at the stage; NaN where the result gives no value.
        rule (DeviationSettings): ``coefficient`` scales the spread, 1.4826
            times the median absolute deviation of the known values;
            ``threshold`` is a percentage of the absolute median. A page
            deviates only when it is further from the median than both.
        unknown_deviates (bool): Whether a page whose value is NaN deviates.

    Returns:
        frozenset[PageId]: The deviating pages; none when fewer than three
        pages have a result.
    """
    if len(values) < _MIN_PAGES:
        return frozenset()
    known = [value for value in values.values() if not math.isnan(value)]
    unknown = {page for page, value in values.items() if math.isnan(value)}
    if len(known) < 2:
        return frozenset(unknown if unknown_deviates else ())
    centre = statistics.median(known)
    distances = distances_from_median(values)
    spread = _MAD_TO_STD * statistics.median(
        distance for distance in distances.values() if not math.isnan(distance)
    )
    allowed = max(rule.coefficient * spread, rule.threshold / 100 * abs(centre))
    deviating = {
        page
        for page, distance in distances.items()
        if not math.isnan(distance) and distance > allowed
    }
    if unknown_deviates:
        deviating |= unknown
    return frozenset(deviating)


def stage_rule(
    stage: FilterStage, settings: ApplicationSettings
) -> tuple[PageValue, DeviationSettings, bool] | None:
    """Return how a stage finds its deviating pages.

    Args:
        stage (FilterStage): The stage shown.
        settings (ApplicationSettings): Source of the coefficients and thresholds.

    Returns:
        tuple[PageValue, DeviationSettings, bool] | None: The page value, the
        rule and whether a page without a value deviates; None for stages that
        mark no page.
    """
    match stage:
        case FilterStage.DESKEW:
            return deskew_angle, settings.deskew_deviation, False
        case FilterStage.SELECT_CONTENT:
            # A page whose content box came out empty is worth a second look.
            return content_diagonal, settings.select_content_deviation, True
        case FilterStage.PAGE_LAYOUT:
            return hard_margins_diagonal, settings.margins_deviation, False
    return None


def deviation_provider(
    sequence: StageSequence,
    settings: Callable[[], ApplicationSettings] = ApplicationSettings.get_instance,
) -> DeviantPagesProvider:
    """Return the deviating-page finder the thumbnail list takes.

    The settings are read on every call, so a changed coefficient or the
    highlighting switch takes effect at the next refresh of the list.

    Args:
        sequence (StageSequence): The stages whose results are compared.
        settings (Callable[[], ApplicationSettings]): Returns the settings to use.

    Returns:
        DeviantPagesProvider: Takes the listed pages and a stage key, the
        integer value of a ``FilterStage`` as text.
    """

    def find(entries: Sequence[PageEntry], stage_key: str) -> frozenset[PageId]:
        current = settings()
        rule = stage_rule(FilterStage(int(stage_key)), current) if stage_key else None
        if not current.highlight_deviation or rule is None:
            return frozenset()
        value, limits, unknown_deviates = rule
        values = {
            entry.page: value(sequence, entry.page)
            for entry in entries
            if not entry.incomplete
        }
        return deviant_pages(values, limits, unknown_deviates)

    return find
