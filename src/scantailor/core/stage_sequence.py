"""Stage sequence for the ScanTailor processing pipeline.

This module provides the StageSequence class that orchestrates the 6-stage
filter pipeline: Fix Orientation -> Page Split -> Deskew -> Select Content
-> Page Layout -> Output.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from scantailor.filters.deskew import Filter as DeskewFilter
    from scantailor.filters.fix_orientation import Filter as FixOrientationFilter
    from scantailor.filters.output import Filter as OutputFilter
    from scantailor.filters.page_layout import Filter as PageLayoutFilter
    from scantailor.filters.page_split import Filter as PageSplitFilter
    from scantailor.filters.select_content import Filter as SelectContentFilter


class FilterStage(IntEnum):
    """Enumeration of processing stages in order."""

    FIX_ORIENTATION = 0
    PAGE_SPLIT = 1
    DESKEW = 2
    SELECT_CONTENT = 3
    PAGE_LAYOUT = 4
    OUTPUT = 5


@runtime_checkable
class FilterProtocol(Protocol):
    """Protocol for filter objects.

    All filters must have a name attribute and a settings property.
    """

    name: str

    @property
    def settings(self) -> object: ...


@dataclass
class StageSequence:
    """Orchestrates the 6-stage filter pipeline.

    The ScanTailor processing pipeline consists of 6 stages that must be
    applied in order:

    1. Fix Orientation - Apply 0/90/180/270 degree rotation
    2. Page Split - Split two-page spreads into individual pages
    3. Deskew - Correct page skew angle
    4. Select Content - Detect and select page content area
    5. Page Layout - Set margins and alignment
    6. Output - Generate final output with binarization

    Example:
        >>> from scantailor.core import StageSequence
        >>> from scantailor.filters.fix_orientation import Filter as FixOrientationFilter
        >>> from scantailor.filters.page_split import Filter as PageSplitFilter
        >>> # ... import other filters
        >>>
        >>> sequence = StageSequence()
        >>> sequence.fix_orientation_filter = FixOrientationFilter()
        >>> sequence.page_split_filter = PageSplitFilter()
        >>> # ... set other filters
        >>>
        >>> # Get filter by stage
        >>> filter = sequence.filter_at(FilterStage.DESKEW)
    """

    # Individual filter instances (lazy initialization supported)
    _fix_orientation_filter: FixOrientationFilter | None = field(default=None)
    _page_split_filter: PageSplitFilter | None = field(default=None)
    _deskew_filter: DeskewFilter | None = field(default=None)
    _select_content_filter: SelectContentFilter | None = field(default=None)
    _page_layout_filter: PageLayoutFilter | None = field(default=None)
    _output_filter: OutputFilter | None = field(default=None)

    @property
    def fix_orientation_filter(self) -> FixOrientationFilter:
        """Get the Fix Orientation filter."""
        if self._fix_orientation_filter is None:
            from scantailor.filters.fix_orientation import Filter

            self._fix_orientation_filter = Filter()
        return self._fix_orientation_filter

    @fix_orientation_filter.setter
    def fix_orientation_filter(self, value: FixOrientationFilter) -> None:
        """Set the Fix Orientation filter."""
        self._fix_orientation_filter = value

    @property
    def page_split_filter(self) -> PageSplitFilter:
        """Get the Page Split filter."""
        if self._page_split_filter is None:
            from scantailor.filters.page_split import Filter

            self._page_split_filter = Filter()
        return self._page_split_filter

    @page_split_filter.setter
    def page_split_filter(self, value: PageSplitFilter) -> None:
        """Set the Page Split filter."""
        self._page_split_filter = value

    @property
    def deskew_filter(self) -> DeskewFilter:
        """Get the Deskew filter."""
        if self._deskew_filter is None:
            from scantailor.filters.deskew import Filter

            self._deskew_filter = Filter()
        return self._deskew_filter

    @deskew_filter.setter
    def deskew_filter(self, value: DeskewFilter) -> None:
        """Set the Deskew filter."""
        self._deskew_filter = value

    @property
    def select_content_filter(self) -> SelectContentFilter:
        """Get the Select Content filter."""
        if self._select_content_filter is None:
            from scantailor.filters.select_content import Filter

            self._select_content_filter = Filter()
        return self._select_content_filter

    @select_content_filter.setter
    def select_content_filter(self, value: SelectContentFilter) -> None:
        """Set the Select Content filter."""
        self._select_content_filter = value

    @property
    def page_layout_filter(self) -> PageLayoutFilter:
        """Get the Page Layout filter."""
        if self._page_layout_filter is None:
            from scantailor.filters.page_layout import Filter

            self._page_layout_filter = Filter()
        return self._page_layout_filter

    @page_layout_filter.setter
    def page_layout_filter(self, value: PageLayoutFilter) -> None:
        """Set the Page Layout filter."""
        self._page_layout_filter = value

    @property
    def output_filter(self) -> OutputFilter:
        """Get the Output filter."""
        if self._output_filter is None:
            from scantailor.filters.output import Filter

            self._output_filter = Filter()
        return self._output_filter

    @output_filter.setter
    def output_filter(self, value: OutputFilter) -> None:
        """Set the Output filter."""
        self._output_filter = value

    @property
    def filters(self) -> list[FilterProtocol]:
        """Get all filters in pipeline order.

        Returns:
            List of all 6 filters in processing order.
        """
        return cast(
            list[FilterProtocol],
            [
                self.fix_orientation_filter,
                self.page_split_filter,
                self.deskew_filter,
                self.select_content_filter,
                self.page_layout_filter,
                self.output_filter,
            ],
        )

    def __len__(self) -> int:
        """Return the number of stages (always 6)."""
        return len(FilterStage)

    def filter_at(self, stage: FilterStage | int) -> FilterProtocol:
        """Get the filter at a specific stage.

        Args:
            stage: The stage index or FilterStage enum.

        Returns:
            The filter for that stage.

        Raises:
            IndexError: If stage is out of range.
        """
        stage_idx = int(stage)
        if stage_idx < 0 or stage_idx >= len(FilterStage):
            raise IndexError(f"Stage index {stage_idx} out of range (0-5)")

        return self.filters[stage_idx]

    def find_filter(self, filter_obj: FilterProtocol) -> FilterStage | None:
        """Find the stage for a given filter instance.

        Args:
            filter_obj: The filter to find.

        Returns:
            The FilterStage if found, None otherwise.
        """
        for stage in FilterStage:
            if self.filter_at(stage) is filter_obj:
                return stage
        return None

    def stage_name(self, stage: FilterStage | int) -> str:
        """Get the human-readable name of a stage.

        Args:
            stage: The stage index or FilterStage enum.

        Returns:
            The filter's name attribute.
        """
        return self.filter_at(stage).name

    @property
    def stage_names(self) -> list[str]:
        """Get all stage names in order.

        Returns:
            List of 6 filter names.
        """
        return [f.name for f in self.filters]

    def get_settings(self, stage: FilterStage | int) -> object:
        """Get the settings object for a specific stage.

        Args:
            stage: The stage index or FilterStage enum.

        Returns:
            The filter's settings object.
        """
        return self.filter_at(stage).settings


# Convenience constants for stage indices
FIX_ORIENTATION_IDX = FilterStage.FIX_ORIENTATION
PAGE_SPLIT_IDX = FilterStage.PAGE_SPLIT
DESKEW_IDX = FilterStage.DESKEW
SELECT_CONTENT_IDX = FilterStage.SELECT_CONTENT
PAGE_LAYOUT_IDX = FilterStage.PAGE_LAYOUT
OUTPUT_IDX = FilterStage.OUTPUT
