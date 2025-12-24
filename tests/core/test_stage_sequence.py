"""Tests for StageSequence."""

import pytest

from scantailor.core import FilterStage, StageSequence


class TestFilterStage:
    """Tests for FilterStage enum."""

    def test_stage_values(self) -> None:
        """Stage values are sequential from 0."""
        assert FilterStage.FIX_ORIENTATION == 0
        assert FilterStage.PAGE_SPLIT == 1
        assert FilterStage.DESKEW == 2
        assert FilterStage.SELECT_CONTENT == 3
        assert FilterStage.PAGE_LAYOUT == 4
        assert FilterStage.OUTPUT == 5

    def test_stage_count(self) -> None:
        """There are exactly 6 stages."""
        assert len(FilterStage) == 6


class TestStageSequence:
    """Tests for StageSequence class."""

    def test_default_creation(self) -> None:
        """Create with default (lazy) filters."""
        sequence = StageSequence()
        assert len(sequence) == 6

    def test_lazy_filter_initialization(self) -> None:
        """Filters are lazily initialized on first access."""
        sequence = StageSequence()
        # Access each filter to trigger lazy initialization
        assert sequence.fix_orientation_filter is not None
        assert sequence.page_split_filter is not None
        assert sequence.deskew_filter is not None
        assert sequence.select_content_filter is not None
        assert sequence.page_layout_filter is not None
        assert sequence.output_filter is not None

    def test_filters_property(self) -> None:
        """filters property returns all 6 filters in order."""
        sequence = StageSequence()
        filters = sequence.filters
        assert len(filters) == 6
        # Verify order
        assert filters[0].name == "Fix Orientation"
        assert filters[1].name == "Page Split"
        assert filters[2].name == "Deskew"
        assert filters[3].name == "Select Content"
        assert filters[4].name == "Page Layout"
        assert filters[5].name == "Output"

    def test_filter_at_by_enum(self) -> None:
        """Get filter by FilterStage enum."""
        sequence = StageSequence()
        assert sequence.filter_at(FilterStage.FIX_ORIENTATION).name == "Fix Orientation"
        assert sequence.filter_at(FilterStage.DESKEW).name == "Deskew"
        assert sequence.filter_at(FilterStage.OUTPUT).name == "Output"

    def test_filter_at_by_index(self) -> None:
        """Get filter by integer index."""
        sequence = StageSequence()
        assert sequence.filter_at(0).name == "Fix Orientation"
        assert sequence.filter_at(2).name == "Deskew"
        assert sequence.filter_at(5).name == "Output"

    def test_filter_at_out_of_range(self) -> None:
        """filter_at raises IndexError for invalid index."""
        sequence = StageSequence()
        with pytest.raises(IndexError):
            sequence.filter_at(-1)
        with pytest.raises(IndexError):
            sequence.filter_at(6)

    def test_find_filter(self) -> None:
        """find_filter returns correct stage for a filter."""
        sequence = StageSequence()
        deskew_filter = sequence.deskew_filter
        assert sequence.find_filter(deskew_filter) == FilterStage.DESKEW

    def test_find_filter_not_found(self) -> None:
        """find_filter returns None for unknown filter."""
        from scantailor.filters.deskew import Filter as DeskewFilter

        sequence = StageSequence()
        other_filter = DeskewFilter()  # Different instance
        assert sequence.find_filter(other_filter) is None

    def test_stage_name(self) -> None:
        """stage_name returns filter name."""
        sequence = StageSequence()
        assert sequence.stage_name(FilterStage.FIX_ORIENTATION) == "Fix Orientation"
        assert sequence.stage_name(FilterStage.PAGE_SPLIT) == "Page Split"
        assert sequence.stage_name(2) == "Deskew"

    def test_stage_names(self) -> None:
        """stage_names returns all filter names."""
        sequence = StageSequence()
        names = sequence.stage_names
        assert names == [
            "Fix Orientation",
            "Page Split",
            "Deskew",
            "Select Content",
            "Page Layout",
            "Output",
        ]

    def test_get_settings(self) -> None:
        """get_settings returns filter settings object."""
        sequence = StageSequence()
        settings = sequence.get_settings(FilterStage.DESKEW)
        assert settings is sequence.deskew_filter.settings

    def test_set_custom_filter(self) -> None:
        """Can set custom filter instance."""
        from scantailor.filters.deskew import Filter as DeskewFilter

        sequence = StageSequence()
        custom_filter = DeskewFilter()
        sequence.deskew_filter = custom_filter
        assert sequence.deskew_filter is custom_filter

    def test_filter_consistency(self) -> None:
        """Same filter instance returned on multiple accesses."""
        sequence = StageSequence()
        filter1 = sequence.deskew_filter
        filter2 = sequence.deskew_filter
        assert filter1 is filter2

    def test_len(self) -> None:
        """len() returns 6."""
        sequence = StageSequence()
        assert len(sequence) == 6
