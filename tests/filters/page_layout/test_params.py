"""Tests for Page Layout parameters."""

from scantailor.core.models import Margins
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.params import ContentRect, ContentSize, Params


class TestContentRect:
    """Tests for ContentRect class."""

    def test_default_values(self):
        """Default values should be zero."""
        rect = ContentRect()
        assert rect.x == 0.0
        assert rect.y == 0.0
        assert rect.width == 0.0
        assert rect.height == 0.0

    def test_is_empty_for_zero_size(self):
        """is_empty should return True for zero dimensions."""
        assert ContentRect().is_empty() is True
        assert ContentRect(width=0, height=10).is_empty() is True
        assert ContentRect(width=10, height=0).is_empty() is True

    def test_is_empty_for_valid_size(self):
        """is_empty should return False for positive dimensions."""
        rect = ContentRect(x=5, y=10, width=100, height=200)
        assert rect.is_empty() is False


class TestContentSize:
    """Tests for ContentSize class."""

    def test_default_values(self):
        """Default values should be zero."""
        size = ContentSize()
        assert size.width_mm == 0.0
        assert size.height_mm == 0.0

    def test_is_empty_for_zero_size(self):
        """is_empty should return True for zero dimensions."""
        assert ContentSize().is_empty() is True
        assert ContentSize(width_mm=0, height_mm=10).is_empty() is True

    def test_is_empty_for_valid_size(self):
        """is_empty should return False for positive dimensions."""
        size = ContentSize(width_mm=100, height_mm=150)
        assert size.is_empty() is False


class TestParams:
    """Tests for Params class."""

    def test_default_values(self):
        """Default values should have zero margins and centered alignment."""
        params = Params()
        assert params.hard_margins_mm.top == 0.0
        assert params.alignment.aligned_with_others() is True
        assert params.auto_margins is True

    def test_hard_width_mm(self):
        """hard_width_mm should sum content width and horizontal margins."""
        params = Params(
            hard_margins_mm=Margins(left=10, right=15),
            content_size_mm=ContentSize(width_mm=100),
        )
        assert params.hard_width_mm() == 125.0  # 10 + 100 + 15

    def test_hard_height_mm(self):
        """hard_height_mm should sum content height and vertical margins."""
        params = Params(
            hard_margins_mm=Margins(top=5, bottom=10),
            content_size_mm=ContentSize(height_mm=200),
        )
        assert params.hard_height_mm() == 215.0  # 5 + 200 + 10

    def test_with_margins(self):
        """with_margins should return new params with updated margins."""
        original = Params(auto_margins=True)
        new_margins = Margins.uniform(10)

        updated = original.with_margins(new_margins)

        assert updated.hard_margins_mm == new_margins
        assert updated.auto_margins is False  # Should disable auto
        assert original.auto_margins is True  # Original unchanged

    def test_with_alignment(self):
        """with_alignment should return new params with updated alignment."""
        original = Params()
        new_alignment = Alignment.top_left()

        updated = original.with_alignment(new_alignment)

        assert updated.alignment == new_alignment
        assert original.alignment != new_alignment  # Original unchanged

    def test_with_auto_margins(self):
        """with_auto_margins should return new params with flag set."""
        original = Params(auto_margins=False)

        updated = original.with_auto_margins(True)

        assert updated.auto_margins is True
        assert original.auto_margins is False

    def test_json_round_trip(self):
        """Should serialize and deserialize correctly."""
        original = Params(
            hard_margins_mm=Margins(top=5, bottom=10, left=15, right=20),
            alignment=Alignment.top_left(),
            content_size_mm=ContentSize(width_mm=100, height_mm=150),
            auto_margins=False,
        )

        json_str = original.model_dump_json()
        restored = Params.model_validate_json(json_str)

        assert restored.hard_margins_mm.top == 5
        assert restored.hard_margins_mm.left == 15
        assert restored.alignment.vertical == original.alignment.vertical
        assert restored.content_size_mm.width_mm == 100
        assert restored.auto_margins is False
