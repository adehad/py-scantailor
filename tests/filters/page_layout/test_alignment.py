"""Tests for Page Layout alignment types."""

from scantailor.filters.page_layout import (
    Alignment,
    HorizontalAlignment,
    VerticalAlignment,
)


class TestVerticalAlignment:
    """Tests for VerticalAlignment enum."""

    def test_values(self):
        """All vertical alignments should have correct values."""
        assert VerticalAlignment.TOP.value == "top"
        assert VerticalAlignment.VCENTER.value == "vcenter"
        assert VerticalAlignment.BOTTOM.value == "bottom"
        assert VerticalAlignment.VAUTO.value == "vauto"
        assert VerticalAlignment.VORIGINAL.value == "voriginal"


class TestHorizontalAlignment:
    """Tests for HorizontalAlignment enum."""

    def test_values(self):
        """All horizontal alignments should have correct values."""
        assert HorizontalAlignment.LEFT.value == "left"
        assert HorizontalAlignment.HCENTER.value == "hcenter"
        assert HorizontalAlignment.RIGHT.value == "right"
        assert HorizontalAlignment.HAUTO.value == "hauto"
        assert HorizontalAlignment.HORIGINAL.value == "horiginal"


class TestAlignment:
    """Tests for Alignment class."""

    def test_default_is_centered(self):
        """Default alignment should be centered."""
        alignment = Alignment()
        assert alignment.vertical == VerticalAlignment.VCENTER
        assert alignment.horizontal == HorizontalAlignment.HCENTER
        assert alignment.is_null is False

    def test_centered_factory(self):
        """centered() should create centered alignment."""
        alignment = Alignment.centered()
        assert alignment.vertical == VerticalAlignment.VCENTER
        assert alignment.horizontal == HorizontalAlignment.HCENTER

    def test_top_left_factory(self):
        """top_left() should create top-left alignment."""
        alignment = Alignment.top_left()
        assert alignment.vertical == VerticalAlignment.TOP
        assert alignment.horizontal == HorizontalAlignment.LEFT

    def test_independent_factory(self):
        """independent() should create null alignment."""
        alignment = Alignment.independent()
        assert alignment.is_null is True

    def test_aligned_with_others(self):
        """aligned_with_others should return opposite of is_null."""
        normal = Alignment()
        independent = Alignment.independent()

        assert normal.aligned_with_others() is True
        assert independent.aligned_with_others() is False

    def test_is_auto_vertical(self):
        """is_auto_vertical should detect VAUTO."""
        auto = Alignment(vertical=VerticalAlignment.VAUTO)
        normal = Alignment(vertical=VerticalAlignment.VCENTER)

        assert auto.is_auto_vertical() is True
        assert normal.is_auto_vertical() is False

    def test_is_auto_horizontal(self):
        """is_auto_horizontal should detect HAUTO."""
        auto = Alignment(horizontal=HorizontalAlignment.HAUTO)
        normal = Alignment(horizontal=HorizontalAlignment.HCENTER)

        assert auto.is_auto_horizontal() is True
        assert normal.is_auto_horizontal() is False

    def test_is_original_vertical(self):
        """is_original_vertical should detect VORIGINAL."""
        original = Alignment(vertical=VerticalAlignment.VORIGINAL)
        normal = Alignment(vertical=VerticalAlignment.VCENTER)

        assert original.is_original_vertical() is True
        assert normal.is_original_vertical() is False

    def test_is_original_horizontal(self):
        """is_original_horizontal should detect HORIGINAL."""
        original = Alignment(horizontal=HorizontalAlignment.HORIGINAL)
        normal = Alignment(horizontal=HorizontalAlignment.HCENTER)

        assert original.is_original_horizontal() is True
        assert normal.is_original_horizontal() is False

    def test_json_round_trip(self):
        """Should serialize and deserialize correctly."""
        original = Alignment(
            vertical=VerticalAlignment.TOP,
            horizontal=HorizontalAlignment.RIGHT,
            is_null=False,
        )

        json_str = original.model_dump_json()
        restored = Alignment.model_validate_json(json_str)

        assert restored.vertical == original.vertical
        assert restored.horizontal == original.horizontal
        assert restored.is_null == original.is_null
