"""Tests for content and page box classes."""

import pytest

from scantailor.filters.select_content import ContentBox, PageBox, PhysicalSize


class TestContentBox:
    """Tests for ContentBox class."""

    def test_default_values(self):
        """Default box should be at origin with zero size."""
        box = ContentBox()
        assert box.x == 0.0
        assert box.y == 0.0
        assert box.width == 0.0
        assert box.height == 0.0

    def test_properties(self):
        """Properties should calculate correctly."""
        box = ContentBox(x=10.0, y=20.0, width=100.0, height=50.0)
        assert box.left == 10.0
        assert box.top == 20.0
        assert box.right == 110.0
        assert box.bottom == 70.0
        assert box.center_x == 60.0
        assert box.center_y == 45.0

    def test_is_empty(self):
        """is_empty should return True for zero-dimension boxes."""
        assert ContentBox().is_empty()
        assert ContentBox(width=0).is_empty()
        assert ContentBox(height=0).is_empty()
        assert ContentBox(width=-1).is_empty()
        assert not ContentBox(x=0, y=0, width=1, height=1).is_empty()

    def test_is_valid(self):
        """is_valid should return True for positive-dimension boxes."""
        assert not ContentBox().is_valid()
        assert ContentBox(x=0, y=0, width=10, height=20).is_valid()

    def test_contains(self):
        """contains should detect point inside box."""
        box = ContentBox(x=10, y=20, width=100, height=50)
        assert box.contains(50, 40)
        assert box.contains(10, 20)  # Edge
        assert box.contains(110, 70)  # Edge
        assert not box.contains(5, 40)  # Left of box
        assert not box.contains(50, 10)  # Above box

    def test_intersects(self):
        """intersects should detect overlapping boxes."""
        box1 = ContentBox(x=0, y=0, width=100, height=100)
        box2 = ContentBox(x=50, y=50, width=100, height=100)
        box3 = ContentBox(x=200, y=200, width=50, height=50)

        assert box1.intersects(box2)
        assert box2.intersects(box1)
        assert not box1.intersects(box3)

    def test_intersection(self):
        """intersection should return overlapping area."""
        box1 = ContentBox(x=0, y=0, width=100, height=100)
        box2 = ContentBox(x=50, y=50, width=100, height=100)

        result = box1.intersection(box2)
        assert result.x == 50
        assert result.y == 50
        assert result.width == 50
        assert result.height == 50

    def test_intersection_no_overlap(self):
        """intersection with no overlap should return empty box."""
        box1 = ContentBox(x=0, y=0, width=50, height=50)
        box2 = ContentBox(x=100, y=100, width=50, height=50)

        result = box1.intersection(box2)
        assert result.is_empty()

    def test_union(self):
        """union should return bounding box of both."""
        box1 = ContentBox(x=0, y=0, width=50, height=50)
        box2 = ContentBox(x=100, y=100, width=50, height=50)

        result = box1.union(box2)
        assert result.x == 0
        assert result.y == 0
        assert result.width == 150
        assert result.height == 150

    def test_union_with_empty(self):
        """union with empty box should return the other box."""
        box = ContentBox(x=10, y=20, width=30, height=40)
        empty = ContentBox()

        assert box.union(empty).to_tuple() == box.to_tuple()
        assert empty.union(box).to_tuple() == box.to_tuple()

    def test_expanded(self):
        """expanded should add margin on all sides."""
        box = ContentBox(x=50, y=50, width=100, height=100)
        result = box.expanded(10)

        assert result.x == 40
        assert result.y == 40
        assert result.width == 120
        assert result.height == 120

    def test_to_tuple(self):
        """to_tuple should return (x, y, width, height)."""
        box = ContentBox(x=1, y=2, width=3, height=4)
        assert box.to_tuple() == (1, 2, 3, 4)

    def test_from_ltrb(self):
        """from_ltrb should create box from edges."""
        box = ContentBox.from_ltrb(10, 20, 110, 70)
        assert box.x == 10
        assert box.y == 20
        assert box.width == 100
        assert box.height == 50


class TestPageBox:
    """Tests for PageBox class."""

    def test_default_values(self):
        """Default box should be at origin with zero size."""
        box = PageBox()
        assert box.x == 0.0
        assert box.y == 0.0
        assert box.width == 0.0
        assert box.height == 0.0

    def test_properties(self):
        """Properties should calculate correctly."""
        box = PageBox(x=10, y=20, width=100, height=50)
        assert box.left == 10
        assert box.top == 20
        assert box.right == 110
        assert box.bottom == 70

    def test_is_empty(self):
        """is_empty should return True for zero-dimension boxes."""
        assert PageBox().is_empty()
        assert not PageBox(x=0, y=0, width=1, height=1).is_empty()

    def test_to_content_box(self):
        """to_content_box should convert to ContentBox."""
        page_box = PageBox(x=10, y=20, width=100, height=50)
        content_box = page_box.to_content_box()

        assert isinstance(content_box, ContentBox)
        assert content_box.x == 10
        assert content_box.y == 20
        assert content_box.width == 100
        assert content_box.height == 50

    def test_from_image_size(self):
        """from_image_size should create box covering full image."""
        box = PageBox.from_image_size(800, 600)
        assert box.x == 0
        assert box.y == 0
        assert box.width == 800
        assert box.height == 600

    def test_from_ltrb(self):
        """from_ltrb should create box from edges."""
        box = PageBox.from_ltrb(10, 20, 110, 70)
        assert box.x == 10
        assert box.y == 20
        assert box.width == 100
        assert box.height == 50


class TestPhysicalSize:
    """Tests for PhysicalSize class."""

    def test_default_values(self):
        """Default size should be zero."""
        size = PhysicalSize()
        assert size.width_mm == 0.0
        assert size.height_mm == 0.0

    def test_is_empty(self):
        """is_empty should return True for zero size."""
        assert PhysicalSize().is_empty()
        assert not PhysicalSize(width_mm=10, height_mm=20).is_empty()

    def test_from_pixels_300dpi(self):
        """from_pixels at 300 DPI should convert correctly."""
        # 300 DPI means 300 pixels per inch
        # 1 inch = 25.4 mm
        # So 300 pixels = 25.4 mm
        size = PhysicalSize.from_pixels(300, 300, dpi=300.0)
        assert pytest.approx(size.width_mm, 0.01) == 25.4
        assert pytest.approx(size.height_mm, 0.01) == 25.4

    def test_from_pixels_different_dpi(self):
        """from_pixels should scale with DPI."""
        size_300 = PhysicalSize.from_pixels(600, 600, dpi=300.0)
        size_600 = PhysicalSize.from_pixels(600, 600, dpi=600.0)

        # At 600 DPI, same pixels = half the physical size
        assert pytest.approx(size_300.width_mm, 0.01) == size_600.width_mm * 2

    def test_validation_rejects_negative(self):
        """Negative values should be rejected."""
        with pytest.raises(ValueError):
            PhysicalSize(width_mm=-1)
        with pytest.raises(ValueError):
            PhysicalSize(height_mm=-1)
