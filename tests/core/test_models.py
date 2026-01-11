"""Tests for core data models.

These tests verify the fundamental data structures: Dpi, ImageId, PageId,
Margins, SubPage, and OrthogonalRotation.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scantailor.core import (
    Dpi,
    ImageId,
    Margins,
    OrthogonalRotation,
    PageId,
    SubPage,
)


class TestDpi:
    """Tests for the Dpi model."""

    def test_default_values_are_zero(self):
        dpi = Dpi()
        assert dpi.horizontal == 0
        assert dpi.vertical == 0

    def test_construction_with_values(self):
        dpi = Dpi(horizontal=300, vertical=600)
        assert dpi.horizontal == 300
        assert dpi.vertical == 600

    def test_uniform_creates_equal_values(self):
        dpi = Dpi.uniform(300)
        assert dpi.horizontal == 300
        assert dpi.vertical == 300

    def test_is_null_when_horizontal_is_zero(self):
        dpi = Dpi(horizontal=0, vertical=300)
        assert dpi.is_null() is True

    def test_is_null_when_vertical_is_zero(self):
        dpi = Dpi(horizontal=300, vertical=0)
        assert dpi.is_null() is True

    def test_is_null_when_both_are_one(self):
        dpi = Dpi(horizontal=1, vertical=1)
        assert dpi.is_null() is True

    def test_is_not_null_when_both_above_one(self):
        dpi = Dpi(horizontal=72, vertical=72)
        assert dpi.is_null() is False

    def test_negative_values_rejected(self):
        with pytest.raises(ValidationError):
            Dpi(horizontal=-1, vertical=300)

    def test_is_immutable(self):
        dpi = Dpi(horizontal=300, vertical=300)
        with pytest.raises(ValidationError):
            dpi.horizontal = 600  # type: ignore[misc]

    def test_equality(self):
        dpi1 = Dpi(horizontal=300, vertical=600)
        dpi2 = Dpi(horizontal=300, vertical=600)
        dpi3 = Dpi(horizontal=300, vertical=300)
        assert dpi1 == dpi2
        assert dpi1 != dpi3

    def test_json_round_trip(self):
        original = Dpi(horizontal=300, vertical=600)
        json_str = original.model_dump_json()
        restored = Dpi.model_validate_json(json_str)
        assert original == restored


class TestImageId:
    """Tests for the ImageId model."""

    def test_construction_with_path(self):
        image_id = ImageId(file_path=Path("/images/scan.tiff"))
        assert image_id.file_path == Path("/images/scan.tiff")
        assert image_id.page == 0

    def test_construction_with_page_number(self):
        image_id = ImageId(file_path=Path("/images/document.pdf"), page=5)
        assert image_id.file_path == Path("/images/document.pdf")
        assert image_id.page == 5

    def test_is_null_for_empty_path(self):
        image_id = ImageId(file_path=Path(""))
        assert image_id.is_null() is True

    def test_is_not_null_for_valid_path(self):
        image_id = ImageId(file_path=Path("/images/scan.tiff"))
        assert image_id.is_null() is False

    def test_is_multi_page_file_when_page_is_positive(self):
        image_id = ImageId(file_path=Path("/doc.pdf"), page=1)
        assert image_id.is_multi_page_file() is True

    def test_is_not_multi_page_file_when_page_is_zero(self):
        image_id = ImageId(file_path=Path("/scan.tiff"), page=0)
        assert image_id.is_multi_page_file() is False

    def test_zero_based_page_for_single_page(self):
        image_id = ImageId(file_path=Path("/scan.tiff"), page=0)
        assert image_id.zero_based_page() == 0

    def test_zero_based_page_for_multi_page(self):
        image_id = ImageId(file_path=Path("/doc.pdf"), page=3)
        assert image_id.zero_based_page() == 2

    def test_negative_page_rejected(self):
        with pytest.raises(ValidationError):
            ImageId(file_path=Path("/scan.tiff"), page=-1)

    def test_is_hashable(self):
        image_id = ImageId(file_path=Path("/scan.tiff"), page=0)
        assert hash(image_id) is not None
        image_set = {image_id}
        assert image_id in image_set

    def test_is_immutable(self):
        image_id = ImageId(file_path=Path("/scan.tiff"), page=0)
        with pytest.raises(ValidationError):
            image_id.page = 1  # type: ignore[misc]

    def test_json_round_trip(self):
        original = ImageId(file_path=Path("/images/scan.tiff"), page=3)
        json_str = original.model_dump_json()
        restored = ImageId.model_validate_json(json_str)
        assert original == restored


class TestSubPage:
    """Tests for the SubPage enum."""

    def test_single_page_value(self):
        assert SubPage.SINGLE_PAGE == 0

    def test_left_page_value(self):
        assert SubPage.LEFT_PAGE == 1

    def test_right_page_value(self):
        assert SubPage.RIGHT_PAGE == 2

    def test_string_representation_single(self):
        assert str(SubPage.SINGLE_PAGE) == "single-page"

    def test_string_representation_left(self):
        assert str(SubPage.LEFT_PAGE) == "left-page"

    def test_string_representation_right(self):
        assert str(SubPage.RIGHT_PAGE) == "right-page"

    def test_from_string_single(self):
        assert SubPage.from_string("single-page") == SubPage.SINGLE_PAGE

    def test_from_string_left(self):
        assert SubPage.from_string("left-page") == SubPage.LEFT_PAGE

    def test_from_string_right(self):
        assert SubPage.from_string("right-page") == SubPage.RIGHT_PAGE

    def test_from_string_uppercase(self):
        assert SubPage.from_string("SINGLE-PAGE") == SubPage.SINGLE_PAGE

    def test_from_string_invalid_raises(self):
        with pytest.raises(KeyError):
            SubPage.from_string("invalid")


class TestPageId:
    """Tests for the PageId model."""

    def test_construction(self):
        image_id = ImageId(file_path=Path("/scan.tiff"))
        page_id = PageId(image_id=image_id, sub_page=SubPage.SINGLE_PAGE)
        assert page_id.image_id == image_id
        assert page_id.sub_page == SubPage.SINGLE_PAGE

    def test_default_sub_page_is_single(self):
        image_id = ImageId(file_path=Path("/scan.tiff"))
        page_id = PageId(image_id=image_id)
        assert page_id.sub_page == SubPage.SINGLE_PAGE

    def test_is_null_when_image_id_is_null(self):
        image_id = ImageId(file_path=Path(""))
        page_id = PageId(image_id=image_id)
        assert page_id.is_null() is True

    def test_is_not_null_when_image_id_is_valid(self):
        image_id = ImageId(file_path=Path("/scan.tiff"))
        page_id = PageId(image_id=image_id)
        assert page_id.is_null() is False

    def test_is_hashable(self):
        image_id = ImageId(file_path=Path("/scan.tiff"))
        page_id = PageId(image_id=image_id, sub_page=SubPage.LEFT_PAGE)
        assert hash(page_id) is not None
        page_set = {page_id}
        assert page_id in page_set

    def test_ordering_by_file_path(self):
        page1 = PageId(image_id=ImageId(file_path=Path("/a.tiff")))
        page2 = PageId(image_id=ImageId(file_path=Path("/b.tiff")))
        assert page1 < page2

    def test_ordering_by_page_number(self):
        page1 = PageId(image_id=ImageId(file_path=Path("/doc.pdf"), page=1))
        page2 = PageId(image_id=ImageId(file_path=Path("/doc.pdf"), page=2))
        assert page1 < page2

    def test_ordering_by_sub_page(self):
        image_id = ImageId(file_path=Path("/scan.tiff"))
        page1 = PageId(image_id=image_id, sub_page=SubPage.LEFT_PAGE)
        page2 = PageId(image_id=image_id, sub_page=SubPage.RIGHT_PAGE)
        assert page1 < page2

    def test_json_round_trip(self):
        original = PageId(
            image_id=ImageId(file_path=Path("/scan.tiff"), page=2),
            sub_page=SubPage.LEFT_PAGE,
        )
        json_str = original.model_dump_json()
        restored = PageId.model_validate_json(json_str)
        assert original == restored


class TestMargins:
    """Tests for the Margins model."""

    def test_default_values_are_zero(self):
        margins = Margins()
        assert margins.top == 0.0
        assert margins.bottom == 0.0
        assert margins.left == 0.0
        assert margins.right == 0.0

    def test_construction_with_values(self):
        margins = Margins(top=10.0, bottom=20.0, left=5.0, right=15.0)
        assert margins.top == 10.0
        assert margins.bottom == 20.0
        assert margins.left == 5.0
        assert margins.right == 15.0

    def test_uniform_creates_equal_values(self):
        margins = Margins.uniform(10.0)
        assert margins.top == 10.0
        assert margins.bottom == 10.0
        assert margins.left == 10.0
        assert margins.right == 10.0

    def test_is_mutable(self):
        margins = Margins(top=10.0)
        margins.top = 20.0
        assert margins.top == 20.0

    def test_json_round_trip(self):
        original = Margins(top=10.0, bottom=20.0, left=5.0, right=15.0)
        json_str = original.model_dump_json()
        restored = Margins.model_validate_json(json_str)
        assert original == restored


class TestOrthogonalRotation:
    """Tests for the OrthogonalRotation model."""

    def test_default_is_zero_degrees(self):
        rotation = OrthogonalRotation()
        assert rotation.degrees == 0

    def test_construction_with_valid_values(self):
        for degrees in [0, 90, 180, 270]:
            rotation = OrthogonalRotation(degrees=degrees)
            assert rotation.degrees == degrees

    def test_invalid_degrees_raises(self):
        with pytest.raises(ValidationError):
            OrthogonalRotation(degrees=45)

    def test_rotate_clockwise_from_zero(self):
        rotation = OrthogonalRotation(degrees=0)
        rotated = rotation.rotate_clockwise()
        assert rotated.degrees == 90

    def test_rotate_clockwise_from_270(self):
        rotation = OrthogonalRotation(degrees=270)
        rotated = rotation.rotate_clockwise()
        assert rotated.degrees == 0

    def test_rotate_counter_clockwise_from_zero(self):
        rotation = OrthogonalRotation(degrees=0)
        rotated = rotation.rotate_counter_clockwise()
        assert rotated.degrees == 270

    def test_rotate_counter_clockwise_from_90(self):
        rotation = OrthogonalRotation(degrees=90)
        rotated = rotation.rotate_counter_clockwise()
        assert rotated.degrees == 0

    def test_rotate_dimensions_at_zero_degrees(self):
        rotation = OrthogonalRotation(degrees=0)
        width, height = rotation.rotate_dimensions(100.0, 200.0)
        assert width == 100.0
        assert height == 200.0

    def test_rotate_dimensions_at_90_degrees(self):
        rotation = OrthogonalRotation(degrees=90)
        width, height = rotation.rotate_dimensions(100.0, 200.0)
        assert width == 200.0
        assert height == 100.0

    def test_rotate_dimensions_at_180_degrees(self):
        rotation = OrthogonalRotation(degrees=180)
        width, height = rotation.rotate_dimensions(100.0, 200.0)
        assert width == 100.0
        assert height == 200.0

    def test_rotate_dimensions_at_270_degrees(self):
        rotation = OrthogonalRotation(degrees=270)
        width, height = rotation.rotate_dimensions(100.0, 200.0)
        assert width == 200.0
        assert height == 100.0

    def test_unrotate_dimensions_reverses_rotation(self):
        rotation = OrthogonalRotation(degrees=90)
        width, height = rotation.rotate_dimensions(100.0, 200.0)
        original_w, original_h = rotation.unrotate_dimensions(width, height)
        assert original_w == 100.0
        assert original_h == 200.0

    def test_rotate_point_at_zero_degrees(self):
        rotation = OrthogonalRotation(degrees=0)
        x, y = rotation.rotate_point(10.0, 20.0, 100.0, 200.0)
        assert x == 10.0
        assert y == 20.0

    def test_rotate_point_at_90_degrees(self):
        rotation = OrthogonalRotation(degrees=90)
        x, y = rotation.rotate_point(10.0, 20.0, 100.0, 200.0)
        assert x == 180.0  # max_y - y = 200 - 20
        assert y == 10.0  # x

    def test_rotate_point_at_180_degrees(self):
        rotation = OrthogonalRotation(degrees=180)
        x, y = rotation.rotate_point(10.0, 20.0, 100.0, 200.0)
        assert x == 90.0  # max_x - x = 100 - 10
        assert y == 180.0  # max_y - y = 200 - 20

    def test_rotate_point_at_270_degrees(self):
        rotation = OrthogonalRotation(degrees=270)
        x, y = rotation.rotate_point(10.0, 20.0, 100.0, 200.0)
        assert x == 20.0  # y
        assert y == 90.0  # max_x - x = 100 - 10

    def test_unrotate_point_reverses_rotation_90_degrees(self):
        """Test that unrotating a 90° rotated point returns the original.

        Original: point (10, 20) in bounds (100, 200)
        After 90° CW: (max_y - y, x) = (180, 10), new bounds (200, 100)
        Unrotate with inverse (270°): should get back (10, 20)
        """
        rotation = OrthogonalRotation(degrees=90)
        # Original point and bounds
        orig_x, orig_y = 10.0, 20.0
        orig_max_x, orig_max_y = 100.0, 200.0

        # After rotation, bounds swap for 90° rotation
        rotated_x, rotated_y = rotation.rotate_point(
            orig_x, orig_y, orig_max_x, orig_max_y
        )
        rotated_max_x, rotated_max_y = orig_max_y, orig_max_x

        # Unrotate using the rotated bounds
        recovered_x, recovered_y = rotation.unrotate_point(
            rotated_x, rotated_y, rotated_max_x, rotated_max_y
        )
        assert abs(recovered_x - orig_x) < 0.001
        assert abs(recovered_y - orig_y) < 0.001

    def test_unrotate_point_reverses_rotation_180_degrees(self):
        """Test that unrotating a 180° rotated point returns the original."""
        rotation = OrthogonalRotation(degrees=180)
        orig_x, orig_y = 10.0, 20.0
        max_x, max_y = 100.0, 200.0

        rotated_x, rotated_y = rotation.rotate_point(orig_x, orig_y, max_x, max_y)
        # Bounds don't swap for 180°
        recovered_x, recovered_y = rotation.unrotate_point(
            rotated_x, rotated_y, max_x, max_y
        )
        assert abs(recovered_x - orig_x) < 0.001
        assert abs(recovered_y - orig_y) < 0.001

    def test_json_round_trip(self):
        original = OrthogonalRotation(degrees=180)
        json_str = original.model_dump_json()
        restored = OrthogonalRotation.model_validate_json(json_str)
        assert original == restored

    def test_equality(self):
        rot1 = OrthogonalRotation(degrees=90)
        rot2 = OrthogonalRotation(degrees=90)
        rot3 = OrthogonalRotation(degrees=180)
        assert rot1 == rot2
        assert rot1 != rot3
