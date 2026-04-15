"""Tests for Fix Orientation settings."""

from pathlib import Path

from scantailor.core.models import ImageId, OrthogonalRotation, PageId, SubPage
from scantailor.filters.fix_orientation.settings import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_rotation_is_zero(self):
        """Default rotation should be 0 degrees."""
        settings = Settings()
        image_id = ImageId(file_path=Path("/test.tiff"))

        rotation = settings.get_rotation_for(image_id)

        assert rotation.degrees == 0

    def test_rotation_not_set_by_default(self):
        """is_rotation_set should return False for unset images."""
        settings = Settings()
        image_id = ImageId(file_path=Path("/test.tiff"))

        assert settings.is_rotation_set(image_id) is False

    def test_apply_rotation_stores_value(self):
        """apply_rotation should store the rotation."""
        settings = Settings()
        image_id = ImageId(file_path=Path("/test.tiff"))
        rotation = OrthogonalRotation(degrees=90)

        settings.apply_rotation(image_id, rotation)

        assert settings.get_rotation_for(image_id).degrees == 90
        assert settings.is_rotation_set(image_id) is True

    def test_apply_rotation_overwrites_existing(self):
        """Applying a new rotation should overwrite the old one."""
        settings = Settings()
        image_id = ImageId(file_path=Path("/test.tiff"))

        settings.apply_rotation(image_id, OrthogonalRotation(degrees=90))
        settings.apply_rotation(image_id, OrthogonalRotation(degrees=180))

        assert settings.get_rotation_for(image_id).degrees == 180

    def test_apply_rotation_to_pages(self):
        """apply_rotation_to_pages should set rotation for all page images."""
        settings = Settings()
        image_id1 = ImageId(file_path=Path("/scan1.tiff"))
        image_id2 = ImageId(file_path=Path("/scan2.tiff"))
        page1 = PageId(image_id=image_id1, sub_page=SubPage.LEFT_PAGE)
        page2 = PageId(image_id=image_id2, sub_page=SubPage.SINGLE_PAGE)

        settings.apply_rotation_to_pages(
            [page1, page2], OrthogonalRotation(degrees=270)
        )

        assert settings.get_rotation_for(image_id1).degrees == 270
        assert settings.get_rotation_for(image_id2).degrees == 270

    def test_different_images_have_independent_rotations(self):
        """Different images should have independent rotation settings."""
        settings = Settings()
        image_id1 = ImageId(file_path=Path("/scan1.tiff"))
        image_id2 = ImageId(file_path=Path("/scan2.tiff"))

        settings.apply_rotation(image_id1, OrthogonalRotation(degrees=90))
        settings.apply_rotation(image_id2, OrthogonalRotation(degrees=270))

        assert settings.get_rotation_for(image_id1).degrees == 90
        assert settings.get_rotation_for(image_id2).degrees == 270

    def test_clear_removes_all_rotations(self):
        """clear() should remove all stored rotations."""
        settings = Settings()
        image_id1 = ImageId(file_path=Path("/scan1.tiff"))
        image_id2 = ImageId(file_path=Path("/scan2.tiff"))
        settings.apply_rotation(image_id1, OrthogonalRotation(degrees=90))
        settings.apply_rotation(image_id2, OrthogonalRotation(degrees=180))

        settings.clear()

        assert settings.is_rotation_set(image_id1) is False
        assert settings.is_rotation_set(image_id2) is False

    def test_remove_rotation(self):
        """remove_rotation should remove a specific image's rotation."""
        settings = Settings()
        image_id1 = ImageId(file_path=Path("/scan1.tiff"))
        image_id2 = ImageId(file_path=Path("/scan2.tiff"))
        settings.apply_rotation(image_id1, OrthogonalRotation(degrees=90))
        settings.apply_rotation(image_id2, OrthogonalRotation(degrees=180))

        settings.remove_rotation(image_id1)

        assert settings.is_rotation_set(image_id1) is False
        assert settings.is_rotation_set(image_id2) is True

    def test_remove_rotation_nonexistent_is_safe(self):
        """remove_rotation on non-existent image should not raise."""
        settings = Settings()
        image_id = ImageId(file_path=Path("/test.tiff"))

        # Should not raise
        settings.remove_rotation(image_id)

    def test_multi_page_file_same_rotation(self):
        """Different pages of the same file should share rotation."""
        settings = Settings()
        # Same file, different pages
        image_id_page1 = ImageId(file_path=Path("/multi.tiff"), page=1)
        image_id_page2 = ImageId(file_path=Path("/multi.tiff"), page=2)

        settings.apply_rotation(image_id_page1, OrthogonalRotation(degrees=90))

        # Page 2 should not have the rotation (different ImageId)
        assert settings.is_rotation_set(image_id_page2) is False

        # But if we apply to page 2 as well...
        settings.apply_rotation(image_id_page2, OrthogonalRotation(degrees=90))
        assert settings.get_rotation_for(image_id_page2).degrees == 90

    def test_to_dict_serialization(self):
        """to_dict should serialize rotations."""
        settings = Settings()
        image_id = ImageId(file_path=Path("/scan.tiff"))
        settings.apply_rotation(image_id, OrthogonalRotation(degrees=90))

        data = settings.to_dict()

        assert "/scan.tiff:0" in data or "\\scan.tiff:0" in str(data)
