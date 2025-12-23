"""Tests for project management.

These tests cover Project, ImageInfo, ImageMetadata, and XML compatibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scantailor.core import (
    Dpi,
    DpiStatus,
    ImageId,
    ImageInfo,
    ImageMetadata,
    PageId,
    Project,
    SubPage,
)


class TestImageMetadata:
    """Tests for ImageMetadata model."""

    def test_default_values(self):
        metadata = ImageMetadata()
        assert metadata.width == 0
        assert metadata.height == 0
        assert metadata.dpi == Dpi()

    def test_construction_with_values(self):
        dpi = Dpi(horizontal=300, vertical=300)
        metadata = ImageMetadata(width=2480, height=3508, dpi=dpi)
        assert metadata.width == 2480
        assert metadata.height == 3508
        assert metadata.dpi.horizontal == 300

    def test_dpi_status_ok(self):
        metadata = ImageMetadata(
            width=2480,
            height=3508,
            dpi=Dpi(horizontal=300, vertical=300),
        )
        assert metadata.horizontal_dpi_status() == DpiStatus.OK
        assert metadata.vertical_dpi_status() == DpiStatus.OK
        assert metadata.is_dpi_ok() is True

    def test_dpi_status_undefined(self):
        metadata = ImageMetadata(width=100, height=100, dpi=Dpi(horizontal=0, vertical=0))
        assert metadata.horizontal_dpi_status() == DpiStatus.UNDEFINED
        assert metadata.is_dpi_ok() is False

    def test_dpi_status_too_large(self):
        metadata = ImageMetadata(
            width=100,
            height=100,
            dpi=Dpi(horizontal=10000, vertical=300),
        )
        assert metadata.horizontal_dpi_status() == DpiStatus.TOO_LARGE
        assert metadata.is_dpi_ok() is False

    def test_dpi_status_too_small(self):
        metadata = ImageMetadata(
            width=100,
            height=100,
            dpi=Dpi(horizontal=20, vertical=300),
        )
        assert metadata.horizontal_dpi_status() == DpiStatus.TOO_SMALL
        assert metadata.is_dpi_ok() is False

    def test_json_round_trip(self):
        original = ImageMetadata(
            width=2480,
            height=3508,
            dpi=Dpi(horizontal=300, vertical=300),
        )
        json_str = original.model_dump_json()
        restored = ImageMetadata.model_validate_json(json_str)
        assert original == restored


class TestImageInfo:
    """Tests for ImageInfo model."""

    def test_default_values(self):
        image_id = ImageId(file_path=Path("/scan.tiff"))
        info = ImageInfo(id=image_id)
        assert info.id == image_id
        assert info.num_sub_pages == 1
        assert info.left_half_removed is False
        assert info.right_half_removed is False

    def test_construction_with_all_values(self):
        image_id = ImageId(file_path=Path("/scan.tiff"))
        metadata = ImageMetadata(width=100, height=200, dpi=Dpi.uniform(300))
        info = ImageInfo(
            id=image_id,
            metadata=metadata,
            num_sub_pages=2,
            left_half_removed=False,
            right_half_removed=False,
        )
        assert info.num_sub_pages == 2
        assert info.metadata.width == 100

    def test_json_round_trip(self):
        original = ImageInfo(
            id=ImageId(file_path=Path("/scan.tiff"), page=1),
            metadata=ImageMetadata(width=100, height=200, dpi=Dpi.uniform(300)),
            num_sub_pages=2,
        )
        json_str = original.model_dump_json()
        restored = ImageInfo.model_validate_json(json_str)
        assert original == restored


class TestProject:
    """Tests for Project model."""

    def test_default_values(self):
        project = Project()
        assert project.version == 1
        assert project.layout_direction == "LTR"
        assert project.images == []
        assert project.selected_page is None

    def test_add_image(self):
        project = Project()
        image_id = ImageId(file_path=Path("/scan1.tiff"))
        info = project.add_image(image_id)
        assert len(project.images) == 1
        assert project.images[0].id == image_id
        assert info.id == image_id

    def test_add_image_with_metadata(self):
        project = Project()
        image_id = ImageId(file_path=Path("/scan1.tiff"))
        metadata = ImageMetadata(width=100, height=200, dpi=Dpi.uniform(300))
        info = project.add_image(image_id, metadata)
        assert info.metadata == metadata

    def test_get_pages_single_page_images(self):
        project = Project()
        project.add_image(ImageId(file_path=Path("/scan1.tiff")))
        project.add_image(ImageId(file_path=Path("/scan2.tiff")))

        pages = project.get_pages()
        assert len(pages) == 2
        assert pages[0].sub_page == SubPage.SINGLE_PAGE
        assert pages[1].sub_page == SubPage.SINGLE_PAGE

    def test_get_pages_two_page_images_ltr(self):
        project = Project(layout_direction="LTR")
        image_id = ImageId(file_path=Path("/scan.tiff"))
        project.images.append(
            ImageInfo(id=image_id, num_sub_pages=2)
        )

        pages = project.get_pages()
        assert len(pages) == 2
        assert pages[0].sub_page == SubPage.LEFT_PAGE
        assert pages[1].sub_page == SubPage.RIGHT_PAGE

    def test_get_pages_two_page_images_rtl(self):
        project = Project(layout_direction="RTL")
        image_id = ImageId(file_path=Path("/scan.tiff"))
        project.images.append(
            ImageInfo(id=image_id, num_sub_pages=2)
        )

        pages = project.get_pages()
        assert len(pages) == 2
        assert pages[0].sub_page == SubPage.RIGHT_PAGE
        assert pages[1].sub_page == SubPage.LEFT_PAGE

    def test_get_pages_left_half_removed(self):
        project = Project()
        image_id = ImageId(file_path=Path("/scan.tiff"))
        project.images.append(
            ImageInfo(id=image_id, num_sub_pages=1, left_half_removed=True)
        )

        pages = project.get_pages()
        assert len(pages) == 1
        assert pages[0].sub_page == SubPage.RIGHT_PAGE

    def test_get_pages_right_half_removed(self):
        project = Project()
        image_id = ImageId(file_path=Path("/scan.tiff"))
        project.images.append(
            ImageInfo(id=image_id, num_sub_pages=1, right_half_removed=True)
        )

        pages = project.get_pages()
        assert len(pages) == 1
        assert pages[0].sub_page == SubPage.LEFT_PAGE

    def test_json_round_trip(self):
        project = Project(
            output_directory=Path("/output"),
            layout_direction="RTL",
        )
        project.add_image(
            ImageId(file_path=Path("/scan1.tiff")),
            ImageMetadata(width=100, height=200, dpi=Dpi.uniform(300)),
        )
        project.selected_page = PageId(
            image_id=ImageId(file_path=Path("/scan1.tiff")),
            sub_page=SubPage.SINGLE_PAGE,
        )

        json_str = project.model_dump_json()
        restored = Project.model_validate_json(json_str)

        assert restored.layout_direction == "RTL"
        assert len(restored.images) == 1
        assert restored.selected_page is not None

    def test_save_and_load_json(self, tmp_path: Path):
        project = Project(output_directory=tmp_path / "out")
        project.add_image(ImageId(file_path=Path("/scan.tiff")))

        project_file = tmp_path / "test.scantailor"
        project.save(project_file)

        loaded = Project.load(project_file)
        assert len(loaded.images) == 1
        assert loaded.images[0].id.file_path == Path("/scan.tiff")


class TestXmlCompatibility:
    """Tests for loading C++ ScanTailor XML projects."""

    def test_load_minimal_xml(self, tmp_path: Path):
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<project version="1" outputDirectory="out" layoutDirection="LTR">
    <directories>
        <directory id="1" path="/images"/>
    </directories>
    <files>
        <file id="1" dirId="1" name="scan.tiff"/>
    </files>
    <images>
        <image id="1" fileId="1" fileImage="0" subPages="1">
            <size width="2480" height="3508"/>
            <dpi horizontal="300" vertical="300"/>
        </image>
    </images>
    <pages>
        <page id="1" imageId="1" subPage="single"/>
    </pages>
</project>
"""
        project_file = tmp_path / "test.scantailor"
        project_file.write_text(xml_content, encoding="utf-8")

        project = Project.load(project_file)

        assert len(project.images) == 1
        assert project.images[0].id.file_path == Path("/images/scan.tiff")
        assert project.images[0].metadata.width == 2480
        assert project.images[0].metadata.height == 3508
        assert project.images[0].metadata.dpi.horizontal == 300

    def test_load_xml_with_rtl_layout(self, tmp_path: Path):
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<project version="1" outputDirectory="out" layoutDirection="RTL">
    <directories/>
    <files/>
    <images/>
    <pages/>
</project>
"""
        project_file = tmp_path / "test.scantailor"
        project_file.write_text(xml_content, encoding="utf-8")

        project = Project.load(project_file)
        assert project.layout_direction == "RTL"

    def test_load_xml_with_two_pages(self, tmp_path: Path):
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<project version="1" outputDirectory="out">
    <directories>
        <directory id="1" path="/images"/>
    </directories>
    <files>
        <file id="1" dirId="1" name="spread.tiff"/>
    </files>
    <images>
        <image id="1" fileId="1" fileImage="0" subPages="2">
            <size width="4960" height="3508"/>
            <dpi horizontal="300" vertical="300"/>
        </image>
    </images>
    <pages>
        <page id="1" imageId="1" subPage="left"/>
        <page id="2" imageId="1" subPage="right"/>
    </pages>
</project>
"""
        project_file = tmp_path / "test.scantailor"
        project_file.write_text(xml_content, encoding="utf-8")

        project = Project.load(project_file)

        assert len(project.images) == 1
        assert project.images[0].num_sub_pages == 2

        pages = project.get_pages()
        assert len(pages) == 2

    def test_load_xml_with_selected_page(self, tmp_path: Path):
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<project version="1" outputDirectory="out">
    <directories>
        <directory id="1" path="/images"/>
    </directories>
    <files>
        <file id="1" dirId="1" name="scan.tiff"/>
    </files>
    <images>
        <image id="1" fileId="1" fileImage="0" subPages="1">
            <size width="100" height="100"/>
            <dpi horizontal="300" vertical="300"/>
        </image>
    </images>
    <pages>
        <page id="1" imageId="1" subPage="single" selected="selected"/>
    </pages>
</project>
"""
        project_file = tmp_path / "test.scantailor"
        project_file.write_text(xml_content, encoding="utf-8")

        project = Project.load(project_file)
        assert project.selected_page is not None
        assert project.selected_page.sub_page == SubPage.SINGLE_PAGE

    def test_load_xml_with_removed_half(self, tmp_path: Path):
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<project version="1" outputDirectory="out">
    <directories>
        <directory id="1" path="/images"/>
    </directories>
    <files>
        <file id="1" dirId="1" name="scan.tiff"/>
    </files>
    <images>
        <image id="1" fileId="1" fileImage="0" subPages="1" removed="L">
            <size width="100" height="100"/>
            <dpi horizontal="300" vertical="300"/>
        </image>
    </images>
    <pages>
        <page id="1" imageId="1" subPage="right"/>
    </pages>
</project>
"""
        project_file = tmp_path / "test.scantailor"
        project_file.write_text(xml_content, encoding="utf-8")

        project = Project.load(project_file)
        assert project.images[0].left_half_removed is True
        assert project.images[0].right_half_removed is False

    def test_load_unknown_format_raises(self, tmp_path: Path):
        project_file = tmp_path / "test.scantailor"
        project_file.write_text("random content", encoding="utf-8")

        with pytest.raises(ValueError, match="Unknown project file format"):
            Project.load(project_file)

    def test_xml_to_json_migration(self, tmp_path: Path):
        """Test loading XML and saving as JSON."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<project version="1" outputDirectory="out">
    <directories>
        <directory id="1" path="/images"/>
    </directories>
    <files>
        <file id="1" dirId="1" name="scan.tiff"/>
    </files>
    <images>
        <image id="1" fileId="1" fileImage="0" subPages="1">
            <size width="2480" height="3508"/>
            <dpi horizontal="300" vertical="300"/>
        </image>
    </images>
    <pages>
        <page id="1" imageId="1" subPage="single"/>
    </pages>
</project>
"""
        xml_file = tmp_path / "original.scantailor"
        xml_file.write_text(xml_content, encoding="utf-8")

        # Load from XML
        project = Project.load(xml_file)

        # Save as JSON
        json_file = tmp_path / "migrated.scantailor"
        project.save(json_file)

        # Load from JSON
        reloaded = Project.load(json_file)

        assert len(reloaded.images) == 1
        assert reloaded.images[0].metadata.width == 2480
