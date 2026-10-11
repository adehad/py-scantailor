"""Tests for project management.

These tests cover Project, ImageInfo, ImageMetadata, and XML compatibility.
"""

import json
import shutil
from pathlib import Path

import pytest

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.project import (
    DpiStatus,
    ImageInfo,
    ImageMetadata,
    Project,
    StageParamsEntry,
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
        metadata = ImageMetadata(
            width=100, height=100, dpi=Dpi(horizontal=0, vertical=0)
        )
        assert metadata.horizontal_dpi_status() == DpiStatus.UNDEFINED
        assert metadata.is_dpi_ok() is False

    def test_dpi_status_too_large(self):
        metadata = ImageMetadata(
            width=100,
            height=100,
            dpi=Dpi(horizontal=10000, vertical=300),
        )
        assert metadata.horizontal_dpi_status() == DpiStatus.TOO_LARGE
        assert metadata.is_dpi_ok() is True

    def test_dpi_status_too_small(self):
        metadata = ImageMetadata(
            width=100,
            height=100,
            dpi=Dpi(horizontal=20, vertical=300),
        )
        assert metadata.horizontal_dpi_status() == DpiStatus.TOO_SMALL
        assert metadata.is_dpi_ok() is True

    def test_is_dpi_ok_rejects_undefined_dpi_on_either_axis(self):
        metadata = ImageMetadata(
            width=100, height=100, dpi=Dpi(horizontal=300, vertical=0)
        )
        assert metadata.is_dpi_ok() is False

    @pytest.mark.parametrize(
        ("dpi", "size", "expected"),
        [
            (0, 100, DpiStatus.UNDEFINED),
            (1, 100, DpiStatus.UNDEFINED),
            (2, 100, DpiStatus.TOO_SMALL),
            (149, 100, DpiStatus.TOO_SMALL),
            (150, 100, DpiStatus.OK),
            (9999, 100, DpiStatus.OK),
            (10000, 100, DpiStatus.TOO_LARGE),
            (300, 5900, DpiStatus.OK),
            (300, 6000, DpiStatus.TOO_SMALL_FOR_PIXEL_SIZE),
            (150, 3000, DpiStatus.TOO_SMALL_FOR_PIXEL_SIZE),
        ],
    )
    def test_dpi_status_thresholds(self, dpi: int, size: int, expected: DpiStatus):
        metadata = ImageMetadata(width=size, height=100, dpi=Dpi.uniform(dpi))

        assert metadata.horizontal_dpi_status() == expected

    def test_dpi_status_uses_height_for_vertical_dpi(self):
        metadata = ImageMetadata(
            width=100, height=6000, dpi=Dpi(horizontal=300, vertical=300)
        )

        assert metadata.horizontal_dpi_status() == DpiStatus.OK
        assert metadata.vertical_dpi_status() == DpiStatus.TOO_SMALL_FOR_PIXEL_SIZE
        assert metadata.is_dpi_ok() is True

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
        project.images.append(ImageInfo(id=image_id, num_sub_pages=2))

        pages = project.get_pages()
        assert len(pages) == 2
        assert pages[0].sub_page == SubPage.LEFT_PAGE
        assert pages[1].sub_page == SubPage.RIGHT_PAGE

    def test_get_pages_two_page_images_rtl(self):
        project = Project(layout_direction="RTL")
        image_id = ImageId(file_path=Path("/scan.tiff"))
        project.images.append(ImageInfo(id=image_id, num_sub_pages=2))

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
        project.add_image(ImageId(file_path=tmp_path / "scan.tiff"))

        project_file = tmp_path / "test.scantailor"
        project.save(project_file)

        loaded = Project.load(project_file)
        assert len(loaded.images) == 1
        assert loaded.images[0].id.file_path == tmp_path / "scan.tiff"


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


class TestPlausibleDpi:
    """Tests for ImageMetadata.has_plausible_dpi."""

    @pytest.mark.parametrize(
        ("dpi", "expected"),
        [
            (Dpi.uniform(300), True),
            (Dpi(), False),
            (Dpi(horizontal=300, vertical=0), False),
            (Dpi.uniform(100), False),
            (Dpi.uniform(12000), False),
        ],
        ids=["normal", "undefined", "one-axis-undefined", "too-small", "too-large"],
    )
    def test_reports_whether_both_axes_are_usable(self, dpi: Dpi, expected: bool):
        metadata = ImageMetadata(width=2480, height=3508, dpi=dpi)

        assert metadata.has_plausible_dpi() is expected

    def test_rejects_dpi_too_low_for_the_pixel_size(self):
        metadata = ImageMetadata(width=20000, height=30000, dpi=Dpi.uniform(150))

        assert not metadata.has_plausible_dpi()


class TestRelativePaths:
    """Image paths are stored relative to the project file when possible."""

    @staticmethod
    def _project_with_params(images: Path, output: Path) -> Project:
        image_id = ImageId(file_path=images / "a.png")
        page = PageId(image_id=image_id)
        project = Project(output_directory=output, selected_page=page)
        project.add_image(image_id)
        project.stage_params = {"deskew": [StageParamsEntry(page=page, params={})]}
        return project

    def test_saved_file_holds_relative_forward_slash_paths(self, tmp_path: Path):
        project = self._project_with_params(tmp_path / "scans", tmp_path / "out")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        project_file = project_dir / "book.ScanTailor"

        project.save(project_file)

        data = json.loads(project_file.read_text(encoding="utf-8"))
        assert data["images"][0]["id"]["file_path"] == "../scans/a.png"
        assert data["output_directory"] == "../out"
        assert data["selected_page"]["image_id"]["file_path"] == "../scans/a.png"
        entry_page = data["stage_params"]["deskew"][0]["page"]
        assert entry_page["image_id"]["file_path"] == "../scans/a.png"

    def test_load_resolves_every_stored_path(self, tmp_path: Path):
        project = self._project_with_params(tmp_path / "scans", tmp_path / "out")
        project_file = tmp_path / "book.ScanTailor"
        project.save(project_file)

        loaded = Project.load(project_file)

        expected = tmp_path / "scans" / "a.png"
        assert loaded.images[0].id.file_path == expected
        assert loaded.output_directory == tmp_path / "out"
        assert loaded.selected_page is not None
        assert loaded.selected_page.image_id.file_path == expected
        assert loaded.stage_params["deskew"][0].page.image_id.file_path == expected

    def test_project_and_images_moved_together_still_resolve(self, tmp_path: Path):
        old_root = tmp_path / "old"
        (old_root / "scans").mkdir(parents=True)
        (old_root / "scans" / "a.png").write_bytes(b"x")
        project = self._project_with_params(old_root / "scans", old_root / "out")
        project.save(old_root / "book.ScanTailor")

        new_root = tmp_path / "new"
        shutil.move(str(old_root), str(new_root))
        loaded = Project.load(new_root / "book.ScanTailor")

        assert loaded.images[0].id.file_path == new_root / "scans" / "a.png"
        assert loaded.images[0].id.file_path.exists()
        assert loaded.output_directory == new_root / "out"

    def test_absolute_paths_in_older_files_are_kept(self, tmp_path: Path):
        project_file = tmp_path / "book.ScanTailor"
        absolute = tmp_path / "elsewhere" / "a.png"
        data = {
            "images": [{"id": {"file_path": str(absolute), "page": 0}}],
            "output_directory": str(tmp_path / "out"),
        }
        project_file.write_text(json.dumps(data), encoding="utf-8")

        loaded = Project.load(project_file)

        assert loaded.images[0].id.file_path == absolute

    def test_malformed_sections_are_left_for_validation(self, tmp_path: Path):
        project_file = tmp_path / "bad.ScanTailor"
        project_file.write_text('{"images": 5, "stage_params": []}', encoding="utf-8")

        with pytest.raises(ValueError, match="images"):
            Project.load(project_file)


class TestRemapImagePaths:
    """Project.remap_image_paths rewrites every place a path is stored."""

    def test_rewrites_images_selected_page_and_stage_params(self, tmp_path: Path):
        old, new = tmp_path / "old", tmp_path / "new"
        image_id = ImageId(file_path=old / "a.png", page=2)
        page = PageId(image_id=image_id, sub_page=SubPage.LEFT_PAGE)
        project = Project(selected_page=page)
        project.add_image(image_id)
        project.stage_params = {
            "output": [StageParamsEntry(page=page, params={"k": 1})]
        }

        project.remap_image_paths(lambda path: new / path.name)

        moved = ImageId(file_path=new / "a.png", page=2)
        assert project.images[0].id == moved
        assert project.selected_page == PageId(
            image_id=moved, sub_page=SubPage.LEFT_PAGE
        )
        entry = project.stage_params["output"][0]
        assert entry.page.image_id == moved
        assert entry.page.sub_page == SubPage.LEFT_PAGE
        assert entry.params == {"k": 1}

    def test_keeps_image_metadata_and_flags(self, tmp_path: Path):
        project = Project()
        info = project.add_image(
            ImageId(file_path=tmp_path / "a.png"),
            ImageMetadata(width=10, height=20, dpi=Dpi.uniform(300)),
        )
        info.num_sub_pages = 2

        project.remap_image_paths(lambda path: tmp_path / "b" / path.name)

        assert project.images[0].metadata.width == 10
        assert project.images[0].num_sub_pages == 2
