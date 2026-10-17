"""Tests for project management.

These tests cover Project, ImageInfo, ImageMetadata, and XML compatibility.
"""

import json
import shutil
from pathlib import Path

import pytest

from scantailor.core.batch import output_file_names
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pdf_source import PdfSource
from scantailor.core.project import (
    DpiStatus,
    ImageInfo,
    ImageMetadata,
    OutputFileLabel,
    PageTextGeometry,
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

    def test_restoring_a_removed_half_brings_back_both_pages_in_order(self):
        project = Project()
        image_id = ImageId(file_path=Path("/scan.tiff"))
        project.images.append(
            ImageInfo(id=image_id, num_sub_pages=1, left_half_removed=True)
        )

        pages = project.restore_removed_half(image_id)

        assert [page.sub_page for page in pages] == [
            SubPage.LEFT_PAGE,
            SubPage.RIGHT_PAGE,
        ]
        assert project.get_pages() == pages
        assert project.images[0].num_sub_pages == 2

    def test_restoring_an_image_with_nothing_removed_changes_nothing(self):
        project = Project()
        image_id = ImageId(file_path=Path("/scan.tiff"))
        project.images.append(ImageInfo(id=image_id))
        other = ImageId(file_path=Path("/other.tiff"))

        assert project.restore_removed_half(image_id) == []
        assert project.restore_removed_half(other) == []
        assert project.images[0].num_sub_pages == 1

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


class TestOutputFileLabels:
    """Labels that keep output file names stable across sessions and removals."""

    @staticmethod
    def _project(tmp_path: Path, files: list[Path]) -> Project:
        project = Project(output_directory=tmp_path / "out")
        for file_path in files:
            project.add_image(ImageId(file_path=file_path))
        return project

    def test_labels_round_trip_through_json_with_relative_paths(self, tmp_path: Path):
        first, second = tmp_path / "a" / "x.png", tmp_path / "b" / "x.png"
        project = self._project(tmp_path, [first, second])
        project.output_file_labels = [
            OutputFileLabel(file_path=first, label=0),
            OutputFileLabel(file_path=second, label=3),
        ]
        project_file = tmp_path / "p.json"

        project.save(project_file)
        stored = json.loads(project_file.read_text(encoding="utf-8"))
        loaded = Project.load(project_file)

        assert [entry["file_path"] for entry in stored["output_file_labels"]] == [
            "a/x.png",
            "b/x.png",
        ]
        assert loaded.output_file_label_map() == {first: 0, second: 3}

    def test_save_registers_labels_for_files_that_have_none(self, tmp_path: Path):
        first, second = tmp_path / "a" / "x.png", tmp_path / "b" / "x.png"
        project = self._project(tmp_path, [first, second])

        project.save(tmp_path / "p.json")

        assert project.output_file_label_map() == {first: 0, second: 1}

    def test_project_file_without_labels_loads_with_none(self, tmp_path: Path):
        project_file = tmp_path / "old.json"
        project_file.write_text(json.dumps({"version": 1}), encoding="utf-8")

        project = Project.load(project_file)

        assert project.output_file_labels == []
        assert project.stage_defaults == {}

    def test_registering_keeps_stored_labels_and_numbers_new_files_after_them(
        self, tmp_path: Path
    ):
        old, new = tmp_path / "a" / "x.png", tmp_path / "b" / "x.png"
        project = self._project(tmp_path, [new])
        project.output_file_labels = [OutputFileLabel(file_path=old, label=4)]

        project.register_output_file_labels()

        assert project.output_file_label_map() == {old: 4, new: 5}

    def test_names_keep_stored_labels_after_a_page_is_removed(self, tmp_path: Path):
        first, second = tmp_path / "a" / "x.png", tmp_path / "b" / "x.png"
        project = self._project(tmp_path, [first, second])
        project.register_output_file_labels()
        removed = project.get_pages()[0]

        project.remove_pages([removed])
        names = output_file_names(
            project.get_pages(),
            project.layout_direction,
            project.output_file_label_map(),
        )

        assert list(names.values()) == ["x(1).tif"]

    def test_names_without_stored_labels_would_have_shifted(self, tmp_path: Path):
        first, second = tmp_path / "a" / "x.png", tmp_path / "b" / "x.png"
        project = self._project(tmp_path, [first, second])
        project.remove_pages([project.get_pages()[0]])

        names = output_file_names(project.get_pages(), project.layout_direction)

        assert list(names.values()) == ["x.tif"]

    def test_stored_labels_win_over_page_order(self, tmp_path: Path):
        first, second = tmp_path / "a" / "x.png", tmp_path / "b" / "x.png"
        pages = [PageId(image_id=ImageId(file_path=p)) for p in (first, second)]

        names = output_file_names(pages, file_labels={first: 1, second: 0})

        assert [names[page] for page in pages] == ["x(1).tif", "x.tif"]

    def test_new_file_is_labelled_after_the_highest_label_of_its_name(
        self, tmp_path: Path
    ):
        old, other = tmp_path / "a" / "x.png", tmp_path / "b" / "y.png"
        new = tmp_path / "c" / "x.png"
        pages = [PageId(image_id=ImageId(file_path=p)) for p in (old, other, new)]

        names = output_file_names(pages, file_labels={old: 2, other: 0})

        assert [names[page] for page in pages] == ["x(2).tif", "y.tif", "x(3).tif"]

    def test_remapping_image_paths_moves_the_labels_too(self, tmp_path: Path):
        old = tmp_path / "a" / "x.png"
        project = self._project(tmp_path, [old])
        project.register_output_file_labels()

        project.remap_image_paths(lambda path: tmp_path / "moved" / path.name)

        assert project.output_file_label_map() == {tmp_path / "moved" / "x.png": 0}


class TestPdfSourceAndTextGeometry:
    @staticmethod
    def _page(path: Path) -> PageId:
        return PageId(image_id=ImageId(file_path=path), sub_page=SubPage.SINGLE_PAGE)

    def test_the_pdf_sources_round_trip_with_relative_paths(self, tmp_path: Path):
        pages = tmp_path / "scans" / "book_pages"
        project = Project(
            output_directory=tmp_path / "out",
            pdf_sources=[
                PdfSource(
                    pdf_path=tmp_path / "scans" / "book.pdf",
                    dpi=300,
                    pages_dir=pages,
                    pages={"page_0001.png": 0},
                ),
                PdfSource(
                    pdf_path=tmp_path / "scans" / "other.pdf",
                    dpi=150,
                    pages_dir=tmp_path / "scans" / "other_pages",
                    pages={"page_0001.png": 0},
                ),
            ],
        )
        project_file = tmp_path / "p.json"

        project.save(project_file)
        stored = json.loads(project_file.read_text(encoding="utf-8"))
        loaded = Project.load(project_file)

        assert stored["pdf_sources"][0]["pdf_path"] == "scans/book.pdf"
        assert stored["pdf_sources"][0]["pages_dir"] == "scans/book_pages"
        assert loaded.pdf_sources == project.pdf_sources

    def test_a_project_file_with_the_single_pdf_source_field_loads_as_a_list(
        self, tmp_path: Path
    ):
        project_file = tmp_path / "old.json"
        legacy = {"pdf_path": "scans/book.pdf", "dpi": 300, "pages": {"a.png": 0}}
        project_file.write_text(
            json.dumps({"version": 1, "pdf_source": legacy}), encoding="utf-8"
        )

        loaded = Project.load(project_file)

        assert loaded.pdf_sources == [
            PdfSource(
                pdf_path=tmp_path / "scans" / "book.pdf", dpi=300, pages={"a.png": 0}
            )
        ]

    def test_a_null_single_pdf_source_field_loads_with_no_sources(self, tmp_path: Path):
        project_file = tmp_path / "old.json"
        project_file.write_text(
            json.dumps({"version": 1, "pdf_source": None}), encoding="utf-8"
        )

        assert Project.load(project_file).pdf_sources == []

    def test_source_of_picks_the_source_whose_folder_holds_the_page(
        self, tmp_path: Path
    ):
        first = PdfSource(
            pdf_path=tmp_path / "one.pdf",
            dpi=300,
            pages_dir=tmp_path / "one_pages",
            pages={"page_0001.png": 0},
        )
        second = PdfSource(
            pdf_path=tmp_path / "two.pdf",
            dpi=300,
            pages_dir=tmp_path / "two_pages",
            pages={"page_0001.png": 0},
        )
        project = Project(pdf_sources=[first, second])

        assert project.source_of(tmp_path / "two_pages" / "page_0001.png") == second
        assert project.source_of(tmp_path / "loose" / "scan.png") is None

    def test_text_geometry_round_trips_with_relative_paths(self, tmp_path: Path):
        image = tmp_path / "pages" / "page_0001.png"
        page = self._page(image)
        project = Project(output_directory=tmp_path / "out")
        project.add_image(ImageId(file_path=image))
        project.set_text_geometry(page, {"matrix": [1.0], "rect_x": 0.0})
        project_file = tmp_path / "p.json"

        project.save(project_file)
        stored = json.loads(project_file.read_text(encoding="utf-8"))
        loaded = Project.load(project_file)

        assert stored["text_geometry"][0]["page"]["image_id"]["file_path"] == (
            "pages/page_0001.png"
        )
        assert loaded.text_geometry_of(page) == {"matrix": [1.0], "rect_x": 0.0}

    def test_setting_the_geometry_of_a_page_again_replaces_it(self, tmp_path: Path):
        page = self._page(tmp_path / "a.png")
        project = Project()

        project.set_text_geometry(page, {"rect_x": 1.0})
        project.set_text_geometry(page, {"rect_x": 2.0})

        assert project.text_geometry == [
            PageTextGeometry(page=page, geometry={"rect_x": 2.0})
        ]

    def test_removing_a_page_drops_its_geometry(self, tmp_path: Path):
        first, second = self._page(tmp_path / "a.png"), self._page(tmp_path / "b.png")
        project = Project()
        project.add_image(first.image_id)
        project.add_image(second.image_id)
        project.set_text_geometry(first, {"rect_x": 1.0})
        project.set_text_geometry(second, {"rect_x": 2.0})

        project.remove_pages([first])

        assert project.text_geometry_of(first) is None
        assert project.text_geometry_of(second) == {"rect_x": 2.0}

    def test_a_project_file_without_either_loads_with_none(self, tmp_path: Path):
        project_file = tmp_path / "old.json"
        project_file.write_text(json.dumps({"version": 1}), encoding="utf-8")

        project = Project.load(project_file)

        assert project.pdf_sources == []
        assert project.text_geometry == []


class TestMostCommonDpi:
    @staticmethod
    def _project(*dpis: Dpi) -> Project:
        project = Project()
        for index, dpi in enumerate(dpis):
            project.add_image(
                ImageId(file_path=Path(f"scan_{index}.png")),
                ImageMetadata(width=1200, height=1600, dpi=dpi),
            )
        return project

    def test_the_resolution_most_images_share_wins(self):
        project = self._project(Dpi.uniform(300), Dpi.uniform(600), Dpi.uniform(300))

        assert project.most_common_dpi() == Dpi.uniform(300)

    def test_one_resolution_for_every_image_is_the_most_common(self):
        project = self._project(Dpi(horizontal=300, vertical=600))

        assert project.most_common_dpi() == Dpi(horizontal=300, vertical=600)

    def test_a_tie_has_no_most_common_resolution(self):
        project = self._project(Dpi.uniform(300), Dpi.uniform(600))

        assert project.most_common_dpi() is None

    def test_images_without_a_usable_resolution_are_not_counted(self):
        project = self._project(Dpi(), Dpi(), Dpi.uniform(400))

        assert project.most_common_dpi() == Dpi.uniform(400)

    def test_an_empty_project_has_no_most_common_resolution(self):
        assert Project().most_common_dpi() is None
