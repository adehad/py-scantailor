"""Tests for importing C++ ScanTailor .ScanTailor (XML) projects with their settings."""

import codecs
import shutil
from pathlib import Path

import pytest
from loguru import logger

from scantailor.core import xml_reader
from scantailor.core.models import Dpi, ImageId, OrthogonalRotation, PageId, SubPage
from scantailor.core.project import Project
from scantailor.core.stage_params import (
    apply_stage_defaults,
    apply_stage_params,
    collect_stage_defaults,
    collect_stage_params,
)
from scantailor.core.stage_sequence import StageSequence
from scantailor.core.xml_compat import (
    NEWEST_PROJECT_VERSION,
    XmlProjectImport,
    XmlUnappliedSettings,
    apply_xml_stage_defaults,
    import_xml_project,
    import_xml_project_file,
)
from scantailor.core.zones import PictureLayer, ZoneCategory
from scantailor.filters.deskew.params import AutoManualMode as DeskewMode
from scantailor.filters.deskew.params import Params as DeskewParams
from scantailor.filters.fix_orientation.xml_import import ImageTrim
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import FillingColor
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.output.picture_shape import PictureShape
from scantailor.filters.page_layout.alignment import (
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_layout.params import Params as PageLayoutParams
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayoutType
from scantailor.filters.page_split.params import AutoManualMode as SplitMode
from scantailor.filters.page_split.params import Params as PageSplitParams
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)
from scantailor.filters.select_content.params import Params as SelectContentParams

FIXTURES = Path(__file__).parents[1] / "fixtures" / "xml_projects"


def _fixture(tmp_path: Path, name: str) -> Path:
    """Copy a fixture project into tmp_path and return its project file."""
    target = tmp_path / name
    shutil.copytree(FIXTURES / name, target)
    return target / f"{name}.ScanTailor"


def _load(tmp_path: Path, name: str) -> XmlProjectImport:
    project_file = _fixture(tmp_path, name)
    return import_xml_project(project_file, project_file.read_text(encoding="utf-8"))


def _params(project: Project, stage: str, model: type) -> dict[PageId, object]:
    """Return the stored parameters of a stage as models, keyed by page."""
    return {
        entry.page: model.model_validate(entry.params)
        for entry in project.stage_params.get(stage, [])
    }


def _page(project: Project, index: int) -> PageId:
    return project.get_pages()[index]


class TestFixturesLoad:
    @pytest.mark.parametrize(
        ("name", "images", "pages"),
        [("simple", 3, 3), ("split_and_mixed", 2, 3), ("multipage_tiff", 2, 2)],
    )
    def test_each_fixture_loads_with_its_images_and_pages(
        self, tmp_path: Path, name: str, images: int, pages: int
    ):
        imported = _load(tmp_path, name)

        assert len(imported.project.images) == images
        assert len(imported.project.get_pages()) == pages
        assert imported.project.version == 3

    def test_image_files_resolve_next_to_the_project_file_and_exist(
        self, tmp_path: Path
    ):
        project_file = _fixture(tmp_path, "simple")
        imported = import_xml_project(
            project_file, project_file.read_text(encoding="utf-8")
        )

        paths = [info.id.file_path for info in imported.project.images]
        assert paths == [
            tmp_path.resolve() / "simple" / "scans" / "page_001.tif",
            tmp_path.resolve() / "simple" / "scans" / "page_002.tif",
            tmp_path.resolve() / "simple" / "extra" / "page_001.tif",
        ]
        assert all(path.exists() for path in paths)
        assert (
            imported.project.output_directory == tmp_path.resolve() / "simple" / "out"
        )

    def test_simple_project_keeps_dpi_size_and_selected_page(self, tmp_path: Path):
        project = _load(tmp_path, "simple").project

        metadata = project.images[0].metadata
        assert (metadata.width, metadata.height) == (60, 80)
        assert metadata.dpi == Dpi.uniform(300)
        assert project.selected_page == _page(project, 2)
        assert project.layout_direction == "LTR"

    def test_import_xml_project_file_reads_xml_and_leaves_json_alone(
        self, tmp_path: Path
    ):
        project_file = _fixture(tmp_path, "simple")
        json_file = tmp_path / "native.ScanTailor"
        Project().save(json_file)

        assert import_xml_project_file(project_file) is not None
        assert import_xml_project_file(json_file) is None

    def test_xml_project_saved_with_a_byte_order_mark_is_still_recognised(
        self, tmp_path: Path
    ):
        project_file = _fixture(tmp_path, "simple")
        text = project_file.read_text(encoding="utf-8")
        project_file.write_bytes(codecs.BOM_UTF8 + text.encode("utf-8"))

        assert import_xml_project_file(project_file) is not None

    def test_project_load_picks_up_the_xml_settings(self, tmp_path: Path):
        project = Project.load(_fixture(tmp_path, "split_and_mixed"))

        assert set(project.stage_params) == {
            "fix_orientation",
            "page_split",
            "deskew",
            "select_content",
            "page_layout",
            "output",
        }


class TestMultiPageTiff:
    def test_pages_of_one_file_become_separate_images_numbered_from_one(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "multipage_tiff").project

        assert [info.id.page for info in project.images] == [1, 2]
        assert project.images[0].id.file_path == project.images[1].id.file_path
        assert all(info.id.is_multi_page_file() for info in project.images)

    def test_settings_stay_with_their_own_page_of_the_file(self, tmp_path: Path):
        project = _load(tmp_path, "multipage_tiff").project
        first, second = project.get_pages()

        deskew = _params(project, "deskew", DeskewParams)
        assert deskew[first].mode is DeskewMode.AUTO
        assert deskew[second].mode is DeskewMode.MANUAL
        assert deskew[second].deskew_angle_deg == pytest.approx(2.25)

        rotations = {
            entry.page.image_id: OrthogonalRotation.model_validate(entry.params)
            for entry in project.stage_params["fix_orientation"]
        }
        assert rotations == {second.image_id: OrthogonalRotation(degrees=270)}

    def test_a_separate_content_record_exists_for_each_page_of_the_file(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "multipage_tiff").project

        select = _params(project, "select_content", SelectContentParams)
        assert set(select) == set(project.get_pages())

    def test_legacy_multi_page_flag_numbers_pages_from_one(self, tmp_path: Path):
        xml = """<project version="3" outputDirectory="out">
<directories><directory id="1" path="scans"/></directories>
<files><file id="2" dirId="1" name="old.tif" multiPage="1"/></files>
<images>
<image id="3" subPages="1" fileId="2" fileImage="0"/>
<image id="4" subPages="1" fileId="2" fileImage="1"/>
</images>
<pages>
<page id="5" imageId="3" subPage="single"/>
<page id="6" imageId="4" subPage="single"/>
</pages>
</project>"""

        project = import_xml_project(tmp_path / "old.ScanTailor", xml).project

        assert [info.id.page for info in project.images] == [1, 2]


class TestVersion4:
    def test_the_newest_version_is_4(self):
        assert NEWEST_PROJECT_VERSION == 4

    def test_version_4_fixture_loads_its_images_pages_and_filters(self, tmp_path: Path):
        imported = _load(tmp_path, "version4")

        assert imported.project.version == 4
        assert len(imported.project.images) == 2
        assert len(imported.project.get_pages()) == 2
        assert set(imported.project.stage_params) >= {"fix_orientation", "deskew"}

    def test_version_4_deskew_pages_keep_the_angle_and_mode(self, tmp_path: Path):
        project = _load(tmp_path, "version4").project

        by_page = {entry.page: entry.params for entry in project.stage_params["deskew"]}
        manual = by_page[_page(project, 0)]
        assert manual["deskew_angle_deg"] == pytest.approx(0.5)
        assert manual["mode"] == "manual"

    def test_trim_only_image_gets_no_rotation_entry_and_keeps_its_trim(
        self, tmp_path: Path
    ):
        imported = _load(tmp_path, "version4")
        first_image = imported.project.images[0].id
        second_image = imported.project.images[1].id

        rotated = {
            entry.page.image_id
            for entry in imported.project.stage_params["fix_orientation"]
        }
        assert rotated == {second_image}
        assert imported.unapplied.fix_orientation_trims == {
            first_image: ImageTrim(left=4, right=5, top=6, bottom=7)
        }

    def test_disabled_trim_is_not_kept(self, tmp_path: Path):
        imported = _load(tmp_path, "version4")

        assert imported.project.images[1].id not in (
            imported.unapplied.fix_orientation_trims
        )

    def test_fill_outside_page_box_pages_are_kept(self, tmp_path: Path):
        imported = _load(tmp_path, "version4")

        assert imported.unapplied.fill_outside_page_box_pages == set(
            imported.project.get_pages()
        )

    def test_content_based_deskew_flag_is_kept_and_defaults_to_on(self, tmp_path: Path):
        assert _load(tmp_path, "version4").unapplied.deskew_content_based is False
        assert _load(tmp_path, "simple").unapplied.deskew_content_based is True

    def test_version_3_projects_have_nothing_unapplied(self, tmp_path: Path):
        unapplied = _load(tmp_path, "simple").unapplied

        assert unapplied == XmlUnappliedSettings()

    def test_dropping_version_4_settings_is_logged_as_a_warning(self, tmp_path: Path):
        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            _load(tmp_path, "version4")
        finally:
            logger.remove(sink)

        text = "".join(messages)
        assert "trim" in text
        assert "fill outside page box" in text
        assert "whole page" in text

    def test_version_3_projects_log_no_dropped_settings(self, tmp_path: Path):
        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            _load(tmp_path, "simple")
        finally:
            logger.remove(sink)

        assert not any("Dropping" in message for message in messages)


class TestVersion:
    def test_newer_version_is_refused_with_a_clear_message(self, tmp_path: Path):
        project_file = _fixture(tmp_path, "newer_version")

        with pytest.raises(ValueError, match=r"version 5.*newer application"):
            Project.load(project_file)

    @pytest.mark.parametrize("version", ["1", "2", "3", "4"])
    def test_versions_up_to_the_newest_are_accepted(self, tmp_path: Path, version: str):
        xml = f'<project version="{version}" outputDirectory="out"/>'

        project = import_xml_project(tmp_path / "p.ScanTailor", xml).project

        assert project.version == int(version)

    @pytest.mark.parametrize(
        "root",
        [
            '<project outputDirectory="out">',
            '<project version="x" outputDirectory="out">',
            '<project version="0" outputDirectory="out">',
            '<project version="99" outputDirectory="out">',
        ],
        ids=["missing", "not-a-number", "zero", "far-future"],
    )
    def test_missing_unreadable_or_unsupported_versions_are_refused(
        self, tmp_path: Path, root: str
    ):
        with pytest.raises(ValueError, match="version"):
            import_xml_project(tmp_path / "p.ScanTailor", root + "</project>")

    def test_a_file_that_is_not_well_formed_xml_raises_value_error(
        self, tmp_path: Path
    ):
        with pytest.raises(ValueError, match="well-formed"):
            import_xml_project(tmp_path / "p.ScanTailor", "<project version='3'>")


class TestSplitAndMixedGeometry:
    def test_split_image_yields_left_and_right_pages_and_keeps_the_selection(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "split_and_mixed").project

        pages = project.get_pages()
        assert [page.sub_page for page in pages] == [
            SubPage.LEFT_PAGE,
            SubPage.RIGHT_PAGE,
            SubPage.SINGLE_PAGE,
        ]
        assert project.images[0].num_sub_pages == 2
        assert project.selected_page == pages[1]

    def test_fix_orientation_rotation_is_stored_per_image(self, tmp_path: Path):
        project = _load(tmp_path, "split_and_mixed").project

        (entry,) = project.stage_params["fix_orientation"]
        assert entry.page == PageId(image_id=project.images[1].id)
        assert OrthogonalRotation.model_validate(entry.params).degrees == 180


class TestPageSplit:
    def test_manual_two_page_layout_keeps_the_split_line_and_dependencies(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "split_and_mixed").project
        spread = project.images[0].id

        split = _params(project, "page_split", PageSplitParams)
        params = split[PageId(image_id=spread)]
        assert params.layout_type is LayoutType.TWO_PAGES
        assert params.split_line_mode is SplitMode.MANUAL
        layout = params.page_layout
        assert layout is not None
        assert layout.layout_type is PageLayoutType.TWO_PAGES
        assert layout.outline == (0.0, 0.0, 120.0, 80.0)
        assert layout.cutter1 == ((58.5, 0.0), (58.5, 80.0))
        assert layout.cutter2 == layout.cutter1
        assert layout.get_split_line_x() == pytest.approx(58.5)
        assert params.dependencies.image_size == (120, 80)
        assert params.dependencies.rotation_degrees == 0
        assert params.dependencies.layout_type is LayoutType.TWO_PAGES

    def test_single_page_record_keeps_rotation_dependency_and_auto_mode(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "split_and_mixed").project
        cover = project.images[1].id

        params = _params(project, "page_split", PageSplitParams)[PageId(image_id=cover)]
        assert params.layout_type is LayoutType.SINGLE_PAGE_UNCUT
        assert params.split_line_mode is SplitMode.AUTO
        assert params.page_layout is not None
        assert params.page_layout.cutter1 is None
        assert params.dependencies.rotation_degrees == 180
        assert params.dependencies.image_size == (60, 80)

    def test_images_without_a_record_follow_a_non_auto_default_layout_type(
        self, tmp_path: Path
    ):
        xml = """<project version="3" outputDirectory="out">
<directories><directory id="1" path="scans"/></directories>
<files>
<file id="2" dirId="1" name="a.tif"/>
<file id="5" dirId="1" name="b.tif"/>
</files>
<images>
<image id="3" subPages="1" fileId="2" fileImage="0"/>
<image id="6" subPages="1" fileId="5" fileImage="0"/>
</images>
<pages>
<page id="4" imageId="3" subPage="single"/>
<page id="7" imageId="6" subPage="single"/>
</pages>
<filters><page-split defaultLayoutType="two-pages">
<image id="3" layoutType="single-cut"><params mode="auto"/></image>
</page-split></filters>
</project>"""

        imported = import_xml_project(tmp_path / "p.ScanTailor", xml)

        split = _params(imported.project, "page_split", PageSplitParams)
        first, second = (PageId(image_id=info.id) for info in imported.project.images)
        assert split[first].layout_type is LayoutType.PAGE_PLUS_OFFCUT
        assert split[second].layout_type is LayoutType.TWO_PAGES
        assert imported.stage_defaults.page_split_layout_type is LayoutType.TWO_PAGES

    def test_auto_default_layout_type_adds_no_records_for_untouched_images(
        self, tmp_path: Path
    ):
        xml = """<project version="3" outputDirectory="out">
<directories><directory id="1" path="scans"/></directories>
<files><file id="2" dirId="1" name="a.tif"/></files>
<images><image id="3" subPages="1" fileId="2" fileImage="0"/></images>
<pages><page id="4" imageId="3" subPage="single"/></pages>
<filters><page-split defaultLayoutType="auto-detect"/></filters>
</project>"""

        project = import_xml_project(tmp_path / "p.ScanTailor", xml).project

        assert "page_split" not in project.stage_params


class TestDeskew:
    def test_manual_and_auto_angles_and_their_dependencies(self, tmp_path: Path):
        project = _load(tmp_path, "split_and_mixed").project
        left, right, cover = project.get_pages()

        deskew = _params(project, "deskew", DeskewParams)
        assert deskew[left].mode is DeskewMode.MANUAL
        assert deskew[left].deskew_angle_deg == pytest.approx(-1.5)
        assert deskew[right].mode is DeskewMode.AUTO
        assert deskew[right].deskew_angle_deg == pytest.approx(0.75)
        assert deskew[right].dependencies.page_outline == (
            (58.5, 0.0),
            (120.0, 0.0),
            (120.0, 80.0),
            (58.5, 80.0),
        )
        assert deskew[cover].dependencies.rotation_degrees == 180


class TestSelectContent:
    def test_boxes_modes_and_dependencies_of_each_page(self, tmp_path: Path):
        project = _load(tmp_path, "split_and_mixed").project
        left, right, cover = project.get_pages()

        select = _params(project, "select_content", SelectContentParams)
        assert select[left].content_detection_mode is ContentDetectionMode.MANUAL
        assert select[left].page_detection_mode is PageDetectionMode.AUTO
        assert select[left].fine_tune_corners
        assert select[left].content_box.to_tuple() == (5.0, 6.0, 48.0, 66.0)
        assert select[left].page_box.to_tuple() == (0.0, 0.0, 58.5, 80.0)
        assert select[left].content_size_mm.width_mm == pytest.approx(4.06)
        assert select[left].content_size_mm.height_mm == pytest.approx(5.59)
        assert select[left].dependencies.page_detection_mode is PageDetectionMode.AUTO
        assert select[left].dependencies.fine_tune_corners
        assert select[left].dependencies.rotated_page_outline[1] == (58.5, 0.0)

        assert select[right].content_detection_mode is ContentDetectionMode.AUTO
        assert select[right].page_detection_mode is PageDetectionMode.DISABLED
        assert not select[right].fine_tune_corners

        assert select[cover].content_detection_mode is ContentDetectionMode.DISABLED
        assert select[cover].page_detection_mode is PageDetectionMode.MANUAL
        assert select[cover].page_box.to_tuple() == (2.0, 3.0, 54.0, 70.0)

    def test_stage_wide_page_detection_box_and_tolerance_are_read(self, tmp_path: Path):
        imported = _load(tmp_path, "split_and_mixed")

        assert imported.stage_defaults.page_detection_box == (210.0, 297.0)
        assert imported.stage_defaults.page_detection_tolerance == pytest.approx(0.15)


class TestPageLayout:
    def test_custom_margins_alignment_and_auto_flag(self, tmp_path: Path):
        project = _load(tmp_path, "split_and_mixed").project
        left, right, cover = project.get_pages()

        layout = _params(project, "page_layout", PageLayoutParams)
        margins = layout[left].hard_margins_mm
        assert (margins.left, margins.top, margins.right, margins.bottom) == (
            12.5,
            6.5,
            8.0,
            10.0,
        )
        assert layout[left].alignment.vertical is VerticalAlignment.TOP
        assert layout[left].alignment.horizontal is HorizontalAlignment.LEFT
        assert not layout[left].alignment.is_null
        assert not layout[left].auto_margins
        assert layout[left].content_rect.width == 48.0
        assert layout[left].page_rect.width == 58.5
        assert layout[left].content_size_mm.height_mm == pytest.approx(5.59)

        assert layout[right].alignment.vertical is VerticalAlignment.VAUTO
        assert layout[right].alignment.horizontal is HorizontalAlignment.HORIGINAL
        assert layout[right].alignment.is_null
        assert layout[right].auto_margins

        assert layout[cover].alignment.vertical is VerticalAlignment.BOTTOM
        assert layout[cover].alignment.horizontal is HorizontalAlignment.RIGHT
        assert layout[cover].hard_margins_mm.left == 0.0


class TestOutput:
    def test_mixed_page_keeps_colour_binarization_splitting_and_picture_options(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "split_and_mixed").project
        left = _page(project, 0)

        params = _params(project, "output", OutputParams)[left]
        assert params.output_dpi == Dpi.uniform(300)
        assert params.color_mode is ColorMode.MIXED
        assert params.despeckle_level is DespeckleLevel.NORMAL
        assert params.black_on_white
        binarization = params.binarization
        assert binarization.method is BinarizationMethod.WOLF
        assert binarization.threshold_adjustment == -5
        assert not binarization.savitzky_golay_smoothing
        assert binarization.morphological_smoothing
        assert not binarization.normalize_illumination
        assert binarization.window_size == 60
        assert binarization.sauvola_coef == pytest.approx(0.4)
        assert binarization.wolf_coef == pytest.approx(0.5)
        assert (binarization.wolf_lower_bound, binarization.wolf_upper_bound) == (
            3,
            250,
        )
        segmenter = binarization.color_segmenter
        assert segmenter.enabled
        assert segmenter.noise_reduction == 9
        assert (
            segmenter.red_threshold_adjustment,
            segmenter.green_threshold_adjustment,
            segmenter.blue_threshold_adjustment,
        ) == (-4, 2, 6)
        assert params.splitting.split_output
        assert params.splitting.color_foreground
        assert params.splitting.original_background
        assert params.picture_shape.shape is PictureShape.RECTANGULAR
        assert params.picture_shape.sensitivity == 80
        assert params.picture_shape.higher_search_sensitivity
        assert params.dewarping.mode is DewarpingMode.OFF

    def test_picture_and_fill_zones_keep_polygons_and_properties(self, tmp_path: Path):
        project = _load(tmp_path, "split_and_mixed").project
        params = _params(project, "output", OutputParams)[_page(project, 0)]

        painter, eraser = params.picture_zones
        assert painter.points == (
            (10.0, 10.0),
            (40.0, 10.0),
            (40.0, 30.0),
            (10.0, 30.0),
        )
        assert painter.properties.layer is PictureLayer.PAINTER2
        assert painter.properties.category is ZoneCategory.MANUAL
        assert eraser.points == ((5.0, 50.0), (25.0, 50.0), (15.0, 70.0))
        assert eraser.properties.layer is PictureLayer.ERASER1
        assert eraser.properties.category is ZoneCategory.AUTO
        assert not painter.is_smooth
        (fill,) = params.fill_zones
        assert fill.points == ((45.0, 40.0), (55.0, 40.0), (55.0, 60.0), (45.0, 60.0))
        assert fill.properties.color == (255, 128, 0)

    def test_zone_with_fewer_than_three_points_is_dropped(self, tmp_path: Path):
        project = _load(tmp_path, "split_and_mixed").project
        params = _params(project, "output", OutputParams)[_page(project, 0)]

        assert len(params.picture_zones) == 2

    def test_manual_dewarping_keeps_the_distortion_model_and_depth(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "split_and_mixed").project
        params = _params(project, "output", OutputParams)[_page(project, 1)]

        assert params.dewarping.mode is DewarpingMode.MANUAL
        assert not params.dewarping.post_deskew
        assert params.dewarping.post_deskew_angle == pytest.approx(-0.5)
        assert params.depth_perception == pytest.approx(2.5)
        model = params.distortion_model
        assert model.is_defined()
        assert model.top == (
            (10.0, 12.0),
            (30.0, 10.0),
            (50.0, 9.5),
            (70.0, 10.0),
            (110.0, 12.0),
        )
        assert model.bottom == (
            (10.0, 68.0),
            (30.0, 70.5),
            (50.0, 71.0),
            (70.0, 70.5),
            (110.0, 68.0),
        )

    def test_edgeplus_method_becomes_edge_div_with_only_the_edge_weight(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "split_and_mixed").project
        binarization = _params(project, "output", OutputParams)[
            _page(project, 1)
        ].binarization

        assert binarization.method is BinarizationMethod.EDGEDIV
        assert binarization.edge_div_kep == pytest.approx(0.25)
        assert binarization.edge_div_kbd == 0.0
        assert binarization.threshold_adjustment == -10
        assert not binarization.morphological_smoothing

    @pytest.mark.parametrize(
        ("method", "kep", "kbd"),
        [("blurdiv", 0.0, 0.25), ("edgediv", 0.25, 0.25)],
    )
    def test_blurdiv_and_edgediv_methods_split_the_coefficient_like_c_plus_plus(
        self, tmp_path: Path, method: str, kep: float, kbd: float
    ):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8").replace(
            'binarizationMethod="edgeplus"', f'binarizationMethod="{method}"'
        )

        project = import_xml_project(project_file, text).project

        binarization = _params(project, "output", OutputParams)[
            _page(project, 1)
        ].binarization
        assert binarization.method is BinarizationMethod.EDGEDIV
        assert (binarization.edge_div_kep, binarization.edge_div_kbd) == (kep, kbd)

    def test_colour_page_keeps_filling_posterization_and_wiener_options(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "split_and_mixed").project
        params = _params(project, "output", OutputParams)[_page(project, 2)]

        assert params.output_dpi == Dpi.uniform(400)
        assert params.color_mode is ColorMode.COLOR_GRAYSCALE
        assert params.despeckle_level is DespeckleLevel.OFF
        assert not params.black_on_white
        common = params.color_common
        assert not common.fill_margins
        assert common.fill_offcut
        assert common.normalize_illumination
        assert common.filling_color is FillingColor.WHITE
        assert common.wiener_coef == pytest.approx(0.5)
        assert common.wiener_window_size == 7
        assert common.posterization.enabled
        assert common.posterization.level == 6
        assert common.posterization.normalization_enabled
        assert not common.posterization.force_black_and_white

    def test_default_output_params_read_back_as_the_c_plus_plus_defaults(
        self, tmp_path: Path
    ):
        project = _load(tmp_path, "simple").project
        params = _params(project, "output", OutputParams)[_page(project, 0)]

        assert params.output_dpi == Dpi.uniform(600)
        assert params.color_mode is ColorMode.BLACK_AND_WHITE
        assert params.despeckle_level is DespeckleLevel.CAUTIOUS
        assert params.depth_perception == 2.0
        assert params.binarization.method is BinarizationMethod.OTSU
        assert params.binarization.window_size == 200
        assert params.picture_shape.shape is PictureShape.FREE
        assert params.picture_zones == ()
        assert not params.distortion_model.is_defined()

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            ("0", DespeckleLevel.OFF),
            ("0.4", DespeckleLevel.OFF),
            ("1", DespeckleLevel.CAUTIOUS),
            ("1.4", DespeckleLevel.CAUTIOUS),
            ("2", DespeckleLevel.NORMAL),
            ("2.4", DespeckleLevel.NORMAL),
            ("2.5", DespeckleLevel.AGGRESSIVE),
            ("3", DespeckleLevel.AGGRESSIVE),
        ],
    )
    def test_despeckle_strength_maps_to_the_nearest_level(
        self, tmp_path: Path, stored: str, expected: DespeckleLevel
    ):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8").replace(
            'despeckleLevel="0"', f'despeckleLevel="{stored}"'
        )

        project = import_xml_project(project_file, text).project

        params = _params(project, "output", OutputParams)[_page(project, 2)]
        assert params.despeckle_level is expected

    def test_depth_perception_is_clamped_to_the_supported_range(self, tmp_path: Path):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8").replace(
            'depthPerception="2.5"', 'depthPerception="9"'
        )

        project = import_xml_project(project_file, text).project

        params = _params(project, "output", OutputParams)[_page(project, 1)]
        assert params.depth_perception == 3.0

    def test_distortion_model_falls_back_to_spline_control_points(self, tmp_path: Path):
        xml = """<project version="3" outputDirectory="out">
<directories><directory id="1" path="s"/></directories>
<files><file id="2" dirId="1" name="a.tif"/></files>
<images><image id="3" subPages="1" fileId="2" fileImage="0"/></images>
<pages><page id="4" imageId="3" subPage="single"/></pages>
<filters><output><page id="4"><params><distortion-model>
<top-curve><xspline>
<point x="0" y="1"/><point x="50" y="0"/><point x="100" y="1"/>
</xspline></top-curve>
<bottom-curve><xspline>
<point x="0" y="90"/><point x="100" y="91"/>
</xspline></bottom-curve>
</distortion-model></params></page></output></filters>
</project>"""

        project = import_xml_project(tmp_path / "p.ScanTailor", xml).project

        model = _params(project, "output", OutputParams)[
            _page(project, 0)
        ].distortion_model
        assert model.top == ((0.0, 1.0), (50.0, 0.0), (100.0, 1.0))
        assert model.bottom == ((0.0, 90.0), (100.0, 91.0))


class TestOutputFileNames:
    def test_files_sharing_a_name_keep_their_disambiguation_labels(
        self, tmp_path: Path
    ):
        imported = _load(tmp_path, "simple")

        labels = {
            path.parent.name + "/" + path.name: label
            for path, label in imported.output_file_labels.items()
        }
        assert labels == {
            "scans/page_001.tif": 0,
            "scans/page_002.tif": 0,
            "extra/page_001.tif": 1,
        }


class TestStageDefaults:
    def test_defaults_are_installed_into_the_stages(self, tmp_path: Path):
        imported = _load(tmp_path, "multipage_tiff")
        stages = StageSequence()

        apply_xml_stage_defaults(stages, imported.stage_defaults)

        assert (
            stages.page_split_filter.settings.get_default_layout_type()
            is LayoutType.SINGLE_PAGE_UNCUT
        )

    def test_page_detection_values_are_installed_into_select_content(
        self, tmp_path: Path
    ):
        imported = _load(tmp_path, "split_and_mixed")
        stages = StageSequence()

        apply_xml_stage_defaults(stages, imported.stage_defaults)

        select_content = stages.select_content_filter.settings
        assert select_content.get_page_detection_box() == (210.0, 297.0)
        assert select_content.get_page_detection_tolerance() == pytest.approx(0.15)

    def test_project_without_filters_has_defaults_and_no_stage_params(
        self, tmp_path: Path
    ):
        xml = """<project version="3" outputDirectory="out">
<directories><directory id="1" path="s"/></directories>
<files><file id="2" dirId="1" name="a.tif"/></files>
<images><image id="3" subPages="1" fileId="2" fileImage="0"/></images>
<pages><page id="4" imageId="3" subPage="single"/></pages>
</project>"""

        imported = import_xml_project(tmp_path / "p.ScanTailor", xml)

        assert imported.project.stage_params == {}
        assert (
            imported.stage_defaults.page_split_layout_type
            is LayoutType.AUTO_LAYOUT_TYPE
        )
        assert imported.stage_defaults.page_detection_tolerance == pytest.approx(0.1)


class TestMalformedInput:
    @pytest.mark.parametrize(
        ("old", "new", "stage", "kept"),
        [
            (
                'mode="manual" angle="-1.5"',
                'mode="manual" angle="abc"',
                "deskew",
                2,
            ),
            (
                'mode="manual" angle="-1.5"',
                'mode="manual" angle="90"',
                "deskew",
                2,
            ),
            (
                '<rotation degrees="180"/></image><image-settings',
                '<rotation degrees="45"/></image><image-settings',
                "fix_orientation",
                0,
            ),
            ('hor="left" null="0"', 'hor="left" null="x"', "page_layout", 2),
            ('windowSize="60"', 'windowSize="sixty"', "output", 2),
            ('<point x="10" y="10"/>', '<point x="abc" y="10"/>', "output", 2),
            (
                '<cutter1><p1 x="58.5" y="0"/>',
                '<cutter1><p1 x="oops" y="0"/>',
                "page_split",
                1,
            ),
        ],
        ids=[
            "deskew-angle-not-a-number",
            "deskew-angle-out-of-range",
            "rotation-not-orthogonal",
            "alignment-null-not-a-number",
            "binarization-window-not-a-number",
            "zone-coordinate-not-a-number",
            "split-line-coordinate-not-a-number",
        ],
    )
    def test_one_malformed_element_is_skipped_and_the_rest_is_kept(
        self, tmp_path: Path, old: str, new: str, stage: str, kept: int
    ):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8")
        baseline = import_xml_project(project_file, text).project
        assert old in text

        project = import_xml_project(project_file, text.replace(old, new, 1)).project

        assert len(project.stage_params.get(stage, [])) == kept
        assert len(project.images) == 2
        assert len(project.get_pages()) == 3
        for other, entries in baseline.stage_params.items():
            if other != stage:
                assert project.stage_params[other] == entries

    def test_malformed_pages_do_not_disturb_the_settings_of_other_pages(
        self, tmp_path: Path
    ):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8").replace(
            'angle="0.75"', 'angle="NaN"'
        )

        project = import_xml_project(project_file, text).project

        deskew = _params(project, "deskew", DeskewParams)
        assert set(deskew) == {_page(project, 0), _page(project, 2)}
        assert deskew[_page(project, 0)].deskew_angle_deg == pytest.approx(-1.5)

    def test_settings_for_unknown_pages_and_ids_are_ignored(self, tmp_path: Path):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8").replace(
            "<deskew>",
            '<deskew><page id="999"><params mode="auto" angle="1"/></page>'
            '<page id="x"><params mode="auto" angle="1"/></page>',
        )

        project = import_xml_project(project_file, text).project

        assert len(project.stage_params["deskew"]) == 3

    def test_unrecognised_filter_elements_are_ignored(self, tmp_path: Path):
        project_file = _fixture(tmp_path, "newer_version")
        text = project_file.read_text(encoding="utf-8").replace(
            'version="5"', 'version="3"'
        )

        project = import_xml_project(project_file, text).project

        assert len(project.images) == 1
        assert project.stage_params == {}

    def test_image_with_an_impossible_sub_page_count_is_skipped(self, tmp_path: Path):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8").replace(
            'subPages="2"', 'subPages="7"'
        )

        project = import_xml_project(project_file, text).project

        assert [info.id.file_path.name for info in project.images] == ["cover.tif"]
        assert [page.sub_page for page in project.get_pages()] == [SubPage.SINGLE_PAGE]

    def test_page_with_an_unknown_sub_page_name_is_skipped(self, tmp_path: Path):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8").replace(
            'subPage="right" selected="selected"',
            'subPage="middle" selected="selected"',
        )

        project = import_xml_project(project_file, text).project

        assert project.selected_page is None
        assert "deskew" in project.stage_params
        assert len(project.stage_params["deskew"]) == 2

    def test_image_pointing_at_an_unknown_file_is_left_out(self, tmp_path: Path):
        project_file = _fixture(tmp_path, "split_and_mixed")
        text = project_file.read_text(encoding="utf-8").replace(
            'fileId="6"', 'fileId="60"'
        )

        project = import_xml_project(project_file, text).project

        assert [info.id.file_path.name for info in project.images] == ["spread.tif"]

    def test_absolute_paths_from_another_platform_are_kept_as_written(
        self, tmp_path: Path
    ):
        xml = """<project version="3" outputDirectory="C:/out">
<directories><directory id="1" path="C:/Users/someone/scans"/></directories>
<files><file id="2" dirId="1" name="a.tif"/></files>
<images><image id="3" subPages="1" fileId="2" fileImage="0"/></images>
<pages><page id="4" imageId="3" subPage="single"/></pages>
</project>"""

        project = import_xml_project(tmp_path / "p.ScanTailor", xml).project

        assert str(project.images[0].id.file_path).replace("\\", "/") == (
            "C:/Users/someone/scans/a.tif"
        )
        assert str(project.output_directory).replace("\\", "/") == "C:/out"


class TestJsonRoundTrip:
    def _stage_params_data(self, project: Project) -> dict[str, object]:
        return project.model_dump(mode="json")["stage_params"]

    def test_json_save_and_reload_keeps_every_setting(self, tmp_path: Path):
        project = Project.load(_fixture(tmp_path, "split_and_mixed"))
        saved = tmp_path / "saved.ScanTailor"

        project.save(saved)
        reloaded = Project.load(saved)

        assert reloaded.stage_params == project.stage_params
        assert reloaded.images == project.images
        assert reloaded.selected_page == project.selected_page
        assert _params(reloaded, "output", OutputParams)[
            _page(reloaded, 1)
        ].distortion_model.is_defined()

    def test_reloaded_settings_reach_the_stage_filters_unchanged(self, tmp_path: Path):
        project = Project.load(_fixture(tmp_path, "split_and_mixed"))
        saved = tmp_path / "saved.ScanTailor"
        project.save(saved)
        reloaded = Project.load(saved)
        stages = StageSequence()

        apply_stage_params(stages, reloaded.stage_params)

        left, right, cover = reloaded.get_pages()
        assert stages.deskew_filter.settings.get_params(left).deskew_angle_deg == (
            pytest.approx(-1.5)
        )
        assert stages.deskew_filter.settings.get_params(left).is_manual()
        layout = stages.page_layout_filter.settings.get_params(left)
        assert layout.hard_margins_mm.left == 12.5
        output = stages.output_filter.settings.get_params(left)
        assert output.color_mode is ColorMode.MIXED
        assert len(output.picture_zones) == 2
        right_output = stages.output_filter.settings.get_params(right)
        assert right_output.dewarping.mode is DewarpingMode.MANUAL
        assert stages.fix_orientation_filter.settings.get_rotation_for(
            cover.image_id
        ) == OrthogonalRotation(degrees=180)
        split = stages.page_split_filter.settings.get_params(left)
        assert split.layout_type is LayoutType.TWO_PAGES
        assert split.page_layout is not None

    def test_stages_collect_back_exactly_what_was_imported(self, tmp_path: Path):
        project = Project.load(_fixture(tmp_path, "split_and_mixed"))
        stages = StageSequence()
        apply_stage_params(stages, project.stage_params)

        collected = collect_stage_params(stages, project.get_pages())

        # Page Split stores one record per image but collects one per page.
        for stage, entries in project.stage_params.items():
            if stage == "page_split":
                assert {(e.page.image_id, str(e.params)) for e in collected[stage]} == {
                    (e.page.image_id, str(e.params)) for e in entries
                }
            else:
                assert collected[stage] == entries

    @pytest.mark.parametrize("name", ["simple", "multipage_tiff"])
    def test_other_fixtures_survive_a_json_round_trip(self, tmp_path: Path, name: str):
        project = Project.load(_fixture(tmp_path, name))
        saved = tmp_path / "saved.ScanTailor"

        project.save(saved)

        assert Project.load(saved).stage_params == project.stage_params

    def test_image_ids_in_entries_match_the_project_images(self, tmp_path: Path):
        project = Project.load(_fixture(tmp_path, "simple"))

        known = set(project.image_ids())
        for entries in project.stage_params.values():
            for entry in entries:
                assert entry.page.image_id in known
        assert isinstance(next(iter(known)), ImageId)


_PROJECT_XML = """{declaration}<project version="3" outputDirectory="out">
<directories><directory id="1" path="{directory}"/></directories>
<files><file id="2" dirId="1" name="{name}"/></files>
<images><image id="3" subPages="1" fileId="2" fileImage="0"/></images>
<pages><page id="4" imageId="3" subPage="single"/></pages>
</project>
"""


def _write_project(
    tmp_path: Path, encoding: str, declaration: str = "", directory: str = "Café"
) -> Path:
    text = _PROJECT_XML.format(
        declaration=declaration, directory=directory, name="scan.tif"
    )
    project_file = tmp_path / "p.ScanTailor"
    project_file.write_bytes(text.encode(encoding))
    return project_file


class TestProjectFileEncoding:
    def test_utf8_file_with_accented_path_is_read_unchanged(self, tmp_path: Path):
        project_file = _write_project(tmp_path, "utf-8")

        imported = import_xml_project_file(project_file)

        assert imported is not None
        assert imported.project.images[0].id.file_path.parent.name == "Café"

    def test_cp1252_file_without_declaration_falls_back_to_the_codepage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(xml_reader, "_ansi_codepage", lambda: "cp1252")
        project_file = _write_project(tmp_path, "cp1252")

        imported = import_xml_project_file(project_file)

        assert imported is not None
        assert imported.project.images[0].id.file_path.parent.name == "Café"

    def test_cp1252_fallback_logs_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(xml_reader, "_ansi_codepage", lambda: "cp1252")
        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            import_xml_project_file(_write_project(tmp_path, "cp1252"))
        finally:
            logger.remove(sink)

        assert any("cp1252" in message for message in messages)

    def test_declared_codepage_is_used_directly(self, tmp_path: Path):
        project_file = _write_project(
            tmp_path,
            "cp1252",
            declaration='<?xml version="1.0" encoding="windows-1252"?>\n',
        )

        imported = import_xml_project_file(project_file)

        assert imported is not None
        assert imported.project.images[0].id.file_path.parent.name == "Café"

    def test_declared_non_latin_codepage_is_honoured(self, tmp_path: Path):
        project_file = _write_project(
            tmp_path,
            "cp1251",
            declaration='<?xml version="1.0" encoding="windows-1251"?>\n',
            directory="Книга",
        )

        imported = import_xml_project_file(project_file)

        assert imported is not None
        assert imported.project.images[0].id.file_path.parent.name == ("Книга")

    def test_unknown_declared_encoding_does_not_crash(self, tmp_path: Path):
        project_file = _write_project(
            tmp_path,
            "utf-8",
            declaration='<?xml version="1.0" encoding="no-such-codec"?>\n',
        )

        imported = import_xml_project_file(project_file)

        assert imported is not None
        assert imported.project.images[0].id.file_path.parent.name == "Café"

    def test_codepage_that_cannot_decode_falls_back_to_cp1252(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(xml_reader, "_ansi_codepage", lambda: "ascii")
        project_file = _write_project(tmp_path, "cp1252")

        imported = import_xml_project_file(project_file)

        assert imported is not None
        assert len(imported.project.images) == 1

    def test_project_load_reads_a_cp1252_xml_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(xml_reader, "_ansi_codepage", lambda: "cp1252")
        project_file = _write_project(tmp_path, "cp1252")

        project = Project.load(project_file)

        assert project.images[0].id.file_path.parent.name == "Café"


class TestProjectFieldsFromXml:
    def test_output_labels_are_stored_in_the_project(self, tmp_path: Path):
        imported = _load(tmp_path, "simple")

        assert imported.project.output_file_label_map() == imported.output_file_labels
        assert sorted(imported.project.output_file_label_map().values()) == [0, 0, 1]

    def test_stage_defaults_are_stored_in_the_project(self, tmp_path: Path):
        project = _load(tmp_path, "split_and_mixed").project

        assert project.stage_defaults["page_detection_box"] == [210.0, 297.0]
        assert project.stage_defaults["page_detection_tolerance"] == pytest.approx(0.15)

    def test_stored_defaults_restore_the_same_stages_as_the_xml_import(
        self, tmp_path: Path
    ):
        imported = _load(tmp_path, "split_and_mixed")
        from_xml = StageSequence()
        apply_xml_stage_defaults(from_xml, imported.stage_defaults)
        from_project = StageSequence()

        apply_stage_defaults(from_project, imported.project.stage_defaults)

        assert collect_stage_defaults(from_project) == collect_stage_defaults(from_xml)

    def test_imported_labels_and_defaults_survive_a_json_round_trip(
        self, tmp_path: Path
    ):
        imported = _load(tmp_path, "split_and_mixed")
        saved = tmp_path / "converted.json"

        imported.project.save(saved)
        loaded = Project.load(saved)

        assert (
            imported.output_file_labels.items()
            <= loaded.output_file_label_map().items()
        )
        assert loaded.stage_defaults == imported.project.stage_defaults
