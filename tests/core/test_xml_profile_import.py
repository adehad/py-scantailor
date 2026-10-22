"""Converting Default Parameters profiles saved by the C++ ScanTailor Advanced."""

from pathlib import Path

import pytest

from scantailor.core.default_params import DefaultParams
from scantailor.core.default_params_profiles import ProfileError, ProfileStore
from scantailor.core.models import Dpi, Margins
from scantailor.core.settings import Units
from scantailor.core.xml_profile_import import parse_xml_profile, read_xml_profile
from scantailor.filters.deskew.params import AutoManualMode
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import FillingColor
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.picture_shape import PictureShape
from scantailor.filters.page_layout.alignment import (
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "xml_profiles" / "Thick books.stp"


def _profile(default_params: str, version: str = "3") -> str:
    return (
        f'<profile version="{version}">'
        f"<default-params>{default_params}</default-params>"
        "</profile>"
    )


@pytest.fixture(scope="module")
def imported() -> DefaultParams:
    return read_xml_profile(FIXTURE)


class TestEveryStageIsRead:
    def test_rotation_deskew_and_layout_type(self, imported: DefaultParams) -> None:
        assert imported.fix_orientation.image_rotation.degrees == 270
        assert imported.deskew.mode is AutoManualMode.MANUAL
        assert imported.deskew.deskew_angle_deg == pytest.approx(1.5)
        assert imported.deskew.auto_oblique
        assert imported.page_split.layout_type is LayoutType.TWO_PAGES

    def test_select_content_modes_and_page_size(self, imported: DefaultParams) -> None:
        select = imported.select_content
        assert select.content_detection_mode is ContentDetectionMode.DISABLED
        assert select.page_detection_mode is PageDetectionMode.MANUAL
        assert select.fine_tune_corners
        assert (
            select.page_rect_size_mm.width_mm,
            select.page_rect_size_mm.height_mm,
        ) == (
            pytest.approx(148.0),
            pytest.approx(210.0),
        )

    def test_page_layout_margins_and_alignment(self, imported: DefaultParams) -> None:
        layout = imported.page_layout
        assert layout.hard_margins_mm == Margins(top=4, bottom=6, left=12, right=8)
        assert layout.alignment.vertical is VerticalAlignment.TOP
        assert layout.alignment.horizontal is HorizontalAlignment.LEFT
        assert not layout.alignment.is_null
        assert not layout.auto_margins

    def test_output_options_under_their_profile_element_names(
        self, imported: DefaultParams
    ) -> None:
        output = imported.output
        assert output.dpi == Dpi.uniform(400)
        assert output.color_mode is ColorMode.MIXED
        assert output.despeckle_level is DespeckleLevel.NORMAL
        assert output.depth_perception == pytest.approx(2.5)
        assert output.color_common.filling_color is FillingColor.WHITE
        assert output.color_common.posterization.level == 6
        assert output.splitting.split_output and output.splitting.color_foreground
        assert output.picture_shape.shape is PictureShape.RECTANGULAR
        assert output.picture_shape.sensitivity == 80
        assert output.dewarping.mode is DewarpingMode.AUTO

    def test_binarization_keeps_the_stored_wolf_bounds(
        self, imported: DefaultParams
    ) -> None:
        bw = imported.output.binarization
        assert bw.method is BinarizationMethod.WOLF
        assert bw.threshold_adjustment == -10
        assert bw.window_size == 120
        assert (bw.wolf_lower_bound, bw.wolf_upper_bound) == (12, 254)

    def test_options_a_profile_lacks_keep_the_port_defaults(
        self, imported: DefaultParams
    ) -> None:
        defaults = DefaultParams().output
        assert imported.output.black_on_white == defaults.black_on_white
        assert (
            imported.output.dewarp_min_span_percent == defaults.dewarp_min_span_percent
        )

    def test_units_are_read(self, imported: DefaultParams) -> None:
        assert imported.units is Units.CENTIMETERS


class TestValuesAreReadAsCxxReadsThem:
    def test_missing_stage_elements_read_like_empty_ones(self) -> None:
        params = parse_xml_profile(_profile(""))

        assert params.deskew.mode is AutoManualMode.AUTO
        assert params.page_split.layout_type is LayoutType.AUTO_LAYOUT_TYPE
        assert params.select_content.page_detection_mode is PageDetectionMode.AUTO
        assert params.units is Units.MILLIMETERS

    def test_a_deskew_angle_beyond_the_dialog_range_is_clamped(self) -> None:
        params = parse_xml_profile(
            _profile('<deskew-params deskewAngleDeg="80" mode="manual"/>')
        )

        assert params.deskew.deskew_angle_deg == 45.0

    def test_a_negative_rotation_is_the_same_turn_counted_forwards(self) -> None:
        params = parse_xml_profile(
            _profile(
                "<fix-orientation-params>"
                '<imageRotation degrees="-90"/>'
                "</fix-orientation-params>"
            )
        )

        assert params.fix_orientation.image_rotation.degrees == 270

    def test_an_older_format_version_is_read(self) -> None:
        params = parse_xml_profile(
            _profile('<page-split-params layoutType="single-cut"/>', version="2")
        )

        assert params.page_split.layout_type is LayoutType.PAGE_PLUS_OFFCUT


class TestFilesThatAreNotProfilesAreRefused:
    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            ("not xml at all", "not valid XML"),
            ('<project version="3"/>', "not <profile>"),
            ("<profile><default-params/></profile>", "no readable version"),
            (_profile("", version="99"), "reads versions"),
            ('<profile version="3"/>', "no <default-params>"),
            (
                _profile(
                    "<fix-orientation-params>"
                    '<imageRotation degrees="45"/>'
                    "</fix-orientation-params>"
                ),
                "invalid value",
            ),
            (_profile('<deskew-params deskewAngleDeg="abc"/>'), "invalid value"),
        ],
    )
    def test_refusal_names_the_reason(self, content: str, reason: str) -> None:
        with pytest.raises(ProfileError, match=reason):
            parse_xml_profile(content)

    def test_a_missing_file_is_a_profile_error(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileError, match="Cannot read"):
            read_xml_profile(tmp_path / "gone.stp")


def test_an_imported_profile_is_stored_as_json_and_reads_back_equal(
    tmp_path: Path, imported: DefaultParams
) -> None:
    store = ProfileStore(tmp_path / "profiles")

    store.write(FIXTURE.stem, imported)

    saved = tmp_path / "profiles" / "saved" / "Thick books.json"
    assert saved.is_file()
    assert store.read("Thick books") == imported
