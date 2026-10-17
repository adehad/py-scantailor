"""Page Layout guides and flag, the despeckle factor, the Sauvola coefficient."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scantailor.core.stage_params import apply_stage_defaults, collect_stage_defaults
from scantailor.core.stage_sequence import StageSequence
from scantailor.core.xml_compat import (
    apply_xml_stage_defaults,
    import_xml_project,
)
from scantailor.core.xml_reader import XmlProjectContext
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.output.xml_import import _read_params as read_output_params
from scantailor.filters.page_layout.guide import Guide
from scantailor.filters.page_layout.xml_import import read_page_layout_stage

FIXTURES = Path(__file__).parent.parent / "fixtures" / "xml_projects"


def _stage(xml: str):
    return read_page_layout_stage(ET.fromstring(xml), XmlProjectContext())


class TestPageLayoutStageSettings:
    def test_guides_are_read_with_orientation_and_position(self):
        stage = _stage(
            '<page-layout showMiddleRect="1"><guides>'
            '<guide orientation="horizontal" position="-12.5"/>'
            '<guide orientation="vertical" position="30"/>'
            "</guides></page-layout>"
        )

        assert stage.guides == (
            Guide(position_mm=-12.5, is_horizontal=True),
            Guide(position_mm=30.0, is_horizontal=False),
        )

    def test_an_unknown_orientation_reads_as_horizontal_like_c_plus_plus(self):
        stage = _stage(
            '<page-layout><guides><guide orientation="diagonal" position="1"/>'
            "</guides></page-layout>"
        )

        assert stage.guides == (Guide(position_mm=1.0, is_horizontal=True),)

    def test_a_guide_with_an_unreadable_position_is_skipped_and_logged(
        self, log_messages: list[str]
    ):
        stage = _stage(
            "<page-layout><guides>"
            '<guide orientation="vertical" position="abc"/>'
            '<guide orientation="vertical" position="4"/>'
            "</guides></page-layout>"
        )

        assert stage.guides == (Guide(position_mm=4.0, is_horizontal=False),)
        assert any("guide" in message for message in log_messages)

    @pytest.mark.parametrize(("flag", "shown"), [("1", True), ("0", False)])
    def test_the_middle_rectangle_flag_is_read(self, flag: str, shown: bool):
        stage = _stage(f'<page-layout showMiddleRect="{flag}"/>')

        assert stage.middle_rect_shown is shown

    def test_an_absent_flag_leaves_the_rectangle_shown(self):
        # C++ reads an absent attribute as hidden, so a project saved before the
        # attribute existed opens with the rectangle hidden.
        assert _stage("<page-layout/>").middle_rect_shown is True

    def test_a_project_without_a_page_layout_element_gets_the_defaults(self):
        stage = read_page_layout_stage(None, XmlProjectContext())

        assert stage.middle_rect_shown is True
        assert stage.guides == ()
        assert stage.entries == []


class TestProjectImportKeepsThem:
    @staticmethod
    def _import(tmp_path: Path, name: str, replace: tuple[str, str] | None = None):
        target = tmp_path / name
        shutil.copytree(FIXTURES / name, target)
        project_file = target / f"{name}.ScanTailor"
        text = project_file.read_text(encoding="utf-8")
        if replace is not None:
            assert replace[0] in text
            text = text.replace(*replace)
        return import_xml_project(project_file, text)

    def test_an_explicit_hidden_rectangle_is_kept(self, tmp_path: Path):
        imported = self._import(tmp_path, "simple")

        assert imported.stage_defaults.page_layout_middle_rect_shown is False
        assert imported.project.stage_defaults["page_layout_middle_rect_shown"] is False

    def test_guides_reach_the_stages_through_the_project_defaults(self, tmp_path: Path):
        imported = self._import(
            tmp_path,
            "simple",
            (
                '<page-layout showMiddleRect="0">',
                '<page-layout showMiddleRect="0"><guides>'
                '<guide orientation="vertical" position="7.5"/></guides>',
            ),
        )
        stages = StageSequence()

        apply_stage_defaults(stages, imported.project.stage_defaults)

        settings = stages.page_layout_filter.settings
        assert settings.get_guides() == (Guide(position_mm=7.5, is_horizontal=False),)
        assert settings.is_middle_rect_shown() is False

    def test_the_direct_xml_installer_sets_them_too(self, tmp_path: Path):
        imported = self._import(tmp_path, "simple")
        stages = StageSequence()

        apply_xml_stage_defaults(stages, imported.stage_defaults)

        assert stages.page_layout_filter.settings.is_middle_rect_shown() is False


class TestStageDefaultsRoundTrip:
    def test_guides_and_the_flag_survive_collect_and_apply(self):
        source = StageSequence()
        layout = source.page_layout_filter.settings
        layout.set_guides([Guide(position_mm=-3.0, is_horizontal=True)])
        layout.set_middle_rect_shown(False)

        target = StageSequence()
        apply_stage_defaults(target, collect_stage_defaults(source))

        restored = target.page_layout_filter.settings
        assert restored.get_guides() == (Guide(position_mm=-3.0, is_horizontal=True),)
        assert restored.is_middle_rect_shown() is False

    def test_a_file_saved_without_the_keys_keeps_the_defaults(self):
        stages = StageSequence()

        apply_stage_defaults(stages, {})

        assert stages.page_layout_filter.settings.get_guides() == ()
        assert stages.page_layout_filter.settings.is_middle_rect_shown() is True


def _output_params(attributes: str = "", bw: str = "") -> OutputParams:
    element = ET.fromstring(
        f'<params {attributes}><color-params colorMode="bw">'
        f'<bw binarizationMethod="sauvola" {bw}/></color-params></params>'
    )
    return read_output_params(element)


class TestDespeckleFactor:
    def test_a_value_between_the_named_levels_is_stored_as_is(self):
        params = _output_params('despeckleLevel="1.7"')

        assert params.despeckle_factor == 1.7
        assert params.despeckle_level is DespeckleLevel.NORMAL
        assert params.despeckle_strength == 1.7

    def test_a_value_the_slider_gives_just_below_a_midpoint_stays_cautious(self):
        params = _output_params('despeckleLevel="1.4"')

        assert params.despeckle_level is DespeckleLevel.CAUTIOUS
        assert params.despeckle_strength == 1.4

    @pytest.mark.parametrize(
        ("text", "level"),
        [
            ("0", DespeckleLevel.OFF),
            ("1", DespeckleLevel.CAUTIOUS),
            ("2", DespeckleLevel.NORMAL),
            ("3", DespeckleLevel.AGGRESSIVE),
        ],
    )
    def test_a_preset_value_needs_no_factor(self, text: str, level: DespeckleLevel):
        params = _output_params(f'despeckleLevel="{text}"')

        assert params.despeckle_level is level
        assert params.despeckle_factor is None

    def test_a_value_beyond_the_slider_range_is_limited(self):
        assert _output_params('despeckleLevel="9"').despeckle_factor == 3.5
        assert _output_params('despeckleLevel="-2"').despeckle_strength == 0.0

    def test_the_factor_survives_a_json_round_trip(self):
        params = _output_params('despeckleLevel="1.7"')

        restored = OutputParams.model_validate(params.model_dump(mode="json"))

        assert restored == params

    def test_a_stored_factor_without_a_level_derives_the_level(self):
        params = OutputParams.model_validate({"despeckle_factor": 2.6})

        assert params.despeckle_level is DespeckleLevel.AGGRESSIVE

    def test_choosing_a_named_level_drops_the_factor(self):
        params = OutputParams().with_despeckle_factor(1.7)

        chosen = params.with_despeckle_level(DespeckleLevel.CAUTIOUS)

        assert chosen.despeckle_factor is None
        assert chosen.despeckle_strength == 1.0

    @pytest.mark.parametrize("factor", [-0.1, 3.6])
    def test_a_factor_outside_the_range_is_rejected(self, factor: float):
        with pytest.raises(ValueError, match="despeckle factor"):
            OutputParams().with_despeckle_factor(factor)

    def test_apply_to_pages_copies_the_factor_with_the_level(self):
        source = OutputParams().with_despeckle_factor(1.7)

        from scantailor.filters.output.params import ApplyGroup

        copied = OutputParams().with_group_from(source, ApplyGroup.DESPECKLE)

        assert copied.despeckle_strength == 1.7


class TestBradleyAndEdgeMethodsReadTheSauvolaCoefficient:
    """C++ gives Bradley, EdgePlus, BlurDiv and EdgeDiv no coefficient of their own."""

    @pytest.mark.parametrize(
        "method",
        ["bradley", "edgeplus", "blurdiv", "edgediv"],
    )
    def test_the_sauvola_coefficient_drives_the_method_not_the_wolf_one(
        self, method: str
    ):
        element = ET.fromstring(
            '<params><color-params colorMode="bw">'
            f'<bw binarizationMethod="{method}" sauvolaCoef="0.21" wolfCoef="0.9"/>'
            "</color-params></params>"
        )

        binarization = read_output_params(element).binarization

        assert binarization.sauvola_coef == 0.21
        if method == "bradley":
            assert binarization.bradley_coef == 0.21
        if method in ("edgeplus", "edgediv"):
            assert binarization.edge_div_kep == 0.21
        if method in ("blurdiv", "edgediv"):
            assert binarization.edge_div_kbd == 0.21
        assert binarization.method is BinarizationMethod(method)
