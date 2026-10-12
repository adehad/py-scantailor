"""Tests for the oblique model: params, settings, persistence and detection."""

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.stage_params import apply_stage_params, collect_stage_params
from scantailor.core.stage_sequence import StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.core.xml_reader import XmlProjectContext
from scantailor.filters.deskew.dependencies import Dependencies
from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.params import MAX_OBLIQUE_DEG, AutoManualMode, Params
from scantailor.filters.deskew.settings import Detection, Settings
from scantailor.filters.deskew.xml_import import read_deskew

from .synthetic_pages import sheared_page, text_page

AUTO = AutoManualMode.AUTO
MANUAL = AutoManualMode.MANUAL


def make_page_id(name: str = "page") -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


DEPENDENCIES = Dependencies(
    page_outline=((0.0, 0.0), (700.0, 0.0), (700.0, 900.0), (0.0, 900.0)),
    rotation_degrees=0,
)
OTHER_DEPENDENCIES = Dependencies(
    page_outline=((0.0, 0.0), (500.0, 0.0), (500.0, 900.0), (0.0, 900.0)),
    rotation_degrees=0,
)


class TestParams:
    """Oblique angle and mode sit next to the rotation's."""

    def test_a_new_page_has_no_oblique_correction(self):
        params = Params()

        assert params.oblique_angle_deg == 0.0
        assert params.oblique_mode is MANUAL

    def test_oblique_mode_is_independent_of_the_rotation_mode(self):
        params = Params(mode=MANUAL, oblique_mode=AUTO)

        assert params.is_manual() and params.is_oblique_auto()
        assert not params.is_auto() and not params.is_oblique_manual()

    def test_oblique_angle_is_limited_to_five_degrees(self):
        with pytest.raises(ValueError):
            Params(oblique_angle_deg=MAX_OBLIQUE_DEG + 0.1)
        with pytest.raises(ValueError):
            Params(oblique_angle_deg=-MAX_OBLIQUE_DEG - 0.1)

    def test_oblique_angle_at_the_limits_is_valid(self):
        assert Params(oblique_angle_deg=MAX_OBLIQUE_DEG).oblique_angle_deg == 5.0
        assert Params(oblique_angle_deg=-MAX_OBLIQUE_DEG).oblique_angle_deg == -5.0

    def test_with_manual_oblique_keeps_the_rotation(self):
        params = Params(deskew_angle_deg=3.0, mode=AUTO)

        changed = params.with_manual_oblique(-2.0)

        assert changed.oblique_angle_deg == -2.0
        assert changed.oblique_mode is MANUAL
        assert changed.deskew_angle_deg == 3.0
        assert changed.mode is AUTO

    def test_with_angle_keeps_the_oblique(self):
        params = Params(oblique_angle_deg=1.5, oblique_mode=AUTO)

        changed = params.with_manual_angle(4.0)

        assert changed.oblique_angle_deg == 1.5
        assert changed.oblique_mode is AUTO

    def test_with_oblique_keeps_the_mode_unless_given(self):
        params = Params(oblique_mode=AUTO)

        assert params.with_oblique(1.0).oblique_mode is AUTO
        assert params.with_oblique(1.0, MANUAL).oblique_mode is MANUAL

    def test_making_a_part_manual_drops_its_redetection_request(self):
        params = Params(redetect_rotation=True, redetect_oblique=True)

        assert params.with_manual_angle(1.0).redetect_rotation is False
        assert params.with_manual_oblique(1.0).redetect_oblique is False

    def test_json_round_trip_keeps_the_oblique(self):
        original = Params(
            deskew_angle_deg=1.25,
            mode=MANUAL,
            oblique_angle_deg=-3.5,
            oblique_mode=AUTO,
            dependencies=DEPENDENCIES,
        )

        restored = Params.model_validate_json(original.model_dump_json())

        assert restored == original

    def test_params_saved_before_the_oblique_existed_load_without_one(self):
        old = {"deskew_angle_deg": 2.0, "mode": "manual", "dependencies": {}}

        restored = Params.model_validate(old)

        assert restored.oblique_angle_deg == 0.0
        assert restored.oblique_mode is MANUAL
        assert restored.mode is MANUAL

    def test_stage_params_round_trip_through_the_project_data(self):
        page = make_page_id()
        stages = StageSequence()
        stages.deskew_filter.set_params(
            page,
            Params(
                deskew_angle_deg=0.5,
                oblique_angle_deg=2.5,
                oblique_mode=MANUAL,
                dependencies=DEPENDENCIES,
            ),
        )
        exported = collect_stage_params(stages, [page])

        restored = StageSequence()
        apply_stage_params(restored, exported)

        stored = restored.deskew_filter.get_params(page)
        assert stored.oblique_angle_deg == 2.5
        assert stored.oblique_mode is MANUAL
        assert stored.deskew_angle_deg == 0.5


class TestXmlImport:
    """The C++ attributes oblique, oblique-mode and the transitional autoOblique."""

    @staticmethod
    def _read(**attributes: str) -> Params:
        page = make_page_id()
        context = XmlProjectContext(page_ids={1: page})
        deskew = ET.Element("deskew")
        page_element = ET.SubElement(deskew, "page", id="1")
        ET.SubElement(page_element, "params", **attributes)
        [entry] = read_deskew(deskew, context)
        return Params.model_validate(entry.params)

    def test_oblique_angle_and_mode_are_read(self):
        params = self._read(
            mode="manual", angle="1.5", oblique="-2.25", **{"oblique-mode": "manual"}
        )

        assert params.oblique_angle_deg == pytest.approx(-2.25)
        assert params.oblique_mode is MANUAL
        assert params.deskew_angle_deg == pytest.approx(1.5)

    def test_oblique_mode_is_independent_of_the_rotation_mode(self):
        params = self._read(
            mode="manual", angle="1", oblique="0.5", **{"oblique-mode": "auto"}
        )

        assert params.mode is MANUAL
        assert params.oblique_mode is AUTO

    def test_oblique_mode_wins_over_the_transitional_flag(self):
        params = self._read(
            mode="auto",
            angle="0",
            oblique="0",
            autoOblique="1",
            **{"oblique-mode": "manual"},
        )

        assert params.oblique_mode is MANUAL

    @pytest.mark.parametrize(("flag", "mode"), [("1", AUTO), ("0", MANUAL)])
    def test_transitional_auto_oblique_flag_sets_the_oblique_mode(
        self, flag: str, mode: AutoManualMode
    ):
        params = self._read(mode="auto", angle="0", oblique="0", autoOblique=flag)

        assert params.oblique_mode is mode

    def test_an_oblique_angle_without_a_mode_follows_the_rotation_mode(self):
        assert self._read(mode="manual", angle="0", oblique="1").oblique_mode is MANUAL
        assert self._read(mode="auto", angle="0", oblique="1").oblique_mode is AUTO

    def test_a_file_without_any_oblique_attribute_stays_unsheared(self):
        params = self._read(mode="auto", angle="1.5")

        assert params.oblique_angle_deg == 0.0
        assert params.oblique_mode is MANUAL

    def test_an_oblique_angle_beyond_the_limit_is_clamped(self):
        params = self._read(
            mode="manual", angle="0", oblique="12", **{"oblique-mode": "manual"}
        )

        assert params.oblique_angle_deg == MAX_OBLIQUE_DEG


class TestSettings:
    """Settings keep rotation and oblique apart."""

    def test_a_page_without_params_follows_the_oblique_default(self):
        settings = Settings()
        page = make_page_id()

        assert settings.get_params(page).oblique_mode is MANUAL
        settings.set_auto_oblique_default(True)
        assert settings.get_params(page).oblique_mode is AUTO

    def test_request_detection_changes_only_that_part(self):
        settings = Settings()
        page = make_page_id()
        settings.set_params(
            page,
            Params(
                deskew_angle_deg=2.0,
                mode=MANUAL,
                oblique_angle_deg=3.0,
                oblique_mode=MANUAL,
            ),
        )

        settings.request_detection(page, Detection.ROTATION)

        stored = settings.get_params(page)
        assert stored.mode is AUTO and stored.redetect_rotation
        assert stored.oblique_angle_deg == 3.0
        assert stored.oblique_mode is MANUAL and not stored.redetect_oblique

    def test_store_detection_keeps_a_manual_oblique(self):
        settings = Settings()
        page = make_page_id()
        settings.set_params(page, Params(oblique_angle_deg=2.0, oblique_mode=MANUAL))

        settings.store_detection(page, DEPENDENCIES, rotation=1.0, oblique=4.0)

        stored = settings.get_params(page)
        assert stored.deskew_angle_deg == 1.0
        assert stored.oblique_angle_deg == 2.0

    def test_store_detection_keeps_a_manual_rotation(self):
        settings = Settings()
        page = make_page_id()
        settings.set_params(
            page,
            Params(deskew_angle_deg=5.0, mode=MANUAL, oblique_mode=AUTO),
        )

        stored_any = settings.store_detection(
            page, DEPENDENCIES, rotation=1.0, oblique=2.0
        )

        stored = settings.get_params(page)
        assert stored_any
        assert stored.deskew_angle_deg == 5.0
        assert stored.oblique_angle_deg == 2.0

    def test_store_detection_reports_nothing_stored_when_everything_is_manual(self):
        settings = Settings()
        page = make_page_id()
        settings.set_params(page, Params(mode=MANUAL, oblique_mode=MANUAL))

        assert not settings.store_detection(
            page, DEPENDENCIES, rotation=1.0, oblique=1.0
        )

    def test_apply_parts_copies_only_the_chosen_manual_part(self):
        settings = Settings()
        source = Params(
            deskew_angle_deg=2.0,
            mode=MANUAL,
            oblique_angle_deg=3.0,
            oblique_mode=MANUAL,
        )
        target = make_page_id("target")
        settings.set_params(
            target, Params(deskew_angle_deg=-1.0, mode=MANUAL, oblique_angle_deg=1.0)
        )

        settings.apply_parts_to_pages(source, [target], rotation=False, oblique=True)

        stored = settings.get_params(target)
        assert stored.deskew_angle_deg == -1.0
        assert stored.oblique_angle_deg == 3.0

    def test_apply_parts_leaves_an_automatic_part_to_be_detected(self):
        settings = Settings()
        source = Params(mode=AUTO, oblique_mode=AUTO)
        target = make_page_id("target")

        settings.apply_parts_to_pages(source, [target])

        stored = settings.get_params(target)
        assert stored.redetect_rotation and stored.redetect_oblique

    def test_apply_parts_to_a_fresh_page_detects_the_part_it_does_not_copy(self):
        settings = Settings()
        source = Params(deskew_angle_deg=2.0, mode=MANUAL)
        target = make_page_id("target")

        settings.apply_parts_to_pages(source, [target], rotation=True, oblique=False)

        stored = settings.get_params(target)
        assert stored.deskew_angle_deg == 2.0 and stored.mode is MANUAL
        assert stored.oblique_mode is MANUAL
        assert not stored.redetect_oblique


class TestDetection:
    """Rotation and oblique detection in the filter, each by its own mode."""

    @staticmethod
    def _detect(filter_: Filter, page: PageId, shear: float = 3.0) -> None:
        filter_.detect_skew(sheared_page(shear), page, dependencies=DEPENDENCIES)

    def test_default_off_detects_the_rotation_only(self):
        filter_ = Filter()
        page = make_page_id()

        self._detect(filter_, page)

        params = filter_.get_params(page)
        assert params.oblique_mode is MANUAL
        assert params.oblique_angle_deg == 0.0

    def test_auto_mode_finds_the_shear_of_a_sheared_page(self):
        filter_ = Filter()
        filter_.auto_oblique_default = True
        page = make_page_id()

        self._detect(filter_, page, shear=3.0)

        params = filter_.get_params(page)
        assert params.oblique_mode is AUTO
        assert params.oblique_angle_deg == pytest.approx(3.0, abs=1.0)
        assert params.dependencies == DEPENDENCIES

    def test_auto_mode_finds_nothing_on_an_upright_page(self):
        filter_ = Filter()
        filter_.auto_oblique_default = True
        page = make_page_id()

        filter_.detect_skew(text_page(), page, dependencies=DEPENDENCIES)

        assert filter_.get_params(page).oblique_angle_deg == pytest.approx(0.0, abs=0.2)

    def test_manual_oblique_is_kept_by_detection(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.set_manual_oblique(page, -1.5)

        self._detect(filter_, page, shear=4.0)

        params = filter_.get_params(page)
        assert params.oblique_angle_deg == -1.5
        assert params.oblique_mode is MANUAL

    def test_manual_rotation_is_kept_while_the_oblique_is_detected(self):
        filter_ = Filter()
        filter_.auto_oblique_default = True
        page = make_page_id()
        filter_.set_manual_angle(page, 0.0)
        filter_.redetect_oblique(page)

        self._detect(filter_, page, shear=3.0)

        params = filter_.get_params(page)
        assert params.mode is MANUAL
        assert params.deskew_angle_deg == 0.0
        assert params.oblique_angle_deg == pytest.approx(3.0, abs=1.0)

    def test_set_manual_oblique_is_clamped_to_the_limit(self):
        filter_ = Filter()
        page = make_page_id()

        filter_.set_manual_oblique(page, 9.0)

        assert filter_.get_params(page).oblique_angle_deg == MAX_OBLIQUE_DEG

    def test_set_manual_angle_keeps_the_oblique(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.set_manual_oblique(page, 2.0)

        filter_.set_manual_angle(page, 1.0)

        assert filter_.get_params(page).oblique_angle_deg == 2.0


class TestNeedsDetection:
    """Each automatic part is detected again when the geometry under it changes."""

    def test_a_page_without_params_needs_detection(self):
        assert Filter().needs_detection(make_page_id(), DEPENDENCIES)

    def test_manual_parts_never_need_detection(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.settings.set_params(
            page,
            Params(mode=MANUAL, oblique_mode=MANUAL, dependencies=DEPENDENCIES),
        )

        assert not filter_.needs_detection(page, OTHER_DEPENDENCIES)

    def test_an_automatic_oblique_needs_detection_when_the_geometry_changes(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.settings.set_params(
            page,
            Params(mode=MANUAL, oblique_mode=AUTO, dependencies=DEPENDENCIES),
        )

        assert filter_.needs_detection(page, OTHER_DEPENDENCIES)
        assert not filter_.needs_detection(page, DEPENDENCIES)

    def test_an_automatic_rotation_needs_detection_when_the_geometry_changes(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.settings.set_params(
            page,
            Params(mode=AUTO, oblique_mode=MANUAL, dependencies=DEPENDENCIES),
        )

        assert filter_.needs_detection(page, OTHER_DEPENDENCIES)

    def test_a_redetection_request_needs_detection_under_the_same_geometry(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.settings.set_params(
            page,
            Params(
                mode=MANUAL,
                oblique_mode=MANUAL,
                dependencies=DEPENDENCIES,
            ),
        )
        filter_.redetect_oblique(page)

        assert filter_.needs_detection(page, DEPENDENCIES)

    def test_detection_answers_the_request(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.set_manual_angle(page, 0.0)
        filter_.redetect_oblique(page)

        filter_.detect_skew(text_page(), page, dependencies=DEPENDENCIES)

        assert not filter_.needs_detection(page, DEPENDENCIES)


class TestApplyGeometry:
    """The filter hands both angles to a transformation."""

    def test_rotation_and_oblique_reach_the_transformation(self):
        xform = ImageTransformation.from_size(400, 600)
        filter_ = Filter()

        filter_.apply_geometry(
            xform, Params(deskew_angle_deg=1.5, oblique_angle_deg=-2.0)
        )

        assert xform.post_rotation == 1.5
        assert xform.post_oblique == -2.0

    def test_a_page_without_oblique_gives_an_unsheared_transformation(self):
        xform = ImageTransformation.from_size(400, 600)

        Filter().apply_geometry(xform, Params(deskew_angle_deg=1.0))

        assert xform.post_oblique == 0.0
