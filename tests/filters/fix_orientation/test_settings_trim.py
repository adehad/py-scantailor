"""Tests for storing, copying and importing the manual trim."""

import json
import shutil
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from scantailor.core.models import ImageId, OrthogonalRotation, PageId, SubPage
from scantailor.core.xml_compat import import_xml_project
from scantailor.filters.fix_orientation.filter import Filter
from scantailor.filters.fix_orientation.image_trim import ImageTrim
from scantailor.filters.fix_orientation.settings import Settings

FIXTURES = Path(__file__).parents[2] / "fixtures" / "xml_projects"

IMAGE = ImageId(file_path=Path("/scans/a.tiff"))
OTHER = ImageId(file_path=Path("/scans/b.tiff"))
TRIM = ImageTrim(enabled=True, left=4, top=5, right=6, bottom=7)


class TestSettingsTrim:
    def test_an_image_has_a_disabled_trim_by_default(self):
        assert Settings().get_trim(IMAGE) == ImageTrim()
        assert not Settings().get_trim(IMAGE).enabled

    def test_a_stored_trim_is_returned(self):
        settings = Settings()

        settings.set_trim(IMAGE, TRIM)

        assert settings.get_trim(IMAGE) == TRIM
        assert settings.get_trim(OTHER) == ImageTrim()

    def test_storing_a_disabled_trim_removes_the_stored_one(self):
        settings = Settings()
        settings.set_trim(IMAGE, TRIM)

        settings.set_trim(IMAGE, ImageTrim(enabled=False, left=9))

        assert settings.trims == {}

    def test_clear_trim_removes_only_that_images_trim(self):
        settings = Settings()
        settings.set_trim(IMAGE, TRIM)
        settings.set_trim(OTHER, TRIM)

        settings.clear_trim(IMAGE)

        assert set(settings.trims) == {OTHER}

    def test_clear_removes_rotations_and_trims(self):
        settings = Settings()
        settings.apply_rotation(IMAGE, OrthogonalRotation(degrees=90))
        settings.set_trim(IMAGE, TRIM)

        settings.clear()

        assert not settings.has_params(IMAGE)

    def test_apply_trim_to_pages_sets_the_trim_of_each_pages_image(self):
        settings = Settings()
        pages = [
            PageId(image_id=IMAGE, sub_page=SubPage.LEFT_PAGE),
            PageId(image_id=IMAGE, sub_page=SubPage.RIGHT_PAGE),
            PageId(image_id=OTHER),
        ]

        settings.apply_trim_to_pages(pages, TRIM)

        assert settings.get_trim(IMAGE) == TRIM
        assert settings.get_trim(OTHER) == TRIM

    def test_applying_a_disabled_trim_clears_the_targets(self):
        settings = Settings()
        settings.set_trim(OTHER, TRIM)

        settings.apply_trim_to_pages([PageId(image_id=OTHER)], ImageTrim())

        assert settings.get_trim(OTHER) == ImageTrim()

    def test_concurrent_writers_leave_a_consistent_store(self):
        settings = Settings()
        images = [ImageId(file_path=Path(f"/scans/{n}.tiff")) for n in range(50)]

        def write(subset: list[ImageId]) -> None:
            for image in subset:
                settings.set_trim(image, TRIM)

        threads = [
            threading.Thread(target=write, args=(images[offset::4],))
            for offset in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert set(settings.trims) == set(images)


class TestJsonRoundTrip:
    def test_rotation_and_trim_survive_a_json_round_trip(self):
        settings = Settings()
        settings.apply_rotation(IMAGE, OrthogonalRotation(degrees=270))
        settings.set_trim(IMAGE, TRIM)

        restored = Settings.from_dict(json.loads(json.dumps(settings.to_dict())))

        assert restored.get_rotation_for(IMAGE).degrees == 270
        assert restored.get_trim(IMAGE) == TRIM

    def test_an_image_with_only_a_trim_is_stored_without_a_rotation(self):
        settings = Settings()
        settings.set_trim(IMAGE, TRIM)

        restored = Settings.from_dict(json.loads(json.dumps(settings.to_dict())))

        assert restored.get_trim(IMAGE) == TRIM
        assert not restored.is_rotation_set(IMAGE)

    def test_an_image_with_only_a_rotation_stores_no_trim(self):
        settings = Settings()
        settings.apply_rotation(IMAGE, OrthogonalRotation(degrees=90))

        (entry,) = settings.to_dict()

        assert "trim" not in entry["params"]

    def test_dump_params_of_an_untouched_image_is_empty(self):
        assert Settings().dump_params(IMAGE) == {}

    def test_load_params_leaves_the_other_setting_alone(self):
        settings = Settings()
        settings.apply_rotation(IMAGE, OrthogonalRotation(degrees=180))

        settings.load_params(IMAGE, {"trim": TRIM.model_dump(mode="json")})

        assert settings.get_rotation_for(IMAGE).degrees == 180
        assert settings.get_trim(IMAGE) == TRIM

    def test_a_params_file_from_before_the_trim_existed_still_loads(self):
        settings = Settings()

        settings.load_params(IMAGE, {"degrees": 90})

        assert settings.get_rotation_for(IMAGE).degrees == 90
        assert settings.get_trim(IMAGE) == ImageTrim()

    def test_a_malformed_trim_is_rejected(self):
        with pytest.raises(ValidationError):
            Settings().load_params(IMAGE, {"trim": {"left": "wide"}})


class TestFilterTrim:
    def test_set_and_get_go_through_the_settings(self):
        filter_ = Filter()

        filter_.set_trim(IMAGE, TRIM)

        assert filter_.settings.get_trim(IMAGE) == TRIM
        assert filter_.get_trim(IMAGE) == TRIM

    def test_trim_crop_area_is_none_without_a_trim(self):
        assert Filter().trim_crop_area(IMAGE, (300, 400)) is None

    def test_trim_crop_area_follows_the_stored_rotation(self):
        filter_ = Filter()
        filter_.set_rotation(IMAGE, OrthogonalRotation(degrees=90))
        filter_.set_trim(IMAGE, ImageTrim(enabled=True, left=100))

        area = filter_.trim_crop_area(IMAGE, (400, 300))

        assert area is not None
        assert area[:, 1].min() == pytest.approx(100)
        assert area[:, 1].max() == pytest.approx(300)

    def test_apply_current_to_pages_copies_rotation_and_trim(self):
        filter_ = Filter()
        filter_.set_rotation(IMAGE, OrthogonalRotation(degrees=90))
        filter_.set_trim(IMAGE, TRIM)

        filter_.apply_current_to_pages(IMAGE, [PageId(image_id=OTHER)])

        assert filter_.get_rotation(OTHER).degrees == 90
        assert filter_.get_trim(OTHER) == TRIM

    def test_apply_current_to_pages_clears_a_target_trim_when_the_source_has_none(self):
        filter_ = Filter()
        filter_.set_trim(OTHER, TRIM)

        filter_.apply_current_to_pages(IMAGE, [PageId(image_id=OTHER)])

        assert filter_.get_trim(OTHER) == ImageTrim()


class TestImportOfVersion4Projects:
    """The trim elements of a version 4 project file become stored trims."""

    @staticmethod
    def _import(tmp_path: Path):
        target = tmp_path / "version4"
        shutil.copytree(FIXTURES / "version4", target)
        project_file = target / "version4.ScanTailor"
        return import_xml_project(
            project_file, project_file.read_text(encoding="utf-8")
        )

    def test_an_enabled_trim_is_stored_with_its_four_margins(self, tmp_path: Path):
        imported = self._import(tmp_path)
        settings = Settings()
        for entry in imported.project.stage_params["fix_orientation"]:
            settings.load_params(entry.page.image_id, entry.params)

        trimmed, rotated = (info.id for info in imported.project.images)
        assert settings.get_trim(trimmed) == ImageTrim(
            enabled=True, left=4, right=5, top=6, bottom=7
        )
        assert not settings.is_rotation_set(trimmed)
        assert settings.get_rotation_for(rotated).degrees == 90

    def test_a_disabled_trim_is_not_stored(self, tmp_path: Path):
        imported = self._import(tmp_path)
        settings = Settings()
        for entry in imported.project.stage_params["fix_orientation"]:
            settings.load_params(entry.page.image_id, entry.params)

        _trimmed, rotated = (info.id for info in imported.project.images)
        assert settings.get_trim(rotated) == ImageTrim()

    def test_imported_trims_survive_saving_as_json_and_reloading(self, tmp_path: Path):
        imported = self._import(tmp_path)
        settings = Settings()
        for entry in imported.project.stage_params["fix_orientation"]:
            settings.load_params(entry.page.image_id, entry.params)

        restored = Settings.from_dict(json.loads(json.dumps(settings.to_dict())))

        assert restored.trims == settings.trims
        assert restored.rotations == settings.rotations
