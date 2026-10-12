"""The profile store: case-insensitive names and an unambiguous active profile."""

from pathlib import Path

import pytest

from scantailor.core.default_params import DefaultParams
from scantailor.core.default_params_profiles import (
    CUSTOM_PROFILE,
    ProfileError,
    ProfileStore,
    active_default_params,
    validate_profile_name,
)
from scantailor.core.settings import ApplicationSettings
from scantailor.filters.deskew.params import AutoManualMode


def _manual(angle: float) -> DefaultParams:
    params = DefaultParams()
    params.deskew.mode = AutoManualMode.MANUAL
    params.deskew.deskew_angle_deg = angle
    return params


@pytest.fixture
def store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(tmp_path / "profiles")


class TestNamesIgnoreCase:
    def test_reading_finds_the_profile_whatever_the_case(
        self, store: ProfileStore
    ) -> None:
        store.write("Thick Books", _manual(2.0))

        assert store.read("thick books") == _manual(2.0)
        assert store.read("THICK BOOKS") == _manual(2.0)

    def test_writing_another_case_replaces_the_profile(
        self, store: ProfileStore
    ) -> None:
        store.write("Thick Books", _manual(2.0))

        store.write("thick books", _manual(3.0))

        assert store.names() == ["thick books"]
        assert store.read("Thick Books") == _manual(3.0)
        assert [p.name for p in (store._saved_folder).iterdir()] == ["thick books.json"]

    def test_deleting_another_case_removes_the_profile(
        self, store: ProfileStore
    ) -> None:
        store.write("Thick Books", _manual(2.0))

        store.delete("THICK books")

        assert store.names() == []

    def test_files_differing_only_in_case_list_one_name(
        self, store: ProfileStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = store._saved_folder
        monkeypatch.setattr(
            store,
            "_saved_files",
            lambda: [folder / "ALPHA.json", folder / "alpha.json"],
        )

        assert store.names() == ["ALPHA"]

    def test_reserved_names_stay_reserved_in_any_case(self) -> None:
        for name in ("default", "SOURCE", " Custom "):
            with pytest.raises(ProfileError):
                validate_profile_name(name)

    @pytest.mark.parametrize("name", ["CON", "nul", "Com1", "LPT9", "aux.v2"])
    def test_windows_device_names_are_refused(self, name: str) -> None:
        with pytest.raises(ProfileError):
            validate_profile_name(name)

    def test_names_that_merely_start_like_a_device_are_accepted(self) -> None:
        for name in ("console", "com10", "nullable", "My CON"):
            assert validate_profile_name(name) == name


class TestActiveProfile:
    def test_remembered_edits_apply_to_their_own_profile(
        self, store: ProfileStore
    ) -> None:
        store.write("books", _manual(1.0))

        store.set_active(_manual(5.0), "books")

        assert store.active_params("books") == _manual(5.0)
        assert store.active_params("BOOKS") == _manual(5.0)

    def test_remembered_edits_do_not_apply_to_another_profile(
        self, store: ProfileStore
    ) -> None:
        store.write("books", _manual(1.0))
        store.write("maps", _manual(2.0))
        store.set_active(_manual(5.0), "books")

        assert store.active_params("maps") == _manual(2.0)

    def test_saving_the_profile_drops_stale_edits(self, store: ProfileStore) -> None:
        store.write("books", _manual(1.0))
        store.set_active(_manual(5.0), "books")

        store.write("books", _manual(7.0))

        assert store.active_params("books") == _manual(7.0)

    def test_deleting_the_profile_drops_its_edits(self, store: ProfileStore) -> None:
        store.write("books", _manual(1.0))
        store.set_active(_manual(5.0), "books")

        store.delete("books")

        assert store.active_params("books") == DefaultParams()

    def test_unsaved_custom_edits_are_remembered(self, store: ProfileStore) -> None:
        store.set_active(_manual(4.0), CUSTOM_PROFILE)

        assert store.active_params(CUSTOM_PROFILE) == _manual(4.0)

    def test_edits_without_an_owner_are_not_kept(self, store: ProfileStore) -> None:
        store.write("books", _manual(1.0))
        store.set_active(_manual(5.0), "books")

        store.set_active(_manual(9.0))

        assert store.active_params("books") == _manual(1.0)

    def test_built_in_profiles_ignore_remembered_edits(
        self, store: ProfileStore
    ) -> None:
        store.set_active(_manual(5.0), "books")

        assert store.active_params("Default") == DefaultParams()
        assert store.active_params("source") == DefaultParams.source()

    def test_a_file_in_the_old_format_is_ignored(self, store: ProfileStore) -> None:
        store.write("books", _manual(1.0))
        (store._folder / "active.json").write_text(
            _manual(5.0).model_dump_json(), encoding="utf-8"
        )

        assert store.active_params("books") == _manual(1.0)

    def test_a_corrupt_file_falls_back_to_the_saved_profile(
        self, store: ProfileStore
    ) -> None:
        store.write("books", _manual(1.0))
        (store._folder / "active.json").write_text("{not json", encoding="utf-8")

        assert store.active_params("books") == _manual(1.0)

    def test_unknown_profile_gives_the_defaults(self, store: ProfileStore) -> None:
        assert store.active_params("nothing here") == DefaultParams()

    def test_new_projects_start_from_the_selected_profile(
        self, store: ProfileStore
    ) -> None:
        store.write("books", _manual(1.0))
        store.set_active(_manual(5.0), "books")
        settings = ApplicationSettings(current_profile="books")

        assert active_default_params(settings, store) == _manual(5.0)
