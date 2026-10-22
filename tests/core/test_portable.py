"""Portable mode: a config folder beside the executable holds the user's files."""

import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6 import QtCore, QtWidgets

from scantailor.core import diagnostics, portable
from scantailor.core.default_params import DefaultParams
from scantailor.core.default_params_profiles import ProfileStore
from scantailor.core.portable import (
    PORTABLE_CONFIG_DIR,
    application_folder,
    find_portable_folder,
    portable_folder,
    use_portable_qsettings,
)
from scantailor.core.settings import ApplicationSettings


@pytest.fixture(autouse=True)
def fresh_portable_lookup() -> Iterator[None]:
    portable_folder.cache_clear()
    yield
    portable_folder.cache_clear()


@pytest.fixture
def frozen_program(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A frozen build whose executable sits in ``tmp_path / "ScanTailor"``.

    Returns:
        Path: The folder of the executable.
    """
    folder = tmp_path / "ScanTailor"
    folder.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(folder / "ScanTailor.exe"))
    ApplicationSettings.reset_instance()
    return folder


class TestFindingTheFolder:
    def test_running_from_source_has_no_program_folder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delattr(sys, "frozen", raising=False)

        assert application_folder() is None
        assert portable_folder() is None

    def test_a_frozen_build_looks_beside_its_executable(
        self, frozen_program: Path
    ) -> None:
        assert application_folder() == frozen_program.resolve()

    def test_without_a_config_folder_the_program_is_not_portable(
        self, tmp_path: Path
    ) -> None:
        assert find_portable_folder(tmp_path) is None

    def test_a_config_folder_switches_portable_mode_on(self, tmp_path: Path) -> None:
        (tmp_path / PORTABLE_CONFIG_DIR).mkdir()

        assert find_portable_folder(tmp_path) == tmp_path / PORTABLE_CONFIG_DIR

    def test_a_config_file_rather_than_a_folder_is_ignored(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / PORTABLE_CONFIG_DIR).write_text("", encoding="utf-8")

        assert find_portable_folder(tmp_path) is None

    def test_a_folder_that_cannot_be_written_is_ignored_with_a_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        log_messages: list[str],
    ) -> None:
        (tmp_path / PORTABLE_CONFIG_DIR).mkdir()

        def refuse(**_: object) -> tempfile.TemporaryDirectory[str]:
            raise PermissionError("read-only")

        monkeypatch.setattr(portable.tempfile, "TemporaryDirectory", refuse)

        assert find_portable_folder(tmp_path) is None
        assert any("Not portable" in message for message in log_messages)

    def test_the_check_leaves_no_file_behind(self, tmp_path: Path) -> None:
        config = tmp_path / PORTABLE_CONFIG_DIR
        config.mkdir()

        find_portable_folder(tmp_path)

        assert list(config.iterdir()) == []


class TestPortableFilesStayBesideTheProgram:
    def test_settings_profiles_and_logs_go_in_the_config_folder(
        self, frozen_program: Path
    ) -> None:
        config = frozen_program.resolve() / PORTABLE_CONFIG_DIR
        config.mkdir()

        assert ApplicationSettings.get_config_path() == config / "settings.json"
        assert diagnostics.log_folder() == config / "logs"
        ProfileStore().write("Books", DefaultParams())
        assert (config / "profiles" / "saved" / "Books.json").is_file()

    def test_saved_settings_are_read_back_from_the_config_folder(
        self, frozen_program: Path
    ) -> None:
        config = frozen_program.resolve() / PORTABLE_CONFIG_DIR
        config.mkdir()

        ApplicationSettings(current_profile="Source").save()
        ApplicationSettings.reset_instance()

        assert (config / "settings.json").is_file()
        assert ApplicationSettings.get_instance().current_profile == "Source"

    def test_the_log_banner_says_the_program_is_portable(
        self, frozen_program: Path
    ) -> None:
        (frozen_program / PORTABLE_CONFIG_DIR).mkdir()

        assert diagnostics.banner_lines(None)[0].endswith(", portable")

    def test_without_the_folder_nothing_is_written_beside_the_program(
        self, frozen_program: Path
    ) -> None:
        assert not ApplicationSettings.get_config_path().is_relative_to(
            frozen_program.resolve()
        )
        assert not diagnostics.log_folder().is_relative_to(frozen_program.resolve())
        assert not diagnostics.banner_lines(None)[0].endswith(", portable")


@pytest.fixture
def restored_qsettings(qapp: QtWidgets.QApplication) -> Iterator[None]:
    previous_format = QtCore.QSettings.defaultFormat()
    previous_names = (
        QtCore.QCoreApplication.organizationName(),
        QtCore.QCoreApplication.applicationName(),
    )
    QtCore.QCoreApplication.setOrganizationName("ScanTailor")
    QtCore.QCoreApplication.setApplicationName("pyScanTailor")
    yield
    QtCore.QSettings.setDefaultFormat(previous_format)
    QtCore.QCoreApplication.setOrganizationName(previous_names[0])
    QtCore.QCoreApplication.setApplicationName(previous_names[1])


@pytest.mark.usefixtures("restored_qsettings")
class TestQSettings:
    def test_portable_qsettings_write_an_ini_file_in_the_config_folder(
        self, tmp_path: Path
    ) -> None:
        config = tmp_path / PORTABLE_CONFIG_DIR
        config.mkdir()

        assert use_portable_qsettings(config)
        settings = QtCore.QSettings()
        settings.setValue("mainWindow/maximized", True)
        settings.sync()

        assert Path(settings.fileName()) == config / "ScanTailor" / "pyScanTailor.ini"
        assert Path(settings.fileName()).is_file()

    def test_a_program_that_is_not_portable_keeps_the_default_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delattr(sys, "frozen", raising=False)
        before = QtCore.QSettings.defaultFormat()

        assert not use_portable_qsettings()
        assert QtCore.QSettings.defaultFormat() == before
