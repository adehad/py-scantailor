"""Tests that the language saved in the settings is the one the UI is shown in."""

import pathlib
from collections.abc import Iterator

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs import settings as settings_dialog
from scantailor.app.ui.dialogs.settings import SettingsDialog
from scantailor.core.settings import ApplicationSettings
from scantailor.main import install_saved_language
from scantailor.translations import load_translation, locale_for_language

_CONTEXT = "AboutDialog"
_SOURCE = "About"
_GERMAN = "Über"
_FRENCH = "À propos"


def shown_text() -> str:
    """Return what the UI currently shows for the About dialog's title.

    Returns:
        str: The installed catalogue's translation, or the English source.
    """
    return QtCore.QCoreApplication.translate(_CONTEXT, _SOURCE)


def save_language(language: str) -> None:
    """Write a language to the isolated settings file, as the Settings dialog does.

    Args:
        language (str): Value to store in the language setting.
    """
    ApplicationSettings(language=language).save()


@pytest.fixture
def system_locale_is_french() -> Iterator[None]:
    """Make QLocale() report French, standing in for a French operating system."""
    previous = QtCore.QLocale()
    QtCore.QLocale.setDefault(QtCore.QLocale("fr_FR"))
    yield
    QtCore.QLocale.setDefault(previous)


def test_saved_language_is_applied_at_startup(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    system_locale_is_french: None,
    isolated_application_settings: pathlib.Path,
) -> None:
    """A saved German setting shows German at startup, not the system's French."""
    save_language("de")
    assert isolated_application_settings.exists()

    installed = install_saved_language(qapp, compiled_qm_dir)

    assert installed
    assert shown_text() == _GERMAN


def test_startup_logs_the_language_and_the_catalogues_loaded(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    log_messages: list[str],
) -> None:
    """The log names the language and each catalogue file, for reports like #55."""
    save_language("de")

    install_saved_language(qapp, compiled_qm_dir)

    line = next(m for m in log_messages if m.startswith("UI language de (de_DE): "))
    assert "scantailor_de.qm" in line


def test_startup_logs_when_no_catalogue_loaded(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    log_messages: list[str],
) -> None:
    """A language without a catalogue is logged as English."""
    save_language("xx")

    install_saved_language(qapp, compiled_qm_dir)

    assert "UI language xx (C): no catalogue loaded, English" in log_messages


@pytest.mark.parametrize(
    "language",
    [
        pytest.param("", id="empty"),
        pytest.param("system", id="system"),
        pytest.param("auto", id="auto"),
        pytest.param(" System ", id="system-padded-mixed-case"),
    ],
)
def test_system_setting_follows_the_system_locale_at_startup(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    system_locale_is_french: None,
    language: str,
) -> None:
    """An empty or system language setting shows the system's language, here French."""
    save_language(language)

    install_saved_language(qapp, compiled_qm_dir)

    assert shown_text() == _FRENCH


@pytest.mark.parametrize(
    "language",
    [
        pytest.param("xx", id="unknown-code"),
        pytest.param("not a locale", id="garbage"),
        pytest.param("en", id="english-source-language"),
    ],
)
def test_language_without_a_catalogue_falls_back_to_english_at_startup(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    system_locale_is_french: None,
    language: str,
) -> None:
    """A code with no catalogue shows English without raising, ignoring the system."""
    save_language(language)

    install_saved_language(qapp, compiled_qm_dir)

    assert shown_text() == _SOURCE


@pytest.mark.parametrize(
    ("language", "expected_name"),
    [
        pytest.param("de", "de_DE", id="bare-language-gets-its-main-region"),
        pytest.param("pt_BR", "pt_BR", id="regional-code-kept"),
        pytest.param("xx", "C", id="unknown-code-is-the-untranslated-c-locale"),
    ],
)
def test_locale_for_language_maps_a_code_to_its_qt_locale(
    language: str, expected_name: str
) -> None:
    """A saved code becomes the Qt locale QTranslator searches catalogues for."""
    assert locale_for_language(language).name() == expected_name


def test_switching_language_replaces_the_previous_catalogue(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
) -> None:
    """Loading French after German shows French, so German no longer answers first."""
    load_translation(qapp, QtCore.QLocale("de"), compiled_qm_dir)
    assert shown_text() == _GERMAN

    installed = load_translation(qapp, QtCore.QLocale("fr"), compiled_qm_dir)

    assert shown_text() == _FRENCH
    assert set(qapp.findChildren(QtCore.QTranslator)) == set(installed)


def test_switching_to_english_removes_the_previous_catalogue(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
) -> None:
    """Choosing a language with no catalogue after German brings back English."""
    load_translation(qapp, QtCore.QLocale("de"), compiled_qm_dir)

    load_translation(qapp, QtCore.QLocale("xx"), compiled_qm_dir)

    assert shown_text() == _SOURCE


@pytest.fixture
def dialog_loads_test_catalogues(
    monkeypatch: pytest.MonkeyPatch, compiled_qm_dir: pathlib.Path
) -> None:
    """Point the Settings dialog at the catalogues compiled for this session."""

    def load_compiled(
        app: QtWidgets.QApplication, locale: QtCore.QLocale
    ) -> list[QtCore.QTranslator]:
        return load_translation(app, locale, compiled_qm_dir)

    monkeypatch.setattr(settings_dialog, "load_translation", load_compiled)


def accept(dialog: SettingsDialog) -> None:
    """Press the dialog's OK button.

    Args:
        dialog (SettingsDialog): Dialog to accept.
    """
    box = dialog.findChild(QtWidgets.QDialogButtonBox)
    assert box is not None
    ok = box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok is not None
    ok.click()


def language_box(dialog: SettingsDialog) -> QtWidgets.QComboBox:
    """Find the dialog's language combo box.

    Args:
        dialog (SettingsDialog): Dialog to search.

    Returns:
        QtWidgets.QComboBox: The language box.
    """
    box = dialog.findChild(QtWidgets.QComboBox, "languageBox")
    assert box is not None
    return box


def test_accepting_a_new_language_applies_it_at_once(
    qtbot: QtBot,
    removed_translators: None,
    dialog_loads_test_catalogues: None,
) -> None:
    """Choosing German and pressing OK translates the running application."""
    settings = ApplicationSettings(language="en")
    dialog = SettingsDialog(settings)
    qtbot.addWidget(dialog)
    box = language_box(dialog)
    box.setCurrentIndex(box.findData("de"))

    accept(dialog)

    assert settings.language == "de"
    assert shown_text() == _GERMAN


def test_accepting_an_unchanged_language_leaves_translators_alone(
    qtbot: QtBot,
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    dialog_loads_test_catalogues: None,
) -> None:
    """Pressing OK without touching the language keeps the installed catalogue."""
    installed = load_translation(qapp, QtCore.QLocale("de"), compiled_qm_dir)
    dialog = SettingsDialog(ApplicationSettings(language="de"))
    qtbot.addWidget(dialog)

    accept(dialog)

    assert set(qapp.findChildren(QtCore.QTranslator)) == set(installed)


@pytest.mark.parametrize(
    ("language", "selected"),
    [
        pytest.param("de", "de", id="bundled-code"),
        pytest.param("pt_BR", "pt", id="regional-code-selects-its-language"),
        pytest.param("system", "fr", id="system-selects-the-system-language"),
        pytest.param("xx", "en", id="unknown-code-selects-english"),
    ],
)
def test_dialog_selects_the_language_the_ui_is_shown_in(
    qtbot: QtBot, system_locale_is_french: None, language: str, selected: str
) -> None:
    """The language box opens on the catalogue the saved setting resolves to."""
    dialog = SettingsDialog(ApplicationSettings(language=language))
    qtbot.addWidget(dialog)

    assert language_box(dialog).currentData() == selected
