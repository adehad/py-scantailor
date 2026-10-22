"""Tests for the locale-to-.qm translation loader."""

import gc
import pathlib

import pytest
from PySide6 import QtCore, QtWidgets

from scantailor.app.ui.widgets.error_widget import load_failure_message
from scantailor.translations import load_translation

_ABOUT_CONTEXT = "AboutDialog"
_ABOUT_SOURCE = "About"


@pytest.mark.parametrize(
    ("locale_name", "expected_text"),
    [
        pytest.param(
            "de_DE", "\u00dcber", id="region-variant-falls-back-to-language-de"
        ),
        pytest.param("de", "\u00dcber", id="bare-language-de"),
        pytest.param("es_ES", "Acerca de", id="spanish-spain"),
        pytest.param("fr_FR", "\u00c0 propos", id="french-with-accent"),
        pytest.param(
            "ko_KR", "ScanTailor Advanced \uc815\ubcf4", id="korean-non-latin"
        ),
        pytest.param("pl_PL", "O programie", id="polish"),
        pytest.param("pt_BR", "Acerca de", id="pt_BR-falls-back-to-pt-file"),
        pytest.param(
            "ru_RU",
            "\u041e \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0435",
            id="russian-cyrillic",
        ),
    ],
)
def test_supported_locale_translates_known_message_from_its_ts(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    locale_name: str,
    expected_text: str,
) -> None:
    """A supported locale, including region variants, shows the .ts translation."""
    load_translation(qapp, QtCore.QLocale(locale_name), compiled_qm_dir)

    translated = qapp.translate(_ABOUT_CONTEXT, _ABOUT_SOURCE)

    assert translated == expected_text


def test_region_variant_loads_language_file_when_no_region_file_exists(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
) -> None:
    """pt_BR has no scantailor_pt_BR.qm, so the loader must use scantailor_pt.qm."""
    assert not (compiled_qm_dir / "scantailor_pt_BR.qm").exists()

    load_translation(qapp, QtCore.QLocale("pt_BR"), compiled_qm_dir)

    assert qapp.translate(_ABOUT_CONTEXT, _ABOUT_SOURCE) == "Acerca de"


def test_unknown_locale_keeps_english_source_text_without_error(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
) -> None:
    """A locale with no .qm file leaves the UI in English and raises nothing."""
    load_translation(qapp, QtCore.QLocale("xx_XX"), compiled_qm_dir)

    assert qapp.translate(_ABOUT_CONTEXT, _ABOUT_SOURCE) == _ABOUT_SOURCE


def test_english_locale_installs_no_application_translator(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
) -> None:
    """English has no scantailor_en.ts, so no application catalogue loads."""
    load_translation(qapp, QtCore.QLocale("en_US"), compiled_qm_dir)

    assert qapp.translate(_ABOUT_CONTEXT, _ABOUT_SOURCE) == _ABOUT_SOURCE


def test_missing_qm_folder_keeps_english_source_text(
    qapp: QtWidgets.QApplication,
    tmp_path: pathlib.Path,
    removed_translators: None,
) -> None:
    """An empty folder with no compiled files falls back to English."""
    load_translation(qapp, QtCore.QLocale("de_DE"), tmp_path)

    assert qapp.translate(_ABOUT_CONTEXT, _ABOUT_SOURCE) == _ABOUT_SOURCE


def test_qt_base_translations_load_alongside_application_translations(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
) -> None:
    """German loads both the Qt base catalogue and the application catalogue."""
    loaded = load_translation(qapp, QtCore.QLocale("de_DE"), compiled_qm_dir)

    assert len(loaded) == 2
    assert qapp.translate("QPlatformTheme", "Cancel") == "Abbrechen"


def test_translator_stays_active_after_garbage_collection(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
) -> None:
    """Dropping the Python references must not revert the UI to English."""
    load_translation(qapp, QtCore.QLocale("de_DE"), compiled_qm_dir)
    gc.collect()
    gc.collect()

    assert qapp.translate(_ABOUT_CONTEXT, _ABOUT_SOURCE) == "\u00dcber"


def test_load_error_panel_takes_upstream_translation_of_its_message(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    tmp_path: pathlib.Path,
) -> None:
    """The error panel's text keeps upstream's context, so its translations load."""
    load_translation(qapp, QtCore.QLocale("de"), compiled_qm_dir)

    text, _format = load_failure_message(tmp_path / "missing.png")

    assert text.startswith("Die folgende Datei existiert nicht:<br>")
    assert 'href="#relink"' in text
