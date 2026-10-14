"""Tests that a real dialog shows the catalogue translation for the chosen locale."""

import pathlib
import xml.etree.ElementTree as ET

import pytest
from PySide6 import QtCore, QtWidgets

from scantailor.app.ui.dialogs.about import AboutDialog
from scantailor.translations import TRANSLATIONS_DIR, load_translation

_CONTEXT = "AboutDialog"
_SOURCE = "Authors"


def catalogue_translation(ts_file: pathlib.Path, context: str, source: str) -> str:
    """Read one translation straight from a .ts file.

    The .ts file is the oracle here, so the test does not depend on the loader
    it is checking.

    Args:
        ts_file (pathlib.Path): The .ts catalogue to read.
        context (str): Name of the context that holds the message.
        source (str): English source text of the message.

    Returns:
        str: The text inside the message's translation element.

    Raises:
        LookupError: If the context or message is missing from the file.
    """
    root = ET.parse(ts_file).getroot()
    for context_element in root.iter("context"):
        if context_element.findtext("name") != context:
            continue
        for message in context_element.iter("message"):
            if message.findtext("source") == source:
                return message.findtext("translation") or ""
    raise LookupError(f"{context}/{source!r} not found in {ts_file.name}")


def tab_texts(dialog: QtWidgets.QDialog) -> list[str]:
    """Return the titles of every tab in the dialog's tab widget.

    Args:
        dialog (QtWidgets.QDialog): Dialog that contains a QTabWidget.

    Returns:
        list[str]: Tab titles in display order.
    """
    tab_widget = dialog.findChild(QtWidgets.QTabWidget)
    assert tab_widget is not None
    return [tab_widget.tabText(index) for index in range(tab_widget.count())]


@pytest.mark.parametrize(
    "language",
    [
        pytest.param("de", id="german"),
        pytest.param("es", id="spanish"),
        pytest.param("fr", id="french"),
        pytest.param("ko", id="korean"),
        pytest.param("pl", id="polish"),
        pytest.param("pt", id="portuguese"),
        pytest.param("ru", id="russian"),
    ],
)
def test_about_dialog_tab_shows_catalogue_translation_for_locale(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
    language: str,
) -> None:
    """Switching to a language makes the About dialog tabs show that catalogue text.

    The expected text comes from the .ts file and must differ from English, so a
    catalogue the loader ignores cannot pass.
    """
    expected = catalogue_translation(
        TRANSLATIONS_DIR / f"scantailor_{language}.ts", _CONTEXT, _SOURCE
    )
    assert expected and expected != _SOURCE

    load_translation(qapp, QtCore.QLocale(language), compiled_qm_dir)
    dialog = AboutDialog(parent=None)
    try:
        assert expected in tab_texts(dialog)
    finally:
        dialog.deleteLater()


def test_about_dialog_tab_falls_back_to_english_for_unknown_locale(
    qapp: QtWidgets.QApplication,
    compiled_qm_dir: pathlib.Path,
    removed_translators: None,
) -> None:
    """An unknown locale keeps the English tab titles and does not raise."""
    load_translation(qapp, QtCore.QLocale("xx_XX"), compiled_qm_dir)
    dialog = AboutDialog(parent=None)
    try:
        assert _SOURCE in tab_texts(dialog)
    finally:
        dialog.deleteLater()
