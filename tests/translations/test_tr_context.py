"""Tests that translated text is looked up in the context lupdate files it under."""

import ast
import pathlib
import xml.etree.ElementTree as ET

import pytest
from PySide6 import QtCore

from scantailor.app.ui.dialogs.apply_to import DESKEW_TITLE, OUTPUT_GROUP_TITLES
from scantailor.core.settings import ApplicationSettings
from scantailor.translations import TRANSLATIONS_DIR, locale_for_language

_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "scantailor"
_GENERATED = _SOURCE_ROOT / "app" / "_ui"
_CATALOGUES = sorted(
    path
    for path in TRANSLATIONS_DIR.glob("scantailor_*.ts")
    if path.stem != "scantailor_untranslated"
)


def _tr_calls_on_other_objects(path: pathlib.Path) -> list[str]:
    """List ``x.tr(...)`` calls whose receiver is not ``self`` or ``cls``.

    lupdate files such a call under the receiver's source text, for instance
    ``self.ui``, while Qt looks it up under the receiver's class name, so the
    translation is never found.

    Args:
        path (pathlib.Path): Python file to scan.

    Returns:
        list[str]: ``file:line receiver`` for each offending call.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tr"
            and not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("self", "cls")
            )
        ):
            receiver = ast.unparse(node.func.value)
            found.append(f"{path.relative_to(_SOURCE_ROOT)}:{node.lineno} {receiver}")
    return found


def test_tr_is_only_called_on_self() -> None:
    """No source file calls tr() on another object, which would never translate."""
    offenders = [
        finding
        for path in _SOURCE_ROOT.rglob("*.py")
        if _GENERATED not in path.parents and "__pycache__" not in path.parts
        for finding in _tr_calls_on_other_objects(path)
    ]

    assert offenders == []


def _active_sources(catalogue: pathlib.Path, context: str) -> set[str]:
    """Return the finished, current source texts of one context in a catalogue."""
    root = ET.parse(catalogue).getroot()
    sources: set[str] = set()
    for element in root.iter("context"):
        if element.findtext("name") != context:
            continue
        for message in element.iter("message"):
            translation = message.find("translation")
            if translation is not None and translation.get("type") is None:
                sources.add(message.findtext("source", ""))
    return sources


@pytest.mark.parametrize("catalogue", _CATALOGUES, ids=lambda path: path.stem)
def test_apply_dialog_titles_are_translated_where_the_window_looks(
    catalogue: pathlib.Path,
) -> None:
    """Every apply dialog title the main window shows has a finished translation."""
    titles = {*OUTPUT_GROUP_TITLES.values(), DESKEW_TITLE}

    assert titles <= _active_sources(catalogue, "MainWindowManager")


def test_first_run_language_follows_the_operating_system() -> None:
    """With nothing saved, the language setting means the system locale."""
    assert locale_for_language(ApplicationSettings().language) == QtCore.QLocale()
