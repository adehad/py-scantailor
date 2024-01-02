"""Translations."""
from __future__ import annotations

import functools
import pathlib
import subprocess

from PySide6 import QtCore, QtWidgets

TS_PATH = pathlib.Path(__file__).parent


@functools.lru_cache
def translation_map() -> dict[str, str]:
    """{LOCALE_NAME: LOCAL_PATH}."""
    return {
        p.name.split("_")[-1]: str(p)
        #
        for p in TS_PATH.glob("*.qm")
    }


def load_translation(app: QtWidgets.QApplication):
    """Load a translation."""
    translator = QtCore.QTranslator()
    user_locale = QtCore.QLocale()
    if translation_file := translation_map().get(user_locale.name()):
        translator.load(locale=user_locale, filename=translation_file)
        app.installTranslator(translator)


if __name__ == "__main__":
    for p in TS_PATH.glob("*.ts"):
        subprocess.call(["pyside6-lrelease", str(p)])
