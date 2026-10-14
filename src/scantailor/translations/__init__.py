"""Load the application's and Qt's translations for a locale.

Compiled .qm files are built from the .ts sources by tools/build_translations.py.
"""

import pathlib

from loguru import logger
from PySide6 import QtCore, QtWidgets

TRANSLATIONS_DIR = pathlib.Path(__file__).parent
_APP_PREFIX = "scantailor"
_QT_BASE_PREFIX = "qtbase"
_FILE_SEPARATOR = "_"


def qt_translations_dir() -> pathlib.Path:
    """Locate the Qt base translations shipped with PySide6.

    Returns:
        pathlib.Path: Folder holding qtbase_<locale>.qm files.
    """
    return pathlib.Path(
        QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.TranslationsPath)
    )


def load_translation(
    app: QtWidgets.QApplication,
    locale: QtCore.QLocale | None = None,
    directory: pathlib.Path = TRANSLATIONS_DIR,
) -> list[QtCore.QTranslator]:
    """Install the translators that match a locale, falling back to English.

    QTranslator.load resolves the locale itself: de_DE tries scantailor_de_DE.qm,
    then scantailor_de.qm; pt_BR tries pt_BR, then pt. A locale with no matching
    file installs nothing and the UI stays in its source English.

    The translators are parented to the application, so Qt keeps them alive for
    the application's lifetime. A Python reference alone would let the garbage
    collector drop them and silently revert the UI to English.

    Args:
        app (QtWidgets.QApplication): Application to install the translators on.
        locale (QtCore.QLocale | None): Locale to load; None uses the system locale.
        directory (pathlib.Path): Folder holding the compiled scantailor_*.qm files.

    Returns:
        list[QtCore.QTranslator]: Translators that were loaded and installed, Qt base
            first. Empty when neither the application nor Qt has the locale.
    """
    target_locale = locale if locale is not None else QtCore.QLocale()
    installed: list[QtCore.QTranslator] = []
    sources = (
        (_QT_BASE_PREFIX, qt_translations_dir()),
        (_APP_PREFIX, directory),
    )
    for prefix, folder in sources:
        translator = QtCore.QTranslator(app)
        if translator.load(target_locale, prefix, _FILE_SEPARATOR, str(folder)):
            app.installTranslator(translator)
            installed.append(translator)
    if not installed:
        logger.info(
            f"No compiled translation for {target_locale.name()}; using English"
        )
    return installed
