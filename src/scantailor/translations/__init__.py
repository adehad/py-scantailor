"""Load the application's and Qt's translations for a locale.

Qt loads compiled .qm catalogues, which are build output made from the .ts
sources. Release bundles and wheels ship them; a source checkout compiles them
on first use into the user's cache folder.
"""

import pathlib
import subprocess
import tempfile

from loguru import logger
from PySide6 import QtCore, QtWidgets

from .catalogues import compile_all, find_lrelease, sources_digest, translation_sources

TRANSLATIONS_DIR = pathlib.Path(__file__).parent
_APP_PREFIX = "scantailor"
_QT_BASE_PREFIX = "qtbase"
_FILE_SEPARATOR = "_"
_TRANSLATOR_NAME = "scantailor-translator"
SYSTEM_LANGUAGE_VALUES = frozenset({"", "system", "auto"})


def locale_for_language(language: str) -> QtCore.QLocale:
    """Turn the saved language setting into the locale to translate to.

    Args:
        language (str): Locale code from the settings, such as "de" or "pt_BR".
            Empty, "system" or "auto" mean the operating system's locale.

    Returns:
        QtCore.QLocale: Locale to pass to load_translation. A code Qt does not
            recognise becomes the C locale, which has no translation, so the UI
            stays in English.
    """
    code = language.strip()
    if code.lower() in SYSTEM_LANGUAGE_VALUES:
        return QtCore.QLocale()
    return QtCore.QLocale(code)


def _remove_installed_translators(app: QtWidgets.QApplication) -> None:
    """Uninstall and release the translators an earlier load_translation installed.

    Args:
        app (QtWidgets.QApplication): Application the translators are parented to.
    """
    for translator in app.findChildren(QtCore.QTranslator, _TRANSLATOR_NAME):
        app.removeTranslator(translator)
        translator.setParent(None)
        translator.deleteLater()


def qt_translations_dir() -> pathlib.Path:
    """Locate the Qt base translations shipped with PySide6.

    Returns:
        pathlib.Path: Folder holding qtbase_<locale>.qm files.
    """
    return pathlib.Path(
        QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.TranslationsPath)
    )


def _compiled_in(sources: list[pathlib.Path], folder: pathlib.Path) -> bool:
    """Whether ``folder`` holds a catalogue for each source."""
    return all((folder / f"{source.stem}.qm").is_file() for source in sources)


def _default_cache_root() -> pathlib.Path:
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.CacheLocation
    )
    base = pathlib.Path(location) if location else pathlib.Path(tempfile.gettempdir())
    return base / "translations"


def compiled_directory(
    directory: pathlib.Path = TRANSLATIONS_DIR,
    cache_root: pathlib.Path | None = None,
) -> pathlib.Path:
    """Find the folder to load catalogues from, compiling them when they are missing.

    A folder with no .ts sources, as in a release bundle, or with a .qm beside
    every source, as in a wheel or after tools/build_translations.py, is used
    as it is. Otherwise the sources are compiled into a folder of the cache
    named after their digest, so that an edited source is compiled again and
    checkouts at other commits do not share catalogues.

    Args:
        directory (pathlib.Path): Folder holding the scantailor_*.ts sources,
            the compiled catalogues, or both.
        cache_root (pathlib.Path | None): Where compiled catalogues are kept;
            None uses the application's cache folder.

    Returns:
        pathlib.Path: Folder holding the catalogues; ``directory`` itself when
            they could not be compiled, which leaves the UI in English.
    """
    sources = translation_sources(directory)
    if _compiled_in(sources, directory):
        return directory
    root = cache_root if cache_root is not None else _default_cache_root()
    target = root / sources_digest(directory)
    if _compiled_in(sources, target):
        return target
    try:
        compile_all(directory, target, find_lrelease())
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning(f"Could not compile the translations: {error}")
        return directory
    logger.info(f"Compiled {len(sources)} translations into {target}")
    return target


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

    Translators a previous call installed are removed first, so switching language
    while running does not leave the old catalogue answering ahead of the new one.
    Qt sends every widget a LanguageChange event as translators come and go.

    Args:
        app (QtWidgets.QApplication): Application to install the translators on.
        locale (QtCore.QLocale | None): Locale to load; None uses the system locale.
        directory (pathlib.Path): Folder holding the scantailor_*.qm catalogues
            or the .ts sources to compile them from; see ``compiled_directory``.

    Returns:
        list[QtCore.QTranslator]: Translators that were loaded and installed, Qt base
            first. Empty when neither the application nor Qt has the locale.
    """
    target_locale = locale if locale is not None else QtCore.QLocale()
    _remove_installed_translators(app)
    installed: list[QtCore.QTranslator] = []
    sources = (
        (_QT_BASE_PREFIX, qt_translations_dir()),
        (_APP_PREFIX, compiled_directory(directory)),
    )
    for prefix, folder in sources:
        translator = QtCore.QTranslator(app)
        translator.setObjectName(_TRANSLATOR_NAME)
        if translator.load(target_locale, prefix, _FILE_SEPARATOR, str(folder)):
            app.installTranslator(translator)
            installed.append(translator)
        else:
            translator.setParent(None)
            translator.deleteLater()
    if not installed:
        logger.info(
            f"No compiled translation for {target_locale.name()}; using English"
        )
    return installed
