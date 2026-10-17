"""Main Entrypoint for the application."""

import pathlib

from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets

from .app._ui import resources_rc
from .app.ui.MainWindow import MainWindowManager
from .app.ui.theme import apply_theme
from .core import diagnostics
from .core.settings import ApplicationSettings
from .translations import TRANSLATIONS_DIR, load_translation, locale_for_language

assert resources_rc, "The import ensures we load in the resources for the rest of Qt."


def setup_qcore():
    """This is set due a warning when running the application without it."""
    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    )


APP_ICON = ":/appicon.svg"


def set_window_icon(app: QtWidgets.QApplication) -> None:
    """Give the main window and every dialog the application icon.

    Without it each title bar shows the platform's generic window icon.

    Args:
        app (QtWidgets.QApplication): Application whose windows get the icon.
    """
    app.setWindowIcon(QtGui.QIcon(APP_ICON))


def install_saved_language(
    app: QtWidgets.QApplication, directory: pathlib.Path = TRANSLATIONS_DIR
) -> list[QtCore.QTranslator]:
    """Translate the application to the language chosen in the Settings dialog.

    Args:
        app (QtWidgets.QApplication): Application to install the translators on.
        directory (pathlib.Path): Folder holding the compiled scantailor_*.qm files.

    Returns:
        list[QtCore.QTranslator]: The translators installed, empty for English.
    """
    language = ApplicationSettings.get_instance().language
    locale = locale_for_language(language)
    translators = load_translation(app, locale, directory)
    catalogues = ", ".join(translator.filePath() for translator in translators)
    logger.info(
        "UI language {} ({}): {}",
        language or "system",
        locale.name(),
        catalogues or "no catalogue loaded, English",
    )
    return translators


def main() -> int:
    """Run the window application.

    Logging is set up before the QApplication, so a failure to create it is
    recorded after the start-up banner.

    Returns:
        int: The exit code of the Qt event loop.
    """
    session = diagnostics.configure_logging("gui", stderr_level="INFO")
    restore_qt_handler = diagnostics.install_qt_message_handler()
    setup_qcore()
    app = QtWidgets.QApplication([])
    app.aboutToQuit.connect(restore_qt_handler)
    logger.info("Qt platform {}, style {}", app.platformName(), app.style().name())
    apply_theme(app)
    set_window_icon(app)
    install_saved_language(app)
    main_window = MainWindowManager()
    main_window.show()
    exit_code = app.exec()
    session.end()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
