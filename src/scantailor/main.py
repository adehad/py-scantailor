"""Main Entrypoint for the application."""

import pathlib

from PySide6 import QtCore, QtWidgets

from .app._ui import resources_rc
from .app.ui.MainWindow import MainWindowManager
from .app.ui.theme import apply_theme
from .core.settings import ApplicationSettings
from .imageproc.numba_warmup import start_warmup
from .translations import TRANSLATIONS_DIR, load_translation, locale_for_language

assert resources_rc, "The import ensures we load in the resources for the rest of Qt."


def setup_qcore():
    """This is set due a warning when running the application without it."""
    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    )


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
    return load_translation(app, locale_for_language(language), directory)


def main() -> int:
    """Run main  UI application."""
    setup_qcore()
    app = QtWidgets.QApplication([])
    apply_theme(app)
    install_saved_language(app)
    main_window = MainWindowManager()
    main_window.show()
    start_warmup()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
