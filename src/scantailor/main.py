"""Main Entrypoint for the application."""

import pathlib
import sys
from collections.abc import Sequence

from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets

from .__about__ import APP_NAME, __version__
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
ORGANIZATION_NAME = "ScanTailor"


def set_application_identity() -> None:
    """Name the organisation and application that QSettings files preferences under.

    Without both names QSettings has nowhere to store values: writes are
    dropped and every preference is back to its default on the next start.
    """
    QtCore.QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    QtCore.QCoreApplication.setApplicationVersion(__version__)


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


def project_file_from_arguments(arguments: Sequence[str]) -> pathlib.Path | None:
    """Pick the project file named on the command line, if any.

    The first argument after the program name is the project, as in ScanTailor
    Advanced; any further arguments are ignored.

    Args:
        arguments (Sequence[str]): The program name and its arguments, as
            QApplication.arguments() returns them with Qt's own options removed.

    Returns:
        pathlib.Path | None: The project file made absolute against the working
            directory, so the recent projects list can find it again, or None if
            no argument was given.
    """
    if len(arguments) < 2:
        return None
    if len(arguments) > 2:
        logger.warning("Ignoring extra command line arguments: {}", arguments[2:])
    return pathlib.Path(arguments[1]).absolute()


def open_project_from_arguments(
    window: MainWindowManager, arguments: Sequence[str]
) -> bool:
    """Open the project named on the command line in the window.

    A file that is missing or cannot be read gets the same warning as one chosen
    through File > Open Project, and the window stays on the welcome panel.

    Args:
        window (MainWindowManager): The shown main window.
        arguments (Sequence[str]): The program name and its arguments, without
            Qt's own options.

    Returns:
        bool: True if a project was named and is now open.
    """
    project_file = project_file_from_arguments(arguments)
    if project_file is None:
        return False
    return window.open_project_file(project_file)


class FileOpenHandler(QtCore.QObject):
    """Open the project file the macOS Finder sends to the application.

    On a cold launch Qt may put the Finder's document in the arguments, which
    open_project_from_arguments() handles, and send no event for it. Otherwise
    the Finder's request reaches the QApplication as a QFileOpenEvent, so
    install this as an event filter on it. The event can arrive before the main
    window exists; the file then waits until attach_window() and opens there.
    Only the latest such file is kept, because the window holds one project.
    """

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Create a handler with no window attached.

        Args:
            parent (QtCore.QObject | None): Owner of the handler.
        """
        super().__init__(parent)
        self._window: MainWindowManager | None = None
        self._pending: pathlib.Path | None = None

    def attach_window(self, window: MainWindowManager) -> None:
        """Send files to the window from now on, opening any that arrived earlier.

        Args:
            window (MainWindowManager): The shown main window.
        """
        self._window = window
        pending, self._pending = self._pending, None
        if pending is not None:
            window.open_project_file(pending)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Consume file open events and pass every other event on.

        Args:
            watched (QtCore.QObject): The object the event is for.
            event (QtCore.QEvent): The event.

        Returns:
            bool: True if the event named a local file and was handled here.
        """
        if not isinstance(event, QtGui.QFileOpenEvent) or not event.file():
            return super().eventFilter(watched, event)
        project_file = pathlib.Path(event.file())
        if self._window is None:
            if self._pending is not None:
                logger.warning("Ignoring project opened before it: {}", self._pending)
            self._pending = project_file
        else:
            self._window.open_project_file(project_file)
        return True


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
    set_application_identity()
    app = QtWidgets.QApplication(sys.argv)
    file_open_handler = FileOpenHandler(app)
    app.installEventFilter(file_open_handler)
    app.aboutToQuit.connect(restore_qt_handler)
    logger.info("Qt platform {}, style {}", app.platformName(), app.style().name())
    apply_theme(app)
    set_window_icon(app)
    install_saved_language(app)
    main_window = MainWindowManager()
    main_window.show()
    open_project_from_arguments(main_window, app.arguments())
    file_open_handler.attach_window(main_window)
    exit_code = app.exec()
    session.end()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
