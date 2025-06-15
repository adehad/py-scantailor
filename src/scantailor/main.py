"""Main Entrypoint for the application."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .app._ui import resources_rc
from .app.ui.MainWindow import MainWindow
from .translations import load_translation

assert resources_rc, "The import ensures we load in the resources for the rest of Qt."


def setup_qcore():
    """This is set due a warning when running the application without it."""
    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    )


def main() -> int:
    """Run main  UI application."""
    setup_qcore()
    app = QtWidgets.QApplication([])
    load_translation(app)
    main_window = MainWindow()
    main_window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
