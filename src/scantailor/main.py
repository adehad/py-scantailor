"""Main Entrypoint for the application."""
from __future__ import annotations

from PySide6 import QtWidgets

from .app.ui import UI_FOLDER, load_ui_widget
from .translations import load_translation


def main() -> int:
    """Run main  UI application."""
    app = QtWidgets.QApplication([])
    load_translation(app)
    main_window = load_ui_widget(UI_FOLDER / "MainWindow.ui")
    main_window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
