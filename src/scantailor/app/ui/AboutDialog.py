"""About Dialog."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from scantailor import __about__
from scantailor.app.ui import UI_FOLDER, load_ui_widget
from scantailor.app.ui.utils import get_cwidget


class AboutDialog(QtWidgets.QDialog):
    """About Dialog."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        """Initialize."""
        super().__init__(parent)
        self.ui = load_ui_widget(UI_FOLDER / "AboutDialog.ui", self)
        self.version_label = get_cwidget(self.ui, QtWidgets.QLabel, "version")
        self.license_viewer = get_cwidget(
            self.ui, QtWidgets.QTextBrowser, "licenseViewer"
        )

        self.set_version()
        self.set_license()

    def set_version(self):
        """Set the version on the dialog."""
        self.version_label.setText("version " + __about__.__version__)

    def set_license(self):
        """Set the license text from the resource file."""
        file = QtCore.QFile(":/GPLv3.html")
        if file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text):  # type: ignore[attr-defined]
            stream = QtCore.QTextStream(file)
            try:
                html_content = stream.readAll()
                self.license_viewer.setHtml(html_content)
            finally:
                file.close()

    def show(self):
        """Show."""
        self.ui.show()
