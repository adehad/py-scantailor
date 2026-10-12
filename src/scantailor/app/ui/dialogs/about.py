"""About Dialog."""

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from scantailor import __about__
from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.utils import get_cwidget

_UI_FOLDER = Path(__file__).parent


_THIRD_PARTY_NOTICES_HTML = (
    "<hr><h3>Third-party assets</h3>"
    "<p><b>Lucide icons</b>: ISC licence, with MIT for the icons derived from "
    "Feather. Full text in the bundled LICENSE file.</p>"
    "<p><b>Inter</b> and <b>Noto Sans KR</b> fonts: SIL Open Font License 1.1.</p>"
)


class AboutDialog(QtWidgets.QDialog):
    """About Dialog."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        """Initialize."""
        super().__init__(parent)
        self.ui = load_ui_widget(_UI_FOLDER / "AboutDialog.ui", self)
        embed_ui(self, self.ui)
        self.version_label = get_cwidget(self.ui, QtWidgets.QLabel, "version")
        self.license_viewer = get_cwidget(
            self.ui, QtWidgets.QTextBrowser, "licenseViewer"
        )

        self.set_version()
        self.set_license()
        self.add_third_party_notices()

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

    def add_third_party_notices(self) -> None:
        """Append the licences of the bundled icons and fonts to the licence tab."""
        self.license_viewer.append(_THIRD_PARTY_NOTICES_HTML)
