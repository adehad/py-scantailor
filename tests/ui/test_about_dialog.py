from pytestqt.qtbot import QtBot

from scantailor.app.ui.AboutDialog import AboutDialog


def test_about_dialog(qtbot: QtBot):
    dialog = AboutDialog(parent=None)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait_active(dialog)
