from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow
from scantailor.app.ui.dialogs.about import AboutDialog


def test_main_window_opens(qtbot: QtBot):
    manager = MainWindow.MainWindowManager()
    window = manager.ui
    qtbot.addWidget(window)
    window.show()
    qtbot.wait_active(window)


def test_show_about(qtbot: QtBot):
    manager = MainWindow.MainWindowManager()
    window = manager.ui
    qtbot.addWidget(window)
    before_qdialog = window.findChild(QtWidgets.QDialog)
    manager.actionAbout.trigger()
    qtbot.wait_until(lambda: window.findChild(QtWidgets.QDialog) is not None)
    after_qdialog = window.findChild(QtWidgets.QDialog)
    assert before_qdialog is None
    assert isinstance(after_qdialog, AboutDialog)
