from __future__ import annotations

from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow


def test_main_window_opens(qtbot: QtBot):
    window = MainWindow.MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait_active(window)
