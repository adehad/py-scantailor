"""Every window shows the application icon in its title bar."""

from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.main import set_window_icon


def test_a_dialog_opened_after_startup_carries_the_app_icon(
    qapp: QtWidgets.QApplication, qtbot: QtBot
) -> None:
    previous = qapp.windowIcon()
    try:
        set_window_icon(qapp)
        dialog = QtWidgets.QDialog()
        qtbot.addWidget(dialog)

        assert not dialog.windowIcon().isNull()
        assert dialog.windowIcon().cacheKey() == qapp.windowIcon().cacheKey()
    finally:
        qapp.setWindowIcon(previous)


def test_the_app_icon_resource_renders(qapp: QtWidgets.QApplication) -> None:
    previous = qapp.windowIcon()
    try:
        set_window_icon(qapp)

        assert not qapp.windowIcon().pixmap(32, 32).isNull()
    finally:
        qapp.setWindowIcon(previous)
