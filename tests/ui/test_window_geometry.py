"""The main window comes back at the size, place and state it closed in."""

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow, window_geometry
from scantailor.app.ui.window_geometry import (
    DEFAULT_SIZE,
    GEOMETRY_KEY,
    MAXIMIZED_KEY,
    is_reachable,
    restore_window_geometry,
    save_window_geometry,
)

SCREEN = QtCore.QRect(0, 0, 1920, 1080)
SECOND_SCREEN = QtCore.QRect(1920, 0, 1280, 1024)


def _primary_area() -> QtCore.QRect:
    screen = QtGui.QGuiApplication.primaryScreen()
    assert screen is not None
    return screen.availableGeometry()


def _window_inside_primary(qtbot: QtBot) -> QtWidgets.QMainWindow:
    """Show a window well inside the primary screen, smaller than the default."""
    area = _primary_area()
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)
    window.resize(area.width() // 2, area.height() // 2)
    window.move(area.left() + 20, area.top() + 40)
    window.show()
    qtbot.wait_exposed(window)
    return window


def test_a_window_fully_on_a_screen_is_reachable() -> None:
    assert is_reachable(QtCore.QRect(100, 100, 800, 600), [SCREEN])


def test_a_window_on_a_screen_that_was_unplugged_is_not_reachable() -> None:
    assert not is_reachable(QtCore.QRect(2000, 100, 800, 600), [SCREEN])


def test_a_window_showing_only_a_sliver_of_its_title_bar_is_not_reachable() -> None:
    assert not is_reachable(QtCore.QRect(1900, 100, 800, 600), [SCREEN])


def test_a_window_whose_title_bar_is_above_every_screen_is_not_reachable() -> None:
    assert not is_reachable(QtCore.QRect(100, -500, 800, 600), [SCREEN])


def test_a_window_on_the_second_screen_is_reachable() -> None:
    assert is_reachable(QtCore.QRect(2000, 100, 800, 600), [SCREEN, SECOND_SCREEN])


def test_a_window_with_no_screens_is_not_reachable() -> None:
    assert not is_reachable(QtCore.QRect(100, 100, 800, 600), [])


def test_saved_geometry_restores_the_same_size_on_a_new_window(qtbot: QtBot) -> None:
    settings = QtCore.QSettings()
    closed = _window_inside_primary(qtbot)
    save_window_geometry(closed, settings)

    reopened = QtWidgets.QMainWindow()
    qtbot.addWidget(reopened)
    maximized = restore_window_geometry(reopened, settings)
    reopened.show()
    qtbot.wait_exposed(reopened)

    assert maximized is False
    assert reopened.size() == closed.size()
    # A window learns its frame margins only once shown, so compare frames of
    # two shown windows. The offscreen platform still offsets a restored frame
    # by a few pixels.
    offset = reopened.frameGeometry().topLeft() - closed.frameGeometry().topLeft()
    assert offset.manhattanLength() <= 10


def test_no_saved_geometry_gives_the_default_size(qtbot: QtBot) -> None:
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    maximized = restore_window_geometry(window, QtCore.QSettings())

    assert maximized is False
    assert window.size() == DEFAULT_SIZE


def test_a_default_window_wider_than_the_screen_keeps_its_corner_on_screen(
    qtbot: QtBot,
) -> None:
    area = _primary_area()
    if area.width() >= DEFAULT_SIZE.width():
        pytest.skip("needs a screen narrower than the default window")
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    restore_window_geometry(window, QtCore.QSettings())

    assert window.frameGeometry().left() == area.left()
    assert area.contains(window.frameGeometry().topLeft())


def test_unreadable_saved_geometry_gives_the_default_size(qtbot: QtBot) -> None:
    settings = QtCore.QSettings()
    settings.setValue(GEOMETRY_KEY, QtCore.QByteArray(b"not a geometry"))
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    restore_window_geometry(window, settings)

    assert window.size() == DEFAULT_SIZE


def test_geometry_saved_far_off_screen_is_brought_back_on_screen(
    qtbot: QtBot,
) -> None:
    settings = QtCore.QSettings()
    closed = _window_inside_primary(qtbot)
    area = _primary_area()
    closed.move(area.right() + 5000, area.bottom() + 5000)
    settings.setValue(GEOMETRY_KEY, closed.saveGeometry())

    reopened = QtWidgets.QMainWindow()
    qtbot.addWidget(reopened)
    restore_window_geometry(reopened, settings)

    assert is_reachable(reopened.frameGeometry(), [area])


def test_geometry_on_a_screen_that_is_gone_opens_at_the_default_size(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = QtCore.QSettings()
    save_window_geometry(_window_inside_primary(qtbot), settings)
    area = _primary_area()
    elsewhere = QtCore.QRect(area.right() + 10_000, 0, 1280, 1024)
    monkeypatch.setattr(window_geometry, "_available_screens", lambda: [elsewhere])

    reopened = QtWidgets.QMainWindow()
    qtbot.addWidget(reopened)
    restore_window_geometry(reopened, settings)

    assert reopened.size() == DEFAULT_SIZE
    assert area.contains(reopened.frameGeometry().center())


def test_a_maximised_window_is_reported_maximised_on_restore(qtbot: QtBot) -> None:
    settings = QtCore.QSettings()
    closed = _window_inside_primary(qtbot)
    closed.showMaximized()
    qtbot.wait_until(closed.isMaximized)
    save_window_geometry(closed, settings)

    reopened = QtWidgets.QMainWindow()
    qtbot.addWidget(reopened)

    assert settings.value(MAXIMIZED_KEY, type=bool) is True
    assert restore_window_geometry(reopened, settings) is True


def test_a_window_that_was_never_shown_saves_nothing(qtbot: QtBot) -> None:
    settings = QtCore.QSettings()
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    save_window_geometry(window, settings)

    assert not settings.contains(GEOMETRY_KEY)
    assert not settings.contains(MAXIMIZED_KEY)


def test_closing_the_main_window_saves_its_size_for_the_next_start(
    qtbot: QtBot,
) -> None:
    area = _primary_area()
    size = QtCore.QSize(area.width() // 2, area.height() // 2)
    first = MainWindow.MainWindowManager()
    qtbot.addWidget(first.ui)
    first.ui.setMinimumSize(1, 1)
    first.show()
    qtbot.wait_exposed(first.ui)
    first.ui.resize(size)
    qtbot.wait_until(lambda: first.ui.size() == size)

    assert first.ui.close()

    second = MainWindow.MainWindowManager()
    qtbot.addWidget(second.ui)
    assert second.ui.size() == size


def test_a_main_window_closed_maximised_opens_maximised(qtbot: QtBot) -> None:
    QtCore.QSettings().setValue(MAXIMIZED_KEY, True)
    manager = MainWindow.MainWindowManager()
    qtbot.addWidget(manager.ui)

    manager.show()

    qtbot.wait_until(manager.ui.isMaximized)


def test_cancelling_the_close_keeps_the_saved_geometry(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = MainWindow.MainWindowManager()
    qtbot.addWidget(manager.ui)
    monkeypatch.setattr(manager, "_close_project_interactive", lambda: False)
    manager.show()
    qtbot.wait_exposed(manager.ui)

    assert not manager.ui.close()
    assert not QtCore.QSettings().contains(GEOMETRY_KEY)
    monkeypatch.undo()
