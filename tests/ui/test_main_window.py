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


def test_navigation_without_a_project_keeps_no_current_page(
    main_window: MainWindow.MainWindowManager,
):
    main_window.goNextPage()
    main_window.goLastPage()

    assert main_window.current_page is None


def test_page_navigation_moves_through_pages_and_clamps_at_the_ends(
    project_window: MainWindow.MainWindowManager,
):
    pages = project_window.pages()

    project_window.goPrevPage()
    assert project_window.current_page == pages[0]
    project_window.goNextPage()
    assert project_window.current_page == pages[1]
    project_window.goLastPage()
    assert project_window.current_page == pages[2]
    project_window.goNextPage()
    assert project_window.current_page == pages[2]
    project_window.goFirstPage()
    assert project_window.current_page == pages[0]


def test_page_navigation_actions_move_the_current_page(
    project_window: MainWindow.MainWindowManager,
):
    pages = project_window.pages()

    project_window.actionNextPage.trigger()
    project_window.actionNextPage.trigger()
    project_window.actionPrevPage.trigger()

    assert project_window.current_page == pages[1]
