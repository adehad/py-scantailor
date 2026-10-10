import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.utils import get_cwidget
from scantailor.app.ui.widgets.stage_list_view import StageListView

STAGE_NUMBERS = [1, 2, 3, 4, 5, 6]
# Shortcuts as defined in MainWindow.ui (not the Ctrl+digit of the original spec).
SHORTCUT_KEYS = {
    1: ("O", QtCore.Qt.Key.Key_O),
    2: ("E", QtCore.Qt.Key.Key_E),
    3: ("N", QtCore.Qt.Key.Key_N),
    4: ("V", QtCore.Qt.Key.Key_V),
    5: ("K", QtCore.Qt.Key.Key_K),
    6: ("P", QtCore.Qt.Key.Key_P),
}


def _action(main_window: MainWindowManager, number: int):
    return getattr(main_window, f"actionSwitchFilter{number}")


def _filter_list(main_window: MainWindowManager) -> QtWidgets.QTableView:
    return get_cwidget(main_window.ui, QtWidgets.QTableView, "filterList")


@pytest.mark.parametrize("number", STAGE_NUMBERS)
def test_switch_filter_action_has_its_ui_file_shortcut(
    main_window: MainWindowManager, number: int
):
    assert (
        _action(main_window, number).shortcut().toString() == SHORTCUT_KEYS[number][0]
    )


@pytest.mark.parametrize("number", STAGE_NUMBERS)
def test_triggering_switch_filter_action_selects_that_stage_row(
    main_window: MainWindowManager, number: int
):
    _action(main_window, number).trigger()

    assert main_window.curFilter == number - 1
    assert _filter_list(main_window).currentIndex().row() == number - 1


@pytest.mark.parametrize("number", STAGE_NUMBERS)
def test_shortcut_key_triggers_matching_switch_filter_action(
    qtbot: QtBot,
    main_window: MainWindowManager,
    number: int,
):
    with qtbot.waitSignal(_action(main_window, number).triggered, timeout=2000):
        qtbot.keyClick(
            main_window.ui,
            SHORTCUT_KEYS[number][1],
        )

    assert _filter_list(main_window).currentIndex().row() == number - 1


@pytest.mark.xfail(
    strict=True,
    reason="actionSwitchFilter1..6 are window-level shortcuts only (as in the C++ "
    "app); MainWindow.ui lists them in no menu",
)
def test_switch_filter_actions_are_listed_in_a_menu(
    main_window: MainWindowManager,
):
    menu_actions = [
        action
        for menu in main_window.ui.menuBar().findChildren(QtWidgets.QMenu)
        for action in menu.actions()
    ]

    for number in STAGE_NUMBERS:
        assert _action(main_window, number) in menu_actions


def test_switch_filter_without_a_project_leaves_window_usable(
    main_window: MainWindowManager,
):
    for number in STAGE_NUMBERS:
        _action(main_window, number).trigger()

    assert main_window.ui.isVisible()
    assert QtWidgets.QApplication.activeModalWidget() is None


def test_filter_list_dock_exists_and_is_visible(main_window: MainWindowManager):
    filter_list = main_window.ui.findChild(QtWidgets.QTableView, "filterList")
    assert filter_list is not None
    assert filter_list.isVisible()


def test_filter_options_container_exists(main_window: MainWindowManager):
    options = main_window.ui.findChild(QtWidgets.QWidget, "filterOptions")
    assert options is not None


def test_filter_list_is_a_stage_list_view(main_window: MainWindowManager):
    assert isinstance(_filter_list(main_window), StageListView)


def test_filter_list_shows_six_stage_rows(main_window: MainWindowManager):
    assert _filter_list(main_window).model().rowCount() == 6


def test_filter_list_starts_on_the_first_stage(main_window: MainWindowManager):
    assert _filter_list(main_window).currentIndex().row() == 0


def test_switch_filter_selects_matching_stage_row(main_window: MainWindowManager):
    _action(main_window, 3).trigger()

    assert _filter_list(main_window).currentIndex().row() == 2


def test_clicking_a_stage_row_updates_the_current_filter(
    main_window: MainWindowManager,
):
    view = _filter_list(main_window)

    view.setCurrentIndex(view.model().index(4, 0))

    assert main_window.curFilter == 4


def test_changing_stage_does_not_change_the_current_page(
    project_window: MainWindowManager,
):
    project_window.goNextPage()

    _action(project_window, 4).trigger()

    assert project_window.curFilter == 3
    assert project_window.current_page == project_window.pages()[1]


def test_filter_options_shows_stage_options_widget(main_window: MainWindowManager):
    _action(main_window, 2).trigger()

    assert main_window.ui.findChild(QtWidgets.QWidget, "filterOptions").findChildren(
        QtWidgets.QWidget
    )
