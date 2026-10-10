import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager
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


@pytest.mark.parametrize("number", STAGE_NUMBERS)
def test_switch_filter_action_has_its_ui_file_shortcut(
    main_window: MainWindowManager, number: int
):
    assert (
        _action(main_window, number).shortcut().toString() == SHORTCUT_KEYS[number][0]
    )


@pytest.mark.parametrize("number", STAGE_NUMBERS)
def test_triggering_switch_filter_action_logs_only_that_stage(
    main_window: MainWindowManager, log_messages: list[str], number: int
):
    _action(main_window, number).trigger()

    assert [m for m in log_messages if m.startswith("Switch filter")] == [
        f"Switch filter {number}"
    ]


@pytest.mark.xfail(
    strict=True,
    reason="actionSwitchFilter1..6 are not attached to any widget or menu, "
    "so their shortcuts never fire",
)
@pytest.mark.parametrize("number", STAGE_NUMBERS)
def test_shortcut_key_triggers_matching_switch_filter_action(
    qtbot: QtBot,
    main_window: MainWindowManager,
    log_messages: list[str],
    number: int,
):
    with qtbot.waitSignal(_action(main_window, number).triggered, timeout=2000):
        qtbot.keyClick(
            main_window.ui,
            SHORTCUT_KEYS[number][1],
        )

    assert f"Switch filter {number}" in log_messages


@pytest.mark.xfail(
    strict=True,
    reason="actionSwitchFilter1..6 are not added to any menu in MainWindow.ui",
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


@pytest.mark.xfail(
    strict=True,
    reason="filterList is a promoted StageListView in MainWindow.ui but loads "
    "as a plain QTableView (StageListView not registered with QUiLoader)",
)
def test_filter_list_is_a_stage_list_view(main_window: MainWindowManager):
    assert isinstance(
        main_window.ui.findChild(QtWidgets.QTableView, "filterList"), StageListView
    )


@pytest.mark.xfail(
    strict=True,
    reason="StageListView is not wired to a StageSequence by MainWindow, "
    "so the filter list model has no six stage rows",
)
def test_filter_list_shows_six_stage_rows(main_window: MainWindowManager):
    assert (
        main_window.ui.findChild(QtWidgets.QTableView, "filterList").model().rowCount()
        == 6
    )


@pytest.mark.xfail(
    strict=True,
    reason="switchFilterN only logs; it does not select a stage row in filterList",
)
def test_switch_filter_selects_matching_stage_row(main_window: MainWindowManager):
    _action(main_window, 3).trigger()

    assert (
        main_window.ui.findChild(QtWidgets.QTableView, "filterList")
        .currentIndex()
        .row()
        == 2
    )


@pytest.mark.xfail(
    strict=True,
    reason="options widgets (app/ui/options_widgets) are not instantiated by "
    "MainWindow, so filterOptions has no child options widget",
)
def test_filter_options_shows_stage_options_widget(main_window: MainWindowManager):
    _action(main_window, 2).trigger()

    assert main_window.ui.findChild(QtWidgets.QWidget, "filterOptions").findChildren(
        QtWidgets.QWidget
    )
