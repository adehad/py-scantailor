"""Tests for the stage list: numbers inside the rows and rows sized from the font."""

import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui import theme
from scantailor.app.ui.theme.tokens import DARK, LIGHT
from scantailor.app.ui.widgets.stage_list_view import StageListView
from scantailor.core.stage_sequence import StageSequence


@pytest.fixture
def stage_list(qtbot) -> StageListView:
    view = StageListView()
    qtbot.addWidget(view)
    view.set_stages(StageSequence())
    view.resize(320, view.height())
    view.show()
    qtbot.waitExposed(view)
    return view


def displayed_text(view: StageListView, row: int) -> str:
    index = view.model().index(row, 0)
    option = QtWidgets.QStyleOptionViewItem()
    view.itemDelegateForColumn(0).initStyleOption(option, index)
    return option.text


def test_each_row_shows_its_number_before_the_name_in_one_cell(stage_list):
    assert not stage_list.verticalHeader().isVisible()
    assert displayed_text(stage_list, 0) == "1\u2007\u2007Fix Orientation"
    assert displayed_text(stage_list, 5) == "6\u2007\u2007Output"


def test_stage_numbers_use_tabular_figures(stage_list):
    index = stage_list.model().index(2, 0)
    option = QtWidgets.QStyleOptionViewItem()
    stage_list.itemDelegateForColumn(0).initStyleOption(option, index)
    assert option.font.isFeatureSet(QtGui.QFont.Tag("tnum"))


def test_rows_and_the_play_button_grow_with_the_font(stage_list, qtbot):
    stage_list.set_batch_processing_possible(True)
    stage_list.selectRow(1)
    row_before = stage_list.rowHeight(0)
    button = stage_list._launch_btn
    assert button is not None
    button_before = button.height()
    assert button_before >= 24

    font = stage_list.font()
    font.setPointSizeF(font.pointSizeF() + 4)
    stage_list.setFont(font)
    qtbot.wait(10)

    assert stage_list.rowHeight(0) > row_before
    assert button.height() > button_before
    assert stage_list.height() >= 6 * stage_list.rowHeight(0)


@pytest.fixture
def themed(qapp: QtWidgets.QApplication, monkeypatch):
    """Theme the application for one test and restore it afterwards."""
    monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
    style = qapp.style().name()
    palette = qapp.palette()
    font = qapp.font()
    yield qapp
    controller = qapp.findChild(QtCore.QObject, "scantailorThemeController")
    if controller is not None:
        shiboken6.delete(controller)
    qapp.setStyleSheet("")
    qapp.setStyle(style)
    qapp.setPalette(palette)
    qapp.setFont(font)


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_the_selected_stage_is_a_neutral_fill_with_an_accent_bar(themed, qtbot, scheme):
    theme.apply_theme(themed, scheme)
    tokens = DARK if scheme == "dark" else LIGHT
    view = StageListView()
    view.setObjectName("filterList")
    qtbot.addWidget(view)
    view.set_stages(StageSequence())
    view.resize(320, view.height())
    view.show()
    qtbot.waitExposed(view)
    view.selectRow(3)
    qtbot.wait(10)

    image = view.viewport().grab().toImage()
    row = view.rowViewportPosition(3) + view.rowHeight(3) // 2
    inside = view.columnWidth(0) - 20
    fill = image.pixelColor(inside, row)
    bar = image.pixelColor(1, row)
    unselected = image.pixelColor(inside, view.rowViewportPosition(0) + 4)
    accent = QtGui.QColor(tokens.accent)

    assert abs(fill.hue() - accent.hue()) > 60 or fill.saturation() < 40
    assert fill != unselected
    assert bar == accent
