"""Numeric Output options render once the value settles, as upstream's 750 ms timer."""

from pathlib import Path

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets.output import OptionsWidget
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.filter import Filter
from scantailor.filters.output.params import Params

PAGE = PageId(image_id=ImageId(file_path=Path("a.tif")), sub_page=SubPage.SINGLE_PAGE)
OTHER = PageId(image_id=ImageId(file_path=Path("b.tif")), sub_page=SubPage.SINGLE_PAGE)


class _Panel:
    def __init__(self, qtbot: QtBot, params: Params | None = None) -> None:
        self.filter = Filter()
        self.filter.set_params(PAGE, params or Params())
        self.widget = OptionsWidget(self.filter)
        qtbot.addWidget(self.widget)
        self.widget.set_current_page(PAGE)
        self.changed: list[PageId] = []
        self.widget.params_changed.connect(self.changed.append)
        self.timer_starts = 0
        timer = self.reload_timer
        start = timer.start

        def counted_start(*args: object) -> None:
            self.timer_starts += 1
            start(*args)

        timer.start = counted_start  # type: ignore[method-assign]

    @property
    def reload_timer(self) -> QtCore.QTimer:
        return self.widget._reload_timer

    def let_the_delay_pass(self) -> None:
        assert self.reload_timer.isActive()
        self.reload_timer.timeout.emit()

    def find[T: QtWidgets.QWidget](self, kind: type[T], name: str) -> T:
        found = self.widget.findChild(kind, name)
        assert found is not None, name
        return found


def test_the_delay_matches_upstream():
    assert OptionsWidget.RELOAD_DELAY_MS == 750


def test_a_burst_of_edits_renders_once_after_it_settles(qtbot: QtBot):
    panel = _Panel(qtbot)
    spin = panel.find(QtWidgets.QSpinBox, "windowSizeSpinBox")
    combo = panel.find(QtWidgets.QComboBox, "methodCombo")
    combo.setCurrentIndex(combo.findData(BinarizationMethod.SAUVOLA))
    panel.changed.clear()

    for value in (120, 130, 140, 150):
        spin.setValue(value)

    assert panel.changed == []
    assert panel.reload_timer.interval() == OptionsWidget.RELOAD_DELAY_MS
    panel.let_the_delay_pass()
    assert panel.changed == [PAGE]
    assert not panel.reload_timer.isActive()
    assert panel.filter.get_params(PAGE).binarization.window_size == 150


def test_each_edit_restarts_the_wait(qtbot: QtBot):
    panel = _Panel(qtbot, Params(color_mode=ColorMode.COLOR_GRAYSCALE))
    spin = panel.find(QtWidgets.QSpinBox, "wienerWindowSpinBox")

    spin.setValue(spin.value() + 2)
    spin.setValue(spin.value() + 2)

    assert panel.timer_starts == 2
    assert panel.changed == []
    panel.let_the_delay_pass()
    assert panel.changed == [PAGE]


def test_an_immediate_edit_sends_the_waiting_one_with_it(qtbot: QtBot):
    panel = _Panel(qtbot, Params(color_mode=ColorMode.COLOR_GRAYSCALE))
    panel.find(QtWidgets.QSpinBox, "wienerWindowSpinBox").setValue(9)

    panel.find(QtWidgets.QCheckBox, "fillMarginsCB").click()

    assert panel.changed == [PAGE]
    assert not panel.reload_timer.isActive()
    QtCore.QCoreApplication.processEvents()
    assert panel.changed == [PAGE]


def test_switching_pages_sends_the_waiting_edit_for_the_old_page(qtbot: QtBot):
    panel = _Panel(qtbot, Params(color_mode=ColorMode.COLOR_GRAYSCALE))
    panel.find(QtWidgets.QSpinBox, "wienerWindowSpinBox").setValue(9)
    assert panel.changed == []

    panel.widget.set_current_page(OTHER)

    assert panel.changed == [PAGE]


def test_flushing_with_nothing_waiting_emits_nothing(qtbot: QtBot):
    panel = _Panel(qtbot)

    panel.widget.flush_pending_reload()

    assert panel.changed == []


@pytest.mark.parametrize(
    ("mode", "name", "kind", "value"),
    [
        (ColorMode.COLOR_GRAYSCALE, "wienerCoefSpinBox", QtWidgets.QDoubleSpinBox, 0.4),
        (ColorMode.COLOR_GRAYSCALE, "posterizeLevelSB", QtWidgets.QSpinBox, 5),
        (ColorMode.BLACK_AND_WHITE, "reduceNoiseSB", QtWidgets.QSpinBox, 11),
        (ColorMode.BLACK_AND_WHITE, "redAdjustmentSB", QtWidgets.QSpinBox, 3),
        (ColorMode.BLACK_AND_WHITE, "greenAdjustmentSB", QtWidgets.QSpinBox, 3),
        (ColorMode.BLACK_AND_WHITE, "blueAdjustmentSB", QtWidgets.QSpinBox, 3),
        (ColorMode.BLACK_AND_WHITE, "windowSizeSpinBox", QtWidgets.QSpinBox, 90),
    ],
)
def test_numeric_controls_wait_for_the_delay(
    qtbot: QtBot,
    mode: ColorMode,
    name: str,
    kind: type[QtWidgets.QAbstractSpinBox],
    value: float,
):
    panel = _Panel(
        qtbot,
        Params(
            color_mode=mode,
            binarization=Params().binarization.with_method(BinarizationMethod.SAUVOLA),
        ),
    )

    panel.find(kind, name).setValue(value)  # type: ignore[attr-defined]

    assert panel.changed == []
    panel.widget.flush_pending_reload()
    assert panel.changed == [PAGE]


@pytest.mark.parametrize("name", ["fillMarginsCB", "blackOnWhiteCB"])
def test_toggles_render_at_once(qtbot: QtBot, name: str):
    panel = _Panel(qtbot, Params(color_mode=ColorMode.COLOR_GRAYSCALE))

    panel.find(QtWidgets.QCheckBox, name).click()

    assert panel.changed == [PAGE]
